#!/usr/bin/env python
"""
MIMIC-Multimodal Step 2: Training Dataset Generation (parquet edition)

Reads Step 1's per-table master dataset (parquet files, one per source
table, already restricted to the ICU-stay cohort) and produces a small
set of training-ready parquet files:

    tabular.parquet          1 row per INCLUDED stay: demographics, age,
                              LOS/mortality/readmission outcome labels
    time_series.parquet      1 row per (stay_id, charttime): pivoted
                              vital signs + frac_charttime, for included
                              stays within their [start, end) window
    cxr_index.parquet        1 row per (stay_id, dicom_id): image/report
                              paths + CheXpert/NegBio labels + StudyDatetime,
                              for included stays within their window
    notes_discharge.parquet  1 row per discharge note, for included stays
                              (not time-windowed -- discharge summaries are
                              written at/after discharge, matching the
                              original pipeline's behavior)
    notes_radiology.parquet  1 row per radiology report, for included
                              stays within their window

Every join, filter, pivot, and the readmission lookahead are single bulk
SQL queries over the whole cohort -- there is no per-patient Python loop
anywhere in this script.

Usage:
    python step2_generate_training_dataset.py \\
        --input-dir ~/MIMICWorkspace/MasterDataset/ \\
        --output-dir ~/MIMICWorkspace/TrainingDataset_24h/ \\
        --age-lower 18 --start-diff 0 --end-diff 24
"""

import argparse
import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_TABULAR_VARIABLES = [
    "subject_id", "hadm_id", "stay_id", "age", "gender", "race",
    "marital_status", "language", "insurance",
]
DEFAULT_VITAL_SIGNS_VARIABLES = [
    "Heart Rate", "Respiratory Rate", "O2 saturation pulseoxymetry",
    "Non Invasive Blood Pressure systolic", "Non Invasive Blood Pressure diastolic",
    "Non Invasive Blood Pressure mean", "Arterial Blood Pressure systolic",
    "Arterial Blood Pressure diastolic", "Arterial Blood Pressure mean",
    "GCS - Eye Opening", "GCS - Verbal Response", "GCS - Motor Response",
    "Temperature Fahrenheit",
]


def register_source_views(con, input_dir):
    """Zero-copy views over Step 1's per-table parquet outputs."""
    input_dir = Path(input_dir)
    for name in [
        "list_ids", "admissions", "patients", "icustays", "chartevents",
        "cxr_metadata", "cxr_image_path", "cxr_text_path", "cxr_chexpert", "cxr_negbio",
        "dsnotes", "radnotes",
    ]:
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Expected Step 1 output not found: {path}")
        con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{path.as_posix()}\')')


# ---------------------------------------------------------------------------
# Stay-level window: demographics, age, and the [start_time, end_time)
# information-collection window for every stay, computed in one bulk query.
# ---------------------------------------------------------------------------

def build_stay_windows(con, start_diff, end_diff, debug_limit=None):
    """One row per ICU stay: demographics + age + metadata + the
    information-collection time window. Vectorized equivalent of the
    original process_patient_ICU's per-patient tab/start_time/end_time
    computation."""

    limit_clause = f"LIMIT {int(debug_limit)}" if debug_limit is not None else ""

    # stay_base MUST be a TABLE, not a VIEW: a view re-evaluates on every
    # referencing query, and `LIMIT n` without ORDER BY has no defined row
    # order -- so a view would hand a DIFFERENT arbitrary subset of stays
    # to the time-series join, the CXR join, the cohort filter, and the
    # final export, and they would not agree on a common cohort.
    # ORDER BY additionally makes the debug subset reproducible.
    con.execute(f"""
        CREATE OR REPLACE TABLE stay_base AS
        SELECT
            a.subject_id, a.hadm_id, i.stay_id,
            a.admission_type, a.admission_location, a.discharge_location,
            i.first_careunit, i.last_careunit,
            a.admittime, a.dischtime, a.edregtime, a.edouttime,
            i.intime, i.outtime, i.los,
            a.hospital_expire_flag, a.deathtime, p.dod,
            p.gender, p.anchor_age, p.anchor_year,
            a.race, a.marital_status, a.language, a.insurance,
            (p.anchor_age + (date_part('year', a.admittime) - p.anchor_year)) AS age,
            (SELECT MIN(t.v) FROM (VALUES (i.intime), (a.admittime), (a.edregtime)) AS t(v)) AS earliest_intime
        FROM icustays i
        JOIN admissions a USING (subject_id, hadm_id)
        JOIN patients p USING (subject_id)
        ORDER BY i.stay_id
        {limit_clause}
    """)

    # start_time / end_time: mirrors the original start_diff/end_diff logic.
    #   start_diff is None      -> earliest_intime
    #   start_diff >= 0         -> intime + start_diff hours (not before earliest_intime)
    #   start_diff <  0         -> intime - |start_diff| hours (not before earliest_intime)
    #   end_diff   is None      -> outtime
    #   end_diff   given        -> intime + end_diff hours (not after outtime)
    if start_diff is None:
        start_expr = "earliest_intime"
    elif start_diff >= 0:
        start_expr = f"GREATEST(intime + INTERVAL '{start_diff} hours', earliest_intime)"
    else:
        start_expr = f"GREATEST(intime - INTERVAL '{abs(start_diff)} hours', earliest_intime)"

    end_expr = "outtime" if end_diff is None else f"LEAST(intime + INTERVAL '{end_diff} hours', outtime)"

    con.execute(f"""
        CREATE OR REPLACE TABLE stay_windows AS
        SELECT *, {start_expr} AS start_time, {end_expr} AS end_time
        FROM stay_base
    """)
    n_stays = con.execute("SELECT COUNT(*) FROM stay_windows").fetchone()[0]
    log.info("stay_windows: %d ICU stays", n_stays)


# ---------------------------------------------------------------------------
# Time series: bulk filter + pivot across the WHOLE cohort in one query.
# ---------------------------------------------------------------------------

def build_time_series(con, vital_signs_variables):
    var_list_sql = ", ".join("'" + v.replace("'", "''") + "'" for v in vital_signs_variables)

    con.execute(f"""
        CREATE OR REPLACE VIEW chart_filtered AS
        SELECT c.hadm_id, c.stay_id, c.charttime, c.label, c.valuenum
        FROM chartevents c
        JOIN stay_windows w USING (hadm_id, stay_id)
        WHERE c.label IN ({var_list_sql})
          AND c.charttime > w.start_time AND c.charttime < w.end_time
    """)

    # DuckDB's PIVOT does the equivalent of the original per-patient
    # df.pivot(index=[...], columns='label', values='valuenum') across
    # every patient in a single pass.
    pivot_cols = ", ".join(f"'{v}'" for v in vital_signs_variables)
    con.execute(f"""
        CREATE OR REPLACE TABLE time_series_pivoted AS
        PIVOT chart_filtered
        ON label IN ({pivot_cols})
        USING FIRST(valuenum)
        GROUP BY hadm_id, stay_id, charttime
    """)

    # Any vital sign never observed anywhere in the cohort won't appear as
    # a column after PIVOT -- add it back as NULL so the schema always
    # matches vital_signs_variables exactly (mirrors the original's
    # `for var in vital_signs_variables: if var not in df_pivot.columns`).
    existing_cols = set(con.execute("SELECT * FROM time_series_pivoted LIMIT 0").df().columns)
    missing = [v for v in vital_signs_variables if v not in existing_cols]
    select_extra = ", ".join(f"NULL AS \"{v}\"" for v in missing)
    select_extra = (", " + select_extra) if select_extra else ""

    con.execute(f"""
        CREATE OR REPLACE VIEW time_series_full AS
        SELECT *{select_extra},
               date_diff('second', MIN(charttime) OVER (PARTITION BY stay_id), charttime) / 3600.0 AS frac_charttime
        FROM time_series_pivoted
    """)


# ---------------------------------------------------------------------------
# CXR: attach stay_id via list_ids, time-window filter, merge in paths/labels.
# ---------------------------------------------------------------------------

def build_cxr_index(con):
    con.execute("""
        CREATE OR REPLACE VIEW cxr_with_stay AS
        SELECT l.stay_id, m.*
        FROM cxr_metadata m
        JOIN list_ids l ON m.subject_id = l.subject_id
            AND m.study_id = l.study_id AND m.dicom_id = l.dicom_id
        WHERE l.stay_id IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE VIEW cxr_windowed AS
        SELECT c.*
        FROM cxr_with_stay c
        JOIN stay_windows w USING (stay_id)
        WHERE c.StudyDatetime > w.start_time AND c.StudyDatetime < w.end_time
    """)
    con.execute("""
        CREATE OR REPLACE VIEW cxr_index_full AS
        SELECT
            c.*,
            ip.path AS image_path,
            tp.path AS text_path,
            ch.* EXCLUDE (subject_id, study_id),
            nb.* EXCLUDE (subject_id, study_id)
        FROM cxr_windowed c
        LEFT JOIN cxr_image_path ip ON c.subject_id = ip.subject_id
            AND c.study_id = ip.study_id AND c.dicom_id = ip.dicom_id
        LEFT JOIN cxr_text_path tp ON c.subject_id = tp.subject_id AND c.study_id = tp.study_id
        LEFT JOIN cxr_chexpert ch ON c.subject_id = ch.subject_id AND c.study_id = ch.study_id
        LEFT JOIN cxr_negbio nb ON c.subject_id = nb.subject_id AND c.study_id = nb.study_id
    """)


# ---------------------------------------------------------------------------
# Notes: attach stay_id via list_ids. Discharge notes are NOT time-windowed
# (matches the original pipeline -- a discharge summary is written at/after
# discharge, so windowing it to an early-ICU collection period would drop
# it entirely). Radiology reports ARE time-windowed.
# ---------------------------------------------------------------------------

def build_notes(con):
    con.execute("""
        CREATE OR REPLACE VIEW notes_discharge_full AS
        SELECT l.stay_id, d.*
        FROM dsnotes d
        JOIN list_ids l ON d.subject_id = l.subject_id AND d.hadm_id = l.hadm_id
            AND d.note_id = l.ds_note_id
        WHERE l.stay_id IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE VIEW radnotes_with_stay AS
        SELECT l.stay_id, r.*
        FROM radnotes r
        JOIN list_ids l ON r.subject_id = l.subject_id AND r.hadm_id = l.hadm_id
            AND r.note_id = l.rad_note_id
        WHERE l.stay_id IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE VIEW notes_radiology_full AS
        SELECT r.*
        FROM radnotes_with_stay r
        JOIN stay_windows w USING (stay_id)
        WHERE r.charttime > w.start_time AND r.charttime < w.end_time
    """)


# ---------------------------------------------------------------------------
# Cohort selection: age, LOS, and missing-modality flags, all vectorized.
# ---------------------------------------------------------------------------

def build_cohort(con, age_lower, age_upper, los_lower, drop_missing_ts, drop_missing_img, drop_missing_text):
    con.execute("""
        CREATE OR REPLACE VIEW modality_presence AS
        SELECT
            w.stay_id,
            (SELECT COUNT(*) FROM time_series_full t WHERE t.stay_id = w.stay_id) AS n_ts,
            (SELECT COUNT(*) FROM cxr_index_full c WHERE c.stay_id = w.stay_id) AS n_img,
            (
                (SELECT COUNT(*) FROM notes_discharge_full d WHERE d.stay_id = w.stay_id)
                + (SELECT COUNT(*) FROM notes_radiology_full r WHERE r.stay_id = w.stay_id)
            ) AS n_text
        FROM stay_windows w
    """)

    ts_clause = "AND m.n_ts > 0" if drop_missing_ts else ""
    img_clause = "AND m.n_img > 0" if drop_missing_img else ""
    text_clause = "AND m.n_text > 0" if drop_missing_text else ""

    con.execute(f"""
        CREATE OR REPLACE VIEW included_stays AS
        SELECT w.stay_id
        FROM stay_windows w
        JOIN modality_presence m USING (stay_id)
        WHERE w.age BETWEEN {age_lower} AND {age_upper}
          AND w.los >= {los_lower}
          {ts_clause} {img_clause} {text_clause}
    """)

    n_total, n_included = con.execute(
        "SELECT (SELECT COUNT(*) FROM stay_windows), (SELECT COUNT(*) FROM included_stays)"
    ).fetchone()
    log.info("Cohort selection: %d / %d stays included", n_included, n_total)


# ---------------------------------------------------------------------------
# Outcomes: mortality, LOS binary, and readmission -- all set-based, no loop.
# ---------------------------------------------------------------------------

def build_outcomes(con, los_range, readmission_range):
    con.execute(f"""
        CREATE OR REPLACE VIEW outcomes AS
        SELECT
            w.stay_id,
            CASE
                WHEN w.hospital_expire_flag = 0 THEN 0
                WHEN w.deathtime IS NOT NULL THEN
                    CASE WHEN w.deathtime > w.intime AND w.deathtime < w.outtime THEN 1 ELSE 0 END
                ELSE
                    CASE WHEN w.dod > w.intime AND w.dod < w.outtime THEN 1 ELSE 0 END
            END AS icu_expire_flag,
            CASE WHEN w.los > {los_range} THEN 1 ELSE 0 END AS los_binary,
            CASE WHEN EXISTS (
                SELECT 1 FROM stay_windows b
                WHERE b.subject_id = w.subject_id
                  AND b.intime > w.intime
                  AND b.intime <= w.outtime + INTERVAL '{readmission_range} days'
            ) THEN 1 ELSE 0 END AS readmission
        FROM stay_windows w
    """)


# ---------------------------------------------------------------------------
# Final export: restrict every table to included_stays and write parquet.
# ---------------------------------------------------------------------------

def export_training_dataset(con, output_dir, tabular_variables):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tab_cols = ", ".join(f'w."{c}"' for c in tabular_variables if c not in ("subject_id", "hadm_id", "stay_id"))
    tab_cols = "w.subject_id, w.hadm_id, w.stay_id" + (", " + tab_cols if tab_cols else "")

    con.execute(f"""
        COPY (
            SELECT {tab_cols},
                   w.admission_type, w.admission_location, w.discharge_location,
                   w.first_careunit, w.last_careunit,
                   w.admittime, w.dischtime, w.edregtime, w.edouttime, w.intime, w.outtime,
                   w.los, w.hospital_expire_flag, w.deathtime, w.dod,
                   o.icu_expire_flag, o.los_binary, o.readmission
            FROM stay_windows w
            JOIN included_stays i USING (stay_id)
            JOIN outcomes o USING (stay_id)
        ) TO '{(output_dir / "tabular.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT t.* FROM time_series_full t
            JOIN included_stays i USING (stay_id)
            ORDER BY stay_id, charttime
        ) TO '{(output_dir / "time_series.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT c.* FROM cxr_index_full c
            JOIN included_stays i USING (stay_id)
            ORDER BY stay_id, StudyDatetime
        ) TO '{(output_dir / "cxr_index.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT d.* FROM notes_discharge_full d
            JOIN included_stays i USING (stay_id)
        ) TO '{(output_dir / "notes_discharge.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT r.* FROM notes_radiology_full r
            JOIN included_stays i USING (stay_id)
            ORDER BY stay_id, charttime
        ) TO '{(output_dir / "notes_radiology.parquet").as_posix()}' (FORMAT PARQUET)
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-Multimodal Step 2: training dataset generation (parquet).")
    parser.add_argument("--input-dir", required=True, type=Path, help="Step 1 output directory.")
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--age-lower", type=int, default=0)
    parser.add_argument("--age-upper", type=int, default=150)
    parser.add_argument("--drop-missing-ts", action="store_true")
    parser.add_argument("--drop-missing-img", action="store_true")
    parser.add_argument("--drop-missing-text", action="store_true")
    parser.add_argument("--los-lower", type=int, default=0)

    parser.add_argument("--start-diff", type=int, default=None)
    parser.add_argument("--end-diff", type=int, default=None)
    parser.add_argument("--tabular-variables", nargs="+", default=DEFAULT_TABULAR_VARIABLES)
    parser.add_argument("--vital-signs-variables", nargs="+", default=DEFAULT_VITAL_SIGNS_VARIABLES)

    parser.add_argument("--los-range", type=int, default=3)
    parser.add_argument("--readmission-range", type=int, default=30)
    parser.add_argument("--debug-limit", type=int, default=None,
                         help="Restrict to the first N ICU stays, for a quick end-to-end test run.")
    return parser.parse_args()


def main():
    args = parse_args()
    con = duckdb.connect(":memory:")

    log.info("Registering views over Step 1 outputs")
    register_source_views(con, args.input_dir)

    log.info("Building stay windows")
    build_stay_windows(con, args.start_diff, args.end_diff, args.debug_limit)

    log.info("Building time series (pivot)")
    build_time_series(con, args.vital_signs_variables)

    log.info("Building CXR index")
    build_cxr_index(con)

    log.info("Building notes")
    build_notes(con)

    log.info("Selecting cohort")
    build_cohort(con, args.age_lower, args.age_upper, args.los_lower,
                 args.drop_missing_ts, args.drop_missing_img, args.drop_missing_text)

    log.info("Computing outcomes")
    build_outcomes(con, args.los_range, args.readmission_range)

    log.info("Exporting training dataset to %s", args.output_dir)
    export_training_dataset(con, args.output_dir, args.tabular_variables)

    log.info("Done.")


if __name__ == "__main__":
    main()