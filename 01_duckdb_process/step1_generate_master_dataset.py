#!/usr/bin/env python
"""
MIMIC-Multimodal Step 1: Master Dataset Generation (parquet edition)

Builds one parquet file per source table, restricted to the ICU-stay
cohort defined by list_ids, and merged with the relevant dictionary
tables (d_items, d_icd_diagnoses, etc.). This replaces the old per-stay
Patient_ICU pickle output: every join here is a single bulk SQL query
across the whole cohort, run once per table -- there is no per-patient
Python loop, because DuckDB can hash-join the small cohort key list
against each large source file in one pass.

Output layout (all under --output-dir):
    list_ids.parquet            <- the stay <-> CXR study <-> note ID map
                                    (also serves as the "core" table)
    cxr_study_datetime.parquet  <- derived CXR StudyDatetime + keys
    admissions.parquet
    patients.parquet
    transfers.parquet
    diagnoses_icd.parquet       <- merged with d_icd_diagnoses
    procedures_icd.parquet      <- merged with d_icd_procedures
    drgcodes.parquet
    services.parquet
    labevents.parquet           <- merged with d_labitems
    hcpcsevents.parquet         <- merged with d_hcpcs
    microbiologyevents.parquet
    emar.parquet                <- merged with emar_detail
    poe.parquet                 <- merged with poe_detail
    prescriptions.parquet       <- merged with pharmacy
    icustays.parquet
    procedureevents.parquet     <- merged with d_items
    outputevents.parquet        <- merged with d_items
    inputevents.parquet         <- merged with d_items
    datetimeevents.parquet      <- merged with d_items
    chartevents.parquet         <- merged with d_items
    ingredientevents.parquet    <- merged with d_items
    cxr_image_path.parquet
    cxr_text_path.parquet
    cxr_metadata.parquet        <- includes StudyDatetime
    cxr_chexpert.parquet
    cxr_negbio.parquet
    cxr_split.parquet
    dsnotes.parquet
    radnotes.parquet            <- merged with radiology_detail

None of the original source files are copied or duplicated -- every
query here reads directly from the source parquet/csv.gz files via
DuckDB views, and only the join RESULT (restricted to the cohort) is
written to disk.

Source data:
  MIMIC-IV        https://physionet.org/content/mimiciv/3.1/
  MIMIC-CXR       https://physionet.org/content/mimic-cxr/2.1.0/
  MIMIC-CXR-JPG   https://physionet.org/content/mimic-cxr-jpg/2.1.0/
  MIMIC-IV-Note   https://physionet.org/content/mimic-iv-note/2.2/note/

Usage:
    python step1_generate_master_dataset.py \\
        --hosp-path ~/MIMICWorkspace/MIMIC-IV-Parquet/.../hosp/ \\
        --icu-path  ~/MIMICWorkspace/MIMIC-IV-Parquet/.../icu/ \\
        --cxr-path  ~/MIMICWorkspace/MIMIC-CXR/2.1.0/ \\
        --cxr-jpg-path ~/MIMICWorkspace/mimic-cxr-jpg/2.1.0/ \\
        --note-path ~/MIMICWorkspace/MIMIC-IV-Note-Parquet/.../note/ \\
        --output-dir ~/MIMICWorkspace/MasterDataset/

Re-running is safe: each output parquet file is skipped if it already
exists, unless --overwrite is passed.
"""

import argparse
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Views: zero-copy access to source folders. DuckDB never stores a copy of
# this data -- every query reads directly from the original parquet /
# csv.gz files on disk.
# ---------------------------------------------------------------------------

_TIME_KEYWORDS = ("time", "date", "dod")


def _source_columns(con, read_expr):
    return con.execute(f"SELECT * FROM {read_expr} LIMIT 0").df().columns.tolist()


def create_views_for_folder(con, folder_path, cast_datetimes):
    """Define a SQL VIEW over every .csv.gz / .parquet file in folder_path,
    named after the file stem. No data is copied to disk."""
    folder_path = Path(folder_path)
    file_list = sorted(os.listdir(folder_path))

    for file_name in tqdm(file_list, desc=f"Register views: {folder_path.name}"):
        file_path = folder_path / file_name

        if file_name.endswith(".csv.gz"):
            table = file_name[: -len(".csv.gz")]
            # quote/escape specified explicitly -- DuckDB's dialect sniffer
            # samples only ~20k rows and can misdetect quote='' if that
            # sample happens to contain few quoted fields, corrupting any
            # later row with an embedded comma inside real quotes.
            read_expr = (
                f"read_csv_auto('{file_path.as_posix()}', compression='gzip', "
                f"quote='\"', escape='\"')"
            )
        elif file_name.endswith(".parquet"):
            table = file_name[: -len(".parquet")]
            read_expr = f"read_parquet('{file_path.as_posix()}')"
        else:
            continue

        if cast_datetimes:
            cols = _source_columns(con, read_expr)
            to_cast = [c for c in cols if any(w in c.lower() for w in _TIME_KEYWORDS)]
            select_parts = [
                f'TRY_CAST("{c}" AS TIMESTAMP) AS "{c}"' if c in to_cast else f'"{c}"'
                for c in cols
            ]
            select_sql = ", ".join(select_parts)
        else:
            select_sql = "*"

        con.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT {select_sql} FROM {read_expr}')


def build_cxr_study_datetime(con, cxr_datetime_path):
    """Compute StudyDatetime from CXR-JPG's int-encoded StudyDate/StudyTime
    columns. Persists ONLY the derived column plus keys -- not a copy of
    the full metadata table."""
    if cxr_datetime_path.exists():
        log.info("Reusing existing cxr_study_datetime at %s", cxr_datetime_path)
    else:
        log.info("Computing cxr_study_datetime -> %s", cxr_datetime_path)
        df = con.execute(
            'SELECT subject_id, study_id, dicom_id, "StudyDate", "StudyTime" FROM "mimic-cxr-2.0.0-metadata"'
        ).df()
        df["StudyDate"] = df["StudyDate"].astype(int)
        study_date = pd.to_datetime(df["StudyDate"], format="%Y%m%d")
        study_time_str = df["StudyTime"].apply(lambda t: "%#010.3f" % t)
        study_time_str = pd.to_datetime(study_time_str, format="%H%M%S.%f").dt.strftime("%H%M%S")
        study_time = pd.to_datetime(study_time_str, format="%H%M%S").dt.time
        study_datetime = [pd.Timestamp.combine(d, t) for d, t in zip(study_date, study_time)]

        out = df[["subject_id", "study_id", "dicom_id"]].copy()
        out["StudyDatetime"] = study_datetime

        cxr_datetime_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cxr_datetime_path, index=False)

    con.execute(f"""
        CREATE OR REPLACE VIEW cxr_study_datetime AS
        SELECT * FROM read_parquet('{cxr_datetime_path.as_posix()}')
    """)


def register_views(hosp_path, icu_path, cxr_path, cxr_jpg_path, note_path, cxr_datetime_path):
    con = duckdb.connect(":memory:")
    create_views_for_folder(con, hosp_path, cast_datetimes=True)
    create_views_for_folder(con, icu_path, cast_datetimes=True)
    create_views_for_folder(con, cxr_path, cast_datetimes=False)
    create_views_for_folder(con, cxr_jpg_path, cast_datetimes=False)
    create_views_for_folder(con, note_path, cast_datetimes=True)
    build_cxr_study_datetime(con, cxr_datetime_path)
    return con


# ---------------------------------------------------------------------------
# ID combination: which ICU stays link to which CXR studies / notes
# ---------------------------------------------------------------------------

def build_list_ids(con, list_ids_path):
    """Join ICU stays against CXR studies and clinical notes by subject,
    admission, and time window. icu_info/note_*_info are query-only
    intermediates; only the final list_ids result is written to disk."""
    con.execute("""
        CREATE OR REPLACE VIEW icu_info AS
        SELECT
            i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
            a.admittime, a.dischtime, a.edregtime, a.edouttime,
            (SELECT MIN(t.v) FROM (VALUES (i.intime), (a.admittime), (a.edregtime)) AS t(v)) AS earliest_intime
        FROM icustays i
        LEFT JOIN admissions a ON i.subject_id = a.subject_id AND i.hadm_id = a.hadm_id
    """)
    con.execute("""
        CREATE OR REPLACE VIEW note_ds_info AS
        SELECT note_id AS ds_note_id, subject_id, hadm_id, charttime AS ds_charttime FROM discharge
    """)
    con.execute("""
        CREATE OR REPLACE VIEW note_rad_info AS
        SELECT note_id AS rad_note_id, subject_id, hadm_id, charttime AS rad_charttime FROM radiology
    """)

    list_ids_df = con.execute("""
        SELECT DISTINCT
            i.subject_id, i.hadm_id, i.stay_id,
            c.study_id, c.dicom_id, ds.ds_note_id, rad.rad_note_id
        FROM icu_info i
        LEFT JOIN cxr_study_datetime c
            ON i.subject_id = c.subject_id
            AND c.StudyDatetime >= i.earliest_intime AND c.StudyDatetime <= i.outtime
        LEFT JOIN note_ds_info ds
            ON i.subject_id = ds.subject_id AND i.hadm_id = ds.hadm_id
        LEFT JOIN note_rad_info rad
            ON i.subject_id = rad.subject_id AND i.hadm_id = rad.hadm_id
            AND rad.rad_charttime >= i.earliest_intime AND rad.rad_charttime <= i.outtime
    """).df()

    list_ids_path.parent.mkdir(parents=True, exist_ok=True)
    list_ids_df.to_parquet(list_ids_path, index=False)
    con.execute(f"CREATE OR REPLACE VIEW list_ids AS SELECT * FROM read_parquet('{list_ids_path.as_posix()}')")

    summary = con.execute("""
        SELECT COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id), COUNT(DISTINCT stay_id),
               COUNT(DISTINCT study_id), COUNT(DISTINCT dicom_id),
               COUNT(DISTINCT ds_note_id), COUNT(DISTINCT rad_note_id)
        FROM list_ids
    """).fetchone()
    log.info(
        "list_ids: %d patients, %d admissions, %d ICU stays, %d CXR studies, "
        "%d CXR images, %d discharge notes, %d radiology reports", *summary,
    )


# ---------------------------------------------------------------------------
# Bulk per-table export: one SQL query per table, restricted to the cohort,
# written straight to parquet. No per-patient loop anywhere below.
# ---------------------------------------------------------------------------

def _export(con, name, sql, output_dir, overwrite):
    out_path = output_dir / f"{name}.parquet"
    if out_path.exists() and not overwrite:
        log.info("Skip %s (exists)", name)
        return
    log.info("Exporting %s", name)
    con.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")


def export_master_dataset(con, output_dir, overwrite, debug_limit=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicated cohort key tables at each grain -- joining against these
    # (rather than list_ids directly) avoids row duplication from list_ids'
    # one-row-per-(stay, study, note) granularity.
    #
    # These MUST be TABLEs, not VIEWs: a view re-evaluates on every query
    # that references it, and `DISTINCT ... LIMIT n` without ORDER BY has
    # no defined row order -- so a view would hand a DIFFERENT arbitrary
    # subset to each export, and the per-table outputs would not share a
    # common cohort. ORDER BY additionally makes the debug subset
    # reproducible across runs.
    limit_clause = f"LIMIT {int(debug_limit)}" if debug_limit is not None else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE key_stay AS
        SELECT DISTINCT subject_id, hadm_id, stay_id FROM list_ids
        ORDER BY subject_id, hadm_id, stay_id {limit_clause}
    """)
    con.execute("CREATE OR REPLACE TABLE key_hadm AS SELECT DISTINCT subject_id, hadm_id FROM key_stay")
    con.execute("CREATE OR REPLACE TABLE key_subject AS SELECT DISTINCT subject_id FROM key_stay")
    # Scoped copy of list_ids restricted to the selected cohort. Also a
    # TABLE, for the same determinism reason as above.
    con.execute("""
        CREATE OR REPLACE TABLE list_ids_scope AS
        SELECT l.* FROM list_ids l JOIN key_stay k USING (subject_id, hadm_id, stay_id)
    """)
    n_stays = con.execute("SELECT COUNT(*) FROM key_stay").fetchone()[0]
    log.info("Cohort for export: %d ICU stays", n_stays)

    # ---- Hosp: subject-level ----
    _export(con, "patients", """
        SELECT p.* FROM patients p JOIN key_subject k USING (subject_id)
    """, output_dir, overwrite)

    # ---- Hosp: hadm-level ----
    _export(con, "admissions", """
        SELECT a.* FROM admissions a JOIN key_hadm k USING (subject_id, hadm_id)
    """, output_dir, overwrite)
    _export(con, "transfers", """
        SELECT t.* FROM transfers t JOIN key_hadm k USING (subject_id, hadm_id)
    """, output_dir, overwrite)
    _export(con, "diagnoses_icd", """
        SELECT d.*, dx.long_title
        FROM diagnoses_icd d
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN d_icd_diagnoses dx USING (icd_code, icd_version)
    """, output_dir, overwrite)
    _export(con, "procedures_icd", """
        SELECT p.*, dp.long_title
        FROM procedures_icd p
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN d_icd_procedures dp USING (icd_code, icd_version)
    """, output_dir, overwrite)
    _export(con, "drgcodes", """
        SELECT d.* FROM drgcodes d JOIN key_hadm k USING (subject_id, hadm_id)
    """, output_dir, overwrite)
    _export(con, "services", """
        SELECT s.* FROM services s JOIN key_hadm k USING (subject_id, hadm_id)
    """, output_dir, overwrite)
    _export(con, "labevents", """
        SELECT l.*, li.label AS lab_label, li.fluid, li.category
        FROM labevents l
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN d_labitems li USING (itemid)
    """, output_dir, overwrite)
    _export(con, "hcpcsevents", """
        SELECT h.*, hc.* EXCLUDE (code)
        FROM hcpcsevents h
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN d_hcpcs hc ON h.hcpcs_cd = hc.code
    """, output_dir, overwrite)
    _export(con, "microbiologyevents", """
        SELECT m.* FROM microbiologyevents m JOIN key_hadm k USING (subject_id, hadm_id)
    """, output_dir, overwrite)
    _export(con, "emar", """
        SELECT e.*, ed.* EXCLUDE (emar_id, subject_id)
        FROM emar e
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN emar_detail ed USING (emar_id)
    """, output_dir, overwrite)
    _export(con, "poe", """
        SELECT p.*, pd.* EXCLUDE (poe_id, subject_id)
        FROM poe p
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN poe_detail pd USING (poe_id)
    """, output_dir, overwrite)
    _export(con, "prescriptions", """
        SELECT rx.*, ph.* EXCLUDE (pharmacy_id, subject_id)
        FROM prescriptions rx
        JOIN key_hadm k USING (subject_id, hadm_id)
        LEFT JOIN pharmacy ph USING (pharmacy_id)
    """, output_dir, overwrite)

    # ---- ICU: stay-level ----
    _export(con, "icustays", """
        SELECT i.* FROM icustays i JOIN key_stay k USING (subject_id, hadm_id, stay_id)
    """, output_dir, overwrite)
    for table in ["procedureevents", "outputevents", "inputevents", "datetimeevents",
                  "chartevents", "ingredientevents"]:
        _export(con, table, f"""
            SELECT e.*, di.label, di.category, di.unitname
            FROM {table} e
            JOIN key_stay k USING (subject_id, hadm_id, stay_id)
            LEFT JOIN d_items di USING (itemid)
        """, output_dir, overwrite)

    # ---- CXR: joined against list_ids_scope directly (already scoped by
    # study_id/dicom_id), deduplicated with DISTINCT ----
    _export(con, "cxr_image_path", """
        SELECT DISTINCT r.*
        FROM "cxr-record-list" r
        JOIN list_ids_scope l ON r.subject_id = l.subject_id
            AND r.study_id = l.study_id AND r.dicom_id = l.dicom_id
    """, output_dir, overwrite)
    _export(con, "cxr_text_path", """
        SELECT DISTINCT s.*
        FROM "cxr-study-list" s
        JOIN list_ids_scope l ON s.subject_id = l.subject_id AND s.study_id = l.study_id
    """, output_dir, overwrite)
    _export(con, "cxr_metadata", """
        SELECT DISTINCT m.*, d.StudyDatetime
        FROM "mimic-cxr-2.0.0-metadata" m
        JOIN list_ids_scope l ON m.subject_id = l.subject_id
            AND m.study_id = l.study_id AND m.dicom_id = l.dicom_id
        LEFT JOIN cxr_study_datetime d ON m.subject_id = d.subject_id
            AND m.study_id = d.study_id AND m.dicom_id = d.dicom_id
    """, output_dir, overwrite)
    _export(con, "cxr_chexpert", """
        SELECT DISTINCT c.*
        FROM "mimic-cxr-2.0.0-chexpert" c
        JOIN list_ids_scope l ON c.subject_id = l.subject_id AND c.study_id = l.study_id
    """, output_dir, overwrite)
    _export(con, "cxr_negbio", """
        SELECT DISTINCT n.*
        FROM "mimic-cxr-2.0.0-negbio" n
        JOIN list_ids_scope l ON n.subject_id = l.subject_id AND n.study_id = l.study_id
    """, output_dir, overwrite)
    _export(con, "cxr_split", """
        SELECT DISTINCT sp.*
        FROM "mimic-cxr-2.0.0-split" sp
        JOIN list_ids_scope l ON sp.subject_id = l.subject_id
            AND sp.study_id = l.study_id AND sp.dicom_id = l.dicom_id
    """, output_dir, overwrite)

    # ---- Notes: joined against list_ids_scope's specific note_id link ----
    _export(con, "dsnotes", """
        SELECT DISTINCT ds.*
        FROM discharge ds
        JOIN list_ids_scope l ON ds.subject_id = l.subject_id AND ds.hadm_id = l.hadm_id
            AND ds.note_id = l.ds_note_id
    """, output_dir, overwrite)
    _export(con, "radnotes", """
        SELECT DISTINCT rad.*, rd.* EXCLUDE (note_id, subject_id)
        FROM radiology rad
        JOIN list_ids_scope l ON rad.subject_id = l.subject_id AND rad.hadm_id = l.hadm_id
            AND rad.note_id = l.rad_note_id
        LEFT JOIN radiology_detail rd USING (note_id)
    """, output_dir, overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC-Multimodal Step 1: master dataset generation (parquet).")
    parser.add_argument("--hosp-path", required=True, type=Path)
    parser.add_argument("--icu-path", required=True, type=Path)
    parser.add_argument("--cxr-path", required=True, type=Path)
    parser.add_argument("--cxr-jpg-path", required=True, type=Path)
    parser.add_argument("--note-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write one parquet file per table.")
    parser.add_argument("--list-ids-path", default=None, type=Path,
                         help="Defaults to <output-dir>/list_ids.parquet.")
    parser.add_argument("--cxr-datetime-path", default=None, type=Path,
                         help="Defaults to <output-dir>/cxr_study_datetime.parquet.")
    parser.add_argument("--rebuild-list-ids", action="store_true",
                         help="Recompute list_ids even if it already exists.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-export a table's parquet file even if it already exists.")
    parser.add_argument("--debug-limit", type=int, default=None,
                         help="Restrict to the first N ICU stays, for a quick end-to-end test run.")
    return parser.parse_args()


def main():
    args = parse_args()
    list_ids_path = args.list_ids_path or (args.output_dir / "list_ids.parquet")
    cxr_datetime_path = args.cxr_datetime_path or (args.output_dir / "cxr_study_datetime.parquet")

    log.info("Registering views over source files (no data copied)")
    con = register_views(
        args.hosp_path, args.icu_path, args.cxr_path, args.cxr_jpg_path, args.note_path,
        cxr_datetime_path,
    )

    if list_ids_path.exists() and not args.rebuild_list_ids:
        log.info("Reusing existing list_ids at %s", list_ids_path)
        con.execute(f"CREATE OR REPLACE VIEW list_ids AS SELECT * FROM read_parquet('{list_ids_path.as_posix()}')")
    else:
        log.info("Building list_ids")
        build_list_ids(con, list_ids_path)

    export_master_dataset(con, args.output_dir, args.overwrite, args.debug_limit)
    log.info("Done. Master dataset written to %s", args.output_dir)


if __name__ == "__main__":
    main()