#!/usr/bin/env python
"""
MIMIC-Multimodal: Master Dataset Generation (DuckDB edition)

Builds one Patient_ICU pickle per unique ICU stay from MIMIC-IV, MIMIC-CXR,
MIMIC-CXR-JPG, and MIMIC-IV-Note source files.

Data structure inspired by:
Soenksen, L. R. et al. Integrated multimodal artificial intelligence framework
for healthcare applications. npj Digit. Med. 5, 149 (2022).
https://physionet.org/content/haim-multimodal/1.0.1/

Source data:
  MIMIC-IV        https://physionet.org/content/mimiciv/3.1/
  MIMIC-CXR       https://physionet.org/content/mimic-cxr/2.1.0/
  MIMIC-CXR-JPG   https://physionet.org/content/mimic-cxr-jpg/2.1.0/
  MIMIC-IV-Note   https://physionet.org/content/mimic-iv-note/2.2/note/

Usage:
    python generate_master_dataset.py \\
        --hosp-path ~/MIMICWorkspace/MIMIC-IV-Parquet/.../hosp/ \\
        --icu-path  ~/MIMICWorkspace/MIMIC-IV-Parquet/.../icu/ \\
        --cxr-path  ~/MIMICWorkspace/MIMIC-CXR/2.1.0/ \\
        --cxr-jpg-path ~/MIMICWorkspace/mimic-cxr-jpg/2.1.0/ \\
        --note-path ~/MIMICWorkspace/MIMIC-IV-Note-Parquet/.../note/ \\
        --output-dir ~/MIMICWorkspace/MasterDataset/

No database file is created -- DuckDB reads your source parquet/csv.gz
files in place via SQL views. The only new files ever written are
list_ids.parquet and cxr_study_datetime.parquet (small derived data that
doesn't exist in the source files). Re-running is safe: those two files
and patient pickles that already exist in --output-dir are all skipped,
so an interrupted run can simply be re-launched to resume.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

from scripts.data_utils import Patient_ICU

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Views: zero-copy access to source folders -- DuckDB never stores a copy
# of this data. Every query reads directly from the original parquet /
# csv.gz files on disk, exactly like your data_utils.py Patient_ICU dict
# structure did, just via SQL instead of dask.
# ---------------------------------------------------------------------------

# Time-like column keywords, applied only to hosp/icu/note tables (matches
# the original convert_datetime heuristic). CXR-JPG's StudyDate/StudyTime
# use a different int/float encoding and are handled separately below.
_TIME_KEYWORDS = ("time", "date", "dod")


def _source_columns(con, read_expr):
    """Cheap column-name lookup without reading the file's data rows."""
    return con.execute(f"SELECT * FROM {read_expr} LIMIT 0").df().columns.tolist()


def create_views_for_folder(con, folder_path, cast_datetimes):
    """Define a SQL VIEW over every .csv.gz / .parquet file in folder_path,
    named after the file stem (mirrors the original dfs[name] dict keys).
    No data is copied or written to disk -- each view reads the source
    file directly at query time."""
    folder_path = Path(folder_path)
    file_list = sorted(os.listdir(folder_path))

    for file_name in tqdm(file_list, desc=f"Register views: {folder_path.name}"):
        file_path = folder_path / file_name

        if file_name.endswith(".csv.gz"):
            table = file_name[: -len(".csv.gz")]
            # quote is specified explicitly rather than left to auto-detection:
            # DuckDB samples only the first ~20k rows to guess CSV dialect, and
            # if that sample happens to contain few/no quoted fields it can
            # conclude quote='' -- then a later row with an embedded comma
            # inside real quotes (e.g. "RIB, UNILAT (NO CXR)") gets split into
            # the wrong number of columns.
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
    columns and persist ONLY the derived column plus keys as a small new
    parquet file -- not a copy of the full metadata table. Everything else
    about a CXR study is still read from the original metadata file via
    its view; this file only adds the one column that doesn't exist yet."""
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
    """Connect an in-memory DuckDB instance and register views over every
    source file. Cheap to call every run -- no data is read or written
    until a query actually touches a view."""
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
    admission, and time window. Equivalent to the original pandasql query,
    executed directly in DuckDB against views over the source files (plus
    the small cxr_study_datetime derived view). icu_info/note_*_info are
    query-only intermediates -- nothing here touches disk except the final
    list_ids parquet, which is new derived data that doesn't exist yet."""
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
        SELECT note_id AS ds_note_id, subject_id, hadm_id, charttime AS ds_charttime
        FROM discharge
    """)
    con.execute("""
        CREATE OR REPLACE VIEW note_rad_info AS
        SELECT note_id AS rad_note_id, subject_id, hadm_id, charttime AS rad_charttime
        FROM radiology
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
        SELECT
            COUNT(DISTINCT subject_id) AS n_patients,
            COUNT(DISTINCT hadm_id) AS n_admissions,
            COUNT(DISTINCT stay_id) AS n_stays,
            COUNT(DISTINCT study_id) AS n_studies,
            COUNT(DISTINCT dicom_id) AS n_images,
            COUNT(DISTINCT ds_note_id) AS n_discharge_notes,
            COUNT(DISTINCT rad_note_id) AS n_radiology_notes
        FROM list_ids
    """).fetchone()
    log.info(
        "list_ids: %d patients, %d admissions, %d ICU stays, %d CXR studies, "
        "%d CXR images, %d discharge notes, %d radiology reports",
        *summary,
    )


# ---------------------------------------------------------------------------
# Per-patient extraction
# ---------------------------------------------------------------------------

def _native(v):
    """Convert a numpy/pandas scalar to a plain Python type. DuckDB's
    parameter binder accepts plain Python int/float/str reliably but can
    raise NotImplementedException on numpy scalar types (numpy.int64,
    numpy.float64, etc.) depending on context -- safest to always convert
    before binding, rather than rely on it working incidentally."""
    return v.item() if hasattr(v, "item") else v


def _native_list(values):
    return [_native(v) for v in values]


def _fetch(con, table, where_sql, params):
    return con.execute(f'SELECT * FROM "{table}" WHERE {where_sql}', _native_list(params)).df()


def _fetch_by_ids(con, table, id_col, ids):
    """Look up rows in a large detail/reference table by a list of foreign
    keys collected from an already-filtered parent table (e.g. emar_detail
    scoped to one patient's emar_id values). Avoids ever loading the full
    detail table into memory."""
    ids = _native_list(i for i in pd.unique(ids) if pd.notna(i))
    if not ids:
        return con.execute(f'SELECT * FROM "{table}" WHERE 1=0').df()
    placeholders = ", ".join(["?"] * len(ids))
    return con.execute(f'SELECT * FROM "{table}" WHERE "{id_col}" IN ({placeholders})', ids).df()


def _fetch_by_id_list(con, table, subject_col, subject_id, id_cols_and_lists):
    """Fetch rows matching subject_id plus one or more IN-list filters,
    e.g. CXR tables filtered by subject_id + study_id (+ dicom_id)."""
    clauses = [f'"{subject_col}" = ?']
    params = [_native(subject_id)]
    for col, values in id_cols_and_lists:
        values = _native_list(v for v in pd.unique(values) if pd.notna(v))
        if not values:
            return con.execute(f'SELECT * FROM "{table}" WHERE 1=0').df()
        clauses.append(f'"{col}" IN ({", ".join(["?"] * len(values))})')
        params.extend(values)
    return con.execute(f'SELECT * FROM "{table}" WHERE {" AND ".join(clauses)}', params).df()


class DictionaryTables:
    """Small tables loaded ONCE as pandas and filtered in-memory per patient,
    instead of querying DuckDB per patient. Two kinds live here:

    - True dictionary/lookup tables (d_items, d_icd_diagnoses, ...) used for
      merges onto every patient's event tables.
    - Small per-patient tables (admissions, patients, icustays, transfers,
      drgcodes, services) that are cheap enough to hold entirely in RAM
      (~hundreds of thousands of rows, tens of MB) -- for these, pandas
      boolean masking on an already-loaded DataFrame beats issuing 30,000+
      separate DuckDB queries, each of which pays fixed per-query overhead.

    Large per-event tables (chartevents, labevents, inputevents, etc.) are
    NOT loaded here -- they stay as DuckDB views queried per patient, since
    they're too large to comfortably fit in memory all at once.
    """

    def __init__(self, con):
        # Dictionary/lookup tables
        self.d_items = con.execute('SELECT * FROM d_items').df()
        self.d_icd_diagnoses = con.execute('SELECT * FROM d_icd_diagnoses').df()
        self.d_icd_procedures = con.execute('SELECT * FROM d_icd_procedures').df()
        self.d_labitems = con.execute('SELECT * FROM d_labitems').df()
        self.d_hcpcs = con.execute('SELECT * FROM d_hcpcs').df()

        # Small per-patient tables -- loaded once, filtered in pandas per stay
        self.admissions = con.execute('SELECT * FROM admissions').df()
        self.patients = con.execute('SELECT * FROM patients').df()
        self.icustays = con.execute('SELECT * FROM icustays').df()
        self.transfers = con.execute('SELECT * FROM transfers').df()
        self.drgcodes = con.execute('SELECT * FROM drgcodes').df()
        self.services = con.execute('SELECT * FROM services').df()


def get_patient_icustay(con, dicts, key_subject_id, key_hadm_id, key_stay_id):
    """Build one Patient_ICU instance for a single ICU stay.

    Small tables (admissions, patients, icustays, transfers, drgcodes,
    services, list_ids' own dictionary lookups) are filtered in pandas
    against DictionaryTables, already preloaded once in memory. Large
    per-event tables (chartevents, labevents, etc.) are queried per patient
    against DuckDB views over the source parquet/csv.gz files, since they're
    too large to hold fully in memory. Output structure matches the
    original Patient_ICU exactly."""

    key_subject_id = _native(key_subject_id)
    key_hadm_id = _native(key_hadm_id)
    key_stay_id = _native(key_stay_id)

    df_core = _fetch(con, "list_ids", "subject_id = ? AND hadm_id = ? AND stay_id = ?",
                      [key_subject_id, key_hadm_id, key_stay_id])

    hosp_where = "subject_id = ? AND hadm_id = ?"
    hosp_params = [key_subject_id, key_hadm_id]

    df_admissions = dicts.admissions[
        (dicts.admissions.subject_id == key_subject_id) & (dicts.admissions.hadm_id == key_hadm_id)
    ]
    df_patients = dicts.patients[dicts.patients.subject_id == key_subject_id]
    df_transfers = dicts.transfers[
        (dicts.transfers.subject_id == key_subject_id) & (dicts.transfers.hadm_id == key_hadm_id)
    ]

    df_diagnoses_icd = _fetch(con, "diagnoses_icd", hosp_where, hosp_params)
    df_diagnoses_icd = df_diagnoses_icd.merge(dicts.d_icd_diagnoses, how="left", on=["icd_code", "icd_version"])

    df_procedures_icd = _fetch(con, "procedures_icd", hosp_where, hosp_params)
    df_procedures_icd = df_procedures_icd.merge(dicts.d_icd_procedures, how="left", on=["icd_code", "icd_version"])

    df_drgcodes = dicts.drgcodes[
        (dicts.drgcodes.subject_id == key_subject_id) & (dicts.drgcodes.hadm_id == key_hadm_id)
    ]
    df_services = dicts.services[
        (dicts.services.subject_id == key_subject_id) & (dicts.services.hadm_id == key_hadm_id)
    ]

    df_labevents = _fetch(con, "labevents", hosp_where, hosp_params)
    df_labevents = df_labevents.merge(dicts.d_labitems, how="left", on="itemid")

    df_hcpcsevents = _fetch(con, "hcpcsevents", hosp_where, hosp_params)
    df_hcpcsevents = df_hcpcsevents.merge(dicts.d_hcpcs, how="left", left_on="hcpcs_cd", right_on="code")

    df_microbiologyevents = _fetch(con, "microbiologyevents", hosp_where, hosp_params)

    df_emar = _fetch(con, "emar", hosp_where, hosp_params)
    df_emar_detail = _fetch_by_ids(con, "emar_detail", "emar_id", df_emar.get("emar_id", []))
    df_emar = df_emar.merge(df_emar_detail, how="left", on="emar_id")

    df_poe = _fetch(con, "poe", hosp_where, hosp_params)
    df_poe_detail = _fetch_by_ids(con, "poe_detail", "poe_id", df_poe.get("poe_id", []))
    df_poe = df_poe.merge(df_poe_detail, how="left", on="poe_id")

    df_prescriptions = _fetch(con, "prescriptions", hosp_where, hosp_params)
    df_pharmacy = _fetch_by_ids(con, "pharmacy", "pharmacy_id", df_prescriptions.get("pharmacy_id", []))
    df_prescriptions = df_prescriptions.merge(df_pharmacy, how="left", on="pharmacy_id")

    icu_where = "subject_id = ? AND hadm_id = ? AND stay_id = ?"
    icu_params = [key_subject_id, key_hadm_id, key_stay_id]

    df_icustays = dicts.icustays[
        (dicts.icustays.subject_id == key_subject_id)
        & (dicts.icustays.hadm_id == key_hadm_id)
        & (dicts.icustays.stay_id == key_stay_id)
    ]
    df_procedureevents = _fetch(con, "procedureevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")
    df_outputevents = _fetch(con, "outputevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")
    df_inputevents = _fetch(con, "inputevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")
    df_datetimeevents = _fetch(con, "datetimeevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")
    df_chartevents = _fetch(con, "chartevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")
    df_ingredientevents = _fetch(con, "ingredientevents", icu_where, icu_params).merge(dicts.d_items, how="left", on="itemid")

    study_id_list = df_core["study_id"]
    dicom_id_list = df_core["dicom_id"]

    df_cxr_image_path = _fetch_by_id_list(
        con, "cxr-record-list", "subject_id", key_subject_id,
        [("study_id", study_id_list), ("dicom_id", dicom_id_list)],
    )
    df_cxr_text_path = _fetch_by_id_list(
        con, "cxr-study-list", "subject_id", key_subject_id,
        [("study_id", study_id_list)],
    )
    study_ids = _native_list(v for v in pd.unique(study_id_list) if pd.notna(v))
    dicom_ids = _native_list(v for v in pd.unique(dicom_id_list) if pd.notna(v))
    if study_ids and dicom_ids:
        placeholders_s = ", ".join(["?"] * len(study_ids))
        placeholders_d = ", ".join(["?"] * len(dicom_ids))
        df_cxr_metadata = con.execute(f"""
            SELECT m.*, d.StudyDatetime
            FROM "mimic-cxr-2.0.0-metadata" m
            LEFT JOIN cxr_study_datetime d
                ON m.subject_id = d.subject_id AND m.study_id = d.study_id AND m.dicom_id = d.dicom_id
            WHERE m.subject_id = ? AND m.study_id IN ({placeholders_s}) AND m.dicom_id IN ({placeholders_d})
        """, [key_subject_id] + study_ids + dicom_ids).df()
    else:
        df_cxr_metadata = con.execute('SELECT *, NULL AS StudyDatetime FROM "mimic-cxr-2.0.0-metadata" WHERE 1=0').df()
    df_cxr_chexpert = _fetch_by_id_list(
        con, "mimic-cxr-2.0.0-chexpert", "subject_id", key_subject_id,
        [("study_id", study_id_list)],
    )
    df_cxr_negbio = _fetch_by_id_list(
        con, "mimic-cxr-2.0.0-negbio", "subject_id", key_subject_id,
        [("study_id", study_id_list)],
    )
    df_cxr_split = _fetch_by_id_list(
        con, "mimic-cxr-2.0.0-split", "subject_id", key_subject_id,
        [("study_id", study_id_list), ("dicom_id", dicom_id_list)],
    )

    df_dsnotes = _fetch(con, "discharge", hosp_where, hosp_params)
    df_dsnotes = df_dsnotes[df_dsnotes["note_id"].isin(df_core["ds_note_id"])]

    df_radnotes = _fetch(con, "radiology", hosp_where, hosp_params)
    df_radnotes = df_radnotes[df_radnotes["note_id"].isin(df_core["rad_note_id"])]
    if len(df_radnotes) > 0:
        df_radiology_detail = _fetch_by_ids(con, "radiology_detail", "note_id", df_radnotes["note_id"])
        df_radnotes = df_radnotes.merge(df_radiology_detail, how="left", on="note_id")

    return Patient_ICU(
        df_core, df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_procedures_icd, df_drgcodes,
        df_services, df_labevents, df_hcpcsevents, df_microbiologyevents, df_emar, df_poe, df_prescriptions,
        df_icustays, df_procedureevents, df_outputevents, df_inputevents, df_datetimeevents, df_chartevents,
        df_ingredientevents,
        df_cxr_split, df_cxr_metadata, df_cxr_chexpert, df_cxr_negbio, df_cxr_image_path, df_cxr_text_path,
        df_dsnotes, df_radnotes,
    )


def generate_master_dataset(con, dicts, key_ids, output_dir):
    """Extract and pickle one Patient_ICU per row in key_ids. Rows whose
    output file already exists are skipped, so an interrupted run can be
    resumed by simply re-launching the script."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(key_ids.itertuples(index=False), total=len(key_ids), desc="Extracting ICU stays"):
        filename = f"ICUstay_{int(row.stay_id)}.pkl"
        out_path = output_dir / filename
        if out_path.exists():
            continue

        icustay = get_patient_icustay(con, dicts, row.subject_id, row.hadm_id, row.stay_id)
        with open(out_path, "wb") as f:
            pickle.dump(icustay, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIMIC-Multimodal master dataset (DuckDB-backed).")
    parser.add_argument("--hosp-path", required=True, type=Path)
    parser.add_argument("--icu-path", required=True, type=Path)
    parser.add_argument("--cxr-path", required=True, type=Path)
    parser.add_argument("--cxr-jpg-path", required=True, type=Path)
    parser.add_argument("--note-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write one ICUstay_<stay_id>.pkl per ICU stay.")
    parser.add_argument("--list-ids-path", default=Path("list_ids.parquet"), type=Path,
                         help="Where to save the subject/admission/stay <-> CXR/note ID mapping "
                              "(new derived data -- not a copy of any source file).")
    parser.add_argument("--cxr-datetime-path", default=Path("cxr_study_datetime.parquet"), type=Path,
                         help="Where to save the derived CXR StudyDatetime column + keys "
                              "(new derived data -- not a copy of the CXR metadata file).")
    parser.add_argument("--rebuild-list-ids", action="store_true",
                         help="Recompute list_ids even if --list-ids-path already exists.")
    parser.add_argument("--start-index", type=int, default=0,
                         help="Start offset into the ICU stay list (for chunked runs).")
    parser.add_argument("--end-index", type=int, default=None,
                         help="End offset into the ICU stay list (default: all).")
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("Registering views over source files (no data copied)")
    con = register_views(
        args.hosp_path, args.icu_path, args.cxr_path, args.cxr_jpg_path, args.note_path,
        args.cxr_datetime_path,
    )

    if args.list_ids_path.exists() and not args.rebuild_list_ids:
        log.info("Reusing existing list_ids at %s", args.list_ids_path)
        con.execute(f"CREATE OR REPLACE VIEW list_ids AS SELECT * FROM read_parquet('{args.list_ids_path.as_posix()}')")
    else:
        log.info("Building list_ids")
        build_list_ids(con, args.list_ids_path)

    key_ids = con.execute(
        "SELECT DISTINCT subject_id, hadm_id, stay_id FROM list_ids ORDER BY subject_id, hadm_id, stay_id"
    ).df()
    key_ids = key_ids.iloc[args.start_index:args.end_index]
    log.info("Extracting %d ICU stays (rows %d:%s) to %s",
             len(key_ids), args.start_index, args.end_index, args.output_dir)

    dicts = DictionaryTables(con)
    generate_master_dataset(con, dicts, key_ids, args.output_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()