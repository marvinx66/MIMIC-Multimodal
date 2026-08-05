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
python generate_master_dataset.py \
  --hosp-path ~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/hosp/ \
  --icu-path ~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/icu/ \
  --cxr-path ~/MIMICWorkspace/MIMIC-CXR/2.1.0/ \
  --cxr-jpg-path ~/MIMICWorkspace/mimic-cxr-jpg/2.1.0/ \
  --note-path ~/MIMICWorkspace/MIMIC-IV-Note-Parquet/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/ \
  --db-path ~/MIMICWorkspace/mimic.duckdb \
  --output-dir ~/MIMICWorkspace/MasterDataset/

Re-running is safe: source folders are only re-ingested if --rebuild-db is
passed, and patient pickles that already exist in --output-dir are skipped
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
# Ingestion: source folders -> indexed DuckDB tables
# ---------------------------------------------------------------------------

# Time-like column keywords, applied only to hosp/icu/note tables (matches
# the original convert_datetime heuristic). CXR-JPG's StudyDate/StudyTime
# use a different int/float encoding and are handled separately below.
_TIME_KEYWORDS = ("time", "date", "dod")

# subject_id / hadm_id / stay_id columns used for per-patient lookups.
# Only these get indexed -- keeps ingestion fast.
_HOSP_INDEX_COLS = {
    "admissions": ["subject_id", "hadm_id"],
    "patients": ["subject_id"],
    "transfers": ["subject_id", "hadm_id"],
    "diagnoses_icd": ["subject_id", "hadm_id"],
    "procedures_icd": ["subject_id", "hadm_id"],
    "drgcodes": ["subject_id", "hadm_id"],
    "services": ["subject_id", "hadm_id"],
    "labevents": ["subject_id", "hadm_id"],
    "hcpcsevents": ["subject_id", "hadm_id"],
    "microbiologyevents": ["subject_id", "hadm_id"],
    "emar": ["subject_id", "hadm_id"],
    "emar_detail": ["emar_id"],
    "poe": ["subject_id", "hadm_id"],
    "poe_detail": ["poe_id"],
    "prescriptions": ["subject_id", "hadm_id"],
    "pharmacy": ["pharmacy_id"],
}
_ICU_INDEX_COLS = {
    "icustays": ["subject_id", "hadm_id", "stay_id"],
    "procedureevents": ["subject_id", "hadm_id", "stay_id"],
    "outputevents": ["subject_id", "hadm_id", "stay_id"],
    "inputevents": ["subject_id", "hadm_id", "stay_id"],
    "datetimeevents": ["subject_id", "hadm_id", "stay_id"],
    "chartevents": ["subject_id", "hadm_id", "stay_id"],
    "ingredientevents": ["subject_id", "hadm_id", "stay_id"],
}
_CXR_INDEX_COLS = {
    "cxr-record-list": ["subject_id", "study_id", "dicom_id"],
    "cxr-study-list": ["subject_id", "study_id"],
}
_CXR_JPG_INDEX_COLS = {
    "mimic-cxr-2.0.0-metadata": ["subject_id", "study_id", "dicom_id"],
    "mimic-cxr-2.0.0-chexpert": ["subject_id", "study_id"],
    "mimic-cxr-2.0.0-negbio": ["subject_id", "study_id"],
    "mimic-cxr-2.0.0-split": ["subject_id", "study_id", "dicom_id"],
}
_NOTE_INDEX_COLS = {
    "discharge": ["subject_id", "hadm_id", "note_id"],
    "radiology": ["subject_id", "hadm_id", "note_id"],
    "radiology_detail": ["note_id"],
}


def _cast_time_columns(con, table):
    """Cast columns whose name suggests a date/time value to TIMESTAMP,
    matching the original per-folder convert_datetime() heuristic."""
    cols = con.execute(f'PRAGMA table_info("{table}")').fetchdf()["name"].tolist()
    to_cast = [c for c in cols if any(w in c.lower() for w in _TIME_KEYWORDS)]
    if not to_cast:
        return
    select_parts = [
        f'TRY_CAST("{c}" AS TIMESTAMP) AS "{c}"' if c in to_cast else f'"{c}"'
        for c in cols
    ]
    con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT {", ".join(select_parts)} FROM "{table}"')


def ingest_folder(con, folder_path, index_cols, cast_datetimes):
    """Read every .csv.gz / .parquet file in folder_path into a DuckDB
    table named after the file stem, then index and (optionally) cast
    time-like columns."""
    folder_path = Path(folder_path)
    file_list = sorted(os.listdir(folder_path))

    for file_name in tqdm(file_list, desc=f"Ingest {folder_path.name}"):
        file_path = folder_path / file_name

        if file_name.endswith(".csv.gz"):
            table = file_name[: -len(".csv.gz")]
            con.execute(f"""
                CREATE OR REPLACE TABLE "{table}" AS
                SELECT * FROM read_csv_auto('{file_path.as_posix()}', compression='gzip')
            """)
        elif file_name.endswith(".parquet"):
            table = file_name[: -len(".parquet")]
            con.execute(f"""
                CREATE OR REPLACE TABLE "{table}" AS
                SELECT * FROM read_parquet('{file_path.as_posix()}')
            """)
        else:
            continue

        if cast_datetimes:
            _cast_time_columns(con, table)

        cols = index_cols.get(table)
        if cols:
            idx_name = f"idx_{table.replace('-', '_')}_{'_'.join(cols)}"
            con.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON "{table}" ({", ".join(cols)})')


def fix_cxr_metadata_datetime(con):
    """Build StudyDatetime from CXR-JPG's int-encoded StudyDate/StudyTime
    columns. Done in pandas (small table, ~fixed-size per MIMIC-CXR-JPG
    release) then written back to DuckDB."""
    df = con.execute('SELECT * FROM "mimic-cxr-2.0.0-metadata"').df()
    df["StudyDate"] = df["StudyDate"].astype(int)
    df["StudyDate"] = pd.to_datetime(df["StudyDate"], format="%Y%m%d")
    df["StudyTime"] = df["StudyTime"].apply(lambda t: "%#010.3f" % t)
    df["StudyTime"] = pd.to_datetime(df["StudyTime"], format="%H%M%S.%f").dt.strftime("%H%M%S")
    df["StudyTime"] = pd.to_datetime(df["StudyTime"], format="%H%M%S").dt.time
    df["StudyDatetime"] = df.apply(lambda r: pd.Timestamp.combine(r["StudyDate"], r["StudyTime"]), axis=1)

    con.register("cxr_metadata_fixed", df)
    con.execute('CREATE OR REPLACE TABLE "mimic-cxr-2.0.0-metadata" AS SELECT * FROM cxr_metadata_fixed')
    con.unregister("cxr_metadata_fixed")

    idx_name = "idx_cxr_metadata_subject_study_dicom"
    con.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON "mimic-cxr-2.0.0-metadata" (subject_id, study_id, dicom_id)')


def build_database(db_path, hosp_path, icu_path, cxr_path, cxr_jpg_path, note_path):
    """One-time ingestion of every source folder into an indexed DuckDB
    database file."""
    con = duckdb.connect(str(db_path))

    ingest_folder(con, hosp_path, _HOSP_INDEX_COLS, cast_datetimes=True)
    ingest_folder(con, icu_path, _ICU_INDEX_COLS, cast_datetimes=True)
    ingest_folder(con, cxr_path, _CXR_INDEX_COLS, cast_datetimes=False)
    ingest_folder(con, cxr_jpg_path, _CXR_JPG_INDEX_COLS, cast_datetimes=False)
    ingest_folder(con, note_path, _NOTE_INDEX_COLS, cast_datetimes=True)

    fix_cxr_metadata_datetime(con)
    return con


# ---------------------------------------------------------------------------
# ID combination: which ICU stays link to which CXR studies / notes
# ---------------------------------------------------------------------------

def build_list_ids(con, list_ids_path):
    """Join ICU stays against CXR studies and clinical notes by subject,
    admission, and time window. Equivalent to the original pandasql query,
    executed directly in DuckDB against the ingested tables."""
    con.execute("""
        CREATE OR REPLACE TABLE icu_info AS
        SELECT
            i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
            a.admittime, a.dischtime, a.edregtime, a.edouttime,
            (SELECT MIN(t.v) FROM (VALUES (i.intime), (a.admittime), (a.edregtime)) AS t(v)) AS earliest_intime
        FROM icustays i
        LEFT JOIN admissions a ON i.subject_id = a.subject_id AND i.hadm_id = a.hadm_id
    """)

    con.execute("""
        CREATE OR REPLACE TABLE note_ds_info AS
        SELECT note_id AS ds_note_id, subject_id, hadm_id, charttime AS ds_charttime
        FROM discharge
    """)
    con.execute("""
        CREATE OR REPLACE TABLE note_rad_info AS
        SELECT note_id AS rad_note_id, subject_id, hadm_id, charttime AS rad_charttime
        FROM radiology
    """)

    con.execute("""
        CREATE OR REPLACE TABLE list_ids AS
        SELECT DISTINCT
            i.subject_id, i.hadm_id, i.stay_id,
            c.study_id, c.dicom_id, ds.ds_note_id, rad.rad_note_id
        FROM icu_info i
        LEFT JOIN "mimic-cxr-2.0.0-metadata" c
            ON i.subject_id = c.subject_id
            AND c.StudyDatetime >= i.earliest_intime AND c.StudyDatetime <= i.outtime
        LEFT JOIN note_ds_info ds
            ON i.subject_id = ds.subject_id AND i.hadm_id = ds.hadm_id
        LEFT JOIN note_rad_info rad
            ON i.subject_id = rad.subject_id AND i.hadm_id = rad.hadm_id
            AND rad.rad_charttime >= i.earliest_intime AND rad.rad_charttime <= i.outtime
    """)
    con.execute('CREATE INDEX IF NOT EXISTS idx_list_ids_keys ON list_ids (subject_id, hadm_id, stay_id)')

    con.execute(f"COPY list_ids TO '{Path(list_ids_path).as_posix()}' (FORMAT PARQUET)")

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

def _fetch(con, table, where_sql, params):
    return con.execute(f'SELECT * FROM "{table}" WHERE {where_sql}', params).df()


def _fetch_by_ids(con, table, id_col, ids):
    """Look up rows in a large detail/reference table by a list of foreign
    keys collected from an already-filtered parent table (e.g. emar_detail
    scoped to one patient's emar_id values). Avoids ever loading the full
    detail table into memory."""
    ids = [i for i in pd.unique(ids) if pd.notna(i)]
    if not ids:
        return con.execute(f'SELECT * FROM "{table}" WHERE 1=0').df()
    placeholders = ", ".join(["?"] * len(ids))
    return con.execute(f'SELECT * FROM "{table}" WHERE "{id_col}" IN ({placeholders})', ids).df()


def _fetch_by_id_list(con, table, subject_col, subject_id, id_cols_and_lists):
    """Fetch rows matching subject_id plus one or more IN-list filters,
    e.g. CXR tables filtered by subject_id + study_id (+ dicom_id)."""
    clauses = [f'"{subject_col}" = ?']
    params = [subject_id]
    for col, values in id_cols_and_lists:
        values = [v for v in pd.unique(values) if pd.notna(v)]
        if not values:
            return con.execute(f'SELECT * FROM "{table}" WHERE 1=0').df()
        clauses.append(f'"{col}" IN ({", ".join(["?"] * len(values))})')
        params.extend(values)
    return con.execute(f'SELECT * FROM "{table}" WHERE {" AND ".join(clauses)}', params).df()


class DictionaryTables:
    """Small reference tables loaded once as pandas and reused across every
    patient (mirrors the original code's dfs_icu['d_items'].compute(), etc.)."""

    def __init__(self, con):
        self.d_items = con.execute('SELECT * FROM d_items').df()
        self.d_icd_diagnoses = con.execute('SELECT * FROM d_icd_diagnoses').df()
        self.d_icd_procedures = con.execute('SELECT * FROM d_icd_procedures').df()
        self.d_labitems = con.execute('SELECT * FROM d_labitems').df()
        self.d_hcpcs = con.execute('SELECT * FROM d_hcpcs').df()


def get_patient_icustay(con, dicts, key_subject_id, key_hadm_id, key_stay_id):
    """Build one Patient_ICU instance for a single ICU stay via indexed
    DuckDB lookups. Output structure matches the original Patient_ICU
    exactly; only the retrieval mechanism (SQL vs. dask filtering) changed."""

    df_core = _fetch(con, "list_ids", "subject_id = ? AND hadm_id = ? AND stay_id = ?",
                      [key_subject_id, key_hadm_id, key_stay_id])

    hosp_where = "subject_id = ? AND hadm_id = ?"
    hosp_params = [key_subject_id, key_hadm_id]

    df_admissions = _fetch(con, "admissions", hosp_where, hosp_params)
    df_patients = _fetch(con, "patients", "subject_id = ?", [key_subject_id])
    df_transfers = _fetch(con, "transfers", hosp_where, hosp_params)

    df_diagnoses_icd = _fetch(con, "diagnoses_icd", hosp_where, hosp_params)
    df_diagnoses_icd = df_diagnoses_icd.merge(dicts.d_icd_diagnoses, how="left", on=["icd_code", "icd_version"])

    df_procedures_icd = _fetch(con, "procedures_icd", hosp_where, hosp_params)
    df_procedures_icd = df_procedures_icd.merge(dicts.d_icd_procedures, how="left", on=["icd_code", "icd_version"])

    df_drgcodes = _fetch(con, "drgcodes", hosp_where, hosp_params)
    df_services = _fetch(con, "services", hosp_where, hosp_params)

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

    df_icustays = _fetch(con, "icustays", icu_where, icu_params)
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
    df_cxr_metadata = _fetch_by_id_list(
        con, "mimic-cxr-2.0.0-metadata", "subject_id", key_subject_id,
        [("study_id", study_id_list), ("dicom_id", dicom_id_list)],
    )
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
    parser.add_argument("--db-path", required=True, type=Path,
                         help="DuckDB database file to create/reuse.")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write one ICUstay_<stay_id>.pkl per ICU stay.")
    parser.add_argument("--list-ids-path", default="list_ids.parquet", type=Path,
                         help="Where to save the subject/admission/stay <-> CXR/note ID mapping.")
    parser.add_argument("--rebuild-db", action="store_true",
                         help="Re-ingest source folders even if --db-path already exists.")
    parser.add_argument("--rebuild-list-ids", action="store_true",
                         help="Recompute list_ids even if --list-ids-path already exists.")
    parser.add_argument("--start-index", type=int, default=0,
                         help="Start offset into the ICU stay list (for chunked runs).")
    parser.add_argument("--end-index", type=int, default=None,
                         help="End offset into the ICU stay list (default: all).")
    return parser.parse_args()


def main():
    args = parse_args()

    db_exists = args.db_path.exists()
    if db_exists and not args.rebuild_db:
        log.info("Reusing existing DuckDB database at %s", args.db_path)
        con = duckdb.connect(str(args.db_path))
    else:
        log.info("Building DuckDB database at %s", args.db_path)
        con = build_database(
            args.db_path, args.hosp_path, args.icu_path,
            args.cxr_path, args.cxr_jpg_path, args.note_path,
        )

    if args.list_ids_path.exists() and not args.rebuild_list_ids:
        log.info("Reusing existing list_ids at %s", args.list_ids_path)
        con.execute(f"CREATE OR REPLACE TABLE list_ids AS SELECT * FROM read_parquet('{args.list_ids_path.as_posix()}')")
        con.execute('CREATE INDEX IF NOT EXISTS idx_list_ids_keys ON list_ids (subject_id, hadm_id, stay_id)')
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
