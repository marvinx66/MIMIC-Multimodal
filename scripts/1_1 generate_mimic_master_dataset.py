"""
generate_mimic_master_dataset.py

Single-file replacement for the "1_1_Master_Dataset_Generation" notebook +
data_utils.py's loading logic. No Dask anywhere. Small tables are read
directly into pandas; every large table (chartevents, labevents,
inputevents, outputevents, procedureevents, datetimeevents,
ingredientevents, diagnoses_icd, procedures_icd, drgcodes, services,
transfers, patients, hcpcsevents, microbiologyevents, emar(+emar_detail),
poe(+poe_detail), prescriptions(+pharmacy)) is queried per-patient directly
from its parquet file with DuckDB, which does row-group-level predicate
pushdown and streams data without ever materializing the full table --
this is what keeps this safe on 16GB RAM regardless of table size.

CAVEAT: DuckDB could not be installed/executed in the environment this was
written in (no network access there), so the SQL is checked for syntax and
against known DuckDB semantics, but not run end-to-end against real MIMIC
data. Dry-run on a handful of patients and spot-check a few fields against
your old pandas-merge output before trusting it for the full cohort.

Usage:
    python generate_mimic_master_dataset.py
(or import the pieces you need into a notebook -- everything below is
plain functions/classes, nothing runs on import except inside
`if __name__ == "__main__":`)
"""

import pickle
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm


# =======================================================================
# CONFIG -- edit these paths for your machine
# =======================================================================
MIMICIV_HOSP_PATH = Path('~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/hosp/').expanduser()
MIMICIV_ICU_PATH = Path('~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/icu/').expanduser()
MIMICIV_CXR_PATH = Path('~/MIMICWorkspace/MIMIC-CXR/2.1.0/').expanduser()
MIMICIV_CXR_JPG_PATH = Path('~/MIMICWorkspace/mimic-cxr-jpg/2.1.0/').expanduser()
MIMICIV_NOTE_PATH = Path('~/MIMICWorkspace/MIMIC-IV-Note-Parquet/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/').expanduser()

STORAGE_PATH = Path('~/MIMICWorkspace/master_dataset/').expanduser()
DUCKDB_TEMP_DIR = Path('~/MIMICWorkspace/duckdb_tmp/').expanduser()
DUCKDB_MEMORY_LIMIT = "10GB"   # leaves headroom for Python/pandas/OS on a 16GB machine
DUCKDB_THREADS = 4


# =======================================================================
# Patient_ICU container (unchanged from your data_utils.py)
# =======================================================================
class Patient_ICU(object):
    def __init__(self, core, admissions, patients, transfers, diagnoses_icd, procedures_icd, drgcodes,
                 services, labevents, hcpcsevents, microbiologyevents, emar, poe, prescriptions,
                 icustays, procedureevents, outputevents, inputevents, datetimeevents, chartevents, ingredientevents,
                 cxr_split, cxr_metadata, cxr_chexpert, cxr_negbio, cxr_image_path, cxr_text_path, dsnotes, radnotes):
        ## CORE
        self.core = core
        ## HOSP
        self.admissions = admissions
        self.patients = patients
        self.transfers = transfers
        self.diagnoses_icd = diagnoses_icd
        self.procedures_icd = procedures_icd
        self.drgcodes = drgcodes
        self.services = services
        self.labevents = labevents
        self.hcpcsevents = hcpcsevents
        self.microbiologyevents = microbiologyevents
        self.emar = emar
        self.poe = poe
        self.prescriptions = prescriptions
        ## ICU
        self.icustays = icustays
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        self.ingredientevents = ingredientevents
        ## CXR
        self.cxr_split = cxr_split
        self.cxr_metadata = cxr_metadata
        self.cxr_chexpert = cxr_chexpert
        self.cxr_negbio = cxr_negbio
        self.cxr_image_path = cxr_image_path
        self.cxr_text_path = cxr_text_path
        ## NOTES
        self.dsnotes = dsnotes
        self.radnotes = radnotes


# =======================================================================
# DuckDB helpers
# =======================================================================
def make_duckdb_con(temp_dir=DUCKDB_TEMP_DIR, memory_limit=DUCKDB_MEMORY_LIMIT, threads=DUCKDB_THREADS):
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{str(temp_dir)}'")
    con.execute(f"PRAGMA threads={threads}")
    return con


def _pq(folder, name):
    return str(Path(folder) / f"{name}.parquet")


def read_table(folder, name):
    """
    Reads a table as pandas regardless of whether it's stored as
    <name>.parquet or <name>.csv.gz in `folder`. Used for the small
    tables loaded fully into pandas. The large tables queried per-patient
    via DuckDB still go through _pq()/read_parquet directly, since those
    ARE confirmed parquet (MIMIC-IV-Parquet / MIMIC-IV-Note-Parquet paths).
    """
    folder = Path(folder)
    pq_path = folder / f"{name}.parquet"
    csv_path = folder / f"{name}.csv.gz"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path, compression='gzip')
    else:
        raise FileNotFoundError(f"Could not find {name}.parquet or {name}.csv.gz in {folder}")


def _q(con, sql, params):
    return con.execute(sql, params).df()


# =======================================================================
# Step 1: load small tables directly into pandas (no Dask needed --
# these are all small enough to just read in full)
# =======================================================================
def convert_datetime_cols(df):
    time_words = ["time", "date", "dod"]
    for col in df.columns:
        if any(w in col.lower() for w in time_words) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def load_small_tables():
    """
    Loads every table that's small enough to fully materialize:
    admissions, icustays, cxr-record-list, cxr-study-list, discharge,
    radiology(+radiology_detail merged in), mimic-cxr-2.0.0-metadata
    (+StudyDatetime construction), mimic-cxr-2.0.0-{chexpert,negbio,split}.

    Everything NOT in this function (labevents, chartevents, etc.) stays
    on disk as parquet and is queried per-patient in get_patient_icustay.
    """
    dfs_hosp = {}
    dfs_icu = {}
    dfs_cxr = {}
    dfs_cxr_jpg = {}
    dfs_note = {}

    dfs_hosp['admissions'] = convert_datetime_cols(read_table(MIMICIV_HOSP_PATH, 'admissions'))
    dfs_icu['icustays'] = convert_datetime_cols(read_table(MIMICIV_ICU_PATH, 'icustays'))

    dfs_cxr['cxr-record-list'] = read_table(MIMICIV_CXR_PATH, 'cxr-record-list')
    dfs_cxr['cxr-study-list'] = read_table(MIMICIV_CXR_PATH, 'cxr-study-list')

    dfs_note['discharge'] = convert_datetime_cols(read_table(MIMICIV_NOTE_PATH, 'discharge'))
    radiology = convert_datetime_cols(read_table(MIMICIV_NOTE_PATH, 'radiology'))
    radiology_detail = read_table(MIMICIV_NOTE_PATH, 'radiology_detail')
    dfs_note['radiology'] = radiology.merge(radiology_detail, how='left', on='note_id')

    # CXR-JPG metadata + StudyDatetime construction (same logic as the notebook)
    metadata = read_table(MIMICIV_CXR_JPG_PATH, 'mimic-cxr-2.0.0-metadata')
    metadata['StudyDate'] = metadata['StudyDate'].astype('int')
    metadata['StudyDate'] = pd.to_datetime(metadata['StudyDate'], format='%Y%m%d')
    metadata['StudyTime'] = metadata.apply(lambda x: '%#010.3f' % x['StudyTime'], axis=1)
    metadata['StudyTime'] = pd.to_datetime(metadata['StudyTime'], format='%H%M%S.%f').dt.strftime('%H%M%S')
    metadata['StudyTime'] = pd.to_datetime(metadata['StudyTime'], format='%H%M%S').dt.time
    metadata['StudyDatetime'] = metadata.apply(lambda r: pd.Timestamp.combine(r['StudyDate'], r['StudyTime']), axis=1)
    dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'] = metadata

    dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'] = read_table(MIMICIV_CXR_JPG_PATH, 'mimic-cxr-2.0.0-chexpert')
    dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'] = read_table(MIMICIV_CXR_JPG_PATH, 'mimic-cxr-2.0.0-negbio')
    dfs_cxr_jpg['mimic-cxr-2.0.0-split'] = read_table(MIMICIV_CXR_JPG_PATH, 'mimic-cxr-2.0.0-split')

    return dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note


# =======================================================================
# Step 2: build list_ids (all combinations of IDs across modules)
# =======================================================================
def build_list_ids(con, dfs_hosp, dfs_icu, dfs_cxr_jpg, dfs_note):
    """
    Same join logic as the notebook's pandasql cell, done in DuckDB instead
    (no extra dependency, and DuckDB can query local pandas DataFrames by
    variable name directly).
    """
    icu_info = dfs_icu['icustays'][['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']].copy()
    icu_info = icu_info.merge(
        dfs_hosp['admissions'][['subject_id', 'hadm_id', 'admittime', 'dischtime', 'edregtime', 'edouttime']],
        on=['subject_id', 'hadm_id'], how='left')
    icu_info['earliest_intime'] = icu_info[['intime', 'admittime', 'edregtime']].min(axis=1)

    cxr_info = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'][
        ['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'StudyDatetime']].copy()

    note_ds_info = dfs_note['discharge'][['note_id', 'subject_id', 'hadm_id', 'charttime']].copy()
    note_ds_info.rename(columns={'note_id': 'ds_note_id', 'charttime': 'ds_charttime'}, inplace=True)

    note_rad_info = dfs_note['radiology'][['note_id', 'subject_id', 'hadm_id', 'charttime']].copy()
    note_rad_info.rename(columns={'note_id': 'rad_note_id', 'charttime': 'rad_charttime'}, inplace=True)

    sql = """
        SELECT DISTINCT
            i.subject_id AS subject_id, i.hadm_id AS hadm_id, i.stay_id,
            c.study_id, c.dicom_id, ds.ds_note_id, rad.rad_note_id
        FROM icu_info i
        LEFT JOIN cxr_info c
            ON i.subject_id = c.subject_id
           AND c.StudyDatetime >= i.earliest_intime AND c.StudyDatetime <= i.outtime
        LEFT JOIN note_ds_info ds
            ON i.subject_id = ds.subject_id AND i.hadm_id = ds.hadm_id
        LEFT JOIN note_rad_info rad
            ON i.subject_id = rad.subject_id AND i.hadm_id = rad.hadm_id
           AND rad.rad_charttime >= i.earliest_intime AND rad.rad_charttime <= i.outtime
    """
    list_ids = con.execute(sql).df()
    return list_ids


# =======================================================================
# Step 3: per-patient extraction (large tables via DuckDB straight from parquet)
# =======================================================================
def get_patient_icustay(con, key_subject_id, key_hadm_id, key_stay_id,
                         list_ids, dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note):
    hp = MIMICIV_HOSP_PATH
    ip = MIMICIV_ICU_PATH

    ## Table of identifiers (small, pandas)
    df_core = list_ids[(list_ids.subject_id == key_subject_id) &
                        (list_ids.hadm_id == key_hadm_id) &
                        (list_ids.stay_id == key_stay_id)]

    ## ---- Hosp: small tables already pandas ----
    df_admissions = dfs_hosp['admissions'][(dfs_hosp['admissions'].subject_id == key_subject_id) &
                                            (dfs_hosp['admissions'].hadm_id == key_hadm_id)]

    ## ---- Hosp: large tables via DuckDB straight from parquet ----
    df_patients = _q(con, """
        SELECT * FROM read_parquet(?) WHERE subject_id = ?
    """, [_pq(hp, 'patients'), key_subject_id])

    df_transfers = _q(con, """
        SELECT * FROM read_parquet(?) WHERE subject_id = ? AND hadm_id = ?
    """, [_pq(hp, 'transfers'), key_subject_id, key_hadm_id])

    df_diagnoses_icd = _q(con, """
        SELECT dx.*, d.long_title
        FROM read_parquet(?) dx
        LEFT JOIN read_parquet(?) d USING (icd_code, icd_version)
        WHERE dx.subject_id = ? AND dx.hadm_id = ?
    """, [_pq(hp, 'diagnoses_icd'), _pq(hp, 'd_icd_diagnoses'), key_subject_id, key_hadm_id])

    df_procedures_icd = _q(con, """
        SELECT p.*, d.long_title
        FROM read_parquet(?) p
        LEFT JOIN read_parquet(?) d USING (icd_code, icd_version)
        WHERE p.subject_id = ? AND p.hadm_id = ?
    """, [_pq(hp, 'procedures_icd'), _pq(hp, 'd_icd_procedures'), key_subject_id, key_hadm_id])

    df_drgcodes = _q(con, """
        SELECT * FROM read_parquet(?) WHERE subject_id = ? AND hadm_id = ?
    """, [_pq(hp, 'drgcodes'), key_subject_id, key_hadm_id])

    df_services = _q(con, """
        SELECT * FROM read_parquet(?) WHERE subject_id = ? AND hadm_id = ?
    """, [_pq(hp, 'services'), key_subject_id, key_hadm_id])

    df_labevents = _q(con, """
        SELECT l.*, d.* EXCLUDE (itemid)
        FROM read_parquet(?) l
        LEFT JOIN read_parquet(?) d USING (itemid)
        WHERE l.subject_id = ? AND l.hadm_id = ?
    """, [_pq(hp, 'labevents'), _pq(hp, 'd_labitems'), key_subject_id, key_hadm_id])

    df_hcpcsevents = _q(con, """
        SELECT h.*, d.*
        FROM read_parquet(?) h
        LEFT JOIN read_parquet(?) d ON h.hcpcs_cd = d.code
        WHERE h.subject_id = ? AND h.hadm_id = ?
    """, [_pq(hp, 'hcpcsevents'), _pq(hp, 'd_hcpcs'), key_subject_id, key_hadm_id])

    df_microbiologyevents = _q(con, """
        SELECT * FROM read_parquet(?) WHERE subject_id = ? AND hadm_id = ?
    """, [_pq(hp, 'microbiologyevents'), key_subject_id, key_hadm_id])

    df_emar = _q(con, """
        SELECT e.*, ed.* EXCLUDE (subject_id)
        FROM read_parquet(?) e
        LEFT JOIN read_parquet(?) ed USING (emar_id)
        WHERE e.subject_id = ? AND e.hadm_id = ?
    """, [_pq(hp, 'emar'), _pq(hp, 'emar_detail'), key_subject_id, key_hadm_id])

    df_poe = _q(con, """
        SELECT p.*, pd.* EXCLUDE (subject_id, poe_id, poe_seq)
        FROM read_parquet(?) p
        LEFT JOIN read_parquet(?) pd USING (poe_id)
        WHERE p.subject_id = ? AND p.hadm_id = ?
    """, [_pq(hp, 'poe'), _pq(hp, 'poe_detail'), key_subject_id, key_hadm_id])

    df_prescriptions = _q(con, """
        SELECT rx.*, ph.* EXCLUDE (subject_id, hadm_id, pharmacy_id)
        FROM read_parquet(?) rx
        LEFT JOIN read_parquet(?) ph USING (pharmacy_id)
        WHERE rx.subject_id = ? AND rx.hadm_id = ?
    """, [_pq(hp, 'prescriptions'), _pq(hp, 'pharmacy'), key_subject_id, key_hadm_id])

    ## ---- ICU: small table already pandas ----
    df_icustays = dfs_icu['icustays'][(dfs_icu['icustays'].subject_id == key_subject_id) &
                                       (dfs_icu['icustays'].hadm_id == key_hadm_id) &
                                       (dfs_icu['icustays'].stay_id == key_stay_id)]

    ## ---- ICU: large event tables via DuckDB, joined with d_items in SQL ----
    def icu_event_query(table_name):
        return _q(con, """
            SELECT ev.*, di.* EXCLUDE (itemid)
            FROM read_parquet(?) ev
            LEFT JOIN read_parquet(?) di USING (itemid)
            WHERE ev.subject_id = ? AND ev.hadm_id = ? AND ev.stay_id = ?
        """, [_pq(ip, table_name), _pq(ip, 'd_items'), key_subject_id, key_hadm_id, key_stay_id])

    df_procedureevents = icu_event_query('procedureevents')
    df_outputevents = icu_event_query('outputevents')
    df_inputevents = icu_event_query('inputevents')
    df_datetimeevents = icu_event_query('datetimeevents')
    df_chartevents = icu_event_query('chartevents')
    df_ingredientevents = icu_event_query('ingredientevents')

    ## ---- CXR (small, already pandas) ----
    study_id_list = df_core['study_id'].unique()
    dicom_id_list = df_core['dicom_id'].unique()

    df_cxr_image_path = dfs_cxr['cxr-record-list'][
        (dfs_cxr['cxr-record-list'].subject_id == key_subject_id) &
        (dfs_cxr['cxr-record-list'].study_id.isin(study_id_list)) &
        (dfs_cxr['cxr-record-list'].dicom_id.isin(dicom_id_list))]
    df_cxr_text_path = dfs_cxr['cxr-study-list'][
        (dfs_cxr['cxr-study-list'].subject_id == key_subject_id) &
        (dfs_cxr['cxr-study-list'].study_id.isin(study_id_list))]

    df_cxr_metadata = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'][
        (dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].subject_id == key_subject_id) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].study_id.isin(study_id_list)) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].dicom_id.isin(dicom_id_list))]
    df_cxr_chexpert = dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'][
        (dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'].subject_id == key_subject_id) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'].study_id.isin(study_id_list))]
    df_cxr_negbio = dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'][
        (dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'].subject_id == key_subject_id) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'].study_id.isin(study_id_list))]
    df_cxr_split = dfs_cxr_jpg['mimic-cxr-2.0.0-split'][
        (dfs_cxr_jpg['mimic-cxr-2.0.0-split'].subject_id == key_subject_id) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-split'].study_id.isin(study_id_list)) &
        (dfs_cxr_jpg['mimic-cxr-2.0.0-split'].dicom_id.isin(dicom_id_list))]

    ## ---- Notes (small, already pandas) ----
    ds_note_id_list = df_core['ds_note_id'].unique()
    rad_note_id_list = df_core['rad_note_id'].unique()
    df_dsnotes = dfs_note['discharge'][(dfs_note['discharge'].subject_id == key_subject_id) &
                                        (dfs_note['discharge'].hadm_id == key_hadm_id) &
                                        (dfs_note['discharge'].note_id.isin(ds_note_id_list))]
    df_radnotes = dfs_note['radiology'][(dfs_note['radiology'].subject_id == key_subject_id) &
                                         (dfs_note['radiology'].hadm_id == key_hadm_id) &
                                         (dfs_note['radiology'].note_id.isin(rad_note_id_list))]
    # radiology_detail was already merged into dfs_note['radiology'] once in load_small_tables()

    return Patient_ICU(
        df_core, df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_procedures_icd,
        df_drgcodes, df_services, df_labevents, df_hcpcsevents, df_microbiologyevents, df_emar,
        df_poe, df_prescriptions, df_icustays, df_procedureevents, df_outputevents, df_inputevents,
        df_datetimeevents, df_chartevents, df_ingredientevents, df_cxr_split, df_cxr_metadata,
        df_cxr_chexpert, df_cxr_negbio, df_cxr_image_path, df_cxr_text_path, df_dsnotes, df_radnotes,
    )


# =======================================================================
# Step 4: batch loop over the whole cohort
# =======================================================================
def generate_master_dataset(key_ids, storage_path, con,
                             list_ids, dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note):
    """
    Single-process by design -- see the explanation in the accompanying
    chat: with 16GB RAM, running multiple DuckDB connections/processes
    against the same large parquet files multiplies memory and I/O
    contention for little guaranteed benefit. DuckDB already parallelizes
    each individual query internally (see PRAGMA threads).
    """
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    nfiles = len(key_ids)
    with tqdm(total=nfiles) as pbar:
        for row in key_ids.itertuples(index=False):
            key_subject_id = row.subject_id
            key_hadm_id = row.hadm_id
            key_stay_id = row.stay_id

            icustay = get_patient_icustay(
                con, key_subject_id, key_hadm_id, key_stay_id,
                list_ids, dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note,
            )

            filename = f'ICUstay_{int(key_stay_id)}.pkl'
            with open(storage_path / filename, 'wb') as f:
                pickle.dump(icustay, f, protocol=pickle.HIGHEST_PROTOCOL)

            pbar.update(1)

    return nfiles


# =======================================================================
# Entry point
# =======================================================================
def main():
    print("Loading small tables into pandas...")
    dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note = load_small_tables()

    print("Setting up DuckDB connection...")
    con = make_duckdb_con()

    print("Building list_ids (ID combinations across modules)...")
    list_ids = build_list_ids(con, dfs_hosp, dfs_icu, dfs_cxr_jpg, dfs_note)
    key_ids = list_ids[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates().reset_index(drop=True)
    print(f"{len(key_ids)} unique ICU stays to extract.")

    print("Generating master dataset...")
    nfiles = generate_master_dataset(
        key_ids, STORAGE_PATH, con,
        list_ids, dfs_hosp, dfs_icu, dfs_cxr, dfs_cxr_jpg, dfs_note,
    )
    print(f"Done. Wrote {nfiles} patient files to {STORAGE_PATH}")


if __name__ == "__main__":
    main()