# %% [markdown]
# # MIMIC-Multimodal: Master Dataset Generation
# 
# The data structure of master dataset is inspired by 
# Soenksen, L. R. et al. Integrated multimodal artificial intelligence framework for healthcare applications. npj Digit. Med. 5, 149 (2022).
# 
# For more details, please visit:
# https://physionet.org/content/haim-multimodal/1.0.1/
# 
# For data access and description, please visit:
# https://mimic.mit.edu/
# 
# MIMIC-IV https://physionet.org/content/mimiciv/2.2/#files-panel \
# MIMIC-CXR https://physionet.org/content/mimic-cxr/2.0.0/#files-panel \
# MIMIC-CXR-JPG https://physionet.org/content/mimic-cxr-jpg/2.0.0/ \
# MIMIC-IV-Note https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel 
# 

# %%


# %%
import numpy as np
import pandas as pd
import pickle
import datetime as dt
from pandasql import sqldf
from data_utils import *

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import time
from functools import partial
from multiprocessing import Pool, cpu_count


# %% [markdown]
# ## Load Data
# For memory efficiency, we first load all files into  **Dask DataFrames** \
# When processing is required, we call **ddf.compute()** to convert the data into a Pandas DataFrame

# %% [markdown]
# ### read files by folder

# %%
# MIMIC-IV hosp module
dfs_hosp = {}
dfs_hosp = read_folder(dfs_hosp, mimiciv_hosp_path)

# %%
# MIMIC-IV icu module
dfs_icu = {}
# Read large dataframes into pandas dataframe since computing such dask dataframe requires a great amount of time and memory
# dfs_icu['chartevents'] = pd.read_csv(mimiciv_icu_path+'chartevents.csv.gz', compression='gzip')
# dfs_icu['chartevents'] = pd.read_parquet(
#     mimiciv_icu_path / 'chartevents.parquet'
# )
dfs_icu = read_folder(dfs_icu,mimiciv_icu_path)

# %%
# MIMIC-IV-CXR
dfs_cxr = {}
dfs_cxr = read_folder(dfs_cxr, mimiciv_cxr_path)
dfs_cxr_jpg = {}
dfs_cxr_jpg = read_folder(dfs_cxr_jpg, mimiciv_cxr_jpg_path)

# %%
# MIMIC-IV-Notes
dfs_note = {}
dfs_note = read_folder(dfs_note, mimiciv_note_path)

# %% [markdown]
# ### datetime conversion

# %%
# Hosp
dfs_hosp = convert_datetime(dfs_hosp)
# ICU
dfs_icu = convert_datetime(dfs_icu)
# Note
dfs_note = convert_datetime(dfs_note)

# %%
# convert time-related variables in CXR metadata
dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'] = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].compute()
df = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata']
df['StudyDate'] = df['StudyDate'].astype('int')
df['StudyDate'] = pd.to_datetime(df['StudyDate'],format='%Y%m%d')
df['StudyTime'] = df.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
df['StudyTime'] = pd.to_datetime(df['StudyTime'], format='%H%M%S.%f').dt.strftime('%H%M%S')
df['StudyTime'] = pd.to_datetime(df['StudyTime'], format='%H%M%S').dt.time
df['StudyDatetime'] = df.apply(lambda r : dt.datetime.combine(r['StudyDate'],r['StudyTime']),1)

# %% [markdown]
# ## ID combinations

# %% [markdown]
# ### get ID lists from each module

# %%
## MIMIC-IV
dfs_hosp['admissions'] = dfs_hosp['admissions'].compute()
dfs_icu['icustays'] = dfs_icu['icustays'].compute()
## MIMIC-IV CXR
dfs_cxr['cxr-record-list'] = dfs_cxr['cxr-record-list'].compute()
## MIMIC-IV Note
dfs_note['discharge'] = dfs_note['discharge'].compute()
dfs_note['radiology'] = dfs_note['radiology'].compute()

# %%
# Get all combinations of IDs in ICU module
icu_info = dfs_icu['icustays'][['subject_id','hadm_id','stay_id','intime','outtime']].copy()
icu_info = icu_info.merge(dfs_hosp['admissions'][['subject_id','hadm_id','admittime','dischtime','edregtime','edouttime']],
                          on=['subject_id','hadm_id'],how='left')
icu_info['earliest_intime'] = icu_info[['intime','admittime','edregtime']].min(axis=1) # earliest entering time for each hospitalization
# Get all combination of IDs in CXR module
cxr_info = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'][['subject_id','study_id','dicom_id','StudyDate','StudyTime','StudyDatetime']].copy()
# Get all combinations of IDs in Note module
note_ds_info = dfs_note['discharge'][['note_id','subject_id','hadm_id','charttime']].copy()
note_ds_info.rename(columns={'note_id':'ds_note_id','charttime':'ds_charttime'},inplace=True)
note_rad_info = dfs_note['radiology'][['note_id','subject_id','hadm_id','charttime']].copy()
note_rad_info.rename(columns={'note_id':'rad_note_id','charttime':'rad_charttime'},inplace=True)

# %% [markdown]
# ### merge IDs by key identifiers and time

# %%
pysqldf = lambda q: sqldf(q, globals())

# %%
# For radiology reports and chest X-ray, we combine the data also by time

## Join on MIMIC-IV,MIMIC-CXR and MIMICIV-Note
sql_query = """
select distinct key_subject_id as subject_id,key_hadm_id as hadm_id,stay_id,study_id,dicom_id,ds_note_id,rad_note_id
from 
(
    select subject_id as key_subject_id,hadm_id as key_hadm_id,stay_id,intime,outtime,admittime,dischtime,earliest_intime
    from icu_info
) as i
left join cxr_info c
on i.key_subject_id = c.subject_id and c.StudyDatetime >= i.earliest_intime and c.StudyDatetime <= i.outtime
left join note_ds_info ds
on i.key_subject_id = ds.subject_id and i.key_hadm_id = ds.hadm_id
left join note_rad_info rad
on i.key_subject_id = rad.subject_id and i.key_hadm_id = rad.hadm_id and rad.rad_charttime >= i.earliest_intime and rad.rad_charttime <= i.outtime
"""
list_ids = pysqldf(sql_query)

# %%
# key identifiers
key_ids = list_ids[['subject_id','hadm_id','stay_id']].drop_duplicates().reset_index(drop=True)

# %% [markdown]
# ### summary

# %%
print('For patients admitted to ICU')
print('Number of unique patients:',list_ids['subject_id'].nunique())
print('Number of unique hospital admissions:',list_ids['hadm_id'].nunique())
print('Number of unique ICU stays:',list_ids['stay_id'].nunique())
print('Number of unique chest xray studies:',list_ids['study_id'].nunique())
print('Number of unique chest xray images:',list_ids['dicom_id'].nunique())
print('Number of unique discharge summaries:',list_ids['ds_note_id'].nunique())
print('Number of unique radiology reports:',list_ids['rad_note_id'].nunique())

# %% [markdown]
# ## Extract information for each unique ICU stay

# %% [markdown]
# ### functions

# Get full MIMIC-IV patient records using key_identifiers
def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id):
    """
    Inputs:
    key_subject_id -> subject_id is unique to a patient
    key_hadm_id    -> hadm_id is unique to a patient hospital stay
    key_stay_id    -> stay_id is unique to a patient ward stay
    Outputs:
    Patient_ICUstay -> ICU patient stay structure
    """
    # Data Extraction

    # debugging
    start = time.time()
    
    ## Table of identifiers
    df_core = list_ids[(list_ids.subject_id == key_subject_id) & (list_ids.hadm_id == key_hadm_id) & 
                       (list_ids.stay_id == key_stay_id)]
    
    ## Hosp - Tables are merged based on subject_id & hadm_id
    # Since miscellaneous information in OMR table is less detailed than in chartevents table, 
    # thus information from OMR table will not be included
    df_admissions = dfs_hosp['admissions'][(dfs_hosp['admissions'].subject_id == key_subject_id) & 
                                           (dfs_hosp['admissions'].hadm_id == key_hadm_id)]
    df_patients = dfs_hosp['patients'][(dfs_hosp['patients'].subject_id == key_subject_id)]
    df_transfers = dfs_hosp['transfers'][(dfs_hosp['transfers'].subject_id == key_subject_id) & 
                                         (dfs_hosp['transfers'].hadm_id == key_hadm_id)]
    df_diagnoses_icd = dfs_hosp['diagnoses_icd'][(dfs_hosp['diagnoses_icd'].subject_id == key_subject_id) &
                                                 (dfs_hosp['diagnoses_icd'].hadm_id == key_hadm_id)]
    df_diagnoses_icd = df_diagnoses_icd.merge(dfs_hosp['d_icd_diagnoses'],
                                              how='left', on=['icd_code', 'icd_version'])
    df_procedures_icd = dfs_hosp['procedures_icd'][(dfs_hosp['procedures_icd'].subject_id == key_subject_id) & 
                                                   (dfs_hosp['procedures_icd'].hadm_id == key_hadm_id)]
    df_procedures_icd = df_procedures_icd.merge(dfs_hosp['d_icd_procedures'], 
                                                how='left', on=['icd_code', 'icd_version'])
    df_drgcodes = dfs_hosp['drgcodes'][(dfs_hosp['drgcodes'].subject_id == key_subject_id) & 
                                       (dfs_hosp['drgcodes'].hadm_id == key_hadm_id)]
    df_services = dfs_hosp['services'][(dfs_hosp['services'].subject_id == key_subject_id) & 
                                       (dfs_hosp['services'].hadm_id == key_hadm_id)]
    df_labevents = dfs_hosp['labevents'][(dfs_hosp['labevents'].subject_id == key_subject_id) & 
                                         (dfs_hosp['labevents'].hadm_id == key_hadm_id)]
    df_labevents = df_labevents.merge(dfs_hosp['d_labitems'], how='left',on='itemid')
    df_hcpcsevents = dfs_hosp['hcpcsevents'][(dfs_hosp['hcpcsevents'].subject_id == key_subject_id) & 
                                             (dfs_hosp['hcpcsevents'].hadm_id == key_hadm_id)]
    df_hcpcsevents = df_hcpcsevents.merge(dfs_hosp['d_hcpcs'], how='left',
                                          left_on='hcpcs_cd',right_on='code')
    df_microbiologyevents = dfs_hosp['microbiologyevents'][(dfs_hosp['microbiologyevents'].subject_id == key_subject_id) & 
                                                           (dfs_hosp['microbiologyevents'].hadm_id == key_hadm_id)]
    df_emar = dfs_hosp['emar'][(dfs_hosp['emar'].subject_id == key_subject_id) & 
                               (dfs_hosp['emar'].hadm_id == key_hadm_id)]
    df_emar = df_emar.merge(dfs_hosp['emar_detail'], how='left', on='emar_id' )
    df_poe = dfs_hosp['poe'][(dfs_hosp['poe'].subject_id == key_subject_id) & (dfs_hosp['poe'].hadm_id == key_hadm_id)]
    df_poe = df_poe.merge(dfs_hosp['poe_detail'], how='left', on='poe_id')
    df_prescriptions = dfs_hosp['prescriptions'][(dfs_hosp['prescriptions'].subject_id == key_subject_id) & 
                                                 (dfs_hosp['prescriptions'].hadm_id == key_hadm_id)]
    df_prescriptions = df_prescriptions.merge(dfs_hosp['pharmacy'], how='left', on='pharmacy_id')
    
    ## ICU - Tables are merged based on subject_id & hadm_id & stay_id
    df_icustays = dfs_icu['icustays'][(dfs_icu['icustays'].subject_id == key_subject_id) & 
                                      (dfs_icu['icustays'].hadm_id == key_hadm_id) & 
                                      (dfs_icu['icustays'].stay_id == key_stay_id)]
    df_procedureevents = dfs_icu['procedureevents'][(dfs_icu['procedureevents'].subject_id == key_subject_id) & 
                                                    (dfs_icu['procedureevents'].hadm_id == key_hadm_id) & 
                                                    (dfs_icu['procedureevents'].stay_id == key_stay_id)]
    df_outputevents = dfs_icu['outputevents'][(dfs_icu['outputevents'].subject_id == key_subject_id) & 
                                              (dfs_icu['outputevents'].hadm_id == key_hadm_id) & 
                                              (dfs_icu['outputevents'].stay_id == key_stay_id)]
    df_inputevents = dfs_icu['inputevents'][(dfs_icu['inputevents'].subject_id == key_subject_id) & 
                                            (dfs_icu['inputevents'].hadm_id == key_hadm_id) & 
                                            (dfs_icu['inputevents'].stay_id == key_stay_id)]
    df_datetimeevents = dfs_icu['datetimeevents'][(dfs_icu['datetimeevents'].subject_id == key_subject_id) & 
                                                  (dfs_icu['datetimeevents'].hadm_id == key_hadm_id) & 
                                                  (dfs_icu['datetimeevents'].stay_id == key_stay_id)]

    # debugging
    t=time.time()
    df_chartevents = dfs_icu['chartevents'][(dfs_icu['chartevents'].subject_id == key_subject_id) & 
                                            (dfs_icu['chartevents'].hadm_id == key_hadm_id) & 
                                            (dfs_icu['chartevents'].stay_id == key_stay_id)]
    # debugging
    print("chartevents:", time.time()-t)

    df_ingredientevents = dfs_icu['ingredientevents'][(dfs_icu['ingredientevents'].subject_id == key_subject_id) & 
                                                      (dfs_icu['ingredientevents'].hadm_id == key_hadm_id) & 
                                                      (dfs_icu['ingredientevents'].stay_id == key_stay_id)]
    # Merge descriptions into each table
    df_procedureevents = df_procedureevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    df_outputevents = df_outputevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    df_inputevents = df_inputevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    df_datetimeevents = df_datetimeevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    df_chartevents = df_chartevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    df_ingredientevents = df_ingredientevents.merge(dfs_icu['d_items'], how='left', on='itemid')
    
    ## CXR
    # Get lists of study_id and dicom_id for each ICU stay
    study_id_list = df_core['study_id'].unique()
    dicom_id_list = df_core['dicom_id'].unique()
    # Extract tables from MIMIC-CXR
    df_cxr_image_path = dfs_cxr['cxr-record-list'][(dfs_cxr['cxr-record-list'].subject_id == key_subject_id) &
                                                   (dfs_cxr['cxr-record-list'].study_id.isin(study_id_list)) &
                                                   (dfs_cxr['cxr-record-list'].dicom_id.isin(dicom_id_list))]
    df_cxr_text_path = dfs_cxr['cxr-study-list'][(dfs_cxr['cxr-study-list'].subject_id == key_subject_id) &
                                                   (dfs_cxr['cxr-study-list'].study_id.isin(study_id_list))]
    # Extract tables from MIMIC-CXR-JPG
    df_cxr_metadata = dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'][(dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].subject_id == key_subject_id) &
                                                              (dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].study_id.isin(study_id_list)) &
                                                              (dfs_cxr_jpg['mimic-cxr-2.0.0-metadata'].dicom_id.isin(dicom_id_list))]
    df_cxr_chexpert = dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'][(dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'].subject_id == key_subject_id) &
                                                              (dfs_cxr_jpg['mimic-cxr-2.0.0-chexpert'].study_id.isin(study_id_list))]
    df_cxr_negbio = dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'][(dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'].subject_id == key_subject_id) & 
                                                          (dfs_cxr_jpg['mimic-cxr-2.0.0-negbio'].study_id.isin(study_id_list))]
    df_cxr_split = dfs_cxr_jpg['mimic-cxr-2.0.0-split'][(dfs_cxr_jpg['mimic-cxr-2.0.0-split'].subject_id == key_subject_id) &
                                                        (dfs_cxr_jpg['mimic-cxr-2.0.0-split'].study_id.isin(study_id_list)) &
                                                        (dfs_cxr_jpg['mimic-cxr-2.0.0-split'].dicom_id.isin(dicom_id_list))]
    
    ## Notes
    ds_note_id_list = df_core['ds_note_id'].unique()
    rad_note_id_list = df_core['rad_note_id'].unique()
    df_dsnotes = dfs_note['discharge'][(dfs_note['discharge'].subject_id == key_subject_id) &
                                       (dfs_note['discharge'].hadm_id == key_hadm_id) &
                                       (dfs_note['discharge'].note_id.isin(ds_note_id_list))]
    df_radnotes = dfs_note['radiology'][(dfs_note['radiology'].subject_id == key_subject_id) &
                                        (dfs_note['radiology'].hadm_id == key_hadm_id) &
                                        (dfs_note['radiology'].note_id.isin(rad_note_id_list))]
    df_radnotes = df_radnotes.merge(dfs_note['radiology_detail'], how='left', on='note_id')

    # debugging
    print(f"32-table operations: {time.time()-start:.2f}s")
    # start = time.time()

    # Create patient object and return
    Patient_ICUstay = Patient_ICU(df_core, df_admissions, df_patients, df_transfers, df_diagnoses_icd, df_procedures_icd, df_drgcodes,
                                  df_services, df_labevents, df_hcpcsevents, df_microbiologyevents, df_emar, df_poe, df_prescriptions, 
                                  df_icustays, df_procedureevents, df_outputevents, df_inputevents, df_datetimeevents, df_chartevents, df_ingredientevents,
                                  df_cxr_split, df_cxr_metadata, df_cxr_chexpert, df_cxr_negbio, df_cxr_image_path, df_cxr_text_path, 
                                  df_dsnotes, df_radnotes)

    # debugging
    # print(f"Patient_ICU: {time.time()-start:.2f}s")
     
    return Patient_ICUstay

# Extract all single ICU stay records
def generate_master_dataset(key_ids, storage_path):
    # Inputs:
    #   key_ids -> Dataframe with all unique available records by key identifiers
    #   storage_path -> Path to structured MIMIC IV databases in pickle files
    
    # Outputs:
    #   nfiles -> Number of single patient files produced
    
    # Extract information for patient
    nfiles = len(key_ids)
    with tqdm(total = nfiles) as pbar:

        #Iterate through all patients

        # Optimization: speed up the process
        for row in key_ids.itertuples(index=False):

            # debugging
            start = time.time()

            key_subject_id = row.subject_id
            key_hadm_id = row.hadm_id
            key_stay_id = row.stay_id
            
            # Save objects
            filename = f'ICUstay_{int(key_stay_id)}'+'.pkl'
            icustay = get_patient_icustay(key_subject_id,key_hadm_id,key_stay_id)

            # debugging
            print(f"get_patient_icustay: {time.time()-start:.2f}s")
            start = time.time()

            with open(storage_path / filename, 'wb') as f:
                pickle.dump(icustay, f, protocol=pickle.HIGHEST_PROTOCOL)

            # debugging
            print(f"pickle: {time.time()-start:.2f}s")

            # Update process bar
            pbar.update(1)

# %% [markdown]
# ### extract and save patient ICU stay information

# %%
dfs_icu['d_items'] = dfs_icu['d_items'].compute()
dfs_note['radiology_detail'] = dfs_note['radiology_detail'].compute()



def test_pickle_file(ICU_path):

    ICUstay_test = pickle.load(open(ICU_path / 'ICUstay_31205490.pkl','rb'))
    ICUstay_test.__dict__.keys()

    with tqdm(total=len(ICUstay_test.__dict__.keys())) as pbar:
        for attribute, value in ICUstay_test.__dict__.items():
            if isinstance(value,pd.DataFrame):
                print(attribute)
                display(value.head())
                pbar.update(1)
            else:
                pbar.update(1) 


if __name__ == "__main__":

    # File path
    # MIMIC-IV
    mimiciv_hosp_path = Path('~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/hosp/').expanduser()
    mimiciv_icu_path = Path('~/MIMICWorkspace/MIMIC-IV-Parquet/physionet.org/files/mimiciv/3.1/icu/').expanduser()
    # MIMIV-CXR & MIMIC-CXR-JPG
    mimiciv_cxr_path = Path('~/MIMICWorkspace/MIMIC-CXR/2.1.0/').expanduser()
    mimiciv_cxr_jpg_path = Path('~/MIMICWorkspace/mimic-cxr-jpg/2.1.0/').expanduser() 
    # MIMIC-IV-Note
    mimiciv_note_path = Path('~/MIMICWorkspace/MIMIC-IV-Note-Parquet/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/').expanduser()
    ICU_path = Path('~/MIMICWorkspace/MasterDataset/').expanduser()

    # test
    # test_pickle_file(ICU_path)

    # multiprocessing
    num_processes = cpu_count() - 1
    chunk_size = len(key_ids) // num_processes
    chunks = [key_ids[i:i + chunk_size] for i in range(0, len(key_ids), chunk_size)]
    
    with Pool(processes=num_processes) as pool:
        pool.map(partial(generate_master_dataset, storage_path=ICU_path), chunks)


    # single processing    
    # debugging
    #generate_master_dataset(key_ids=key_ids[0:10],storage_path=ICU_path)

    # production part1 ID 0 ~ 30000
    # generate_master_dataset(key_ids=key_ids[0:30000],storage_path=ICU_path)

    # production part1 ID 30000 ~
    #generate_master_dataset(key_ids=key_ids[30000:],storage_path=ICU_path)    