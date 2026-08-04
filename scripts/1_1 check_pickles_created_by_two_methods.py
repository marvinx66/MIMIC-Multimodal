import hashlib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from data_utils import Patient_ICU


def check_by_file_hash(path1, path2):
    h1 = hashlib.sha256()
    h2 = hashlib.sha256()
    with open(path1, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h1.update(chunk)
    with open(path2, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h2.update(chunk)
    return h1.hexdigest(), h2.hexdigest()


def check_py_content(path1, path2):
    with open(path1, "rb") as f:
        obj1 = pickle.load(f)
    with open(path2, "rb") as f:
        obj2 = pickle.load(f)

    if isinstance(obj1, Patient_ICU) and isinstance(obj2, Patient_ICU):
        try:
            pd.testing.assert_frame_equal(obj1, obj2, check_like=False)
            print("Identical")
        except AssertionError as e:
            print("Different:")
            print(e)
    else:
        print(obj1 == obj2)


def dataframe_fingerprint(df):
    """Order-independent content hash for a DataFrame — same content,
    different row order, still matches."""
    df_sorted_cols = df[sorted(df.columns)]
    row_hashes = pd.util.hash_pandas_object(df_sorted_cols, index=False).values
    return np.sort(row_hashes)


def compare_dataframes(df_a, df_b, name):
    if df_a is None and df_b is None:
        return True, "both None"
    if df_a is None or df_b is None:
        return False, f"one is None (A_is_None={df_a is None}, B_is_None={df_b is None})"
    if df_a.shape != df_b.shape:
        return False, f"shape mismatch: A={df_a.shape} vs B={df_b.shape}"
    if set(df_a.columns) != set(df_b.columns):
        return False, (f"column mismatch: A-only={set(df_a.columns)-set(df_b.columns)}, "
                        f"B-only={set(df_b.columns)-set(df_a.columns)}")

    fp_a = dataframe_fingerprint(df_a)
    fp_b = dataframe_fingerprint(df_b)
    if np.array_equal(fp_a, fp_b):
        return True, "identical (fingerprint match)"
    return False, "content differs (fingerprint mismatch) — needs detailed diff"


def compare_patient_icu(a, b, verbose=True):
    """Compare every attribute of two Patient_ICU instances."""
    attrs_a, attrs_b = vars(a), vars(b)
    all_attrs = sorted(set(attrs_a) | set(attrs_b))
    report = {}

    for attr in all_attrs:
        if attr not in attrs_a:
            report[attr] = (False, "missing in A"); continue
        if attr not in attrs_b:
            report[attr] = (False, "missing in B"); continue

        val_a, val_b = attrs_a[attr], attrs_b[attr]

        if isinstance(val_a, pd.DataFrame) or isinstance(val_b, pd.DataFrame):
            ok, msg = compare_dataframes(val_a, val_b, attr)
        elif isinstance(val_a, (list, tuple)) and isinstance(val_b, (list, tuple)):
            ok = sorted(map(str, val_a)) == sorted(map(str, val_b))
            msg = "identical" if ok else f"len A={len(val_a)} vs len B={len(val_b)}"
        else:
            try:
                ok = val_a == val_b
            except Exception as e:
                ok, val_a_repr = False, f"comparison error: {e}"
            msg = "identical" if ok else f"A={val_a!r} vs B={val_b!r}"

        report[attr] = (ok, msg)

    if verbose:
        for attr, (ok, msg) in report.items():
            print(f"[{'OK  ' if ok else 'DIFF'}] {attr}: {'' if ok else msg}")

    return report

if __name__ == "__main__":
    path1 = Path("~\OneDrive - University of Wollongong\PickleFilesForCheck\ICUstay_39553978.pkl").expanduser()
    path2 = Path("~\MIMICWorkspace\MasterDataset\ICUstay_39553978.pkl").expanduser()

    # Check by file hash
    # hash1, hash2 = check_by_file_hash(path1, path2)
    # print(f"Hash of {path1}: {hash1}")
    # print(f"Hash of {path2}: {hash2}")
    # print("Files are identical by hash:", hash1 == hash2)

    # Check by content
    # check_py_content(path1, path2)

    patient_a = pickle.load(open(path1, "rb"))
    patient_b = pickle.load(open(path2, "rb"))
    report = compare_patient_icu(patient_a, patient_b)


