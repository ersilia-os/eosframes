import os
import h5py
import pandas as pd

from ..utils.utils import get_model_title, get_model_slug, get_run_columns

from ..default import ERSILIA_GDRIVE_DATAFRAMES_BASE_URL


def read_chunked_csvs(dir_path: str):
    """
    Read CSV files from a folder, assuming they have a suffix that determines their order.
    """
    batch_ids = []
    zfill = 0
    prefixes = []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".csv") and not fn.startswith("chunk"):
            raise Exception("The folder contains files that are not CSV. Please use a clean folder containing only CSV files in the format chunk_000000.csv")
        batch_id = fn.split("_")[-1].split(".")[0]
        zfill = len(batch_id)
        batch_ids += [int(batch_id)]
        prefix = "_".join(fn.split("_")[0:-1])
        prefixes += [prefix]
    if len(set(prefixes)) > 1:
        raise Exception("Multiple file prefixes specified. It is not save to merge them.")
    prefix = list(prefixes)[0]
    df = None
    for batch_id in batch_ids:
        fn = "{0}_{1}.csv".format(prefix, str(batch_id).zfill(zfill))
        if df is None:
            df = pd.read_csv(fn)
            continue
        df = pd.concat([df, pd.read_csv(fn)], axis=1)
    return df


def save_chunked_csvs(df: pd.DataFrame, dir_path: str, chunksize: int) -> None:
    """
    
    """
    if chunksize > 100000:
        raise Exception("Chunksize at Ersilia is currently limited to 100000")
    df = df.reset_index(drop=True)
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        raise Exception("Folder {0} exists. Please remove the folder before saving files in there".format(dir_path))
    os.mkdir(dir_path)
    num_chunks = df.shape[0] / chunksize + 1
    if num_chunks > 999999:
        raise Exception("Too many chunks ({0}). Maximum number of chunks is 999999. Increase the chunksize if you want to process your full daataset".format(num_chunks))
    for i, chunk in chunker(df, chunksize):
        file_name = "chunk_{0}.csv".format(str(i).zfill(6))
        df_ = df.iloc[chunk]
        df_.to_csv(os.path.join(dir_path, file_name), index=False)


def stack_chunked_csvs_to_h5(dir_path, h5_path, dtype):
    """
    Read CSV files from directory into an HDF5 file
    """
    file_names = []
    for fn in os.listdir(dir_path):
        if not fn.startswith("chunk") and not fn.endswith(".csv"):
            raise Exception("Only files ending in the format chunk_000000.csv are accepted")
        file_names += [fn]
    batch_ids = []
    for fn in file_names:
        batch_id = int(fn.split("_")[-1].split(".csv")[0])
        batch_ids += [batch_id]
    batch_ids = sorted(batch_ids)
    with h5py.File(h5_path, "w") as f:
        features = None
        for batch_id in batch_ids:
            fn = "chunk_{0}.csv".format(str(batch_id).zfill(6))
            df = pd.read_csv(os.path.join(dir_path, fn))
            if features is None:
                features = [c for c in df.columns.tolist() if c not in set(["key", "input"])]
            input_list = df["input"].tolist()
            key_list = df["key"].tolist()
            data = df[features]


def save_xlsx(df: pd.DataFrame, file_path: str) -> None:
    """
    Save as spreadsheet.
    """
    data_sheet_name = "Data"
    legend_sheet_name = "Legend"
    columns = [c for c in df.columns.tolist() if c not in set(["key", "input"])]
    model_ids = []
    for c in columns:
        model_id = c.split(".")[-1]
        if model_id not in model_ids:
            model_ids += [model_id]
    colors = get_colors(len(model_ids)) # get colors from model ids
    R = []
    for model_id in model_ids:
        r = [model_id, get_model_slug(model_id), get_model_title(model_id), "https://github.com/ersilia-os/{0}".format(model_id)]
        R += [r]
    dl = pd.DataFrame(R, columns=["model_id", "slug", "title", "link"])

    dc = None
    columns_colors = []
    for i, model_id in enumerate(model_ids):
        dc_ = get_run_columns(model_id)
        columns_colors += [colors[i]]*dc_.shape[0]
        if dc is None:
            dc = dc_
        else:
            dc = pd.concat([dc, dc_], axis=1).reset_index()

    

def upload_xlsx(xlsx_file: str) -> None:
    """
    Upload XLSX file to Ersilia's Google Drive. There is a dedicated folder to this under
    Projects/Deliverables. 
    """
