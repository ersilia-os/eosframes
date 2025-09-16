import os
import h5py
import pandas as pd


def write_h5(df: pd.DataFrame, h5_path: str, dtype: any) -> None:
    """
    Save DataFrame as HDF5 file in Ersilia
    
    ---
    Parameters
    df: pd.DataFrame
        DataFrame to save
    h5_path: str
        Path to the HDF5 file to create
    dtype: data type
        Data type for the feature values

    ---
    Returns
    None
    """
    if os.path.exists(h5_path):
        raise Exception("File {0} exists. Please remove it before saving".format(h5_path))
    df = df.reset_index(drop=True)
    with h5py.File(h5_path, "w") as f:
        if "key" in df.columns:
            keys = df["key"].astype(str).tolist()
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset("key", data=keys, dtype=dt)
        inputs = df["input"].astype(str).tolist()
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("input", data=inputs, dtype=dt)
        feature_columns = [c for c in df.columns if c not in set(["key", "input"])]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("features", data=feature_columns, dtype=dtype)
        values = df[feature_columns].values
        f.create_dataset("values", data=values, dtype=dtype)


def write_chunked_csvs(df: pd.DataFrame, dir_path: str, chunksize: int) -> None:
    """
    TODO: Write this text.
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


def write_xlsx(df: pd.DataFrame, file_path: str) -> None:
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