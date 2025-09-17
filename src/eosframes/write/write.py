import os
import h5py
import pandas as pd

from ..utils.utils import chunker, get_model_id_from_path, is_model_id_valid, get_colors, get_model_slug, get_model_title, get_run_columns


def write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """
    Save DataFrame as CSV file in Ersilia
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to save
    file_path: str
        Path to the CSV file to create
    
    Returns
    -------
    None
    """
    if os.path.exists(csv_path):
        raise Exception("File {0} exists. Please remove it before saving".format(csv_path))
    if not csv_path.endswith(".csv"):
        raise Exception("File {0} must have a .csv extension".format(csv_path))
    model_id_0 = get_model_id_from_path(csv_path)
    if model_id_0 is None:
        raise Exception("Could not extract model_id from file name {0}! The file name must contain the model identifier".format(file_path))
    model_id_1 = getattr(df, "model_id", None)
    if model_id_1 is None:
        raise Exception("DataFrame does not have a model_id attribute")
    if model_id_0 != model_id_1:
        raise Exception("Model_id from file name ({0}) does not match model_id from DataFrame ({1})".format(model_id_0, model_id_1))
    df = df.reset_index(drop=True)
    df.to_csv(csv_path, index=False)


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
    model_id_0 = get_model_id_from_path(h5_path)
    if model_id_0 is None:
        raise Exception("Could not extract model_id from file name {0}! The file name must contain the model identifier".format(h5_path))
    model_id_1 = getattr(df, "model_id", None)
    if model_id_1 is None:
        raise Exception("DataFrame does not have a model_id attribute")
    if model_id_0 != model_id_1:
        raise Exception("Model_id from file name ({0}) does not match model_id from DataFrame ({1})".format(model_id_0, model_id_1))
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
        f.create_dataset("features", data=feature_columns, dtype=dt)
        values = df[feature_columns].values
        f.create_dataset("values", data=values, dtype=dtype)


def write_chunked_csvs(df: pd.DataFrame, dir_path: str, chunksize: int) -> None:
    """
    This function splits a dataframe into multiple CSV files, each containing a chunk of the original dataframe.
    The CSV files are saved in a specified directory, with filenames indicating their chunk number.
    
    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe to be chunked and saved.
    dir_path: str
        The directory path where the chunked CSV files will be saved.
    chunksize: int
        The number of rows per chunk.
    
    Returns
    -------
    None
    """
    if chunksize > 100000:
        raise Exception("Chunksize at Ersilia is currently limited to 100000")
    model_id_0 = get_model_id_from_path(dir_path)
    if model_id_0 is None:
        raise Exception("Could not extract model_id from directory {0}! The directory must contain the model identifier".format(dir_path))
    model_id_1 = getattr(df, "model_id", None)
    if model_id_1 is None:
        raise Exception("DataFrame does not have a model_id attribute")
    if model_id_0 != model_id_1:
        raise Exception("Model_id from file name ({0}) does not match model_id from DataFrame ({1})".format(model_id_0, model_id_1))
    df = df.reset_index(drop=True)
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        raise Exception("Folder {0} exists. Please remove the folder before saving files in there".format(dir_path))
    os.mkdir(dir_path)
    num_chunks = df.shape[0] / chunksize + 1
    if num_chunks > 999999:
        raise Exception("Too many chunks ({0}). Maximum number of chunks is 999999. Increase the chunksize if you want to process your full daataset".format(num_chunks))
    for i, chunk in enumerate(chunker(df, chunksize)):
        file_name = "chunk_{0}.csv".format(str(i).zfill(6))
        chunk.to_csv(os.path.join(dir_path, file_name), index=False)


def write_xlsx(df: pd.DataFrame, xlsx_path: str) -> None:
    """
    Save dataframe as spreadsheet in Ersilia format.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to save
    xlsx_path: str
        Path to the XLSX file to create
    
    Returns
    -------
    None
    """
    if not xlsx_path.endswith(".xlsx"):
        raise Exception("File {0} must have a .xlsx extension".format(xlsx_path))
    if os.path.exists(xlsx_path):
        #raise Exception("File {0} exists. Please remove it before saving".format(xlsx_path))
        os.remove(xlsx_path)
    df.model_id = getattr(df, "model_id", None)

    data_sheet_name = "Data"
    legend_sheet_name = "Legend"
    columns = [c for c in df.columns.tolist() if c not in set(["key", "input"])]
    model_ids = []
    for c in columns:
        model_id = c.split(".")[-1]
        if not is_model_id_valid(model_id):
            raise Exception("Column {0} does not have a valid model_id suffix".format(c))
        if model_id not in model_ids:
            model_ids += [model_id]
    colors = get_colors(len(model_ids))
    R = []
    for model_id in model_ids:
        r = [model_id, get_model_slug(model_id), get_model_title(model_id), "https://github.com/ersilia-os/{0}".format(model_id)]
        R += [r]
    dl = pd.DataFrame(R, columns=["model_id", "slug", "title", "link"])

    dc = None
    columns_colors = []
    for i, model_id in enumerate(model_ids):
        dc_ = get_run_columns(model_id)
        dc_ = pd.concat([pd.DataFrame([model_id]*dc_.shape[0], columns=["model_id"]), dc_], axis=1)
        columns_colors += [colors[i]]*dc_.shape[0]
        if dc is None:
            dc = dc_
        else:
            dc = pd.concat([dc, dc_], axis=0).reset_index(drop=True)
    
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        # Data sheet
        df.to_excel(writer, sheet_name=data_sheet_name, index=False, startrow=0, startcol=0)
        worksheet = writer.sheets[data_sheet_name]
        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, 0, len(df.columns) - 1)
        for i, column in enumerate(df.columns):
            max_length = min(max(df[column].astype(str).map(len).max(), len(str(column))) + 2, 50)
            worksheet.set_column(i, i, max_length)
        # Legend sheet
        dl.to_excel(writer, sheet_name=legend_sheet_name, index=False, startrow=1, startcol=0)
        dc.to_excel(writer, sheet_name=legend_sheet_name, index=False, startrow=1, startcol=dl.shape[1] + 1)
        worksheet = writer.sheets[legend_sheet_name]
        worksheet.merge_range(0, 0, 0, dl.shape[1] - 1, "Ersilia models", writer.book.add_format({'align': 'center', 'bold': True}))
        worksheet.merge_range(0, dl.shape[1] + 1, 0, dl.shape[1] + dc.shape[1], "Columns", writer.book.add_format({'align': 'center', 'bold': True}))
        worksheet.freeze_panes(2, 0)
        worksheet.set_column(0, dl.shape[1] - 1, 30)
        worksheet.set_column(dl.shape[1], dl.shape[1] + dc.shape[1] - 1, 30)
