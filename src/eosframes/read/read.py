import os
import h5py
import pandas as pd


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read CSV file into a Pandas DataFrame
    This file is assumed to have the standard Ersilia format, containing columns "key", "input", and feature columns.
    
    Parameters
    ----------
    file_path: str
        Path to the CSV file
    
    Returns
    -------
    df: pd.DataFrame
        DataFrame containing the data from the CSV file
    """
    if not os.path.exists(file_path):
        raise Exception("File {0} does not exist".format(file_path))
    df = pd.read_csv(file_path)
    if "key" not in df.columns:
        raise Exception("File {0} does not contain a column named 'key'".format(file_path))
    if "input" not in df.columns:
        raise Exception("File {0} does not contain a column named 'input'".format(file_path))
    return df


def read_h5(h5_path: str) -> pd.DataFrame:
    """
    Read HDF5 file into a Pandas DataFrame
    This file is assumed to have the standard Ersilia format, containing values, features, key (optional), and input datasets.
    
    Parameters
    ----------
    h5_path: str
        Path to the HDF5 file

    Returns
    -------
    df: pd.DataFrame
        DataFrame containing the data from the HDF5 file
    """
    if not os.path.exists(h5_path):
        raise Exception("File {0} does not exist".format(h5_path))
    with h5py.File(h5_path, "r") as f:
        print(f.keys())
        if "values" not in f.keys():
            raise Exception("File {0} does not contain a dataset named 'values'".format(h5_path))
        values = f["values"][:]
        columns = [x.decode("utf-8") for x in f["features"][:]]
        if "key" in f.keys():
            keys = [x.decode("utf-8") for x in f["key"][:]]
        else:
            keys = None
        inputs = [x.decode("utf-8") for x in f["input"][:]]
    if keys is None:
        df = pd.DataFrame({"input": inputs})
    else:
        df = pd.DataFrame({"key": keys, "input": inputs})
    df_ = pd.DataFrame(values, columns=columns)
    df = pd.concat([df, df_], axis=1)
    return df


def read_chunked_csvs(dir_path: str) -> pd.DataFrame:
    """
    Read CSV files from a folder, assuming they have a suffix that determines their order.
    Files must be in the standard Ersilia format, containing columns "key" (optional), "input", and feature columns.
    
    Parameters
    ----------
    dir_path: str
        Path to the directory containing the CSV files
    
    Returns
    -------
    df: pd.DataFrame
        DataFrame containing the concatenated data from the CSV files
    """
    if not os.path.exists(dir_path):
        raise Exception("Directory {0} does not exist".format(dir_path))
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
    batch_ids = sorted(batch_ids)
    for batch_id in batch_ids:
        fn = "{0}_{1}.csv".format(prefix, str(batch_id).zfill(zfill))
        if df is None:
            df = pd.read_csv(fn)
            continue
        df = pd.concat([df, pd.read_csv(fn)], axis=1)
    return df