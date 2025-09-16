import re
import pandas as pd
import numpy as np
import requests


def get_model_id_from_file_name(file_name: str) -> str:
    """
    Given a file name, extract the model identifier if it exists.
    
    The model identifier is assumed to follow the pattern 'eosXYYY' where X is a digit and YYY are alphanumeric characters.
    Example valid identifiers: eos42ez, eos4e40, eos3804

    Parameters
    ----------
    file_name : str
        The name of the file from which to extract the model identifier
    
    Returns
    -------
    str
        The extracted model identifier if found, otherwise None
    """
    if not file_name.endswith(".csv") and not file_name.endswith(".h5"):
        raise Exception("File name must end with .csv or .h5")
    text = file_name.split("/")[-1]
    pattern = r'(?<![A-Za-z0-9])eos\d[A-Za-z0-9]{3}(?![A-Za-z0-9])'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None


def is_model_id_valid(model_id: str) -> bool:
    """
    Simple function to determine if a model identifier is valid or not.
    """
    if len(model_id) != 7:
        return False
    if not model_id.startswith("eos"):
        return False
    return True


def get_run_columns(model_id: str) -> pd.DataFrame:
    """
    Fetch run_columns.csv from a given repository under ersilia-os.

    Parameters
    ----------
    model_id : str
        Repository name inside the ersilia-os org (e.g. "eos4e40").

    Returns
    -------
    pd.DataFrame
        The CSV contents as a pandas DataFrame.
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/model/framework/columns/run_columns.csv"
    return pd.read_csv(url)


def get_model_slug(model_id: str) -> str:
    """
    Get the model slug from the GitHub README.md file in ersilia-os/{model_id}.
    Assumes README.md contains a line like: `slug`: some-slug
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/README.md"
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    try:
        slug = text.split("**Slug:** `")[1].split("`")[0].strip()
        return slug
    except IndexError:
        raise ValueError(f"No slug found in README.md for {model_id}")


def get_model_title(model_id: str) -> str:
    """
    Get the model title from the GitHub README.md file in ersilia-os/{model_id}.
    Assumes the README contains a line like: "# Title".
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/README.md"
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    try:
        title = text.split("# ")[1].split("\n")[0].strip()
        return title
    except IndexError:
        raise ValueError(f"No title found in README.md for {model_id}")


def resolve_datatype(df: pd.DataFrame, run_columns: pd.DataFrame):
    """
    Resolve the datatype of a given dataframe.
    """
    data_columns = run_columns["name"].tolist()
    if any(data_columns != [c for c in df.columns.tolist() if c != {"key", "input"}]):
        raise Exception("Columns do not match")
    declared_dtype = set(df["type"].tolist())
    if declared_dtype == {"string"}:
        dtype = str
        return df.astype(dtype), dtype
    di = df[[c for c in df.columns.tolist() if c in {"key", "input"}]]
    data = df[[c for c in df.columns.tolist() if c not in {"key", "input"}]]
    if declared_dtype == {"integer"}:
        min_v, max_v = np.min(data), np.max(data)
        if any(data.isna()):
            dtype = np.int16
            return pd.concat([di, data.astype(dtype)], axis=1), dtype
        if min_v >= -128 and max_v <= 127:
            dtype = np.int8
            return pd.concat([di, data.astype(dtype)], axis=1), dtype
        else:
            dtype = np.int16
            return pd.concat([di, data.astype(dtype)], axis=1), dtype
    if declared_dtype == {"float"} or declared_dtype == {"integer", "float"}:
        dtype = np.float32
        return pd.concat([di, data.astype(dtype)], axis=1), dtype

