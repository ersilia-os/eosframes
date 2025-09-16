import pandas as pd
import numpy as np
import requests


def is_model_id_valid(model_id):
    if len(model_id) != 7:
        return False
    if not model_id.startswith("eos"):
        return False
    return True


def get_run_columns(model_id: str) -> pd.DataFrame:
    url = "https://raw.githubusercontent./{0}/model/framework/columns/run_columns.csv".format(model_id)

    df = pd.DataFrame()
    return df


def get_model_slug(model_id: str) -> str:
    """
    Gets model slug from GitHub README page.
    """
    url = "https://raw.githubusercontent...{0}/README.md".format(model_id)
    
    slug = text.split("`slug`: ")[1].split("\n")[0]
    return slug


def get_model_title(model_id: str) -> str:
    """
    Gets model title from GitHub README page.
    """
    url = 

    title = text.split("# ")[1].split("\n")[0]
    return title


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

