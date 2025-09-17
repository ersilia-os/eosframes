import pandas as pd
from typing import List

from ..utils.utils import is_model_id_valid


def hstack(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Stack Ersilia dataframes horizontally

    Parameters
    ----------
    df_list: List[pd.DataFrame]
        List of dataframes to stack
    
    Returns
    -------
    df: pd.DataFrame
        Horizontally stacked dataframe
    """
    prev_input_list = None
    for df in df_list:
        cur_input_list = df["input"].tolist()
        if prev_input_list is None:
            prev_input_list = cur_input_list
        if not prev_input_list == cur_input_list:
            raise Exception("Input columns do not match!")
        prev_input_list = cur_input_list

    input_list = df_list[0]["input"].tolist()
    key_list = None
    for df in df_list:
        columns = list(df.columns)
        if "key" in columns:
            key_list = df["key"].tolist()
            break

    model_ids = [getattr(df, "model_id", None) for df in df_list]
    for model_id in model_ids:
        if model_id is None:
            raise Exception("One of the dataframes does not have a model_id attribute")
        if not is_model_id_valid(model_id):
            raise Exception("Invalid model_id: {0}".format(model_id))

    if key_list is None:
        do = pd.DataFrame({"input": input_list})
    else:
        do = pd.DataFrame({"key": key_list, "input": input_list})

    for model_id, df in zip(model_ids, df_list):
        columns = [c for c in df.columns.tolist() if c not in {"key", "input"}]
        rename = {c: c + "." + model_id for c in columns}
        do = pd.concat([do, df[columns].reset_index(drop=True).rename(columns=rename)], axis=1)

    return do


def vstack(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Stack Ersilia dataframe vertically

    Parameters
    ----------
    df_list: List[pd.DataFrame]
        List of dataframes to stack

    Returns
    -------
    df: pd.DataFrame
        Vertically stacked dataframe
    """
    prev_cols = None
    for df in df_list:
        if prev_cols is None:
            prev_cols = df.columns.tolist()
        cur_cols = df.columns.tolist()
        if not prev_cols == cur_cols:
            raise Exception("Columns do not match")
        prev_cols = cur_cols
    do = None
    for df in df_list:
        if do is None:
            do = df
            continue
        do = pd.concat([do, df], axis=0)
    model_ids = [getattr(df, "model_id", None) for df in df_list]
    for model_id in model_ids:
        if model_id is None:
            raise Exception("One of the dataframes does not have a model_id attribute")
    model_id = list(set(model_ids))[0]
    do.model_id = model_id
    return do

