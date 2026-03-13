"""Stack Ersilia DataFrames horizontally or vertically."""

from typing import List

import pandas as pd

from ..exceptions import EosframesError
from ..naming import is_model_id_valid


def hstack(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Stack Ersilia DataFrames horizontally (one model per frame).

    All frames must share the same ``input`` column in the same order.
    Feature columns are suffixed with ``.model_id``.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        DataFrames to stack. Each must have a ``model_id`` attribute.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    EosframesError
        If inputs do not match or a DataFrame has an invalid ``model_id``.
    """
    model_ids = [getattr(df, "model_id", None) for df in df_list]
    for i, model_id in enumerate(model_ids):
        if model_id is None:
            raise EosframesError(
                f"DataFrame #{i} does not have a 'model_id' attribute."
            )
        if not is_model_id_valid(model_id):
            raise EosframesError(f"Invalid model_id: {model_id!r}")

    reference_inputs = df_list[0]["input"].tolist()
    for i, df in enumerate(df_list[1:], start=2):
        if df["input"].tolist() != reference_inputs:
            raise EosframesError(
                f"Input mismatch: DataFrame #{i} has different inputs or row order "
                "than DataFrame #1."
            )

    key_list = None
    for df in df_list:
        if "key" in df.columns:
            key_list = df["key"].tolist()
            break

    meta = (
        {"input": reference_inputs}
        if key_list is None
        else {"key": key_list, "input": reference_inputs}
    )
    result = pd.DataFrame(meta)

    for model_id, df in zip(model_ids, df_list):
        feature_cols = [c for c in df.columns if c not in {"key", "input"}]
        block = (
            df[feature_cols]
            .reset_index(drop=True)
            .rename(columns={c: f"{c}.{model_id}" for c in feature_cols})
        )
        result = pd.concat([result, block], axis=1)

    return result


def vstack(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Stack Ersilia DataFrames vertically (same model, multiple batches).

    All frames must share the same columns and the same ``model_id``.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        DataFrames to stack. Each must have a ``model_id`` attribute.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    EosframesError
        If columns differ, ``model_id`` attributes are missing, or model IDs
        do not all match.
    """
    model_ids = [getattr(df, "model_id", None) for df in df_list]
    for i, model_id in enumerate(model_ids):
        if model_id is None:
            raise EosframesError(
                f"DataFrame #{i} does not have a 'model_id' attribute."
            )

    unique_ids = set(model_ids)
    if len(unique_ids) > 1:
        raise EosframesError(
            f"Cannot vstack DataFrames with different model IDs: {sorted(unique_ids)}"
        )

    reference_cols = df_list[0].columns.tolist()
    for i, df in enumerate(df_list[1:], start=2):
        if df.columns.tolist() != reference_cols:
            raise EosframesError(
                f"Column mismatch: DataFrame #{i} has columns {df.columns.tolist()} "
                f"but expected {reference_cols}."
            )

    result = pd.concat(df_list, axis=0).reset_index(drop=True)
    result.model_id = model_ids[0]
    return result
