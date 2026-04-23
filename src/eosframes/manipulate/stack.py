"""Stack Ersilia DataFrames horizontally or vertically."""

from typing import List

import pandas as pd

from ..exceptions import EosframesError
from ..naming import is_model_id_valid

_STACK_MODES = ("eosmix", "explicit")


def hstack(df_list: List[pd.DataFrame], mode: str) -> pd.DataFrame:
    """Stack Ersilia DataFrames horizontally (one model per frame).

    All frames must share the same ``input`` column in the same order. Two
    naming conventions are available, matched to the two stack filename
    modes (see :func:`eosframes.naming.is_valid_stack_mix_name` and
    :func:`eosframes.naming.is_valid_stack_explicit_name`):

    * ``mode="eosmix"`` — feature column names are suffixed with
      ``_<model_id>_<version>`` so provenance lives in the columns.
    * ``mode="explicit"`` — feature column names are kept as-is (provenance
      lives in the filename that will carry every model's id + version).

    Parameters
    ----------
    df_list : list of pd.DataFrame
        DataFrames to stack. Each must have ``model_id`` and ``version``
        attributes (set automatically by ``read_csv`` / ``read_h5`` when
        the filename follows the naming convention).
    mode : {"eosmix", "explicit"}
        Column-naming strategy.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    EosframesError
        If ``mode`` is unknown, any DataFrame is missing ``model_id`` or
        ``version``, the same ``(model_id, version)`` pair appears twice,
        or row inputs don't match.
    """
    if mode not in _STACK_MODES:
        raise EosframesError(
            f"Unknown stack mode: {mode!r}. Expected one of {_STACK_MODES}."
        )

    pairs = []
    for i, df in enumerate(df_list):
        model_id = getattr(df, "model_id", None)
        version = getattr(df, "version", None)
        if model_id is None:
            raise EosframesError(
                f"DataFrame #{i} does not have a 'model_id' attribute."
            )
        if not is_model_id_valid(model_id):
            raise EosframesError(f"Invalid model_id: {model_id!r}")
        if version is None:
            raise EosframesError(
                f"DataFrame #{i} (model_id={model_id}) does not have a 'version' "
                "attribute. Read the file with a name that encodes the version "
                "(e.g. eos4e40_v1.csv) so df.version is set."
            )
        pairs.append((model_id, version))

    # Same (model_id, version) must not appear twice — columns would collide
    # in eosmix mode, and the explicit-mode filename would be ambiguous.
    seen: set = set()
    for p in pairs:
        if p in seen:
            raise EosframesError(
                f"Duplicate (model_id, version) in stack inputs: {p}. "
                "Each model/version combination may appear at most once."
            )
        seen.add(p)

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

    for (model_id, version), df in zip(pairs, df_list):
        feature_cols = [c for c in df.columns if c not in {"key", "input"}]
        block = df[feature_cols].reset_index(drop=True)
        if mode == "eosmix":
            block = block.rename(
                columns={c: f"{c}_{model_id}_{version}" for c in feature_cols}
            )
        # mode == "explicit": leave column names as-is.
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
