"""Standard scaler for Ersilia model outputs.

Fits per-column mean/std on numeric feature columns, saves parameters to a
JSON file, and applies the transform to new data with model_id/version
cross-validation.
"""

import json
import os
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from .exceptions import EosframesError
from .logger import get_logger
from .naming import (
    is_valid_name,
    is_valid_transformer_name,
    parse_name,
    parse_transformer_name,
)

_META_COLS = {"key", "input"}
SUPPORTED_METHODS = ("standard",)


# ---------------------------------------------------------------------------
# Low-level DataFrame API
# ---------------------------------------------------------------------------


def fit(df: pd.DataFrame, method: str = "standard") -> dict:
    """Fit a scaler on the numeric feature columns of a DataFrame.

    Non-numeric columns and columns with more than 25 % missing values are
    skipped and listed in ``skipped_columns``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. ``key`` and ``input`` columns are ignored.
    method : str
        Scaling method. Currently only ``"standard"`` is supported.

    Returns
    -------
    dict
        Keys: ``method``, ``columns``, ``skipped_columns``, ``parameters``.

    Raises
    ------
    EosframesError
        If no numeric columns are available after filtering.
    """
    if method not in SUPPORTED_METHODS:
        raise EosframesError(
            f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}"
        )

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        raise EosframesError("No numeric feature columns found to fit the scaler.")

    skipped = []
    fitted_cols = []
    parameters: dict = {}

    for col in numeric_cols:
        if df[col].isna().mean() > 0.25:
            skipped.append(col)
            continue
        series = df[col].dropna().astype(float)
        fitted_cols.append(col)
        if method == "standard":
            parameters[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
            }

    if not fitted_cols:
        raise EosframesError(
            "All numeric columns were skipped (too many missing values)."
        )

    return {
        "method": method,
        "columns": fitted_cols,
        "skipped_columns": skipped,
        "parameters": parameters,
    }


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply scaler parameters to a DataFrame.

    ``key``, ``input``, and any columns not in ``params["columns"]`` pass
    through unchanged. The feature columns must match exactly — same set and
    same order — as the columns the scaler was fitted on.

    Parameters
    ----------
    df : pd.DataFrame
    params : dict
        As returned by :func:`fit` (or loaded from the JSON file).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with scaled feature columns.

    Raises
    ------
    EosframesError
        If the feature columns do not match the fitted columns exactly.
    """
    fitted_cols = params["columns"]
    df_feature_cols = [c for c in df.columns if c not in _META_COLS]

    if df_feature_cols != fitted_cols:
        raise EosframesError(
            f"Column mismatch: input has feature columns {df_feature_cols} "
            f"but transformer was fitted on {fitted_cols}."
        )

    method = params["method"]
    parameters = params["parameters"]

    result = df.copy()
    for col in fitted_cols:
        p = parameters[col]
        series = result[col].astype(float)
        if method == "standard":
            std = p["std"]
            result[col] = 0.0 if std == 0 else (series - p["mean"]) / std

    return result


# ---------------------------------------------------------------------------
# File-level API
# ---------------------------------------------------------------------------


def _write_df(df: pd.DataFrame, output_path: str) -> None:
    """Write a DataFrame to CSV or H5, bypassing the naming convention check."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".h5":
        feat_cols = [c for c in df.columns if c not in _META_COLS]
        with h5py.File(output_path, "w") as f:
            dt = h5py.string_dtype(encoding="utf-8")
            if "key" in df.columns:
                f.create_dataset("key", data=df["key"].astype(str).tolist(), dtype=dt)
            if "input" in df.columns:
                f.create_dataset(
                    "input", data=df["input"].astype(str).tolist(), dtype=dt
                )
            f.create_dataset("features", data=feat_cols, dtype=dt)
            f.create_dataset("values", data=df[feat_cols].values, dtype=np.float32)
    else:
        raise EosframesError(f"Unsupported output format '{ext}'. Expected .csv or .h5")


def fit_file(
    input_path: str,
    scaler_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Fit a scaler on an Ersilia output file and save the parameters.

    When *output_path* is provided the scaled data is also written immediately
    (fit-transform in one call).

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file. Must follow the Ersilia naming convention.
    scaler_path : str
        Path where the JSON parameter file will be written. Must not already
        exist. Must follow ``[prefix_]<model_id>_<version>_transformer.json``
        with model ID and version matching *input_path*.
    output_path : str, optional
        If provided, also write the scaled data here (fit-transform). The
        file must not already exist.

    Returns
    -------
    str
        Absolute path of the saved scaler JSON file.

    Raises
    ------
    EosframesError
        On naming convention violations, pre-existing files, or no numeric
        columns.
    """
    logger = get_logger()

    if not is_valid_name(input_path):
        raise EosframesError(
            f"'{input_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext>"
        )

    if not is_valid_transformer_name(scaler_path):
        raise EosframesError(
            f"'{scaler_path}' does not follow the scaler naming convention. "
            "Expected: [prefix_]<model_id>_<version>_transformer.json"
        )

    parsed = parse_name(input_path)
    scaler_parsed = parse_transformer_name(scaler_path)

    if scaler_parsed["model_id"] != parsed["model_id"]:
        raise EosframesError(
            f"Scaler model ID '{scaler_parsed['model_id']}' does not match "
            f"input model ID '{parsed['model_id']}'. "
            "Scaler filename must encode the same model ID as the input file."
        )
    if scaler_parsed["version"] != parsed["version"]:
        raise EosframesError(
            f"Scaler version '{scaler_parsed['version']}' does not match "
            f"input version '{parsed['version']}'. "
            "Scaler filename must encode the same version as the input file."
        )

    if os.path.exists(scaler_path):
        raise EosframesError(
            f"Scaler file '{scaler_path}' already exists. Remove it first."
        )

    if output_path is not None and os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    from .ops import _read_file

    df = _read_file(input_path)

    logger.info("Fitting scaler on %d rows from %s", len(df), input_path)
    fitted = fit(df)

    transformer = {
        "model_id": parsed["model_id"],
        "version": parsed["version"],
        "n_rows": len(df),
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        **fitted,
    }
    with open(scaler_path, "w") as fh:
        json.dump(transformer, fh, indent=2)
    logger.info("Scaler saved to %s", scaler_path)

    if output_path is not None:
        scaled_df = transform(df, fitted)
        _write_df(scaled_df, output_path)
        logger.info("Scaled output written to %s", output_path)

    return scaler_path


def transform_file(
    input_path: str,
    scaler_path: str,
    output_path: str,
) -> str:
    """Apply a saved scaler to an Ersilia output file.

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file. Must follow the Ersilia naming convention.
    scaler_path : str
        Path to a JSON scaler file produced by :func:`fit_file`.
    output_path : str
        Where to write the scaled data. Must not already exist.

    Returns
    -------
    str
        Absolute path of the scaled output file.

    Raises
    ------
    EosframesError
        On naming convention violations, missing scaler file, pre-existing
        output path, model_id/version mismatch, or column mismatch.
    """
    logger = get_logger()

    if not is_valid_name(input_path):
        raise EosframesError(
            f"'{input_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext>"
        )

    if not is_valid_transformer_name(scaler_path):
        raise EosframesError(
            f"'{scaler_path}' does not follow the scaler naming convention. "
            "Expected: [prefix_]<model_id>_<version>_transformer.json"
        )

    parsed = parse_name(input_path)

    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    if not os.path.exists(scaler_path):
        raise EosframesError(f"Scaler file '{scaler_path}' not found.")

    with open(scaler_path) as fh:
        transformer = json.load(fh)

    t_model_id = transformer.get("model_id")
    t_version = transformer.get("version")
    f_model_id = parsed["model_id"]
    f_version = parsed["version"]

    if f_model_id != t_model_id:
        raise EosframesError(
            f"Model ID mismatch: file has '{f_model_id}' but scaler "
            f"was fitted on '{t_model_id}'."
        )
    if f_version != t_version:
        raise EosframesError(
            f"Version mismatch: file has '{f_version}' but scaler "
            f"was fitted on '{t_version}'."
        )

    from .ops import _read_file

    df = _read_file(input_path)

    logger.info("Applying scaler to %d rows from %s", len(df), input_path)
    scaled_df = transform(df, transformer)

    _write_df(scaled_df, output_path)
    logger.info("Scaled output written to %s", output_path)
    return output_path
