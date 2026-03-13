"""File-level operations for splitting, converting, stacking, appending, and deduplicating."""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from .exceptions import EosframesError
from .logger import get_logger
from .naming import is_valid_name, parse_name
from .read.read import read_csv, read_h5
from .utils.utils import chunker
from .write.write import write_csv, write_h5


def _read_file(path: str) -> pd.DataFrame:
    """Read a CSV or H5 file into a DataFrame with model_id set."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv(path)
    if ext == ".h5":
        return read_h5(path)
    raise EosframesError(
        f"Unsupported format '{ext}' for '{path}'. Expected .csv or .h5"
    )


def split_csv(input_path: str, output_folder: str, chunksize: int = 10000) -> int:
    """Split a CSV file into numbered chunk files inside a folder.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file. No naming convention required.
    output_folder : str
        Path to the folder to create. Must not already exist.
    chunksize : int
        Number of rows per chunk (default 10000).

    Returns
    -------
    int
        Number of chunk files written.

    Raises
    ------
    EosframesError
        If the output folder already exists.
    """
    logger = get_logger()
    if os.path.exists(output_folder):
        raise EosframesError(
            f"Output folder '{output_folder}' already exists. "
            "Remove it or choose a different name."
        )
    df = pd.read_csv(input_path)
    total_rows = len(df)
    num_chunks = (total_rows + chunksize - 1) // chunksize
    zfill = 6 if num_chunks >= 1000 else 3
    os.makedirs(output_folder)
    logger.info(
        "Splitting %d rows into %d chunks (chunksize=%d) → %s",
        total_rows,
        num_chunks,
        chunksize,
        output_folder,
    )
    for i, chunk in enumerate(chunker(df, chunksize)):
        fname = f"chunk_{str(i).zfill(zfill)}.csv"
        chunk.to_csv(os.path.join(output_folder, fname), index=False)
    logger.info("Split complete: %d chunks written to %s", num_chunks, output_folder)
    return num_chunks


def convert_file(input_path: str, output_path: str) -> None:
    """Convert a file between formats, or assemble a folder of chunks.

    Supported conversions:

    * folder of CSVs  → ``.csv`` or ``.h5``
    * ``.csv``        → ``.h5``
    * ``.h5``         → ``.csv``

    Parameters
    ----------
    input_path : str
        A CSV file, an H5 file, or a folder of chunk CSVs.
    output_path : str
        Output file path. Must follow the Ersilia naming convention.

    Raises
    ------
    EosframesError
        On naming convention violations, existing output, or unsupported formats.
    """
    logger = get_logger()
    if not is_valid_name(output_path):
        raise EosframesError(
            f"Output '{output_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )
    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    parsed = parse_name(output_path)
    model_id = parsed["model_id"]
    out_ext = parsed["extension"]

    if os.path.isdir(input_path):
        csv_files = sorted(f for f in os.listdir(input_path) if f.endswith(".csv"))
        if not csv_files:
            raise EosframesError(f"No CSV files found in '{input_path}'")
        logger.info("Reading %d chunk files from %s", len(csv_files), input_path)
        frames = [pd.read_csv(os.path.join(input_path, f)) for f in csv_files]
        df = pd.concat(frames, axis=0).reset_index(drop=True)
    else:
        in_ext = os.path.splitext(input_path)[1].lower()
        if in_ext == ".csv":
            if is_valid_name(input_path):
                df = read_csv(input_path)
            else:
                logger.info("Reading %s", input_path)
                df = pd.read_csv(input_path)
        elif in_ext == ".h5":
            df = read_h5(input_path)
        else:
            raise EosframesError(
                f"Unsupported input format '{in_ext}'. Expected .csv or .h5"
            )

    logger.info("Converting %s → %s", input_path, output_path)
    df.model_id = model_id

    if out_ext == "csv":
        write_csv(df, output_path)
    else:
        write_h5(df, output_path, dtype=np.float32)

    logger.info("Done: %s", output_path)


def stack_files(input_paths: List[str], output_path: str, suffix: bool = True) -> None:
    """Horizontally stack outputs from multiple Ersilia models into one CSV.

    All input files must contain the same inputs in the same order. Each
    model ID may appear only once.

    Parameters
    ----------
    input_paths : list of str
        Two or more CSV or H5 files, each following the naming convention.
    output_path : str
        Output CSV path (no naming convention required).
    suffix : bool
        If True (default), append ``.model_id`` to feature column names.

    Raises
    ------
    EosframesError
        On duplicate models, input mismatch, or invalid naming.
    """
    logger = get_logger()
    if len(input_paths) < 2:
        raise EosframesError("At least two input files are required for stacking.")
    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )
    if not output_path.endswith(".csv"):
        raise EosframesError("Output must be a .csv file.")

    dfs = []
    seen_model_ids: List[str] = []
    for path in input_paths:
        if not is_valid_name(path):
            raise EosframesError(
                f"'{path}' does not follow the naming convention. "
                "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv)"
            )
        logger.info("Reading %s", path)
        df = _read_file(path)
        model_id = getattr(df, "model_id", None)
        if model_id in seen_model_ids:
            raise EosframesError(
                f"Model '{model_id}' appears more than once in the input list. "
                "Each model must be unique when stacking."
            )
        seen_model_ids.append(model_id)
        dfs.append(df)

    reference_inputs = dfs[0]["input"].tolist()
    for i, df in enumerate(dfs[1:], start=2):
        if "input" not in df.columns:
            raise EosframesError(f"File #{i} does not contain an 'input' column.")
        if df["input"].tolist() != reference_inputs:
            raise EosframesError(
                f"Input mismatch: file #{i} has different inputs or a different row order "
                "than file #1. Stacking requires all files to have identical inputs in the same order."
            )

    meta_cols = [c for c in ("key", "input") if c in dfs[0].columns]
    result = dfs[0][meta_cols].reset_index(drop=True).copy()
    for df in dfs:
        model_id = getattr(df, "model_id", None)
        feature_cols = [c for c in df.columns if c not in {"key", "input"}]
        block = df[feature_cols].reset_index(drop=True)
        if suffix:
            block = block.rename(columns={c: f"{c}.{model_id}" for c in feature_cols})
        result = pd.concat([result, block], axis=1)

    logger.info(
        "Stacked %d files × %d rows → %d feature columns",
        len(dfs),
        len(result),
        len(result.columns) - len(meta_cols),
    )
    result.to_csv(output_path, index=False)
    logger.info("Done: %s", output_path)


def append_files(input_paths: List[str], output_path: str) -> None:
    """Vertically concatenate files from the same Ersilia model.

    All input files must share the same model ID and identical columns.
    Rows are appended in the order given.

    Parameters
    ----------
    input_paths : list of str
        Two or more CSV or H5 files.
    output_path : str
        Output file path. Must follow the naming convention.

    Raises
    ------
    EosframesError
        On model ID mismatch, column mismatch, or invalid naming.
    """
    logger = get_logger()
    if len(input_paths) < 2:
        raise EosframesError("At least two input files are required for appending.")
    if not is_valid_name(output_path):
        raise EosframesError(
            f"Output '{output_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )
    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    out_parsed = parse_name(output_path)
    expected_model_id = out_parsed["model_id"]
    out_ext = out_parsed["extension"]

    dfs = []
    reference_columns = None
    for path in input_paths:
        logger.info("Reading %s", path)
        df = _read_file(path)
        model_id = getattr(df, "model_id", None)
        if model_id != expected_model_id:
            raise EosframesError(
                f"Model ID mismatch: '{path}' has model '{model_id}' "
                f"but output expects '{expected_model_id}'."
            )
        cols = list(df.columns)
        if reference_columns is None:
            reference_columns = cols
        elif cols != reference_columns:
            raise EosframesError(
                f"Column mismatch: '{path}' has columns {cols} "
                f"but expected {reference_columns}."
            )
        dfs.append(df)

    result = pd.concat(dfs, axis=0).reset_index(drop=True)
    result.model_id = expected_model_id
    logger.info("Appended %d files → %d rows total", len(dfs), len(result))

    if out_ext == "csv":
        write_csv(result, output_path)
    else:
        write_h5(result, output_path, dtype=np.float32)

    logger.info("Done: %s", output_path)


def dedupe_file(input_path: str, output_path: str) -> Tuple[int, int]:
    """Remove duplicate rows by key, keeping the first occurrence.

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file.
    output_path : str
        Output file path. Must follow the naming convention.

    Returns
    -------
    tuple of (int, int)
        ``(rows_before, rows_after)``

    Raises
    ------
    EosframesError
        On naming convention violations or model ID mismatch.
    """
    logger = get_logger()
    if not is_valid_name(output_path):
        raise EosframesError(
            f"Output '{output_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )
    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    out_parsed = parse_name(output_path)
    expected_model_id = out_parsed["model_id"]
    out_ext = out_parsed["extension"]

    logger.info("Reading %s", input_path)
    df = _read_file(input_path)

    model_id = getattr(df, "model_id", None)
    if model_id != expected_model_id:
        raise EosframesError(
            f"Model ID mismatch: '{input_path}' has model '{model_id}' "
            f"but output expects '{expected_model_id}'."
        )
    if "key" not in df.columns:
        raise EosframesError(f"'{input_path}' does not contain a 'key' column.")

    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    after = len(df)
    logger.info("Removed %d duplicate(s), %d rows remaining", before - after, after)

    df.model_id = expected_model_id

    if out_ext == "csv":
        write_csv(df, output_path)
    else:
        write_h5(df, output_path, dtype=np.float32)

    logger.info("Done: %s", output_path)
    return before, after
