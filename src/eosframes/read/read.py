"""Readers for Ersilia output files (CSV, H5, chunked CSVs)."""

import os

import h5py
import pandas as pd

from ..exceptions import EosframesError
from ..logger import get_logger
from ..naming import get_model_id_from_path, get_version_from_path


def read_csv(file_path: str) -> pd.DataFrame:
    """Read an Ersilia-format CSV file into a DataFrame.

    The file is expected to have at least ``key`` and ``input`` columns
    followed by one or more feature columns. The model ID is extracted
    from the filename and attached as ``df.model_id``.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    EosframesError
        If the file does not exist, has no recognisable model ID in its
        name, or is missing required columns.
    """
    logger = get_logger()
    if not os.path.exists(file_path):
        raise EosframesError(f"File not found: '{file_path}'")
    model_id = get_model_id_from_path(file_path)
    if model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from filename '{file_path}'. "
            "The filename must contain an Ersilia model identifier (e.g. eos4e40)."
        )
    logger.info("Reading CSV: %s", file_path)
    df = pd.read_csv(file_path)
    for col in ("key", "input"):
        if col not in df.columns:
            raise EosframesError(
                f"'{file_path}' is missing the required '{col}' column."
            )
    df.model_id = model_id
    df.version = get_version_from_path(file_path)
    logger.info("Loaded %d rows (model_id=%s)", len(df), model_id)
    return df


def read_h5(h5_path: str) -> pd.DataFrame:
    """Read an Ersilia-format HDF5 file into a DataFrame.

    Expected datasets: ``values`` (N×F float), ``features`` (F strings),
    ``input`` (N strings), and optionally ``key`` (N strings).

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    EosframesError
        If the file does not exist, has no recognisable model ID, or is
        missing the ``values`` dataset.
    """
    logger = get_logger()
    if not os.path.exists(h5_path):
        raise EosframesError(f"File not found: '{h5_path}'")
    model_id = get_model_id_from_path(h5_path)
    if model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from filename '{h5_path}'. "
            "The filename must contain an Ersilia model identifier (e.g. eos4e40)."
        )
    logger.info("Reading H5: %s", h5_path)
    with h5py.File(h5_path, "r") as f:
        if "values" not in f:
            raise EosframesError(
                f"'{h5_path}' is missing the required 'values' dataset."
            )
        values = f["values"][:]
        columns = [x.decode("utf-8") for x in f["features"][:]]
        keys = [x.decode("utf-8") for x in f["key"][:]] if "key" in f else None
        inputs = [x.decode("utf-8") for x in f["input"][:]]

    meta = {"input": inputs} if keys is None else {"key": keys, "input": inputs}
    df = pd.concat([pd.DataFrame(meta), pd.DataFrame(values, columns=columns)], axis=1)
    df.model_id = model_id
    df.version = get_version_from_path(h5_path)
    logger.info("Loaded %d rows (model_id=%s)", len(df), model_id)
    return df


def read_chunked_csvs(dir_path: str) -> pd.DataFrame:
    """Read a folder of chunk CSVs produced by :func:`~eosframes.split_csv`.

    Files must be named ``chunk_<N>.csv`` (zero-padded index) and all
    share the same column layout. The model ID is
    extracted from the directory name.

    Parameters
    ----------
    dir_path : str
        Path to the directory containing the chunk CSV files.

    Returns
    -------
    pd.DataFrame
        All chunks concatenated in numeric order.

    Raises
    ------
    EosframesError
        If the directory does not exist, has no recognisable model ID,
        or contains unexpected non-CSV files.
    """
    logger = get_logger()
    if not os.path.exists(dir_path):
        raise EosframesError(f"Directory not found: '{dir_path}'")
    model_id = get_model_id_from_path(dir_path)
    if model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from directory name '{dir_path}'."
        )
    logger.info("Reading chunked CSVs from: %s", dir_path)

    filenames = os.listdir(dir_path)
    batch_ids = []
    zfill = 0
    prefixes = []
    for fn in filenames:
        if not fn.endswith(".csv") or not fn.startswith("chunk"):
            raise EosframesError(
                f"Unexpected file '{fn}' in '{dir_path}'. "
                "Chunk folders should contain only files named chunk_<N>.csv."
            )
        parts = fn.split("_")
        batch_id_str = parts[-1].split(".")[0]
        zfill = len(batch_id_str)
        batch_ids.append(int(batch_id_str))
        prefixes.append("_".join(parts[:-1]))

    if len(set(prefixes)) > 1:
        raise EosframesError(
            f"Multiple file prefixes found in '{dir_path}': {set(prefixes)}. "
            "All chunk files must share the same prefix."
        )

    prefix = prefixes[0]
    frames = [
        pd.read_csv(os.path.join(dir_path, f"{prefix}_{str(i).zfill(zfill)}.csv"))
        for i in sorted(batch_ids)
    ]
    df = pd.concat(frames, axis=0).reset_index(drop=True)
    df.model_id = model_id
    df.version = get_version_from_path(dir_path)
    logger.info(
        "Loaded %d rows from %d chunks (model_id=%s)", len(df), len(frames), model_id
    )
    return df
