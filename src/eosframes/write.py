"""Writers for Ersilia output files (CSV, H5, chunked CSVs).

Every write path in this module enforces three invariants:

1. **No silent overwrite.** Writers refuse to clobber an existing file or
   directory and raise :class:`~eosframes.EosframesError` with guidance
   to delete the existing target first.
2. **Model-ID match.** Writers extract the model ID from the destination
   path with :func:`eosframes.naming.get_model_id_from_path` and compare
   it against ``df.model_id``; mismatches raise.
3. **Required attributes.** Every DataFrame passed in must have a
   ``model_id`` attribute. Set it via ``df.model_id = "..."`` after
   transformations that drop loose attributes (``concat`` /
   ``drop_duplicates``).
"""

import os
from typing import Union

import h5py
import numpy as np
import pandas as pd

from .exceptions import EosframesError
from .logger import get_logger
from .naming import get_model_id_from_path
from .utils import chunker


def write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Save a DataFrame as a CSV file in Ersilia format.

    The CSV is written with no row index. The model ID encoded in
    *csv_path* must match ``df.model_id`` (extracted leniently from the
    path basename — the strict canonical naming pattern is enforced by
    the higher-level operations in :mod:`eosframes.ops`, not here).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save. Must have a ``model_id`` attribute set.
    csv_path : str
        Destination path. Must end in ``.csv`` and contain a valid
        Ersilia model identifier somewhere in its basename.

    Raises
    ------
    EosframesError
        If the file already exists, the extension is wrong, the model
        ID cannot be resolved from the path, ``df.model_id`` is missing,
        or the path-encoded and DataFrame model IDs disagree.
    """
    logger = get_logger()
    if os.path.exists(csv_path):
        raise EosframesError(
            f"File '{csv_path}' already exists. Remove it before saving."
        )
    if not csv_path.endswith(".csv"):
        raise EosframesError(f"Output path must end in '.csv', got: '{csv_path}'")
    path_model_id = get_model_id_from_path(csv_path)
    if path_model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from '{csv_path}'. "
            "The filename must contain an Ersilia model identifier."
        )
    df_model_id = getattr(df, "model_id", None)
    if df_model_id is None:
        raise EosframesError("DataFrame does not have a 'model_id' attribute.")
    if path_model_id != df_model_id:
        raise EosframesError(
            f"Model ID mismatch: filename encodes '{path_model_id}' "
            f"but DataFrame has model_id='{df_model_id}'."
        )
    logger.info("Writing CSV: %s (%d rows)", csv_path, len(df))
    df.reset_index(drop=True).to_csv(csv_path, index=False)
    logger.info("Done: %s", csv_path)


def write_h5(df: pd.DataFrame, h5_path: str, dtype: Union[np.dtype, str]) -> None:
    """Save a DataFrame as an HDF5 file in Ersilia format.

    Writes four datasets:

    * ``key``      — UTF-8 strings, ``(N,)`` (only if ``df`` has a ``key`` column)
    * ``input``    — UTF-8 strings, ``(N,)``
    * ``features`` — UTF-8 strings, ``(F,)`` (the feature column names)
    * ``values``   — numeric, ``(N, F)``, with the dtype provided by *dtype*

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save. Must have a ``model_id`` attribute and an
        ``input`` column.
    h5_path : str
        Destination path. Must contain a valid Ersilia model identifier
        somewhere in its basename.
    dtype : numpy.dtype or str
        NumPy dtype for the ``values`` dataset (e.g. ``numpy.float32``,
        ``numpy.int8``). Non-float dtypes will quantize the values.

    Raises
    ------
    EosframesError
        If the file already exists, the model ID cannot be resolved,
        ``df.model_id`` is missing, or the path-encoded and DataFrame
        model IDs disagree.
    """
    logger = get_logger()
    if os.path.exists(h5_path):
        raise EosframesError(
            f"File '{h5_path}' already exists. Remove it before saving."
        )
    path_model_id = get_model_id_from_path(h5_path)
    if path_model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from '{h5_path}'. "
            "The filename must contain an Ersilia model identifier."
        )
    df_model_id = getattr(df, "model_id", None)
    if df_model_id is None:
        raise EosframesError("DataFrame does not have a 'model_id' attribute.")
    if path_model_id != df_model_id:
        raise EosframesError(
            f"Model ID mismatch: filename encodes '{path_model_id}' "
            f"but DataFrame has model_id='{df_model_id}'."
        )
    df = df.reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    logger.info("Writing H5: %s (%d rows)", h5_path, len(df))
    with h5py.File(h5_path, "w") as f:
        dt_str = h5py.string_dtype(encoding="utf-8")
        if "key" in df.columns:
            f.create_dataset("key", data=df["key"].astype(str).tolist(), dtype=dt_str)
        f.create_dataset("input", data=df["input"].astype(str).tolist(), dtype=dt_str)
        f.create_dataset("features", data=feature_cols, dtype=dt_str)
        f.create_dataset("values", data=df[feature_cols].values, dtype=dtype)
    logger.info("Done: %s", h5_path)


def write_chunked_csvs(df: pd.DataFrame, dir_path: str, chunksize: int) -> None:
    """Split a DataFrame into chunk CSV files inside *dir_path*.

    Creates *dir_path* (which must not already exist) and writes
    ``chunk_<N>.csv`` files into it, where ``<N>`` is a zero-padded
    integer wide enough to accommodate the largest chunk index. Every
    chunk has the same column header.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to chunk. Must have a ``model_id`` attribute.
    dir_path : str
        Directory to create. Must contain a valid Ersilia model
        identifier in its basename and must not already exist.
    chunksize : int
        Rows per chunk. Hard-capped at 100 000.

    Raises
    ------
    EosframesError
        If the directory already exists, the model ID cannot be
        resolved, ``df.model_id`` is missing, the path-encoded and
        DataFrame model IDs disagree, or *chunksize* exceeds the limit.
    """
    logger = get_logger()
    if chunksize > 100_000:
        raise EosframesError(f"chunksize {chunksize} exceeds the limit of 100 000.")
    path_model_id = get_model_id_from_path(dir_path)
    if path_model_id is None:
        raise EosframesError(
            f"Could not extract a model ID from '{dir_path}'. "
            "The directory name must contain an Ersilia model identifier."
        )
    df_model_id = getattr(df, "model_id", None)
    if df_model_id is None:
        raise EosframesError("DataFrame does not have a 'model_id' attribute.")
    if path_model_id != df_model_id:
        raise EosframesError(
            f"Model ID mismatch: directory encodes '{path_model_id}' "
            f"but DataFrame has model_id='{df_model_id}'."
        )
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        raise EosframesError(
            f"Directory '{dir_path}' already exists. Remove it before saving."
        )
    os.mkdir(dir_path)
    num_chunks = (len(df) + chunksize - 1) // chunksize
    zfill = len(str(max(num_chunks - 1, 0)))
    logger.info(
        "Writing %d rows to %s in %d chunks of up to %d rows each",
        len(df),
        dir_path,
        num_chunks,
        chunksize,
    )
    for i, chunk in enumerate(chunker(df.reset_index(drop=True), chunksize)):
        chunk.to_csv(
            os.path.join(dir_path, f"chunk_{str(i).zfill(zfill)}.csv"), index=False
        )
    logger.info("Done: %s (%d chunk files)", dir_path, num_chunks)
