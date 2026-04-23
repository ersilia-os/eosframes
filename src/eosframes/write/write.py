"""Writers for Ersilia output files (CSV, H5, chunked CSVs, XLSX)."""

import os

import h5py
import pandas as pd

from ..exceptions import EosframesError
from ..hub import _fetch_model_slug, _fetch_model_title, _fetch_run_columns
from ..logger import get_logger
from ..naming import get_model_id_from_path, is_model_id_valid
from ..utils.utils import chunker

# Google Sheets colour palette used to highlight model groups in XLSX output
_PALETTE = [
    "#3366CC",
    "#DC3912",
    "#FF9900",
    "#109618",
    "#990099",
    "#3B3EAC",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
    "#316395",
    "#994499",
    "#22AA99",
    "#AAAA11",
    "#6633CC",
    "#E67300",
    "#8B0707",
    "#329262",
    "#5574A6",
    "#3B3EAC",
]


def _get_colors(n: int) -> list:
    """Return *n* colours from the palette, cycling if necessary."""
    return (_PALETTE * (n // len(_PALETTE) + 1))[:n]


def write_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Save a DataFrame as a CSV file in Ersilia format.

    The model ID encoded in *csv_path* must match ``df.model_id``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save. Must have a ``model_id`` attribute.
    csv_path : str
        Destination path. Must end in ``.csv`` and contain a valid
        Ersilia model identifier.

    Raises
    ------
    EosframesError
        If the file already exists, the extension is wrong, or the model
        ID cannot be resolved / does not match the DataFrame.
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


def write_h5(df: pd.DataFrame, h5_path: str, dtype) -> None:
    """Save a DataFrame as an HDF5 file in Ersilia format.

    The model ID encoded in *h5_path* must match ``df.model_id``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save. Must have a ``model_id`` attribute.
    h5_path : str
        Destination path. Must contain a valid Ersilia model identifier.
    dtype
        NumPy dtype for the ``values`` dataset (e.g. ``np.float32``).

    Raises
    ------
    EosframesError
        If the file already exists or the model ID cannot be resolved /
        does not match the DataFrame.
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

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to chunk. Must have a ``model_id`` attribute.
    dir_path : str
        Directory to create. Must contain a valid Ersilia model identifier
        and must not already exist.
    chunksize : int
        Rows per chunk. Capped at 100 000.

    Raises
    ------
    EosframesError
        If the directory already exists, the model ID cannot be resolved,
        or *chunksize* exceeds the limit.
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
    logger.info("Writing %d rows to %s in chunks of %d", len(df), dir_path, chunksize)
    for i, chunk in enumerate(chunker(df.reset_index(drop=True), chunksize)):
        chunk.to_csv(
            os.path.join(dir_path, f"chunk_{str(i).zfill(zfill)}.csv"), index=False
        )
    logger.info("Done: %s", dir_path)


def write_xlsx(df: pd.DataFrame, xlsx_path: str) -> None:
    """Save a stacked DataFrame as a formatted Excel workbook.

    Expects feature columns to have ``.model_id`` suffixes (as produced
    by :func:`~eosframes.stack_files` with ``suffix=True``). Fetches
    model slugs, titles, and column definitions from GitHub to populate
    a *Legend* sheet.

    Parameters
    ----------
    df : pd.DataFrame
        Stacked DataFrame with ``key``, ``input``, and suffixed feature
        columns.
    xlsx_path : str
        Destination path (must end in ``.xlsx``).

    Raises
    ------
    EosframesError
        If the extension is wrong or a column has an invalid model ID
        suffix.
    """
    if not xlsx_path.endswith(".xlsx"):
        raise EosframesError(f"Output path must end in '.xlsx', got: '{xlsx_path}'")
    if os.path.exists(xlsx_path):
        os.remove(xlsx_path)

    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    model_ids = []
    for col in feature_cols:
        mid = col.split(".")[-1]
        if not is_model_id_valid(mid):
            raise EosframesError(
                f"Column '{col}' does not have a valid model ID suffix."
            )
        if mid not in model_ids:
            model_ids.append(mid)

    colors = _get_colors(len(model_ids))

    legend_rows = [
        {
            "model_id": mid,
            "slug": _fetch_model_slug(mid),
            "title": _fetch_model_title(mid),
            "link": f"https://github.com/ersilia-os/{mid}",
        }
        for mid in model_ids
    ]
    dl = pd.DataFrame(legend_rows)

    col_frames = []
    col_colors = []
    for i, mid in enumerate(model_ids):
        dc_ = _fetch_run_columns(mid)
        dc_.insert(0, "model_id", mid)
        col_colors.extend([colors[i]] * len(dc_))
        col_frames.append(dc_)
    dc = pd.concat(col_frames, axis=0).reset_index(drop=True)

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        # Data sheet
        df.to_excel(writer, sheet_name="Data", index=False, startrow=0, startcol=0)
        ws = writer.sheets["Data"]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, 0, len(df.columns) - 1)
        for i, col in enumerate(df.columns):
            width = min(max(df[col].astype(str).map(len).max(), len(str(col))) + 2, 50)
            ws.set_column(i, i, width)

        # Legend sheet
        dl.to_excel(writer, sheet_name="Legend", index=False, startrow=1, startcol=0)
        dc.to_excel(
            writer,
            sheet_name="Legend",
            index=False,
            startrow=1,
            startcol=dl.shape[1] + 1,
        )
        ws_leg = writer.sheets["Legend"]
        bold_center = writer.book.add_format({"align": "center", "bold": True})
        ws_leg.merge_range(0, 0, 0, dl.shape[1] - 1, "Ersilia models", bold_center)
        ws_leg.merge_range(
            0, dl.shape[1] + 1, 0, dl.shape[1] + dc.shape[1], "Columns", bold_center
        )
        ws_leg.freeze_panes(2, 0)
        ws_leg.set_column(0, dl.shape[1] - 1, 30)
        ws_leg.set_column(dl.shape[1], dl.shape[1] + dc.shape[1] - 1, 30)
