"""File-level operations: split, convert, stack, unstack, append, dedupe.

Every public function in this module follows the same shape:

1. Validate the destination path against the naming convention
   (:func:`eosframes.naming.is_valid_name` and friends).
2. Refuse to overwrite an existing output.
3. Read the input(s) with :func:`eosframes.read.read_csv` /
   :func:`eosframes.read.read_h5`, which attach ``df.model_id`` /
   ``df.version``.
4. Cross-validate model IDs (and versions where relevant) between input
   and output before performing the operation.
5. Re-set ``df.model_id`` / ``df.version`` after any pandas operation
   that may drop loose attributes (``concat`` / ``drop_duplicates``),
   then write via :func:`eosframes.write.write_csv` /
   :func:`eosframes.write.write_h5`.

The CLI in :mod:`eosframes.cli` is a thin Click adapter on top of these
functions; business logic lives here.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import hub
from .exceptions import EosframesError
from .logger import get_logger
from .naming import (
    is_model_id_valid,
    is_valid_name,
    is_valid_stack_explicit_name,
    is_valid_stack_mix_name,
    make_stack_explicit_name,
    make_stack_mix_name,
    parse_name,
    parse_stack_explicit_name,
    parse_stack_mix_name,
)
from .read import read_csv, read_h5
from .stack import hstack
from .utils import chunker
from .write import write_csv, write_h5


def _read_file(path: str) -> pd.DataFrame:
    """Read a CSV or H5 file into a DataFrame with ``model_id`` set.

    Internal dispatcher that delegates to :func:`eosframes.read.read_csv`
    or :func:`eosframes.read.read_h5` based on the file extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv(path)
    if ext == ".h5":
        return read_h5(path)
    raise EosframesError(
        f"Unsupported format '{ext}' for '{path}'. Expected .csv or .h5"
    )


def _require_no_overwrite(path: str, *, kind: str = "file") -> None:
    """Refuse to clobber an existing output.

    The two-message split (``"Remove it first."`` vs ``"Remove it or choose
    a different name."``) is preserved by the *kind* argument so error
    text users may already script against doesn't shift.
    """
    if not os.path.exists(path):
        return
    if kind == "folder":
        raise EosframesError(
            f"Output folder '{path}' already exists. "
            "Remove it or choose a different name."
        )
    raise EosframesError(f"Output file '{path}' already exists. Remove it first.")


def _compute_summary_stats(df: pd.DataFrame) -> List[Dict]:
    """Per-feature summary statistics for an Ersilia output frame.

    Returns one dict per non-meta column, with keys
    ``column / dtype / missing / min / mean / max``. Numeric ``min`` /
    ``mean`` / ``max`` are ``None`` for non-numeric columns or columns
    that are fully missing — pretty-printers handle the rendering.

    Lifted out of :func:`eosframes.cli.summary` so the CLI command shrinks
    to argument parsing + Rich rendering and the math can be tested
    directly.
    """
    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    stats_rows: List[Dict] = []
    for col in feature_cols:
        series = df[col]
        row: Dict = {
            "column": col,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "min": None,
            "mean": None,
            "max": None,
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean):
                row["min"] = float(clean.min())
                row["mean"] = float(clean.mean())
                row["max"] = float(clean.max())
        stats_rows.append(row)
    return stats_rows


def _require_valid_output_name(path: str) -> Dict:
    """Validate a data-file output path against the naming convention.

    Returns the :func:`parse_name` dict on success; raises with the
    standard ``"Expected: <model_id>_<version>.<ext>"`` message otherwise.
    Used by every op whose output is a single data file (convert,
    append, dedupe).
    """
    if not is_valid_name(path):
        raise EosframesError(
            f"Output '{path}' does not follow the naming convention. "
            "Expected: [prefix_]<model_id>_<version>.<ext> "
            "with ext in {csv, h5}."
        )
    return parse_name(path)


def split_csv(input_path: str, output_folder: str, chunksize: int = 10000) -> int:
    """Split a CSV file into numbered chunk files inside a folder.

    This is the only operation in the module that accepts inputs without
    a model ID — its purpose is to pre-process inputs *before* a model
    run, when the model is not yet known. Each chunk preserves the
    original header. The chunk index is zero-padded to a width that
    accommodates the largest index (``chunk_000.csv`` / ``chunk_007.csv``
    for up to 1 000 chunks; ``chunk_000000.csv`` for >1 000).

    Parameters
    ----------
    input_path : str
        Path to the input CSV file. No naming convention required.
    output_folder : str
        Path to the folder to create. Must not already exist.
    chunksize : int, default ``10000``
        Number of rows per chunk. Smaller values produce more files;
        larger values mean each chunk takes longer to run downstream.

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
    _require_no_overwrite(output_folder, kind="folder")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    num_chunks = (total_rows + chunksize - 1) // chunksize
    zfill = len(str(max(num_chunks - 1, 0)))
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

    * Folder of chunk CSVs → ``.csv`` (concatenate)
    * Folder of chunk CSVs → ``.h5``  (concatenate, write HDF5)
    * ``.csv``             → ``.h5``
    * ``.h5``              → ``.csv``

    The output is always a single file (the inverse — splitting a
    single file into chunks — is :func:`split_csv`). When the input is
    a folder, every CSV inside it is read with raw ``pandas.read_csv``
    (no naming-convention check on the chunk files); the model ID is
    taken from *output_path*.

    Parameters
    ----------
    input_path : str
        A CSV file, an H5 file, or a folder of chunk CSVs.
    output_path : str
        Output file path. Must follow the Ersilia naming convention
        (``[prefix_]<model_id>_<version>.csv`` or ``.h5``).

    Raises
    ------
    EosframesError
        On naming convention violations of *output_path*, existing
        output, an empty input folder, or unsupported file extensions.
    """
    logger = get_logger()
    parsed = _require_valid_output_name(output_path)
    _require_no_overwrite(output_path)

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


def stack_files(input_paths: List[str], output_path: str) -> None:
    """Horizontally stack outputs from multiple Ersilia models into one CSV.

    The *output filename* selects the column-naming mode:

    * **Mode A (eosmix)** — ``[prefix_]eosmix.csv``. Feature columns are
      suffixed with ``_<model_id>_<version>`` so column names carry the
      provenance. The output filename does not embed the model list.
    * **Mode B (explicit)** — ``[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv``.
      Feature columns stay bare. The output filename must list every
      stacked ``(model_id, version)`` in the same order as
      *input_paths*.

    All input files must follow the Ersilia naming convention and
    contain the same molecules in the same row order. Duplicate
    ``(model_id, version)`` pairs across inputs are rejected — they
    would collide in Mode A and produce ambiguous filenames in Mode B.

    Parameters
    ----------
    input_paths : list of str
        Two or more CSV or H5 files, each following the naming
        convention.
    output_path : str
        Output CSV path. Must follow either Mode A or Mode B.

    Raises
    ------
    EosframesError
        On naming violations, duplicate ``(model_id, version)`` pairs,
        Mode B model-order mismatch between filename and input order,
        pre-existing output, or input row mismatch.
    """
    logger = get_logger()
    if len(input_paths) < 2:
        raise EosframesError("At least two input files are required for stacking.")
    _require_no_overwrite(output_path)

    dfs = []
    input_pairs: List[Tuple[str, str]] = []  # (model_id, version) per input
    for path in input_paths:
        parsed = parse_name(path)
        if parsed is None or parsed["name_type"] not in {"csv", "h5"}:
            raise EosframesError(
                f"'{path}' does not follow the naming convention. "
                "Expected: [prefix_]<model_id>_<version>.<ext> "
                "with ext in {csv, h5}."
            )
        logger.info("Reading %s", path)
        df = _read_file(path)
        input_pairs.append((parsed["model_id"], parsed["version"]))
        dfs.append(df)

    # Resolve mode from the output filename.
    mix_suggestion = make_stack_mix_name()
    explicit_suggestion = make_stack_explicit_name(input_pairs)

    if is_valid_stack_mix_name(output_path):
        mode = "eosmix"
    elif is_valid_stack_explicit_name(output_path):
        out_parsed = parse_stack_explicit_name(output_path) or {"models": []}
        out_pairs = out_parsed["models"]
        if out_pairs != input_pairs:
            raise EosframesError(
                f"Model order mismatch in output filename '{os.path.basename(output_path)}'.\n"
                f"  From --input:  {input_pairs}\n"
                f"  From --output: {out_pairs}\n"
                "The output filename must list each (model_id, version) in the "
                "same order as the inputs.\n"
                f"Try: {explicit_suggestion}"
            )
        mode = "explicit"
    else:
        raise EosframesError(
            f"'{os.path.basename(output_path)}' does not follow a stack "
            "naming convention.\n\n"
            "Choose exactly one of:\n"
            "  Mode A (eosmix):   [prefix]_eosmix.csv\n"
            "      Feature columns are suffixed with _<model_id>_<version>.\n"
            f"      Try: {mix_suggestion}\n\n"
            "  Mode B (explicit): [prefix]_<m1>_<v1>_..._<mN>_<vN>.csv\n"
            "      Each stacked (model_id, version) appears in the filename "
            "in -i order. Columns stay bare.\n"
            f"      Try: {explicit_suggestion}"
        )

    result = hstack(dfs, mode=mode)

    meta_cols = [c for c in ("key", "input") if c in result.columns]
    logger.info(
        "Stacked %d files × %d rows → %d feature columns (mode=%s)",
        len(dfs),
        len(result),
        len(result.columns) - len(meta_cols),
        mode,
    )
    result.to_csv(output_path, index=False)
    logger.info("Done: %s", output_path)


def append_files(input_paths: List[str], output_path: str) -> None:
    """Vertically concatenate files from the same Ersilia model.

    All input files must share the same model ID (encoded in their
    filenames) and have identical column layouts. Rows are appended in
    the order given. Duplicate keys, if any, are *not* removed —
    follow with :func:`dedupe_file` if needed.

    Parameters
    ----------
    input_paths : list of str
        Two or more CSV or H5 files. Each must follow the naming
        convention.
    output_path : str
        Output file path. Must follow the naming convention; its
        encoded model ID determines the expected model ID for all
        inputs.

    Raises
    ------
    EosframesError
        On invalid output naming, pre-existing output, model-ID
        mismatch between an input and the output, or column mismatch
        across inputs.
    """
    logger = get_logger()
    if len(input_paths) < 2:
        raise EosframesError("At least two input files are required for appending.")
    out_parsed = _require_valid_output_name(output_path)
    _require_no_overwrite(output_path)

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
    """Remove duplicate rows by ``key``, keeping the first occurrence.

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file. Must follow the naming convention and
        contain a ``key`` column.
    output_path : str
        Output file path. Must follow the naming convention; its
        encoded model ID must match the input.

    Returns
    -------
    rows_before : int
        Row count of the input file.
    rows_after : int
        Row count after deduplication. ``rows_before - rows_after`` is
        the number of duplicate rows that were dropped.

    Raises
    ------
    EosframesError
        On naming convention violations, pre-existing output,
        model-ID mismatch, or a missing ``key`` column.
    """
    logger = get_logger()
    out_parsed = _require_valid_output_name(output_path)
    _require_no_overwrite(output_path)

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


# Matches an eosmix column suffix at the END of a name:
#   "<original>_<model_id>_<version>" where model_id = eos<d><3 alnum>, version = v<digits>
_EOSMIX_COL_RE = re.compile(
    r"^(?P<original>.+)_(?P<model_id>eos\d[A-Za-z0-9]{3})_(?P<version>v\d+)$"
)


def _classify_stack_columns_mode_a(
    feature_cols: List[str],
) -> List[Tuple[str, str, str, str]]:
    """Parse Mode A feature columns into (original, model_id, version, suffixed).

    Raises ``EosframesError`` listing any columns whose name doesn't match
    the eosmix suffix pattern.
    """
    parsed = []
    bad: List[str] = []
    for col in feature_cols:
        m = _EOSMIX_COL_RE.match(col)
        if not m or not is_model_id_valid(m.group("model_id")):
            bad.append(col)
            continue
        parsed.append(
            (m.group("original"), m.group("model_id"), m.group("version"), col)
        )
    if bad:
        raise EosframesError(
            "Some feature columns do not follow the Mode A suffix pattern "
            "'<original>_<model_id>_<version>':\n  "
            + ", ".join(bad[:10])
            + (" ..." if len(bad) > 10 else "")
        )
    return parsed


def _classify_stack_columns_mode_b(
    feature_cols: List[str], models: List[Tuple[str, str]]
) -> List[Tuple[str, str, str]]:
    """Assign each bare feature column to a (model_id, version) via run_columns.csv.

    Fetches run_columns.csv for each model, checks that every feature column
    maps to exactly one model, and returns ``[(col, model_id, version), ...]``.

    Raises ``EosframesError`` on ambiguous (a column listed for 2+ models),
    unmatched (a column not listed for any model), or missing (a model's
    run_columns lists columns not present in the stack) cases.
    """
    # Fetch run_columns.csv for each model → dict of model -> set of column names.
    model_cols: List[Tuple[Tuple[str, str], set]] = []
    for model_id, version in models:
        df_cols = hub.fetch_columns(model_id, version)
        if "name" not in df_cols.columns:
            raise EosframesError(
                f"run_columns.csv for {model_id} {version} is missing a 'name' column."
            )
        model_cols.append(((model_id, version), set(df_cols["name"].astype(str))))

    feat_set = set(feature_cols)
    assignments: List[Tuple[str, str, str]] = []

    # 1. Check ambiguity (a column listed for 2+ models) and unmatched (a
    #    column not listed for any model).
    column_owners: dict = {}
    for (model_id, version), cols in model_cols:
        for c in cols:
            column_owners.setdefault(c, []).append((model_id, version))
    ambiguous = {
        c: owners
        for c, owners in column_owners.items()
        if len(owners) > 1 and c in feat_set
    }
    if ambiguous:
        msg_lines = [f"  {c!r}: {owners}" for c, owners in list(ambiguous.items())[:5]]
        raise EosframesError(
            "Ambiguous columns — the following appear in run_columns.csv of multiple models:\n"
            + "\n".join(msg_lines)
        )
    unmatched = [c for c in feature_cols if c not in column_owners]
    if unmatched:
        raise EosframesError(
            "Unmatched feature columns — none of the stacked models' "
            "run_columns.csv lists these:\n  "
            + ", ".join(unmatched[:10])
            + (" ..." if len(unmatched) > 10 else "")
        )

    # 2. Check every model's expected columns are present in the stack.
    missing_report = []
    for (model_id, version), cols in model_cols:
        missing = [c for c in cols if c not in feat_set]
        if missing:
            missing_report.append(
                f"  {model_id} {version} is missing: {', '.join(missing[:10])}"
                + (" ..." if len(missing) > 10 else "")
            )
    if missing_report:
        raise EosframesError(
            "Stack file is missing feature columns required by the models' run_columns.csv:\n"
            + "\n".join(missing_report)
        )

    # 3. Build the assignment in the input order, respecting the model list order.
    for (model_id, version), cols in model_cols:
        for c in feature_cols:
            if c in cols:
                assignments.append((c, model_id, version))

    return assignments


def unstack_file(input_path: str, output_folder: str) -> List[str]:
    """Split a horizontally stacked CSV back into per-model files.

    The mode is resolved from the input filename:

    * Mode A (``[prefix]_eosmix.csv``) — column names carry the model
      provenance. Columns are grouped by the ``_<model_id>_<version>``
      suffix; the suffix is stripped when writing each per-model file.
    * Mode B (``[prefix]_<m1>_<v1>_..._<mN>_<vN>.csv``) — column names are
      bare. Each model's ``run_columns.csv`` is fetched from GitHub
      (via :func:`eosframes.fetch_columns`) and columns are distributed
      by name.

    The output folder must not already exist and is created fresh. Each
    per-model file is written as ``<prefix>_<model_id>_<version>.csv`` with
    ``prefix`` inherited from the stacked filename (dropped when the input
    is unprefixed).

    Parameters
    ----------
    input_path : str
        Path to a stacked CSV (Mode A or Mode B).
    output_folder : str
        Destination folder; must not exist.

    Returns
    -------
    list of str
        Absolute paths of the per-model files that were written, in
        the order of the stacked models.

    Raises
    ------
    EosframesError
        On invalid filename, missing ``key`` / ``input`` columns, ambiguous
        or unmatched columns (Mode B), or pre-existing output folder.
    """
    logger = get_logger()
    _require_no_overwrite(output_folder, kind="folder")

    mix = parse_stack_mix_name(input_path)
    explicit = parse_stack_explicit_name(input_path)
    if mix is not None:
        mode = "eosmix"
        prefix = mix["prefix"]
    elif explicit is not None:
        mode = "explicit"
        prefix = explicit["prefix"]
    else:
        raise EosframesError(
            f"'{os.path.basename(input_path)}' does not follow a stack naming "
            "convention.\n\n"
            "Expected one of:\n"
            "  Mode A: [prefix]_eosmix.csv\n"
            "  Mode B: [prefix]_<m1>_<v1>_..._<mN>_<vN>.csv"
        )

    logger.info("Reading stacked CSV: %s (mode=%s)", input_path, mode)
    df = pd.read_csv(input_path)
    for col in ("key", "input"):
        if col not in df.columns:
            raise EosframesError(
                f"'{input_path}' is missing the required '{col}' column."
            )
    feature_cols = [c for c in df.columns if c not in {"key", "input"}]

    # Assemble per-(model, version) column lists. Each assignment entry is a
    # tuple of stacked_col_name, model_id, version, output_col_name.
    assignments: List[Tuple[str, str, str, str]]
    if mode == "eosmix":
        parsed = _classify_stack_columns_mode_a(feature_cols)
        assignments = [
            (stacked, mid, ver, original) for original, mid, ver, stacked in parsed
        ]
    else:
        pairs = explicit["models"]
        mode_b = _classify_stack_columns_mode_b(feature_cols, pairs)
        # Column names stay as-is in Mode B.
        assignments = [(col, mid, ver, col) for col, mid, ver in mode_b]

    # Group assignments by (model_id, version), preserving model order.
    per_model: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    order: List[Tuple[str, str]] = []
    for stacked, mid, ver, output_name in assignments:
        key = (mid, ver)
        if key not in per_model:
            per_model[key] = []
            order.append(key)
        per_model[key].append((stacked, output_name))

    # Create the destination folder and write each per-model CSV.
    os.makedirs(output_folder)
    written: List[str] = []
    for model_id, version in order:
        cols = per_model[(model_id, version)]
        stacked_names = [s for s, _ in cols]
        output_names = [o for _, o in cols]
        sub = df[["key", "input", *stacked_names]].copy()
        # Rename the suffixed columns back to their original names.
        sub.rename(columns=dict(zip(stacked_names, output_names)), inplace=True)
        sub.model_id = model_id
        sub.version = version
        out_basename = (
            f"{prefix}_{model_id}_{version}.csv"
            if prefix
            else f"{model_id}_{version}.csv"
        )
        out_path = os.path.abspath(os.path.join(output_folder, out_basename))
        write_csv(sub, out_path)
        written.append(out_path)

    logger.info(
        "Unstacked %s → %d per-model files in %s",
        input_path,
        len(written),
        output_folder,
    )
    return written
