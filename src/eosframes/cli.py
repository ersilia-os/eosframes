import os
from typing import Optional

import click
import pandas as pd

from . import hub, ops
from . import scale as _scale
from .exceptions import EosframesError
from .logger import get_logger
from .naming import (
    is_valid_columns_name,
    is_valid_info_name,
    is_valid_summary_name,
    make_columns_name,
    make_info_name,
    make_summary_name,
    parse_name,
)


def _err(e: EosframesError) -> click.ClickException:
    return click.ClickException(str(e))


def _parse_input_or_fail(input_path: str) -> dict:
    """Parse a data file/dir path or raise a ClickException with a helpful message."""
    parsed = parse_name(input_path)
    basename = os.path.basename(input_path.rstrip("/\\"))
    if parsed is None:
        raise click.ClickException(
            f"'{basename}' does not follow the Ersilia naming convention.\n\n"
            "Expected one of (prefix optional):\n"
            "  [prefix]_<model_id>_<version>.csv   e.g. eos4e40_v1.csv, example_eos4e40_v1.csv\n"
            "  [prefix]_<model_id>_<version>.h5    e.g. eos4e40_v1.h5\n"
            "  [prefix]_<model_id>_<version>_chunks/  e.g. eos4e40_v1_chunks/\n\n"
            "model_id = 'eos' + 1 digit + 3 alphanumeric (e.g. eos4e40).\n"
            "version  = 'v' + integer             (e.g. v1).\n\n"
            f"Try renaming to something like 'eos4e40_v1{os.path.splitext(basename)[1] or '.csv'}'."
        )
    if parsed["name_type"] not in {"csv", "h5", "chunks_dir"}:
        raise click.ClickException(
            f"'{basename}' is a '{parsed['name_type']}' sidecar file, not a data file.\n"
            "Pass the original .csv / .h5 / _chunks/ file instead."
        )
    return parsed


def _resolve_sidecar_output(
    output: Optional[str], parsed_input: dict, kind: str
) -> Optional[str]:
    """Validate an info/columns/summary sidecar output path.

    ``kind`` is one of ``"info"``, ``"columns"``, ``"summary"``. Returns
    ``output`` on success, ``None`` if ``output`` is ``None``, and raises
    ``ClickException`` with a suggested filename on any violation.
    """
    if output is None:
        return None

    model_id = parsed_input["model_id"]
    version = parsed_input["version"]
    make_name = {
        "info": make_info_name,
        "columns": make_columns_name,
        "summary": make_summary_name,
    }[kind]
    validator = {
        "info": is_valid_info_name,
        "columns": is_valid_columns_name,
        "summary": is_valid_summary_name,
    }[kind]
    suggested = make_name(model_id, version)
    suggested_with_prefix = make_name(model_id, version, prefix="example")

    out_basename = os.path.basename(output)
    if not validator(output):
        raise click.ClickException(
            f"'{out_basename}' is not a valid {kind}-sidecar filename.\n\n"
            f"The filename must end with the literal suffix '_{kind}.csv' — the "
            f"'_{kind}' token is required, not optional.\n\n"
            f"Expected: [prefix]_<model_id>_<version>_{kind}.csv\n"
            "  where:\n"
            "    [prefix]   optional, e.g. 'example' or '260313_gardp'\n"
            f"    model_id   must match --input ('{model_id}')\n"
            f"    version    must match --input ('{version}')\n"
            f"    _{kind}     literal token (required)\n\n"
            f"Try: {suggested}\n"
            f"Or with a prefix: {suggested_with_prefix}"
        )

    out_parsed = parse_name(output)
    if out_parsed["model_id"] != model_id:
        raise click.ClickException(
            f"Model ID mismatch: input file is for '{model_id}' but '{out_basename}' "
            f"encodes '{out_parsed['model_id']}'.\n"
            f"Try: {suggested}"
        )
    if out_parsed["version"] != version:
        raise click.ClickException(
            f"Version mismatch: input file is '{version}' but '{out_basename}' "
            f"encodes '{out_parsed['version']}'.\n"
            f"Try: {suggested}"
        )

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file '{output}' already exists. Remove it first."
        )

    return output


@click.group()
def main():
    """eosframes — Ersilia output data utilities.

    Manipulate inputs and outputs from the Ersilia Model Hub.
    File naming convention for outputs: <model_id>_<version>.<ext>
    (e.g. eos4e40_v1.csv, eos4e40_v1.h5, eos4e40_v1_chunks/)
    """


@main.command(short_help="Split a CSV into numbered chunk files.")
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    "output_folder",
    required=True,
    type=click.Path(),
    help="Destination folder (must not exist; will be created).",
)
@click.option(
    "--chunksize",
    default=10000,
    show_default=True,
    help="Number of rows per chunk file.",
)
def split(input_csv: str, output_folder: str, chunksize: int) -> None:
    """Split the input CSV into numbered chunk files inside the output folder.

    Works with any CSV file — raw input data, Ersilia model outputs, or
    any other tabular file. The column header is preserved in every chunk.
    No model ID is required in the input filename.

    Chunk files are named chunk_<N>.csv with zero-padding sized to fit the
    total chunk count (e.g. chunk_0.csv for 1–9, chunk_00.csv for 10–99).
    """
    try:
        ops.split_csv(input_csv, output_folder, chunksize)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Convert between CSV, H5, and chunk folders.")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output file. Must follow the naming convention (eos4e40_v1.csv or eos4e40_v1.h5).",
)
def convert(input_path: str, output_path: str) -> None:
    """Convert between CSV, H5, and chunk folders, inferring format from file extensions.

    The input can be a CSV file, an H5 file, or a folder of chunk CSV files.
    The output must follow the naming convention: eos4e40_v1.csv or eos4e40_v1.h5.

    Supported conversions:

    \b
      folder/  → eos4e40_v1.csv   (assemble chunks into CSV)
      folder/  → eos4e40_v1.h5    (assemble chunks into H5)
      .csv     → eos4e40_v1.h5    (CSV to H5)
      .h5      → eos4e40_v1.csv   (H5 to CSV)
    """
    try:
        ops.convert_file(input_path, output_path)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Horizontally stack outputs from multiple models.")
@click.argument(
    "inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help=(
        "Output CSV. Must follow one of the two stack conventions: "
        "[prefix]_eosmix.csv (columns suffixed with _<model_id>_<version>) "
        "or [prefix]_<m1>_<v1>_..._<mN>_<vN>.csv in input order (columns bare)."
    ),
)
def stack(inputs: tuple, output: str) -> None:
    """Horizontally stack outputs from multiple Ersilia models into one CSV.

    Each INPUT must be a CSV or H5 file following the naming convention
    (e.g. eos4e40_v1.csv). All files must contain the same inputs in the
    same order — this is validated before stacking.

    The 'key' and 'input' columns are kept only once in the output.

    The naming of the output file picks the column convention (pick one):

    \b
      Mode A:  [prefix]_eosmix.csv
               → each feature column becomes <column>_<model_id>_<version>
      Mode B:  [prefix]_<m1>_<v1>_..._<mN>_<vN>.csv
               → feature columns stay bare; each model appears in the
                 filename in the same order as the INPUTS.

    \b
    Examples:
      eosframes stack eos4e40_v1.csv eos3804_v1.csv -o project_eosmix.csv
      eosframes stack eos4e40_v1.csv eos3804_v1.csv -o eos4e40_v1_eos3804_v1.csv
    """
    try:
        ops.stack_files(list(inputs), output)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Split a stacked CSV back into per-model files.")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    "output_folder",
    required=True,
    type=click.Path(),
    help="Destination folder (must not exist; will be created).",
)
def unstack(input_file: str, output_folder: str) -> None:
    """Split a stacked CSV back into per-model files in a fresh folder.

    The mode is resolved from the input filename:

    \b
      Mode A ([prefix]_eosmix.csv):
        Column names carry the provenance. Columns are grouped by their
        _<model_id>_<version> suffix; the suffix is stripped when writing
        each per-model file.
      Mode B ([prefix]_<m1>_<v1>_..._<mN>_<vN>.csv):
        Column names are bare. Each model's run_columns.csv is fetched
        from GitHub and columns are distributed by name.

    Each per-model file is written as <prefix>_<model_id>_<version>.csv
    with the prefix inherited from the stacked filename.

    \b
    Examples:
      eosframes unstack project_eosmix.csv -o ./split/
      eosframes unstack eos4e40_v1_eos7m30_v1.csv -o ./split/
    """
    try:
        ops.unstack_file(input_file, output_folder)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Vertically concatenate files from the same model.")
@click.argument(
    "inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output file path (must follow naming convention, e.g. eos4e40_v1.csv or eos4e40_v1.h5).",
)
def append(inputs: tuple, output: str) -> None:
    """Vertically concatenate files from the same Ersilia model.

    All INPUT files must belong to the same model (same model ID). Rows are
    appended in the order the files are given. All files must have identical
    columns.

    The output must follow the naming convention and its model ID must match
    the inputs. Format is inferred from the output extension (.csv or .h5).

    \b
    Example:
      eosframes append eos4e40_v1_batch1.csv eos4e40_v1_batch2.csv -o eos4e40_v1.csv
      eosframes append eos4e40_v1_part1.h5 eos4e40_v1_part2.h5    -o eos4e40_v1.h5
    """
    try:
        ops.append_files(list(inputs), output)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Remove duplicate rows by key.")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    "output_file",
    required=True,
    type=click.Path(),
    help="Output file (must follow naming convention; same model ID as input).",
)
def dedupe(input_file: str, output_file: str) -> None:
    """Remove duplicate rows by key, keeping the first occurrence.

    Reads the input, drops any row whose 'key' value has already appeared,
    and writes the result to the output file. Both files must share the
    same model ID. Output format is inferred from the extension (.csv or .h5).

    \b
    Example:
      eosframes dedupe eos4e40_v1_raw.csv -o eos4e40_v1.csv
    """
    try:
        ops.dedupe_file(input_file, output_file)
    except EosframesError as e:
        raise _err(e) from e


@main.command(short_help="Summarize an Ersilia output file.")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help=(
        "Optional sidecar CSV to write with per-feature stats (column, dtype, "
        "missing, min, mean, max). Must follow "
        "[prefix]_<model_id>_<version>_summary.csv with matching model_id / version. "
        "Omit to print only."
    ),
)
def summary(input_file: str, output: str) -> None:
    """Summarize an Ersilia output file.

    Displays file metadata, row/column counts, duplicate/missing-data flags,
    and per-feature statistics (dtype, missing, min, mean, max).

    With --output, the per-feature stats are also written as a sidecar CSV
    (one row per feature column).

    \b
    Examples:
      eosframes summary eos4e40_v1.csv
      eosframes summary eos4e40_v1.csv -o eos4e40_v1_summary.csv
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    df = ops._read_file(input_file)

    # Resolve filename-based metadata (model_id, version) — required for the
    # -o naming-convention check and surfaced in the header block.
    parsed_in = parse_name(input_file)
    resolved_out = (
        _resolve_sidecar_output(output, parsed_in, "summary") if parsed_in else None
    )
    if output is not None and parsed_in is None:
        # -o given but input filename doesn't follow the convention — we
        # can't validate/build a matching sidecar name. Reject explicitly.
        _parse_input_or_fail(input_file)  # raises with the verbose error

    model_id = getattr(df, "model_id", "unknown")
    version = parsed_in["version"] if parsed_in else "unknown"
    fmt = os.path.splitext(input_file)[1].lstrip(".")
    file_size_kb = os.path.getsize(input_file) / 1024

    meta_cols = [c for c in ("key", "input") if c in df.columns]
    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    uniq_col = (
        "key" if "key" in df.columns else ("input" if "input" in df.columns else None)
    )

    # Compute per-feature stats once — reused by pretty-print and CSV output.
    def _fmt(v: float) -> str:
        return f"{v:.0f}" if v == int(v) else f"{v:.4g}"

    stats_rows = []
    for col in feature_cols:
        series = df[col]
        n_missing = int(series.isna().sum())
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "missing": n_missing,
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

    def _label_row(label: str, value: str) -> None:
        # Pad label to 14 chars so values line up (longest label = "Unique inputs:" = 14).
        console.print(f"  [bold]{label + ':':<15}[/bold]{value}")

    console.print()
    console.rule(f"[bold cyan]{os.path.basename(input_file)}[/bold cyan]")
    _label_row("Model ID", model_id)
    _label_row("Version", version)
    _label_row("Format", fmt.upper())
    _label_row("Size", f"{file_size_kb:.1f} KB")
    _label_row(
        "Columns",
        f"{len(df.columns)} ({len(meta_cols)} meta + {len(feature_cols)} features)",
    )
    _label_row("Rows", f"{len(df):,}")
    if uniq_col is not None:
        total = len(df)
        unique = int(df[uniq_col].nunique())
        _label_row(
            "Unique keys" if uniq_col == "key" else "Unique inputs",
            f"{unique:,}",
        )
        if unique == total:
            dup_str = "[green]no[/green]"
        else:
            dup_str = f"[yellow]yes ({total - unique:,})[/yellow]"
        _label_row("Duplicates", dup_str)

    total_missing = sum(r["missing"] for r in stats_rows)
    if total_missing == 0:
        miss_str = "[green]no[/green]"
    else:
        n_cols_with_missing = sum(1 for r in stats_rows if r["missing"] > 0)
        miss_str = (
            f"[yellow]yes ({total_missing:,} cells in "
            f"{n_cols_with_missing} column{'s' if n_cols_with_missing != 1 else ''})[/yellow]"
        )
    _label_row("Missing data", miss_str)

    if not feature_cols:
        console.print("\n  [dim]No feature columns found.[/dim]")
        return

    console.print()
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        show_edge=False,
    )
    table.add_column("column", style="cyan", no_wrap=True)
    table.add_column("dtype", justify="center")
    table.add_column("missing", justify="right")
    table.add_column("min", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("max", justify="right")

    for row in stats_rows:
        series = df[row["column"]]
        missing_str = (
            str(row["missing"])
            if row["missing"] == 0
            else f"[yellow]{row['missing']}[/yellow]"
        )
        if pd.api.types.is_numeric_dtype(series):
            if row["min"] is None:
                min_s = mean_s = max_s = "[dim]—[/dim]"
            else:
                min_s = _fmt(row["min"])
                mean_s = _fmt(row["mean"])
                max_s = _fmt(row["max"])
        else:
            n_unique = series.nunique()
            min_s = "[dim]—[/dim]"
            mean_s = f"[dim]{n_unique} unique[/dim]"
            max_s = "[dim]—[/dim]"
        table.add_row(row["column"], row["dtype"], missing_str, min_s, mean_s, max_s)

    console.print(table)

    if resolved_out is not None:
        pd.DataFrame(stats_rows).to_csv(resolved_out, index=False)
        get_logger().info(
            "Summary written to %s (%d feature(s))", resolved_out, len(stats_rows)
        )
        click.echo(resolved_out)


@main.command(short_help="Show metadata for a model.")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help=(
        "Optional sidecar CSV to write. Must follow "
        "[prefix]_<model_id>_<version>_info.csv, with model_id and version "
        "matching the input file. Omit to print only."
    ),
)
def info(input_file: str, output: str) -> None:
    """Show metadata for the model identified by the input file.

    The input must follow the Ersilia naming convention; the model ID and
    version are resolved from its name.

    With --output, a sidecar CSV is also written. The output must follow
    [prefix]_<model_id>_<version>_info.csv and its model_id / version must
    match the input.

    \b
    Examples:
      eosframes info data/example_eos4e40_v1.csv
      eosframes info data/example_eos4e40_v1.csv -o example_eos4e40_v1_info.csv
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    parsed_in = _parse_input_or_fail(input_file)
    resolved_out = _resolve_sidecar_output(output, parsed_in, "info")
    model_id = parsed_in["model_id"]

    try:
        metadata = hub.fetch_metadata(model_id)
    except EosframesError as e:
        raise _err(e) from e

    def _flatten(v) -> str:
        import json

        if isinstance(v, list):
            return " | ".join(str(item) for item in v)
        if isinstance(v, dict):
            return json.dumps(v)
        if v is None:
            return ""
        return str(v)

    rows = [(k, _flatten(v)) for k, v in metadata.items()]

    console = Console()
    console.print()
    console.rule(f"[bold cyan]{os.path.basename(input_file)}[/bold cyan]")
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        show_edge=False,
    )
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    for field, value in rows:
        table.add_row(field, value if value else "[dim]—[/dim]")
    console.print(table)

    if resolved_out is not None:
        pd.DataFrame(
            {"field": [k for k, _ in rows], "value": [v for _, v in rows]}
        ).to_csv(resolved_out, index=False)
        logger = get_logger()
        logger.info("Metadata written to %s (%d fields)", resolved_out, len(rows))
        click.echo(resolved_out)


@main.command(short_help="Show feature column definitions for a model version.")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help=(
        "Optional sidecar CSV to write. Must follow "
        "[prefix]_<model_id>_<version>_columns.csv, with model_id and version "
        "matching the input file. Omit to print only."
    ),
)
def columns(input_file: str, output: str) -> None:
    """Show the feature column definitions for the model version of the input.

    The input must follow the Ersilia naming convention; the model ID and
    version are resolved from its name.

    With --output, a sidecar CSV is also written. The output must follow
    [prefix]_<model_id>_<version>_columns.csv and its model_id / version must
    match the input.

    \b
    Examples:
      eosframes columns data/example_eos4e40_v1.csv
      eosframes columns data/example_eos4e40_v1.csv -o example_eos4e40_v1_columns.csv
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    parsed_in = _parse_input_or_fail(input_file)
    resolved_out = _resolve_sidecar_output(output, parsed_in, "columns")
    model_id = parsed_in["model_id"]
    version = parsed_in["version"]

    try:
        df = hub.fetch_columns(model_id, version)
    except EosframesError as e:
        raise _err(e) from e

    console = Console()
    console.print()
    console.rule(f"[bold cyan]{os.path.basename(input_file)}[/bold cyan]")
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold magenta",
        show_edge=False,
    )
    for col in df.columns:
        table.add_column(
            str(col),
            style="cyan" if col == "name" else None,
            overflow="fold",
            no_wrap=(col == "name"),
        )
    for _, row in df.iterrows():
        table.add_row(*(str(v) if pd.notna(v) else "[dim]—[/dim]" for v in row))
    console.print(table)

    if resolved_out is not None:
        df.to_csv(resolved_out, index=False)
        logger = get_logger()
        logger.info("Columns written to %s (%d column(s))", resolved_out, len(df))
        click.echo(resolved_out)


@main.command(short_help="Fit a scaler and save parameters to a JSON file.")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--scaler",
    "-s",
    required=True,
    type=click.Path(),
    help="Path where the scaler JSON file will be saved.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="If provided, also write the scaled data here (fit-transform).",
)
def fit(input_file: str, scaler: str, output: str) -> None:
    """Fit a scaler on INPUT_FILE and save parameters to SCALER.

    Only numeric feature columns are fitted. Columns with more than 25 %
    missing values are skipped. The key and input columns are ignored.

    When -o is given the scaled output is also written immediately (fit-transform).

    \b
    Examples:
      eosframes fit eos4e40_v1.csv -s eos4e40_v1_transformer.json
      eosframes fit eos4e40_v1.csv -s eos4e40_v1_transformer.json -o eos4e40_v1_scaled.csv
    """
    try:
        _scale.fit_file(input_file, scaler, output_path=output)
    except EosframesError as e:
        raise _err(e) from e
    click.echo(output if output is not None else scaler)


@main.command(short_help="Apply a saved scaler to scale numeric feature columns.")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--scaler",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Scaler JSON file produced by `eosframes fit`.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output file path.",
)
def transform(input_file: str, scaler: str, output: str) -> None:
    """Apply a saved scaler to INPUT_FILE and write scaled data to OUTPUT.

    Loads the scaler parameters from SCALER and applies them to the numeric
    feature columns of INPUT_FILE. The key and input columns pass through
    unchanged.

    \b
    Example:
      eosframes transform new_eos4e40_v1.csv -s eos4e40_v1_transformer.json -o scaled.csv
    """
    try:
        out = _scale.transform_file(input_file, scaler, output)
    except EosframesError as e:
        raise _err(e) from e
    click.echo(out)
