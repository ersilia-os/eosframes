import os

import click
import pandas as pd

from . import hub, ops
from . import scale as _scale
from .exceptions import EosframesError
from .logger import get_logger


def _err(e: EosframesError) -> click.ClickException:
    return click.ClickException(str(e))


@click.group()
def main():
    """eosframes — Ersilia output data utilities.

    Manipulate inputs and outputs from the Ersilia Model Hub.
    File naming convention for outputs: <model_id>_<version>.<ext>
    (e.g. eos4e40_v1.csv, eos4e40_v1.h5, eos4e40_v1_chunks/)
    """


@main.command()
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder", type=click.Path())
@click.option(
    "--chunksize",
    default=10000,
    show_default=True,
    help="Number of rows per chunk file.",
)
def split(input_csv: str, output_folder: str, chunksize: int) -> None:
    """Split INPUT_CSV into numbered chunk files inside OUTPUT_FOLDER.

    Works with any CSV file — raw input data, Ersilia model outputs, or
    any other tabular file. The column header is preserved in every chunk.
    No model ID is required in the input filename.

    Chunk files are named chunk_000.csv (3-digit padding) or
    chunk_000000.csv (6-digit) when more than 999 chunks are produced.
    """
    try:
        ops.split_csv(input_csv, output_folder, chunksize)
    except EosframesError as e:
        raise _err(e) from e


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output")
def convert(input: str, output: str) -> None:
    """Convert INPUT to OUTPUT, inferring format from file extensions.

    INPUT can be a CSV file, an H5 file, or a folder of chunk CSV files.
    OUTPUT must follow the naming convention: eos4e40_v1.csv or eos4e40_v1.h5.

    Supported conversions:

    \b
      folder/  → eos4e40_v1.csv   (assemble chunks into CSV)
      folder/  → eos4e40_v1.h5    (assemble chunks into H5)
      .csv     → eos4e40_v1.h5    (CSV to H5)
      .h5      → eos4e40_v1.csv   (H5 to CSV)
    """
    try:
        ops.convert_file(input, output)
    except EosframesError as e:
        raise _err(e) from e


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output CSV file path.",
)
@click.option(
    "--suffix/--no-suffix",
    default=True,
    show_default=True,
    help=(
        "Append the model identifier as a suffix to feature column names "
        "(e.g. 'score' → 'score.eos4e40'). "
        "Recommended when stacking outputs from multiple models."
    ),
)
def stack(inputs: tuple, output: str, suffix: bool) -> None:
    """Horizontally stack outputs from multiple Ersilia models into one CSV.

    Each INPUT must be a CSV or H5 file following the naming convention
    (e.g. eos4e40_v1.csv). All files must contain the same inputs in the
    same order — this is validated before stacking.

    The 'key' and 'input' columns are kept only once in the output.
    Feature columns from each model are appended side by side.

    \b
    Example:
      eosframes stack eos4e40_v1.csv eos3804_v1.csv -o stacked.csv
      eosframes stack eos4e40_v1.csv eos3804_v1.h5  -o stacked.csv --no-suffix
    """
    try:
        ops.stack_files(list(inputs), output, suffix=suffix)
    except EosframesError as e:
        raise _err(e) from e


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output file path (must follow naming convention, e.g. eos4e40_v1.csv or eos4e40_v1.h5).",
)
def append(inputs: tuple, output: str) -> None:
    """Vertically concatenate files from the same Ersilia model.

    All INPUT files must belong to the same model (same model ID). Rows are
    appended in the order the files are given. All files must have identical
    columns.

    OUTPUT must follow the naming convention and its model ID must match the
    inputs. Format is inferred from the output extension (.csv or .h5).

    \b
    Example:
      eosframes append eos4e40_v1_batch1.csv eos4e40_v1_batch2.csv -o eos4e40_v1.csv
      eosframes append eos4e40_v1_part1.h5 eos4e40_v1_part2.h5    -o eos4e40_v1.h5
    """
    try:
        ops.append_files(list(inputs), output)
    except EosframesError as e:
        raise _err(e) from e


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file")
def dedupe(input_file: str, output_file: str) -> None:
    """Remove duplicate rows by key, keeping the first occurrence.

    Reads INPUT_FILE, drops any row whose 'key' value has already appeared,
    and writes the result to OUTPUT_FILE. Both files must share the same model
    ID. Output format is inferred from the extension (.csv or .h5).

    \b
    Example:
      eosframes dedupe eos4e40_v1_raw.csv eos4e40_v1.csv
    """
    try:
        ops.dedupe_file(input_file, output_file)
    except EosframesError as e:
        raise _err(e) from e


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def summary(input_file: str) -> None:
    """Print a summary of an Ersilia output file.

    Displays file metadata, row/column counts, and per-feature statistics
    (dtype, missing values, min, mean, max) for numeric columns.

    \b
    Example:
      eosframes summary eos4e40_v1.csv
      eosframes summary eos4e40_v1.h5
    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from .naming import parse_name

    console = Console()
    df = ops._read_file(input_file)

    model_id = getattr(df, "model_id", "unknown")
    parsed = parse_name(input_file)
    version = parsed["version"] if parsed else "unknown"
    fmt = os.path.splitext(input_file)[1].lstrip(".")
    file_size_kb = os.path.getsize(input_file) / 1024

    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    has_key = "key" in df.columns

    console.print()
    console.rule(f"[bold cyan]{os.path.basename(input_file)}[/bold cyan]")
    console.print(f"  [bold]Model ID:[/bold]  {model_id}")
    console.print(f"  [bold]Version:[/bold]   {version}")
    console.print(f"  [bold]Format:[/bold]    {fmt.upper()}")
    console.print(f"  [bold]Size:[/bold]      {file_size_kb:.1f} KB")
    console.print(f"  [bold]Rows:[/bold]      {len(df):,}")
    console.print(f"  [bold]Features:[/bold]  {len(feature_cols)} column(s)")
    console.print(f"  [bold]Has key:[/bold]   {'yes' if has_key else 'no'}")

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
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("dtype", justify="center")
    table.add_column("Missing", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Max", justify="right")

    for col in feature_cols:
        series = df[col]
        n_missing = series.isna().sum()
        missing_str = str(n_missing) if n_missing == 0 else f"[yellow]{n_missing}[/yellow]"

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) == 0:
                min_s = mean_s = max_s = "[dim]—[/dim]"
            else:
                def _fmt(v):
                    return f"{v:.0f}" if v == int(v) else f"{v:.4g}"
                min_s  = _fmt(clean.min())
                mean_s = _fmt(clean.mean())
                max_s  = _fmt(clean.max())
        else:
            n_unique = series.nunique()
            min_s  = "[dim]—[/dim]"
            mean_s = f"[dim]{n_unique} unique[/dim]"
            max_s  = "[dim]—[/dim]"

        table.add_row(col, str(series.dtype), missing_str, min_s, mean_s, max_s)

    console.print(table)


@main.command()
@click.argument("model_id")
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output CSV path. Defaults to <model_id>_metadata.csv.",
)
def info(model_id: str, output: str) -> None:
    """Fetch metadata for a model from GitHub and save it as a CSV.

    Retrieves the metadata.json (or metadata.yml) from the model's GitHub
    repository at https://github.com/ersilia-os/<MODEL_ID> and writes all
    fields as a two-column CSV (field, value).

    List-valued fields are joined with ' | '. The output file does not need
    to follow the Ersilia naming convention.

    \b
    Examples:
      eosframes info eos4e40
      eosframes info eos4e40 -o my_metadata.csv
    """
    logger = get_logger()

    if output is None:
        output = f"{model_id}_metadata.csv"
    if not output.endswith(".csv"):
        raise click.ClickException("Output must be a .csv file.")
    if os.path.exists(output):
        raise click.ClickException(f"Output file '{output}' already exists. Remove it first.")

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

    rows = [{"field": k, "value": _flatten(v)} for k, v in metadata.items()]
    pd.DataFrame(rows).to_csv(output, index=False)
    logger.info("Metadata written to %s (%d fields)", output, len(rows))
    click.echo(output)


@main.command()
@click.argument("model_id")
@click.argument("version")
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output CSV path. Defaults to <model_id>_<version>_columns.csv.",
)
def columns(model_id: str, version: str, output: str) -> None:
    """Fetch the run_columns.csv for a model version from GitHub.

    Downloads model/framework/columns/run_columns.csv from the model's
    GitHub repository. The version is used to resolve the git ref
    (e.g. 'v1' → tag 'v1.0.0', falling back to 'main').

    \b
    Examples:
      eosframes columns eos4e40 v1
      eosframes columns eos4e40 v1 -o my_columns.csv
    """
    logger = get_logger()

    if output is None:
        output = f"{model_id}_{version}_columns.csv"
    if not output.endswith(".csv"):
        raise click.ClickException("Output must be a .csv file.")
    if os.path.exists(output):
        raise click.ClickException(f"Output file '{output}' already exists. Remove it first.")

    try:
        df = hub.fetch_columns(model_id, version)
    except EosframesError as e:
        raise _err(e) from e

    df.to_csv(output, index=False)
    logger.info("Columns written to %s (%d column(s))", output, len(df))
    click.echo(output)


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output file path. Defaults to <input_stem>_scaled.<ext>.",
)
@click.option(
    "--params",
    default=None,
    type=click.Path(),
    help="JSON file for transform parameters. With --fit: save fitted params here. Without --fit: load params from here (forward pass).",
)
@click.option(
    "--fit",
    is_flag=True,
    default=False,
    help="Fit a new scaler and save parameters to --params. Requires --params.",
)
@click.option(
    "--method",
    default="standard",
    show_default=True,
    type=click.Choice(_scale.SUPPORTED_METHODS),
    help="Scaling method. Ignored in forward-pass mode.",
)
def transform(input_file: str, output: str, params: str, fit: bool, method: str) -> None:
    """Scale the numeric feature columns of INPUT_FILE.

    Three modes depending on --params and --fit:

    \b
      No --params          fit on INPUT_FILE, discard parameters.
      --params FILE --fit  fit on INPUT_FILE, save parameters to FILE.
      --params FILE        load parameters from FILE, apply (forward pass).

    Only numeric feature columns are scaled. Columns with more than 25 %
    missing values are skipped. The key and input columns pass through
    unchanged.

    \b
    Examples:
      eosframes transform eos4e40_v1.csv
      eosframes transform eos4e40_v1.csv --params scaler.json --fit
      eosframes transform new_eos4e40_v1.csv --params scaler.json -o scaled.csv
    """
    if fit and params is None:
        raise click.UsageError("--fit requires --params to specify where to save the parameters.")
    try:
        out = _scale.transform_file(
            input_file,
            output_path=output,
            params=params,
            fit=fit,
            method=method,
        )
    except EosframesError as e:
        raise _err(e) from e
    click.echo(out)
