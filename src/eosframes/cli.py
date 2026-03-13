import json
import os

import click
import numpy as np
import pandas as pd
import requests

from .logger import get_logger
from .naming import is_valid_name, parse_name
from .utils.utils import chunker


@click.group()
def main():
    """eosframes — Ersilia output data utilities.

    Manipulate inputs and outputs from the Ersilia Model Hub.
    File naming convention for outputs: <model_id>_<version>.<ext>
    (e.g. eos4e40_v1.csv, eos4e40_v1.h5, eos4e40_v1_chunks/)
    """
    pass


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
    logger = get_logger()

    df = pd.read_csv(input_csv)
    total_rows = len(df)
    num_chunks = (total_rows + chunksize - 1) // chunksize
    zfill = 6 if num_chunks >= 1000 else 3

    if os.path.exists(output_folder):
        raise click.ClickException(
            f"Output folder '{output_folder}' already exists. "
            "Remove it or choose a different name."
        )
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
    logger = get_logger()

    # Validate output naming convention
    if not is_valid_name(output):
        raise click.ClickException(
            f"OUTPUT '{output}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> "
            "(e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file '{output}' already exists. Remove it first."
        )

    parsed = parse_name(output)
    model_id = parsed["model_id"]
    out_ext = parsed["extension"]  # "csv" or "h5"

    # Read input
    if os.path.isdir(input):
        csv_files = sorted(f for f in os.listdir(input) if f.endswith(".csv"))
        if not csv_files:
            raise click.ClickException(f"No CSV files found in '{input}'")
        logger.info("Reading %d chunk files from %s", len(csv_files), input)
        frames = [pd.read_csv(os.path.join(input, f)) for f in csv_files]
        df = pd.concat(frames, axis=0).reset_index(drop=True)
        df.model_id = model_id
    else:
        in_ext = os.path.splitext(input)[1].lower()
        if in_ext == ".csv":
            # If input follows naming convention, use read_csv (validates model_id);
            # otherwise load raw and take model_id from output name.
            if is_valid_name(input):
                from .read.read import read_csv
                df = read_csv(input)
            else:
                logger.info("Reading %s", input)
                df = pd.read_csv(input)
                df.model_id = model_id
        elif in_ext == ".h5":
            from .read.read import read_h5
            df = read_h5(input)
        else:
            raise click.ClickException(
                f"Unsupported input format '{in_ext}'. Expected .csv or .h5"
            )

    logger.info("Converting %s → %s", input, output)

    if out_ext == "csv":
        # Ensure model_id attribute is set on df before writing
        df.model_id = model_id
        from .write.write import write_csv
        write_csv(df, output)
    else:
        df.model_id = model_id
        from .write.write import write_h5
        write_h5(df, output, dtype=np.float32)

    logger.info("Done: %s", output)


def _read_file(path: str) -> pd.DataFrame:
    """Read a CSV or H5 file, returning a DataFrame with model_id set."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        from .read.read import read_csv
        return read_csv(path)
    elif ext == ".h5":
        from .read.read import read_h5
        return read_h5(path)
    else:
        raise click.ClickException(
            f"Unsupported file format '{ext}' for '{path}'. Expected .csv or .h5"
        )


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
    logger = get_logger()

    if len(inputs) < 2:
        raise click.ClickException("At least two input files are required for stacking.")

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file '{output}' already exists. Remove it first."
        )

    if not output.endswith(".csv"):
        raise click.ClickException("OUTPUT must be a .csv file.")

    # Read all input files
    dfs = []
    seen_model_ids = []
    for path in inputs:
        if not is_valid_name(path):
            raise click.ClickException(
                f"'{path}' does not follow the naming convention. "
                "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv)"
            )
        logger.info("Reading %s", path)
        df = _read_file(path)
        model_id = getattr(df, "model_id", None)
        if model_id in seen_model_ids:
            raise click.ClickException(
                f"Model '{model_id}' appears more than once in the input list. "
                "Each model must be unique when stacking."
            )
        seen_model_ids.append(model_id)
        dfs.append(df)

    # Validate that all inputs are identical and in the same order
    reference_inputs = dfs[0]["input"].tolist()
    for i, df in enumerate(dfs[1:], start=2):
        if "input" not in df.columns:
            raise click.ClickException(
                f"File #{i} does not contain an 'input' column."
            )
        if df["input"].tolist() != reference_inputs:
            raise click.ClickException(
                f"Input mismatch: file #{i} has different inputs or a different row order "
                f"than file #1. Stacking requires all files to have identical inputs in the same order."
            )

    # Build the stacked dataframe
    # Start with key + input from the first file
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
    result.to_csv(output, index=False)
    logger.info("Done: %s", output)


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
    logger = get_logger()

    if len(inputs) < 2:
        raise click.ClickException("At least two input files are required for appending.")

    if not is_valid_name(output):
        raise click.ClickException(
            f"OUTPUT '{output}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file '{output}' already exists. Remove it first."
        )

    out_parsed = parse_name(output)
    expected_model_id = out_parsed["model_id"]
    out_ext = out_parsed["extension"]

    # Read all input files, validating model_id consistency
    dfs = []
    reference_columns = None
    for path in inputs:
        logger.info("Reading %s", path)
        df = _read_file(path)
        model_id = getattr(df, "model_id", None)
        if model_id != expected_model_id:
            raise click.ClickException(
                f"Model ID mismatch: '{path}' has model '{model_id}' "
                f"but output expects '{expected_model_id}'."
            )
        cols = list(df.columns)
        if reference_columns is None:
            reference_columns = cols
        elif cols != reference_columns:
            raise click.ClickException(
                f"Column mismatch: '{path}' has columns {cols} "
                f"but expected {reference_columns}."
            )
        dfs.append(df)

    result = pd.concat(dfs, axis=0).reset_index(drop=True)
    result.model_id = expected_model_id

    logger.info(
        "Appended %d files → %d rows total",
        len(dfs),
        len(result),
    )

    if out_ext == "csv":
        from .write.write import write_csv
        write_csv(result, output)
    else:
        from .write.write import write_h5
        write_h5(result, output, dtype=np.float32)

    logger.info("Done: %s", output)


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
    logger = get_logger()

    if not is_valid_name(output_file):
        raise click.ClickException(
            f"OUTPUT '{output_file}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext> (e.g. eos4e40_v1.csv or eos4e40_v1.h5)"
        )

    if os.path.exists(output_file):
        raise click.ClickException(
            f"Output file '{output_file}' already exists. Remove it first."
        )

    out_parsed = parse_name(output_file)
    expected_model_id = out_parsed["model_id"]
    out_ext = out_parsed["extension"]

    logger.info("Reading %s", input_file)
    df = _read_file(input_file)

    model_id = getattr(df, "model_id", None)
    if model_id != expected_model_id:
        raise click.ClickException(
            f"Model ID mismatch: '{input_file}' has model '{model_id}' "
            f"but output expects '{expected_model_id}'."
        )

    if "key" not in df.columns:
        raise click.ClickException(
            f"'{input_file}' does not contain a 'key' column."
        )

    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate(s), %d rows remaining", removed, len(df))

    df.model_id = expected_model_id

    if out_ext == "csv":
        from .write.write import write_csv
        write_csv(df, output_file)
    else:
        from .write.write import write_h5
        write_h5(df, output_file, dtype=np.float32)

    logger.info("Done: %s", output_file)


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
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    df = _read_file(input_file)

    model_id = getattr(df, "model_id", "unknown")
    parsed = parse_name(input_file)
    version = parsed["version"] if parsed else "unknown"
    fmt = os.path.splitext(input_file)[1].lstrip(".")
    file_size_kb = os.path.getsize(input_file) / 1024

    feature_cols = [c for c in df.columns if c not in {"key", "input"}]
    has_key = "key" in df.columns

    # ── File info panel ──────────────────────────────────────────────────────
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

    # ── Per-feature statistics table ─────────────────────────────────────────
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
                # Format integers without decimal point
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


_GITHUB_RAW = "https://raw.githubusercontent.com/ersilia-os/{model_id}/main/{filename}"
_METADATA_CANDIDATES = ["metadata.json", "metadata.yml", "metadata.yaml"]


def _fetch_metadata(model_id: str) -> dict:
    """Fetch model metadata from GitHub raw content. Tries JSON then YAML."""
    for filename in _METADATA_CANDIDATES:
        url = _GITHUB_RAW.format(model_id=model_id, filename=filename)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            if filename.endswith(".json"):
                return json.loads(resp.text)
            else:
                try:
                    import yaml
                    return yaml.safe_load(resp.text)
                except ImportError:
                    raise click.ClickException(
                        f"Model '{model_id}' has a YAML metadata file but 'pyyaml' is not installed. "
                        "Install it with: pip install pyyaml"
                    )
    raise click.ClickException(
        f"Could not fetch metadata for model '{model_id}'. "
        "Make sure the model ID is correct and the repo exists at "
        f"https://github.com/ersilia-os/{model_id}"
    )


def _flatten_value(v) -> str:
    """Convert a metadata value to a plain string for CSV output."""
    if isinstance(v, list):
        return " | ".join(str(item) for item in v)
    if isinstance(v, dict):
        return json.dumps(v)
    if v is None:
        return ""
    return str(v)


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
    fields as columns in a single-row CSV file.

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

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file '{output}' already exists. Remove it first."
        )

    logger.info("Fetching metadata for '%s' from GitHub...", model_id)
    metadata = _fetch_metadata(model_id)

    row = {k: _flatten_value(v) for k, v in metadata.items()}
    df = pd.DataFrame([row])
    df.to_csv(output, index=False)

    logger.info("Metadata written to %s (%d fields)", output, len(row))
    click.echo(output)
