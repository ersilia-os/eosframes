![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Manipulating Ersilia's dataframes

`eosframes` is a Python library and CLI tool for manipulating inputs and outputs from the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia). It handles splitting, assembling, converting, scaling, and summarizing tabular model output files.

---

## Naming Convention

Naming rules must be followed. Every write path (data files, chunk directories, sidecar CSVs) refuses a filename that doesn't match the convention, and every read path extracts the model ID and version directly from the filename.

### The canonical pattern

```
[prefix]_<model_id>_<version>[_<kind>].<ext>
```

| Component   | Format                                        | Required | Examples                          |
|-------------|-----------------------------------------------|----------|-----------------------------------|
| `prefix`    | Alphanumeric tokens (`_` is allowed)          | Optional | `example`, `260313_gardp`         |
| `model_id`  | `eos` + 1 digit + 3 alphanumeric chars        | Required | `eos4e40`, `eos7m30`, `eos3804`   |
| `version`   | `v` + integer                                 | Required | `v1`, `v2`, `v10`                 |
| `kind`      | Sidecar tag (`info`, `columns`, `summary`)    | For sidecars only | `info`, `columns`, `summary` |
| `ext`       | `csv` (or `h5`)                               | Required for files (omitted for chunk dirs) | `csv`, `h5` |

The underscores (`_`) are separators between components. `example_eos4e40_v1.csv` parses as prefix `example`, model_id `eos4e40`, version `v1`, ext `csv`.

### File types at a glance

| Filename pattern                                        | What it holds                               | Produced by                           |
|---------------------------------------------------------|---------------------------------------------|---------------------------------------|
| `[prefix]_<model_id>_<version>.csv`                     | Ersilia data (rows × features)              | `convert`, `append`, `dedupe`, any model run |
| `[prefix]_<model_id>_<version>.h5`                      | Same data in HDF5                           | `convert`, `append`, `dedupe`         |
| `[prefix]_<model_id>_<version>_chunks/`                 | Folder of `chunk_NNN.csv` slices            | `split` (with Ersilia-named input), `convert` |
| `[prefix]_<model_id>_<version>_info.csv`                | GitHub metadata for the model               | `eosframes info -o`                   |
| `[prefix]_<model_id>_<version>_columns.csv`             | `run_columns.csv` for the model version     | `eosframes columns -o`                |
| `[prefix]_<model_id>_<version>_summary.csv`             | Per-feature stats (column/dtype/missing/min/mean/max) | `eosframes summary -o`      |
| `[prefix]_eosmix.csv`                                   | Horizontal stack, Mode A (columns suffixed with `_<model_id>_<version>`) | `eosframes stack -o` |
| `[prefix]_<m1>_<v1>_..._<mN>_<vN>.csv` (N≥2)            | Horizontal stack, Mode B (bare columns; every model listed in filename in input order) | `eosframes stack -o` |

### When the convention is enforced

The naming convention is always enforced. The only exception is `eosframes split INPUT`, which accepts any CSV since its purpose is pre-processing before a model run, and the model ID isn't known yet.

---

## Installation

Simply run:

```bash
pip install eosframes
```

Or install from source:

```bash
git clone https://github.com/ersilia-os/eosframes.git
cd eosframes
pip install -e .
```

---

## File Formats

### CSV

Every Ersilia CSV file has this column layout:

| Column          | Type              | Description                                          |
|-----------------|-------------------|------------------------------------------------------|
| `key`           | string            | Unique molecule identifier (MD5 hash of SMILES)      |
| `input`         | string            | Molecule SMILES fed to the model                     |
| feature columns | numeric or string | One or more columns of model output values           |

Example (from `example_eos4e40_v1.csv`, inhibition at 50 µM):

```
key,input,inhibition_50um
0a41432aef1339039da0fa52f4c47dfa,CC(C)(Cc1c[nH]c2ccc(Cl)cc12)NCCOc1ccccc1OCC1CC1,0.017
2a8c41c161b12a1779a1aaf1614639de,Cc1ccc(C)c(Oc2ccncc2C(=O)N2CCN(C3CC3)c3ccccc23)c1,0.007
```

Example (from `example_eos7m30_v1.csv`, 49 ADMET properties):

```
key,input,molecular_weight,logp,hydrogen_bond_acceptors,...
0a41432aef...,CC(C)(...),412.96,5.60,3.0,...
```

### H5 (HDF5)

Ersilia H5 files store the same data in a binary format suited for large datasets:

```
eos4e40_v1.h5
├── key      (N,)    — UTF-8 string, one per molecule
├── input    (N,)    — UTF-8 string, one per molecule
├── features (F,)    — UTF-8 string, feature column names
└── values   (N, F)  — float32, model output values
```

---

## CLI Reference

### `eosframes split`

Split a CSV file into numbered chunk files.

```bash
eosframes split INPUT_CSV -o OUTPUT_FOLDER [--chunksize N]
```

```bash
# Split 100,000 compounds into chunks of 10,000 for parallel model runs
eosframes split compounds.csv -o compounds_chunks/ --chunksize 10000
# → compounds_chunks/chunk_000.csv ... chunk_009.csv

# Works on any CSV, including Ersilia model outputs
eosframes split example_eos4e40_v1.csv -o eos4e40_v1_chunks/ --chunksize 25
```

Chunk files are named `chunk_000.csv` (3-digit padding) or `chunk_000000.csv` (6-digit, for >999 chunks). The header is preserved in every chunk.

---

### `eosframes convert`

Convert between CSV, H5, and chunk folders.

```bash
eosframes convert INPUT -o OUTPUT
```

| `INPUT`       | `--output`           | Description              |
|---------------|----------------------|--------------------------|
| `folder/`     | `eos4e40_v1.csv`     | Assemble chunks → CSV    |
| `folder/`     | `eos4e40_v1.h5`      | Assemble chunks → H5     |
| `.csv`        | `eos4e40_v1.h5`      | CSV → H5                 |
| `.h5`         | `eos4e40_v1.csv`     | H5 → CSV                 |

```bash
# CSV to H5
eosframes convert example_eos4e40_v1.csv -o eos4e40_v1.h5

# H5 back to CSV
eosframes convert eos4e40_v1.h5 -o eos4e40_v1_restored.csv

# Assemble chunks into H5
eosframes convert eos4e40_v1_chunks/ -o eos4e40_v1.h5
```

---

### `eosframes stack`

**Horizontally** stack outputs from multiple Ersilia models into one CSV.

```bash
eosframes stack INPUT1 INPUT2 [...] -o OUTPUT
```

Pass each input file as a positional argument, space-separated. All input files must contain the **same molecules in the same order**. The `key` and `input` columns are kept only once.

Model provenance never gets lost. The output filename selects one of two naming modes, and each mode preserves provenance in a different place.

**Mode A — `[prefix]_eosmix.csv`**

Feature columns are suffixed with `_<model_id>_<version>`. The mixture filename itself carries no model list — the columns do.

```bash
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv -o project_eosmix.csv
# → columns: key, input, inhibition_50um_eos4e40_v1, molecular_weight_eos7m30_v1, logp_eos7m30_v1, ...
```

**Mode B — `[prefix]_<model_id>_<version>_..._<model_id>_<version>.csv`**

Every stacked `(model_id, version)` appears in the filename, in the same order as the positional inputs. Feature columns stay bare.

```bash
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv -o eos4e40_v1_eos7m30_v1.csv
# → columns: key, input, inhibition_50um, molecular_weight, logp, ...
```

No other output name is accepted. Duplicate `(model_id, version)` pairs across inputs are always rejected (column collisions in Mode A; ambiguous filenames in Mode B). Two versions of the same model (e.g. `eos4e40_v1` + `eos4e40_v2`) *are* allowed.

---

### `eosframes unstack`

Split a stacked CSV back into per-model files (the inverse of `stack`).

```bash
eosframes unstack STACKED_CSV -o OUTPUT_FOLDER
```

The mode is resolved from the input filename. The output folder must not already exist and is created fresh. Each per-model file is written as `<prefix>_<model_id>_<version>.csv`, with the prefix inherited from the stacked filename (dropped when the input is unprefixed).

**Mode A (`[prefix]_eosmix.csv`)** — column names carry the provenance. Columns are grouped by the `_<model_id>_<version>` suffix, and the suffix is stripped on the way out.

```bash
# project_eosmix.csv has columns: key, input, score_eos4e40_v1, logp_eos7m30_v1, ...
eosframes unstack project_eosmix.csv -o ./split/
# → split/project_eos4e40_v1.csv  (columns: key, input, score)
# → split/project_eos7m30_v1.csv  (columns: key, input, logp, ...)
```

**Mode B (`[prefix]_<m1>_<v1>_..._<mN>_<vN>.csv`)** — column names are bare, so `unstack` fetches each model's `run_columns.csv` from GitHub (via `eosframes columns`) and distributes columns by name. Ambiguous columns (listed for 2+ models) and unmatched columns (not listed for any) are reported as errors.

```bash
eosframes unstack eos4e40_v1_eos7m30_v1.csv -o ./split/
# → split/eos4e40_v1.csv
# → split/eos7m30_v1.csv
```

---

### `eosframes append`

Vertically concatenate files from the **same model** (same model ID, same columns).

```bash
eosframes append INPUT1 INPUT2 [...] -o OUTPUT
```

Pass each input file as a positional argument, space-separated.

```bash
# Combine two batches from the same model run
eosframes append eos4e40_v1_batch1.csv eos4e40_v1_batch2.csv -o eos4e40_v1.csv

# Append H5 files
eosframes append eos4e40_v1_part1.h5 eos4e40_v1_part2.h5 -o eos4e40_v1.h5
```

---

### `eosframes dedupe`

Remove duplicate rows by `key`, keeping the first occurrence.

```bash
eosframes dedupe INPUT_FILE -o OUTPUT_FILE
```

```bash
eosframes dedupe eos4e40_v1_raw.csv -o eos4e40_v1.csv
```

---

### `eosframes summary`

Print a formatted summary of an Ersilia output file. With `--output`, also writes the per-feature stats as a sidecar CSV.

```bash
eosframes summary INPUT_FILE [-o OUTPUT]
```

If `--output/-o` is omitted, the summary is only printed. When provided, it must follow `[prefix]_<model_id>_<version>_summary.csv` with matching model ID / version. The CSV contains one row per feature column (`column`, `dtype`, `missing`, `min`, `mean`, `max`).

```bash
# Print only
eosframes summary example_eos7m30_v1.csv

# Print and save as example_eos7m30_v1_summary.csv
eosframes summary example_eos7m30_v1.csv -o example_eos7m30_v1_summary.csv
```

---

### `eosframes info`

Fetch metadata for the model identified by an Ersilia data file and pretty-print it. Optionally writes a sidecar CSV.

```bash
eosframes info INPUT_FILE [-o OUTPUT]
```

`INPUT_FILE` is any file following the naming convention (`.csv`, `.h5`, or `_chunks/`); the model ID and version are resolved from its name. The metadata itself is fetched from `https://github.com/ersilia-os/<model_id>`.

If `--output/-o` is omitted, the metadata is only printed to the console. When provided, it must follow `[prefix]_<model_id>_<version>_info.csv` and the model ID and version must match the input.

```bash
# Print only (no file written)
eosframes info data/example_eos4e40_v1.csv

# Print and save as example_eos4e40_v1_info.csv
eosframes info data/example_eos4e40_v1.csv -o example_eos4e40_v1_info.csv
```

---

### `eosframes columns`

Fetch the `run_columns.csv` (feature column definitions) for the model version identified by an Ersilia data file. Pretty-prints the columns table and optionally writes a sidecar CSV.

```bash
eosframes columns INPUT_FILE [-o OUTPUT]
```

`INPUT_FILE` follows the naming convention; the model ID and version are resolved from its name. The version is mapped to a git ref (`v1` → `v1.0.0`, falling back to `main`).

If `--output/-o` is omitted, the table is only printed. When provided, it must follow `[prefix]_<model_id>_<version>_columns.csv` with matching model ID / version.

```bash
# Print only
eosframes columns data/example_eos7m30_v1.csv

# Print and save
eosframes columns data/example_eos4e40_v1.csv -o example_eos4e40_v1_columns.csv
```

---

### `eosframes fit`

```bash
eosframes fit INPUT_FILE -s TRANSFORMER_FILE
```

Fits a standard transformer on the numeric feature columns of `INPUT_FILE` and saves
the parameters to `TRANSFORMER_FILE`. The transformer file must follow the naming convention
`[prefix_]<model_id>_<version>_transformer.json` and its model ID / version must
match the input file. Columns with more than 25 % missing values are skipped.

```bash
eosframes fit eos7m30_v1.csv -s eos7m30_v1_transformer.json
```

### `eosframes transform`

```bash
eosframes transform INPUT_FILE -s TRANSFORMER_FILE -o OUTPUT_FILE
```

Applies a saved transformer to `INPUT_FILE`. `TRANSFORMER_FILE` must be a JSON file produced
by `eosframes fit`. The output defaults to `<input_stem>_scaled.<ext>`.

```bash
eosframes transform new_eos7m30_v1.csv -s eos7m30_v1_transformer.json -o new_eos7m30_v1_scaled.csv
```

---

## End-to-End Example

Complete workflow from raw inputs to a multi-model combined dataset.

```bash
# 1. Split 100K compounds into chunks for parallel runs
eosframes split compounds.csv -o chunks/ --chunksize 10000

# 2. Run models (outside eosframes — e.g., using Ersilia)
#    ersilia run -i chunks/chunk_000.csv -o eos4e40_v1_000.csv
#    ...

# 3. Assemble chunks from each model
eosframes append eos4e40_v1_part1.csv eos4e40_v1_part2.csv -o eos4e40_v1.csv
eosframes append eos7m30_v1_part1.csv eos7m30_v1_part2.csv -o eos7m30_v1.csv

# 4. Deduplicate (in case of overlapping batches)
eosframes dedupe eos4e40_v1.csv -o eos4e40_v1_clean.csv
eosframes dedupe eos7m30_v1.csv -o eos7m30_v1_clean.csv

# 5. Summarize
eosframes summary eos7m30_v1_clean.csv

# 6. Fit a transformer on the training set and save parameters
eosframes fit eos7m30_v1_clean.csv -s eos7m30_v1_transformer.json

# 7. Apply transformer to new data
eosframes transform new_eos7m30_v1.csv -s eos7m30_v1_transformer.json -o new_eos7m30_v1_scaled.csv

# 8. Stack both model outputs side by side (Mode A — columns carry provenance)
eosframes stack eos4e40_v1_clean.csv eos7m30_v1_clean.csv -o combined_eosmix.csv

# 8b. Or equivalently Mode B (filename carries provenance; columns stay bare)
eosframes stack eos4e40_v1_clean.csv eos7m30_v1_clean.csv -o eos4e40_v1_eos7m30_v1.csv
```

## Python API

Every CLI command has a Python counterpart. Import from `eosframes` directly:

```python
from eosframes import (
    split_csv, convert_file,
    append_files, dedupe_file,
    stack_files, unstack_file,
    fit_file, transform_file,
    hstack, vstack,
)
```

### File-level operations

| Function | CLI equivalent | Notes |
|---|---|---|
| `split_csv(input, output_folder, chunksize=10000)` | `split` | Returns number of chunks written |
| `convert_file(input_path, output_path)` | `convert` | CSV↔H5, CSV↔chunks, H5↔chunks |
| `append_files(*input_paths, output_path)` | `append` | Vertically stack same-model files |
| `dedupe_file(input_path, output_path)` | `dedupe` | Deduplicate by `key` column |
| `stack_files(*input_paths, output_path)` | `stack` | Horizontally stack multiple models |
| `unstack_file(input_path, output_folder)` | `unstack` | Split a stacked file back per model |
| `fit_file(input_path, scaler_path, output_path=None)` | `fit` | Fit scaler; pass `output_path` for fit-transform |
| `transform_file(input_path, scaler_path, output_path)` | `transform` | Apply saved scaler |

### DataFrame-level operations

For working directly with in-memory DataFrames (all require `df.model_id` to be set):

| Function | What it does |
|---|---|
| `hstack(df1, df2, …)` | Horizontal stack — mirrors `stack_files` |
| `vstack(df1, df2, …)` | Vertical stack — mirrors `append_files` |
| `fit_scaler(df)` | Fit and return scaler params dict |
| `apply_scaler(df, params)` | Apply params dict to DataFrame |

### Error handling

All violations of the naming convention, model ID mismatches, or attempts to overwrite existing files raise `EosframesError`.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest tests/ -v
```

---

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
