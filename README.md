![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Manipulating Ersilia's dataframes

`eosframes` is a Python library and CLI tool for manipulating inputs and outputs from the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia). It handles splitting, assembling, converting, scaling, and summarizing tabular model outputs in CSV and HDF5 formats — all while enforcing Ersilia's file naming conventions.

---

## Naming Convention

Naming is **the** core concept in `eosframes`. Every write path (data files, chunk directories, sidecar CSVs) refuses a filename that doesn't match the convention, and every read path extracts the model ID and version directly from the filename. This eliminates whole classes of mistakes — mixing outputs from different models, scaling a `v2` file with a `v1` scaler, writing an info CSV that points to the wrong model — without relying on metadata stored inside the file.

### The canonical pattern

```
[prefix]_<model_id>_<version>[_<kind>].<ext>
```

| Component   | Format                                        | Required | Examples                          |
|-------------|-----------------------------------------------|----------|-----------------------------------|
| `prefix`    | Alphanumeric tokens joined by `_`             | Optional | `example`, `260313_gardp`         |
| `model_id`  | `eos` + 1 digit + 3 alphanumeric chars        | Required | `eos4e40`, `eos7m30`, `eos3804`   |
| `version`   | `v` + integer                                 | Required | `v1`, `v2`, `v10`                 |
| `kind`      | Sidecar tag (`info`, `columns`, `summary`)    | For sidecars only | `info`, `columns`, `summary` |
| `ext`       | `csv` or `h5` (no `kind` allowed for `h5`)    | Required for files (omitted for chunk dirs) | `csv`, `h5` |

The underscores (`_`) are **separators** between components, not part of the prefix. `example_eos4e40_v1.csv` parses as prefix `example`, model_id `eos4e40`, version `v1`, ext `csv` — **not** as prefix `example_` + `eos4e40_v1.csv`.

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

### Worked examples

Valid:

```
eos4e40_v1.csv                       # canonical data file
eos4e40_v1.h5                        # canonical H5 data file
eos4e40_v1_chunks/                   # chunk directory
example_eos4e40_v1.csv               # descriptive prefix
260313_gardp_eos4e40_v1.csv          # date + project prefix
eos4e40_v1_info.csv                  # info sidecar
example_eos7m30_v1_columns.csv       # columns sidecar with prefix
eos4e40_v1_summary.csv               # summary sidecar
project_eosmix.csv                   # horizontal stack, Mode A
eos4e40_v1_eos7m30_v1.csv            # horizontal stack, Mode B (2 models)
```

Invalid (the CLI will reject these and suggest a corrected filename):

```
results.csv                   # no model_id
eos4e40.csv                   # missing version
eos4e40_1.csv                 # version must be v-prefixed (v1, not 1)
my_eos4e40_v1_extra.csv       # unknown trailing token (not _info, _columns, _summary)
eos4e40_v1_info.h5            # sidecars are csv-only
eos4e40 v1.csv                # spaces not allowed
```

### When the convention is enforced

- **Always enforced** on outputs (`convert`, `append`, `dedupe`, `info -o`, `columns -o`, `summary -o`, the Python writers). The CLI rejects invalid output paths with a message that suggests a correct filename.
- **Enforced on inputs that need a model context** — the `info`, `columns`, `summary`, and `transform` commands parse model ID and version out of the input filename.
- **Not enforced on `eosframes split` input** — `split` accepts any CSV since its purpose is pre-processing before a model run, and the model ID isn't known yet.

---

## Installation

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

Use `eosframes convert eos4e40_v1.h5 -o eos4e40_v1.csv` to convert to CSV, or `eosframes summary` to summarize without converting.

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

Horizontally stack outputs from multiple Ersilia models into one CSV.

```bash
eosframes stack INPUT1 INPUT2 [...] -o OUTPUT
```

Pass each input file as a positional argument, space-separated. All input files must contain the **same molecules in the same order**. The `key` and `input` columns are kept only once.

Model provenance never gets lost: the **output filename selects one of two naming modes**, and each mode preserves provenance in a different place.

**Mode A — `[prefix]_eosmix.csv`**

Feature columns are suffixed with `_<model_id>_<version>`. The mixture filename itself carries no model list — the columns do.

```bash
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv -o project_eosmix.csv
# → columns: key, input, inhibition_50um_eos4e40_v1, molecular_weight_eos7m30_v1, logp_eos7m30_v1, ...
```

**Mode B — `[prefix]_<m1>_<v1>_..._<mN>_<vN>.csv`**

Every stacked `(model_id, version)` appears in the filename, in the same order as the positional inputs. Feature columns stay bare.

```bash
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv -o eos4e40_v1_eos7m30_v1.csv
# → columns: key, input, inhibition_50um, molecular_weight, logp, ...
```

No other output name is accepted — the CLI rejects anything else and suggests a valid filename for each mode. Duplicate `(model_id, version)` pairs across inputs are always rejected (column collisions in Mode A; ambiguous filenames in Mode B). Two versions of the same model (e.g. `eos4e40_v1` + `eos4e40_v2`) *are* allowed.

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

```
──────────────────── example_eos7m30_v1.csv ────────────────────
  Model ID:      eos7m30
  Version:       v1
  Format:        CSV
  Size:          95.3 KB
  Columns:       51 (2 meta + 49 features)
  Rows:          100
  Unique keys:   100
  Duplicates:    no
  Missing data:  no

 column                          dtype    missing     min    mean     max
 molecular_weight               float64        0    198.2   373.8   594.1
 logp                           float64        0     0.61    3.57    6.66
 hydrogen_bond_acceptors        float64        0      1.0    5.12    13.0
 ...
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

### `eosframes transform`

Scale the numeric feature columns of an Ersilia output file. Three modes depending on `--params` and `--fit`:

```bash
eosframes transform INPUT_FILE [-o OUTPUT] [--params JSON] [--fit] [--method standard]
```

| Flags                  | Behaviour                                              |
|------------------------|--------------------------------------------------------|
| *(none)*               | Fit on the input, discard parameters                   |
| `--params FILE --fit`  | Fit on the input, save parameters to `FILE`            |
| `--params FILE`        | Load parameters from `FILE`, apply (forward pass)      |

```bash
# Fit a scaler on the eos7m30 ADMET outputs (49 features), discard params
eosframes transform example_eos7m30_v1.csv

# Fit and save parameters for later reuse
eosframes transform example_eos7m30_v1.csv --params eos7m30_v1_scaler.json --fit

# Apply saved parameters to new compounds (forward pass)
eosframes transform new_eos7m30_v1.csv --params eos7m30_v1_scaler.json -o new_eos7m30_v1_scaled.csv
```

The saved JSON file contains:

```json
{
  "model_id": "eos7m30",
  "version": "v1",
  "n_rows": 100,
  "fitted_at": "2026-03-13T10:00:00",
  "method": "standard",
  "columns": ["molecular_weight", "logp", ...],
  "skipped_columns": [],
  "parameters": {
    "molecular_weight": {"mean": 373.8, "std": 82.4},
    "logp": {"mean": 3.57, "std": 1.21},
    ...
  }
}
```

Columns with more than 25% missing values are skipped and listed in `skipped_columns`. The output defaults to `<input_stem>_scaled.<ext>` when `-o` is not provided.

---

## Python API

All CLI operations are also available as Python functions.

### Reading and writing

```python
import numpy as np
from eosframes import read_csv, read_h5, write_csv, write_h5

# Read (model_id is extracted from the filename automatically)
df = read_csv("example_eos4e40_v1.csv")
print(df.model_id)      # "eos4e40"
print(df.shape)         # (100, 3)  — key, input, inhibition_50um

df = read_h5("eos7m30_v1.h5")
print(df.shape)         # (100, 51) — key, input, 49 ADMET features

# Write
write_csv(df, "eos4e40_v1.csv")
write_h5(df, "eos4e40_v1.h5", dtype=np.float32)
```

### Stacking

```python
from eosframes import hstack, vstack, read_csv

df4e40 = read_csv("example_eos4e40_v1.csv")
df7m30 = read_csv("example_eos7m30_v1.csv")

# Horizontal: combine features from two models (same 100 molecules).
# `mode` is required and matches the two CLI naming conventions.
mix = hstack([df4e40, df7m30], mode="eosmix")
# columns: key, input, inhibition_50um_eos4e40_v1, molecular_weight_eos7m30_v1, ...

explicit = hstack([df4e40, df7m30], mode="explicit")
# columns: key, input, inhibition_50um, molecular_weight, ...
# (provenance lives in the filename you save this DataFrame to)

# Vertical: concatenate rows from the same model
batch1 = read_csv("eos4e40_v1_batch1.csv")
batch2 = read_csv("eos4e40_v1_batch2.csv")
all_rows = vstack([batch1, batch2])
```

### File operations

```python
from eosframes import (
    split_csv,
    convert_file,
    stack_files,
    unstack_file,
    append_files,
    dedupe_file,
)

# Split
n_chunks = split_csv("compounds.csv", "compounds_chunks/", chunksize=10000)

# Convert
convert_file("example_eos4e40_v1.csv", "eos4e40_v1.h5")

# Stack files
stack_files(
    ["example_eos4e40_v1.csv", "example_eos7m30_v1.csv"],
    "project_eosmix.csv",   # or e.g. "eos4e40_v1_eos7m30_v1.csv" for Mode B
)

# Unstack (inverse of stack). Mode is detected from the input filename.
# Output folder must not exist — unstack creates it.
written = unstack_file("project_eosmix.csv", "./split/")
# → ["./split/project_eos4e40_v1.csv", "./split/project_eos7m30_v1.csv"]

# Append
append_files(["eos4e40_v1_batch1.csv", "eos4e40_v1_batch2.csv"], "eos4e40_v1.csv")

# Deduplicate — returns (rows_before, rows_after)
before, after = dedupe_file("eos4e40_v1_raw.csv", "eos4e40_v1.csv")
```

### Scaling

```python
from eosframes import fit_scaler, apply_scaler, transform_file
import pandas as pd

# DataFrame API
df = pd.read_csv("example_eos7m30_v1.csv")
params = fit_scaler(df)           # dict with method, columns, parameters
scaled = apply_scaler(df, params)

# File API (with model_id/version cross-validation)
transform_file("example_eos7m30_v1.csv", params="eos7m30_v1_scaler.json", fit=True)
transform_file("new_eos7m30_v1.csv", params="eos7m30_v1_scaler.json", output_path="new_eos7m30_v1_scaled.csv")
```

### Hub data

```python
from eosframes import fetch_metadata, fetch_columns

# Low-level fetchers — take a model_id (and version) directly.
# The CLI `info` / `columns` commands wrap these and resolve the identifiers
# from a filename for you.
metadata = fetch_metadata("eos4e40")
print(metadata["task"])          # "Classification"

columns_df = fetch_columns("eos7m30", "v1")
print(columns_df.head())
```

### Naming utilities

```python
from eosframes import (
    parse_name,
    make_output_name,
    make_info_name,
    make_columns_name,
    make_summary_name,
    is_valid_name,
    is_valid_info_name,
    is_valid_columns_name,
    is_valid_summary_name,
    get_version_from_path,
)

parse_name("example_eos4e40_v1.csv")
# → {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

parse_name("260313_gardp_eos7m30_v2.h5")
# → {"model_id": "eos7m30", "version": "v2", "extension": "h5", "name_type": "h5"}

parse_name("example_eos4e40_v1_info.csv")
# → {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "info"}

parse_name("output.csv")   # → None (no model_id found)

make_output_name("eos4e40", "v1", "csv")            # → "eos4e40_v1.csv"
make_info_name("eos4e40", "v1")                     # → "eos4e40_v1_info.csv"
make_columns_name("eos4e40", "v1", prefix="example")# → "example_eos4e40_v1_columns.csv"
make_summary_name("eos4e40", "v1")                  # → "eos4e40_v1_summary.csv"

is_valid_name("example_eos4e40_v1.csv")       # True  (data file)
is_valid_name("eos4e40_v1_info.csv")          # False (sidecar — use is_valid_info_name)
is_valid_info_name("example_eos4e40_v1_info.csv")     # True
is_valid_columns_name("eos4e40_v1_columns.csv")       # True

get_version_from_path("eos7m30_v2.h5")   # "v2"
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

# 6. Fit a scaler on the training set and save parameters
eosframes transform eos7m30_v1_clean.csv --params eos7m30_v1_scaler.json --fit

# 7. Apply scaler to new data (forward pass)
eosframes transform new_eos7m30_v1.csv --params eos7m30_v1_scaler.json -o new_eos7m30_v1_scaled.csv

# 8. Stack both model outputs side by side (Mode A — columns carry provenance)
eosframes stack eos4e40_v1_clean.csv eos7m30_v1_clean.csv -o combined_eosmix.csv

# 8b. Or equivalently Mode B (filename carries provenance; columns stay bare)
eosframes stack eos4e40_v1_clean.csv eos7m30_v1_clean.csv -o eos4e40_v1_eos7m30_v1.csv
```

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
