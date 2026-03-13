# Manipulating Ersilia's dataframes

`eosframes` is a Python library and CLI tool for manipulating inputs and outputs from the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia). It handles splitting, assembling, converting, scaling, and inspecting tabular model outputs in CSV and HDF5 formats — all while enforcing Ersilia's file naming conventions.

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

## Naming Convention

All **output** files must follow this naming convention:

```
[prefix_]<model_id>_<version>.<ext>
```

| Component   | Format                                  | Example                        |
|-------------|----------------------------------------|--------------------------------|
| `prefix`    | Optional — any alphanumeric prefix     | `260313_gardp_`, `example_`   |
| `model_id`  | `eos` + digit + 3 alphanumeric chars  | `eos4e40`, `eos7m30`           |
| `version`   | `v` + integer                          | `v1`, `v2`                     |
| `ext`       | `csv` or `h5`                          | `csv`, `h5`                    |

Valid examples:

```
eos4e40_v1.csv                   # canonical
eos4e40_v1.h5
eos4e40_v1_chunks/               # folder of chunk CSVs
260313_gardp_eos4e40_v1.csv      # date + project prefix
example_eos7m30_v1.csv           # descriptive prefix
```

> **Input files** (used with `eosframes split`) do **not** need to follow this convention.

---

## File Formats

### CSV

Every Ersilia CSV file has this column layout:

| Column          | Type              | Description                                          |
|-----------------|-------------------|------------------------------------------------------|
| `key`           | string            | Unique molecule identifier (MD5 hash of SMILES)      |
| `input`         | string            | Molecule SMILES fed to the model                     |
| feature columns | numeric or string | One or more columns of model output values            |

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

Use `eosframes convert eos4e40_v1.h5 eos4e40_v1.csv` to convert to CSV, or `eosframes summary` to inspect without converting.

---

## CLI Reference

### `eosframes split`

Split a CSV file into numbered chunk files.

```bash
eosframes split INPUT_CSV OUTPUT_FOLDER [--chunksize N]
```

```bash
# Split 100,000 compounds into chunks of 10,000 for parallel model runs
eosframes split compounds.csv compounds_chunks/ --chunksize 10000
# → compounds_chunks/chunk_000.csv ... chunk_009.csv

# Works on any CSV, including Ersilia model outputs
eosframes split example_eos4e40_v1.csv eos4e40_v1_chunks/ --chunksize 25
```

Chunk files are named `chunk_000.csv` (3-digit padding) or `chunk_000000.csv` (6-digit, for >999 chunks). The header is preserved in every chunk.

---

### `eosframes convert`

Convert between CSV, H5, and chunk folders.

```bash
eosframes convert INPUT OUTPUT
```

| Input         | Output               | Description              |
|---------------|----------------------|--------------------------|
| `folder/`     | `eos4e40_v1.csv`     | Assemble chunks → CSV    |
| `folder/`     | `eos4e40_v1.h5`      | Assemble chunks → H5     |
| `.csv`        | `eos4e40_v1.h5`      | CSV → H5                 |
| `.h5`         | `eos4e40_v1.csv`     | H5 → CSV                 |

```bash
# CSV to H5
eosframes convert example_eos4e40_v1.csv eos4e40_v1.h5

# H5 back to CSV
eosframes convert eos4e40_v1.h5 eos4e40_v1_restored.csv

# Assemble chunks into H5
eosframes convert eos4e40_v1_chunks/ eos4e40_v1.h5
```

---

### `eosframes stack`

Horizontally stack outputs from multiple Ersilia models into one CSV.

```bash
eosframes stack INPUT1 INPUT2 [...] --output OUTPUT [--suffix/--no-suffix]
```

All input files must contain the **same molecules in the same order**. The `key` and `input` columns are kept only once.

```bash
# Stack eos4e40 (1 feature) and eos7m30 (49 ADMET features) side by side
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv --output stacked.csv
# → columns: key, input, inhibition_50um.eos4e40, molecular_weight.eos7m30, logp.eos7m30, ...

# Without model ID suffix in column names
eosframes stack example_eos4e40_v1.csv example_eos7m30_v1.csv --output stacked.csv --no-suffix
```

---

### `eosframes append`

Vertically concatenate files from the **same model** (same model ID, same columns).

```bash
eosframes append INPUT1 INPUT2 [...] --output OUTPUT
```

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
eosframes dedupe INPUT_FILE OUTPUT_FILE
```

```bash
eosframes dedupe eos4e40_v1_raw.csv eos4e40_v1.csv
```

---

### `eosframes summary`

Print a formatted summary of an Ersilia output file.

```bash
eosframes summary INPUT_FILE
```

```bash
eosframes summary example_eos7m30_v1.csv
```

```
────────────────── example_eos7m30_v1.csv ──────────────────
  Model ID:  eos7m30
  Version:   v1
  Format:    CSV
  Size:      51.8 KB
  Rows:      100
  Features:  49 column(s)
  Has key:   yes

 Column                          dtype    Missing     Min    Mean     Max
 molecular_weight               float64        0   198.2   373.8   594.1
 logp                           float64        0    0.61    3.57    6.66
 hydrogen_bond_acceptors        float64        0     1.0    5.12    13.0
 ...
```

---

### `eosframes info`

Fetch metadata for a model from GitHub and save as CSV.

```bash
eosframes info MODEL_ID [--output OUTPUT]
```

```bash
eosframes info eos4e40
# → eos4e40_metadata.csv (field, value)

eosframes info eos7m30 -o admet_metadata.csv
```

---

### `eosframes columns`

Fetch the `run_columns.csv` (feature column definitions) for a model version from GitHub.

```bash
eosframes columns MODEL_ID VERSION [--output OUTPUT]
```

```bash
eosframes columns eos7m30 v1
# → eos7m30_v1_columns.csv

eosframes columns eos4e40 v1 -o eos4e40_columns.csv
```

---

### `eosframes transform`

Scale the numeric feature columns of an Ersilia output file. Three modes depending on `--params` and `--fit`:

```bash
eosframes transform INPUT_FILE [-o OUTPUT] [--params JSON] [--fit] [--method standard]
```

| Flags                  | Behaviour                                              |
|------------------------|--------------------------------------------------------|
| *(none)*               | Fit on `INPUT_FILE`, discard parameters                |
| `--params FILE --fit`  | Fit on `INPUT_FILE`, save parameters to `FILE`         |
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

# Horizontal: combine features from two models (same 100 molecules)
combined = hstack([df4e40, df7m30])
# columns: key, input, inhibition_50um.eos4e40, molecular_weight.eos7m30, ...

# Vertical: concatenate rows from the same model
batch1 = read_csv("eos4e40_v1_batch1.csv")
batch2 = read_csv("eos4e40_v1_batch2.csv")
all_rows = vstack([batch1, batch2])
```

### File operations

```python
from eosframes import split_csv, convert_file, stack_files, append_files, dedupe_file

# Split
n_chunks = split_csv("compounds.csv", "compounds_chunks/", chunksize=10000)

# Convert
convert_file("example_eos4e40_v1.csv", "eos4e40_v1.h5")

# Stack files
stack_files(["example_eos4e40_v1.csv", "example_eos7m30_v1.csv"], "stacked.csv")

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

# Fetch model metadata from GitHub
metadata = fetch_metadata("eos4e40")
print(metadata["task"])          # "Classification"

# Fetch run_columns.csv for a model version
columns_df = fetch_columns("eos7m30", "v1")
print(columns_df.head())
```

### Naming utilities

```python
from eosframes import parse_name, make_output_name, is_valid_name, get_version_from_path

parse_name("example_eos4e40_v1.csv")
# → {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

parse_name("260313_gardp_eos7m30_v2.h5")
# → {"model_id": "eos7m30", "version": "v2", "extension": "h5", "name_type": "h5"}

parse_name("output.csv")   # → None (no model_id found)

make_output_name("eos4e40", "v1", "csv")  # → "eos4e40_v1.csv"

is_valid_name("example_eos4e40_v1.csv")  # True
is_valid_name("results.csv")              # False

get_version_from_path("eos7m30_v2.h5")   # "v2"
```

---

## End-to-End Example

Complete workflow from raw inputs to a multi-model combined dataset.

```bash
# 1. Split 100K compounds into chunks for parallel runs
eosframes split compounds.csv chunks/ --chunksize 10000

# 2. Run models (outside eosframes — e.g., using Ersilia)
#    ersilia run -i chunks/chunk_000.csv -o eos4e40_v1_000.csv
#    ...

# 3. Assemble chunks from each model
eosframes append eos4e40_v1_*.csv -o eos4e40_v1.csv
eosframes append eos7m30_v1_*.csv -o eos7m30_v1.csv

# 4. Deduplicate (in case of overlapping batches)
eosframes dedupe eos4e40_v1.csv eos4e40_v1_clean.csv
eosframes dedupe eos7m30_v1.csv eos7m30_v1_clean.csv

# 5. Inspect
eosframes summary eos7m30_v1_clean.csv

# 6. Fit a scaler on the training set and save parameters
eosframes transform eos7m30_v1_clean.csv --params eos7m30_v1_scaler.json --fit

# 7. Apply scaler to new data (forward pass)
eosframes transform new_eos7m30_v1.csv --params eos7m30_v1_scaler.json -o new_eos7m30_v1_scaled.csv

# 8. Stack both model outputs side by side
eosframes stack eos4e40_v1_clean.csv eos7m30_v1_clean.csv --output combined.csv

# 9. Convert to H5 for compact storage
eosframes convert combined.csv combined_eos4e40_eos7m30_v1.h5
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
