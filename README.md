# Manipulate Ersilia's data frames

`eosframes` is a Python library and CLI tool for manipulating inputs and outputs from the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia). It provides utilities to split large input datasets into chunks, assemble model outputs back into single files, and convert between CSV and HDF5 formats — all while enforcing Ersilia's file naming conventions.

## Installation

Create a Conda environment and install via pip:

```bash
conda create -n eosframes python=3.12
conda activate eosframes
pip install git+https://github.com/ersilia-os/eosframes.git
```

## Naming Convention

All **output** files must follow this naming convention:

```
<model_id>_<version>.<ext>
```

| Component   | Format              | Example      |
|-------------|---------------------|--------------|
| `model_id`  | `eos` + digit + 3 alphanumeric | `eos4e40`    |
| `version`   | `v` + integer       | `v1`, `v2`   |
| `ext`       | `csv` or `h5`       | `csv`        |

Examples:

```
eos4e40_v1.csv        # CSV output from model eos4e40, version 1
eos4e40_v1.h5         # H5 output from model eos4e40, version 1
eos4e40_v1_chunks/    # folder of chunk CSVs from model eos4e40, version 1
```

> **Input files** (used with `eosframes split`) do **not** need to follow this convention.

---

## File Formats

### CSV

Every Ersilia CSV file has this column layout:

| Column | Type | Required | Description |
|---|---|---|---|
| `key` | string | yes | Unique molecule identifier (e.g. `EOSK000001`) |
| `input` | string | yes | Molecule representation fed to the model (typically a SMILES string) |
| feature columns | numeric or string | yes | One or more columns of model output values |

Example:

```
key,input,score,probability
EOSK000001,CCO,0.812,0.934
EOSK000002,c1ccccc1,0.341,0.102
EOSK000003,CC(=O)O,0.567,0.445
```

### H5 (HDF5)

Ersilia H5 files store the same information as CSV in a binary format suited for large datasets. Each file contains four top-level datasets:

| Dataset | dtype | Description |
|---|---|---|
| `key` | UTF-8 string | Molecule identifiers (one per row) |
| `input` | UTF-8 string | Molecule inputs (one per row) |
| `features` | UTF-8 string | Feature column names (one per feature) |
| `values` | numeric (e.g. `float32`) | 2-D array of shape `(n_rows, n_features)` |

The `key` dataset is optional; all others are required. Feature column names and values are stored separately and recombined into a DataFrame on read.

```
eos4e40_v1.h5
├── key      (N,)  — string
├── input    (N,)  — string
├── features (F,)  — string  ["score", "probability"]
└── values   (N, F) — float32
```

Use `eosframes convert eos4e40_v1.h5 eos4e40_v1.csv` to inspect an H5 file as plain text, or `eosframes summary eos4e40_v1.h5` to print statistics without converting.

---

## CLI Reference

### `eosframes split`

Split a CSV file into numbered chunk files inside a folder.

```bash
eosframes split INPUT_CSV OUTPUT_FOLDER [--chunksize N]
```

| Argument / Option | Description |
|---|---|
| `INPUT_CSV` | Any CSV file — raw inputs, model outputs, or any tabular data |
| `OUTPUT_FOLDER` | Path to the folder that will be created with chunk files |
| `--chunksize N` | Rows per chunk (default: 10000) |

The column header is preserved in every chunk. No model ID is required in the input filename. Chunk files are named `chunk_000.csv`, `chunk_001.csv`, … (3-digit padding for ≤999 chunks; 6-digit for larger datasets).

**Examples:**

```bash
# Split a large input dataset before running Ersilia models
eosframes split compounds.csv compounds_chunks/ --chunksize 10000

# Split a model output file into smaller pieces
eosframes split eos4e40_v1.csv eos4e40_v1_chunks/ --chunksize 5000
```

---

### `eosframes convert`

Convert INPUT to a named output file. The output format is inferred from the extension.

```bash
eosframes convert INPUT OUTPUT
```

| Argument | Description |
|---|---|
| `INPUT` | A CSV file, an H5 file, or a folder of chunk CSV files |
| `OUTPUT` | Output file path — must follow the naming convention |

Supported conversions:

| Input | Output | Description |
|---|---|---|
| folder/ | `eos4e40_v1.csv` | Assemble chunks into CSV |
| folder/ | `eos4e40_v1.h5` | Assemble chunks into H5 |
| `.csv` | `eos4e40_v1.h5` | Convert CSV to H5 |
| `.h5` | `eos4e40_v1.csv` | Convert H5 to CSV |

**Examples:**

```bash
# Assemble model output chunks into a single CSV
eosframes convert eos4e40_v1_chunks/ eos4e40_v1.csv

# Assemble chunks directly into H5
eosframes convert eos4e40_v1_chunks/ eos4e40_v1.h5

# CSV → H5
eosframes convert eos4e40_v1.csv eos4e40_v1.h5

# H5 → CSV
eosframes convert eos4e40_v1.h5 eos4e40_v1.csv
```

---

### `eosframes stack`

Horizontally stack outputs from multiple Ersilia models into a single CSV.

```bash
eosframes stack INPUT1 INPUT2 [INPUT3 ...] --output OUTPUT [--suffix/--no-suffix]
```

| Argument / Option | Description |
|---|---|
| `INPUT1 INPUT2 ...` | Two or more CSV or H5 files, each following the naming convention |
| `--output / -o` | Output CSV file path (no naming convention required) |
| `--suffix` | Append `.<model_id>` to feature column names (default: on) |
| `--no-suffix` | Keep original column names without model ID suffix |

All input files must contain the **same inputs in the same order**. The `key` and `input` columns are written only once in the output.

Using `--suffix` (the default) is recommended when stacking outputs from multiple models, as it lets you trace each column back to its source model.

**Examples:**

```bash
# Stack two model outputs, column names suffixed with model ID
eosframes stack eos4e40_v1.csv eos3804_v1.csv --output stacked.csv

# Stack three files, mixing CSV and H5 inputs
eosframes stack eos4e40_v1.csv eos3804_v1.h5 eos2r5a_v2.csv --output stacked.csv

# Suppress model ID suffix in column names
eosframes stack eos4e40_v1.csv eos3804_v1.csv --output stacked.csv --no-suffix
```

---

### `eosframes append`

Vertically concatenate files from the same Ersilia model into a single output.

```bash
eosframes append INPUT1 INPUT2 [INPUT3 ...] --output OUTPUT
```

| Argument / Option | Description |
|---|---|
| `INPUT1 INPUT2 ...` | Two or more CSV or H5 files, each following the naming convention |
| `--output / -o` | Output file path — must follow the naming convention |

All input files must share the same **model ID** and have **identical columns**. Rows are appended in the order the files are given. The model ID in `OUTPUT` must match the inputs. Output format is inferred from the extension (`.csv` or `.h5`).

**Examples:**

```bash
# Append two CSV batches from the same model into one CSV
eosframes append eos4e40_v1_batch1.csv eos4e40_v1_batch2.csv -o eos4e40_v1.csv

# Append H5 files and write to H5
eosframes append eos4e40_v1_part1.h5 eos4e40_v1_part2.h5 -o eos4e40_v1.h5
```

---

### `eosframes dedupe`

Remove duplicate rows from a file, keeping the first occurrence of each `key`.

```bash
eosframes dedupe INPUT_FILE OUTPUT_FILE
```

Both files must share the same model ID. Output format is inferred from the extension (`.csv` or `.h5`). The input file does not need to follow the strict naming convention as long as the model ID is present in the filename.

**Example:**

```bash
eosframes dedupe eos4e40_v1_raw.csv eos4e40_v1.csv
```

---

### `eosframes summary`

Print a summary of an Ersilia output file.

```bash
eosframes summary INPUT_FILE
```

Displays:
- Model ID, version, format, file size
- Row and feature column count
- Per-feature statistics: dtype, missing values, min, mean, max

**Example:**

```bash
eosframes summary eos4e40_v1.csv
eosframes summary eos4e40_v1.h5
```

---

## Python API

`eosframes` can also be used as a library.

### Reading data

```python
from eosframes import read_csv, read_h5, read_chunked_csvs

# Read a CSV output (model ID must be in the filename)
df = read_csv("eos4e40_v1.csv")

# Read an H5 file
df = read_h5("eos4e40_v1.h5")

# Assemble a folder of chunks
df = read_chunked_csvs("eos4e40_v1_chunks/")

print(df.model_id)  # "eos4e40"
```

### Writing data

```python
import numpy as np
from eosframes import write_csv, write_h5, write_chunked_csvs

# Write as CSV (model ID must appear in the filename)
write_csv(df, "eos4e40_v1.csv")

# Write as H5
write_h5(df, "eos4e40_v1.h5", dtype=np.float32)

# Write as chunks (model ID must appear in the folder name)
write_chunked_csvs(df, "eos4e40_v1_chunks/", chunksize=10000)
```

### Naming utilities

```python
from eosframes import parse_name, make_output_name, is_valid_name

# Parse a filename
parse_name("eos4e40_v1.csv")
# → {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

parse_name("eos4e40_v1_chunks")
# → {"model_id": "eos4e40", "version": "v1", "extension": None, "name_type": "chunks_dir"}

parse_name("output.csv")
# → None

# Build a canonical filename
make_output_name("eos4e40", "v1", "csv")
# → "eos4e40_v1.csv"

# Validate
is_valid_name("eos4e40_v1.h5")   # True
is_valid_name("results.csv")     # False
```

### Stacking multiple model outputs

```python
from eosframes import hstack, vstack

# Horizontal stack: combine features from multiple models (same inputs)
df_combined = hstack([df_eos4e40, df_eos3804])

# Vertical stack: combine rows from the same model (same columns)
df_all = vstack([df_batch1, df_batch2])
```

### Logging

`eosframes` uses the standard Python `logging` module with a logger named `"eosframes"`. To see log output, the logger is pre-configured with a console handler. You can retrieve it to adjust settings:

```python
from eosframes import get_logger
import logging

logger = get_logger()
logger.setLevel(logging.DEBUG)  # increase verbosity
```

---

## End-to-End Example

A complete workflow from raw inputs to a multi-model stacked output.

### 1. Split a large dataset

```bash
# Split input molecules into chunks for parallel model runs
eosframes split compounds.csv compounds_chunks/ --chunksize 10000
# → compounds_chunks/chunk_000.csv ... chunk_009.csv

# Or split a model output for downstream processing
eosframes split eos4e40_v1_full.csv eos4e40_v1_chunks/ --chunksize 5000
```

Each chunk retains the original column headers.

### 2. Run the Ersilia models (outside eosframes)

```bash
ersilia run -i compounds_chunks/chunk_000.csv -o eos4e40_v1_chunk_000.csv
ersilia run -i compounds_chunks/chunk_001.csv -o eos4e40_v1_chunk_001.csv
# ... repeat for all chunks and all models
```

### 3. Append output chunks from the same model

```bash
eosframes append eos4e40_v1_chunk_*.csv -o eos4e40_v1.csv
eosframes append eos3804_v1_chunk_*.csv -o eos3804_v1.csv
```

### 4. Deduplicate (in case of re-runs or overlapping inputs)

```bash
eosframes dedupe eos4e40_v1.csv eos4e40_v1_clean.csv
eosframes dedupe eos3804_v1.csv eos3804_v1_clean.csv
```

### 5. Inspect each output

```bash
eosframes summary eos4e40_v1_clean.csv
```

```
────────────────────── eos4e40_v1_clean.csv ──────────────────────
  Model ID:  eos4e40
  Version:   v1
  Format:    CSV
  Rows:      100,000
  Features:  3 column(s)

 Column     dtype    Missing    Min     Mean     Max
────────────────────────────────────────────────────
 score     float32        0   0.001   0.4821   0.999
 prob      float32        0       0   0.5013       1
 label     object         0       —   2 unique    —
```

### 6. Convert to H5 for compact storage

```bash
eosframes convert eos4e40_v1_clean.csv eos4e40_v1.h5
eosframes convert eos3804_v1_clean.csv eos3804_v1.h5
```

### 7. Stack outputs from multiple models

```bash
# Combine eos4e40 and eos3804 outputs side by side
eosframes stack eos4e40_v1.h5 eos3804_v1.h5 --output combined_v1.csv
```

The result has columns: `key`, `input`, `score.eos4e40`, `prob.eos4e40`, `label.eos4e40`, `activity.eos3804`, …

---

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
