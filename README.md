![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Manipulating Ersilia's dataframes

`eosframes` is a library for manipulating inputs and outputs from the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia). It splits, assembles, converts, scales, and summarises tabular model output files.

## Installation

Python ≥ 3.8 is required.

```bash
pip install eosframes
```

Or from source:

```bash
git clone https://github.com/ersilia-os/eosframes.git
cd eosframes
pip install -e .
```

## Quick start

Every file the library reads or writes encodes a model ID and version in its filename, e.g. `eos4e40_v1.csv` (model `eos4e40`, version `v1`).

```bash
# Slice a big input CSV into chunks for parallel model runs
eosframes split compounds.csv -o chunks/ --chunksize 10000

# Stitch the per-batch outputs back into one file
eosframes append eos4e40_v1_000.csv eos4e40_v1_001.csv -o eos4e40_v1.csv

# Combine outputs from multiple models, side by side
eosframes stack eos4e40_v1.csv eos7m30_v1.csv -o project_eosmix.csv
```

Everything the CLI does is also importable:

```python
from eosframes import read_csv, hstack, fit, transform

df = read_csv("eos4e40_v1.csv")
params = fit(df)
scaled = transform(df, params)
```

Run `eosframes --help` (or `eosframes <command> --help`) for inline help.

## Commands

| Command     | Purpose                                                       |
|-------------|---------------------------------------------------------------|
| `split`     | Slice any CSV into chunk files for parallel model runs.       |
| `convert`   | CSV ↔ H5, or assemble a chunks folder.                        |
| `append`    | Vertically concatenate batches from the same model.           |
| `dedupe`    | Drop duplicate rows by `key`.                                 |
| `stack`     | Horizontally combine outputs from different models.           |
| `unstack`   | Split a stacked file back into per-model files.               |
| `summary`   | Per-feature stats from a local file.                          |
| `info`      | Model metadata fetched from GitHub.                           |
| `columns`   | Feature definitions fetched from GitHub.                      |
| `fit`       | Fit a type-aware robust scaler and save its parameters.       |
| `transform` | Apply a saved scaler to a file.                               |

See [`docs/cli.md`](docs/cli.md) for every flag, example, and refusal condition.

`fit` and `transform` stream in bounded memory — `fit` walks one column at a
time, `transform` one row-chunk at a time — so they handle files far larger
than RAM (wide fingerprint frames, tens of GB). Tune `--chunksize` to trade
memory for throughput.

## Scripts

[`scripts/build_scaler.sh`](scripts/build_scaler.sh) builds and packages a scaler
for one model in a single step: it pulls the model's precalculations from the
[isaura](https://github.com/ersilia-os/isaura) store over the Ersilia reference
library (`data/ersilia_reference_library_v0.csv`), fits an `eosframes` scaler, and
compresses the transformer into a versioned zip.

```bash
scripts/build_scaler.sh <model_id> <version>      # e.g. eos4e40 v1
```

The artifact is written to:

```
output/ersilia_reference_library_v0/<model_id>/<version>/scaler-<eosframes-major>.zip
```

(e.g. `output/ersilia_reference_library_v0/eos4e40/v1/scaler-1.zip`), containing a
single `<model_id>_<version>_transformer.json`. The bucket defaults to
`isaura-public`; override it with the `PROJECT_NAME` environment variable.

**Prerequisites:** `isaura` and `eosframes` on `PATH`, and a running local isaura
MinIO engine (`isaura engine --start`).

## Documentation

- [`docs/cli.md`](docs/cli.md) — every CLI command, all flags, examples, and error patterns.
- [`docs/nomenclature.md`](docs/nomenclature.md) — every recognised filename / directory pattern, the strict/lenient contract, and the two stack modes.
- [`docs/scaling.md`](docs/scaling.md) — the type-aware robust scaler: column kinds, how each is picked, quantization / imputation, and the bounded-memory streaming fit/transform for large files.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
