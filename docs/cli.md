# CLI reference

The complete reference for the `eosframes` command-line tool. Every flag,
every command, with copy-pasteable examples. The README has a quick
grouped table that's enough for daily use; this doc is for when you need
the exact behaviour of a flag or want to know which error you just
triggered.

> All commands accept `--help` for inline help. Default log level is INFO.
> Set `EOSFRAMES_LOG_LEVEL=DEBUG` (or call `eosframes.set_verbosity(True)`
> from Python) for verbose output.

**Contents**

1. [Conventions and global behaviour](#conventions-and-global-behaviour)
2. [Pipeline commands](#pipeline-commands) — `split`, `convert`, `append`, `dedupe`
3. [Multi-model stacking](#multi-model-stacking) — `stack`, `unstack`
4. [Inspection and sidecars](#inspection-and-sidecars) — `summary`, `info`, `columns`
5. [Scaling](#scaling) — `fit`, `transform`
6. [Common error patterns](#common-error-patterns)
7. [Exit codes and logging](#exit-codes-and-logging)
8. [Where to look in the code](#where-to-look-in-the-code)

## Conventions and global behaviour

- **Input paths** must point at existing files (or directories, for chunks).
  Click rejects missing inputs with its own error before eosframes runs.
- **Output paths** are validated against the [naming convention][nom]; on
  every command, eosframes refuses to overwrite an existing destination
  ("Remove it first."). Choose a new name or delete the old one yourself.
- **Model-ID cross-check.** Every command that writes a data file or
  sidecar extracts the model ID from the destination and validates it
  against the input. Mismatches raise with a "Try: …" suggestion.
- **Format inference.** When a command takes a `-o OUTPUT` flag with a
  free extension, the format is inferred from the extension (`.csv`,
  `.h5`, `.json` where applicable).
- **`--help` everywhere.** `eosframes --help` lists every command;
  `eosframes <command> --help` shows the per-command help with examples.

[nom]: nomenclature.md
[scale]: scaling.md

## Pipeline commands

The four commands you'll use to slice, assemble, and tidy raw model
outputs.

### `split` — slice a CSV into chunk files

```
eosframes split INPUT.csv -o FOLDER/ [--chunksize 10000]
```

The **only** command that accepts any CSV — there's no model ID to
validate yet. Splits `INPUT.csv` into `chunk_NNN.csv` files in `FOLDER/`,
zero-padded to fit the total chunk count.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Destination folder. Must not exist; will be created. |
| `--chunksize INTEGER` | no | `10000` | Rows per chunk file. |

```bash
eosframes split compounds.csv -o chunks/
eosframes split compounds.csv -o chunks/ --chunksize 5000
```

Refuses: if `FOLDER/` already exists.

### `convert` — switch between CSV, H5, and chunk folders

```
eosframes convert INPUT -o OUTPUT
```

Format is inferred from extensions. `INPUT` may be a CSV, an H5, or a
folder of chunk CSVs.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Output file. Must follow the naming convention. |

Supported conversions:

```
folder/  → eos4e40_v1.csv   (assemble chunks → CSV)
folder/  → eos4e40_v1.h5    (assemble chunks → H5)
.csv     → eos4e40_v1.h5
.h5      → eos4e40_v1.csv
```

```bash
eosframes convert eos4e40_v1_chunks/ -o eos4e40_v1.csv
eosframes convert eos4e40_v1.csv -o eos4e40_v1.h5
```

Refuses: `OUTPUT` doesn't match the convention; `OUTPUT` already exists;
input folder is empty; unrecognised extensions.

### `append` — vertically concatenate batches of the same model

```
eosframes append IN1 IN2 ... -o OUTPUT
```

All inputs must share the same model ID and column layout. Rows are
appended in the given order. Duplicates are **not** dropped — pipe
through `dedupe` if needed.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Output file. Must follow the convention; encoded model ID must match the inputs. |

```bash
eosframes append eos4e40_v1_part1.csv eos4e40_v1_part2.csv -o eos4e40_v1.csv
eosframes append eos4e40_v1_part1.h5  eos4e40_v1_part2.h5  -o eos4e40_v1.h5
```

Refuses: fewer than 2 inputs; column or model-ID mismatch across inputs;
output naming-convention violation; output exists.

### `dedupe` — drop duplicate rows by `key`

```
eosframes dedupe INPUT -o OUTPUT
```

Keeps the first occurrence of each `key`. Output extension determines
format (`.csv` or `.h5`); model ID must match the input.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Output file. Same model ID as input. |

```bash
eosframes dedupe eos4e40_v1_raw.csv -o eos4e40_v1.csv
```

Refuses: input lacks a `key` column; model-ID mismatch; output exists.

## Multi-model stacking

Two commands for combining outputs across models, and the inverse.

### `stack` — horizontally combine outputs from different models

```
eosframes stack IN1 IN2 ... -o OUTPUT
```

All inputs must have the same `key` / `input` columns in the same row
order — this is validated before stacking. The shared `key` / `input`
columns appear once in the output; feature columns are appended in input
order.

The **output filename picks the column-naming mode** (there is no flag):

- `[prefix_]eosmix.csv` → Mode A. Feature columns get suffixed
  `_<model_id>_<version>`.
- `[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv` → Mode B. Feature columns stay
  bare; every input's `(model_id, version)` must appear in the filename
  in the same order as the inputs.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Output CSV. Must follow Mode A or Mode B naming. |

```bash
# Mode A (provenance in column names)
eosframes stack eos4e40_v1.csv eos7m30_v1.csv -o project_eosmix.csv

# Mode B (provenance in filename)
eosframes stack eos4e40_v1.csv eos7m30_v1.csv -o eos4e40_v1_eos7m30_v1.csv
```

Refuses: fewer than 2 inputs; any input doesn't follow the convention;
duplicate `(model_id, version)` pairs across inputs; row inputs differ
between inputs; Mode B output filename lists the wrong models or wrong
order; output exists.

See [docs/nomenclature.md][nom] for the full Mode A vs Mode B contract.

### `unstack` — split a stacked CSV back into per-model files

```
eosframes unstack STACKED -o FOLDER/
```

The mode is resolved from the input filename. In **Mode A**, columns are
grouped by their `_<model_id>_<version>` suffix and the suffix is
stripped on write. In **Mode B**, each model's `run_columns.csv` is
fetched from GitHub and columns are distributed by name. The prefix
(when present) is inherited by every per-model file.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | yes | — | Destination folder. Must not exist; will be created. |

```bash
eosframes unstack project_eosmix.csv -o ./per_model/
eosframes unstack eos4e40_v1_eos7m30_v1.csv -o ./per_model/
```

Refuses: input doesn't match a stack convention; input missing `key` or
`input` column; Mode B columns can't be unambiguously distributed
(unmatched / ambiguous / missing — see the relevant section in
`unstack_file` in `ops.py`); output folder exists.

## Inspection and sidecars

The "look at this file or model" commands. All three optionally write a
matching sidecar CSV when `-o` is given.

### `summary` — per-feature stats from a local file

```
eosframes summary INPUT [-o SUMMARY.csv]
```

Pretty-prints model metadata (model ID, version, row / column counts,
duplicates, missing data) and per-feature statistics (dtype, missing,
min, mean, max).

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | no | — | Sidecar CSV. Must match `[prefix]_<model_id>_<version>_summary.csv` with the input's model ID / version. |

```bash
eosframes summary eos4e40_v1.csv
eosframes summary eos4e40_v1.csv -o eos4e40_v1_summary.csv
```

Refuses (when `-o` is given): output doesn't match the sidecar
convention; model ID / version mismatch; output exists.

### `info` — model metadata from GitHub

```
eosframes info INPUT [-o INFO.csv]
```

Fetches `metadata.json` (or `metadata.yml` / `.yaml`) from
`github.com/ersilia-os/<model_id>` and prints a table of fields.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | no | — | Sidecar CSV. Must match `[prefix]_<model_id>_<version>_info.csv` with the input's model ID / version. |

```bash
eosframes info data/example_eos4e40_v1.csv
eosframes info data/example_eos4e40_v1.csv -o example_eos4e40_v1_info.csv
```

Refuses: input doesn't follow the convention; metadata file not found on
GitHub; (with `-o`) sidecar mismatch as for `summary`.

### `columns` — per-version feature definitions from GitHub

```
eosframes columns INPUT [-o COLUMNS.csv]
```

Fetches `run_columns.csv` for the input's `(model_id, version)`. Tries
the semver tag (`v1` → `v1.0.0`) first, falls back to `main`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-o`, `--output PATH` | no | — | Sidecar CSV. Must match `[prefix]_<model_id>_<version>_columns.csv` with the input's model ID / version. |

```bash
eosframes columns data/example_eos4e40_v1.csv
eosframes columns data/example_eos4e40_v1.csv -o example_eos4e40_v1_columns.csv
```

Refuses: input doesn't follow the convention; `run_columns.csv` not
found on any candidate ref; (with `-o`) sidecar mismatch as for `summary`.

## Scaling

Two commands for the type-aware robust scaler. See
[docs/scaling.md][scale] for the mental model — what kinds exist, when
each is picked, and the on-disk JSON schema.

### `fit` — fit a scaler and save its parameters

```
eosframes fit INPUT -s TRANSFORMER.json [-o SCALED] [--quantize] [--impute] [--chunksize N]
```

Auto-classifies every numeric feature column and writes a dtype-agnostic
scaler JSON. The JSON's filename must match the
[transformer convention](nomenclature.md#the-patterns):
`[prefix_]<model_id>_<version>_transformer.json` with the input's model
ID / version. If `-o` is given, also runs the inline transform once
(fit-transform).

**Memory.** The fit never loads the whole file. It walks one feature
column at a time (~`n_rows` values), so a frame with thousands of columns
fits in a fraction of its on-disk size. H5 inputs are column-sliced
directly; CSV inputs are first streamed — in row-chunks of `--chunksize`
— into a temporary columnar H5, which is removed after the fit. (For
repeated work on a huge CSV, `eosframes convert INPUT.csv -o INPUT.h5`
once, then fit/transform on the H5 to skip the per-run staging pass.)

| Flag | Required | Default | Description |
|---|---|---|---|
| `-s`, `--scaler PATH` | yes | — | Where to save the scaler JSON. Must follow the transformer convention. |
| `-o`, `--output PATH` | no | — | If set, also write the scaled output here (fit-transform). |
| `--quantize` | no | off | Only meaningful with `-o`: write the inline output as int8 in `[-127, 127]` (`-128` for NaN). The saved scaler JSON is **always** dtype-agnostic. |
| `--impute` | no | off | Only meaningful with `-o`: replace input NaN with each column's fit-time median before the inline transform. The scaler JSON always records the impute value; this flag only opts in for the inline output. |
| `--chunksize INTEGER` | no | `50000` | Rows per chunk for CSV→H5 staging and the inline transform. Bounds peak memory; lower for very wide frames, raise for narrow ones. |

```bash
eosframes fit eos4e40_v1.csv -s eos4e40_v1_transformer.json
eosframes fit eos4e40_v1.csv -s eos4e40_v1_transformer.json -o eos4e40_v1_scaled.csv
eosframes fit eos4e40_v1.csv -s eos4e40_v1_transformer.json -o eos4e40_v1_scaled.csv --quantize --impute
```

Notes:
- `--quantize` / `--impute` without `-o` emit a warning and are otherwise
  no-ops.
- The CLI prints the path of the file it just wrote (`-o` if given,
  otherwise the scaler).

Refuses: scaler filename doesn't follow the transformer convention or
disagrees with the input's model ID / version; `-s` already exists; `-o`
already exists.

### `transform` — apply a saved scaler

```
eosframes transform INPUT -s TRANSFORMER.json -o OUTPUT [--quantize] [--impute] [--chunksize N]
```

Loads the scaler JSON, validates compatibility, and writes the scaled
output. `key` and `input` columns pass through unchanged.

**Memory.** Input is read and output written one row-chunk of
`--chunksize` at a time, so peak memory is one chunk in plus one chunk
out — independent of file size. Works for any CSV/H5 in-and-out
combination.

| Flag | Required | Default | Description |
|---|---|---|---|
| `-s`, `--scaler PATH` | yes | — | Scaler JSON produced by `eosframes fit`. Must exist. |
| `-o`, `--output PATH` | yes | — | Output file. Format inferred from extension (`.csv` / `.h5`). |
| `--quantize` | no | off | Quantize output to int8 in `[-127, 127]` (`-128` for NaN). |
| `--impute` | no | off | Replace input NaN with each column's recorded `impute_value` before applying the transform. The output will have no NaN entries. |
| `--chunksize INTEGER` | no | `50000` | Rows per streamed chunk. Bounds peak memory; lower for very wide frames, raise for narrow ones. |

```bash
eosframes transform new_eos4e40_v1.csv -s eos4e40_v1_transformer.json -o scaled.csv
eosframes transform new_eos4e40_v1.csv -s eos4e40_v1_transformer.json -o scaled.h5 --quantize
eosframes transform new_eos4e40_v1.csv -s eos4e40_v1_transformer.json -o scaled.csv --impute
```

Refuses (see [docs/scaling.md → What `transform` refuses][scale]): version
mismatch (major component), method mismatch, model-ID or version
mismatch between scaler and input, feature-column mismatch, invalid
`--quantize` / dtype combination, output exists.

## Common error patterns

The error message always carries a concrete suggestion — read the
"Try: …" line if there is one. The most common rejections:

- **"…does not follow the naming convention."** The destination filename
  doesn't match any pattern. See [docs/nomenclature.md][nom].
- **"…already exists. Remove it first."** Every writer refuses to
  clobber. Delete the existing path or pick a new one.
- **"Model ID mismatch: …"** The filename's encoded model ID disagrees
  with the input file's `df.model_id`. Either the filename is wrong, or
  you're writing one model's output to another model's filename.
- **"Model order mismatch in output filename …"** (Mode B `stack` only.)
  The output filename's model list doesn't match the input order.
- **"Re-fit the scaler — the schema only carries across the same
  eosframes major version."** (`transform`.) Your saved scaler was fitted
  with an incompatible eosframes version.
- **"Feature columns mismatch …"** (`transform`.) The input file's
  feature columns don't exactly match the scaler's recorded list.

## Exit codes and logging

- **Exit codes.** `0` on success. Non-zero on any `EosframesError` (Click
  translates the wrapped exception into its standard exit-code path) or
  on Click-level argument validation errors.
- **Logging.** Default level is INFO — every read / write / step prints a
  one-line status. `EOSFRAMES_LOG_LEVEL=DEBUG` (or
  `eosframes.set_verbosity(True)` from Python) opens up the DEBUG stream:
  HTTP probe traces, per-chunk index lines, per-column classifier
  decisions in `fit`.
- **Progress.** The streaming `fit` / `transform` report progress two ways,
  picked automatically (both at INFO or more verbose; silent under
  `EOSFRAMES_LOG_LEVEL=WARNING`):
  - **Interactive terminal** — an in-place progress bar on stderr (the fit
    bar counts columns, the transform bar counts row-chunks).
  - **Piped / non-interactive** (a log file, `nohup`, CI, `build_scaler.sh`)
    — periodic INFO log lines instead, so a long job on a huge file stays
    legible. You'll see `Staging: N rows written…` during CSV→H5 staging,
    `Fitting columns: k/N (p%)` through the fit, and `Transforming chunks:
    k/N (p%)` through the transform (percentages whenever the row count is
    known up front, i.e. H5 inputs).
- **stderr vs stdout.** Logs go to stderr. The only thing on stdout is
  the path of the file a command just wrote (when applicable) — so you
  can pipe `eosframes fit ... -o file | xargs ...`.

## Where to look in the code

| Concern | Location |
|---|---|
| Click commands (one function per command) | `src/eosframes/cli.py` |
| Business logic (per-command implementation) | `src/eosframes/ops.py`, `src/eosframes/scale.py` |
| Naming-convention parsing & gates | `src/eosframes/naming.py` |
| Readers / writers (used by every command) | `src/eosframes/read.py`, `src/eosframes/write.py` |
| GitHub fetcher (used by `info`, `columns`, `unstack` Mode B) | `src/eosframes/hub.py` |
| Sidecar output validation (`-o` on `summary`, `info`, `columns`) | `_resolve_sidecar_output` in `cli.py` |
| Error translation (`EosframesError` → Click) | `_err` in `cli.py` |
