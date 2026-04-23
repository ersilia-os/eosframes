# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"             # install with dev deps (ruff, pytest)
ruff check src/                     # lint
pytest tests/ -v                    # run full test suite
pytest tests/test_eosframes.py::test_name -v   # run a single test
eosframes --help                    # CLI entrypoint (installed via poetry script)
```

Python `>=3.8` is supported; ruff is pinned to `target-version = "py38"`, so don't use newer-version-only syntax (`X | Y` unions, `type` built-in as annotation).

## Working conventions

Keep `README.md` in sync as improvements land. When you change CLI flags, Python API surface, naming rules, file formats, or workflows, update the corresponding section of `README.md` in the same change — don't let it drift.

## Architecture

### The `model_id` attribute contract

Every DataFrame that flows through the library carries a `model_id` attribute (set as `df.model_id = "..."`, using pandas' loose attribute assignment). This is the load-bearing invariant of the codebase:

- Readers (`read_csv`, `read_h5`, `read_chunked_csvs` in `src/eosframes/read/read.py`) parse the model ID out of the filename via `get_model_id_from_path` and attach it.
- Writers (`write_csv`, `write_h5` in `src/eosframes/write/write.py`) require `df.model_id` **and** cross-validate it against the model ID encoded in the destination path — mismatches raise `EosframesError`.
- DataFrame ops (`hstack`, `vstack` in `src/eosframes/manipulate/stack.py`) rely on the attribute for column suffixing and same-model checks.
- Ops that accept arbitrary CSV/H5 files (`ops._read_file`, `ops.append_files`, `ops.dedupe_file`) parse the output path with `parse_name` and compare against the incoming `df.model_id`.

When editing or adding I/O paths, preserve this contract: after any transformation, re-set `df.model_id` before writing (see `ops.py` for the pattern — pandas may drop the attribute across `concat`/`drop_duplicates`).

### Naming convention (enforced, not advisory)

`src/eosframes/naming.py` is the single source of truth. Output files must match `[prefix_]<model_id>_<version>.<ext>` where:

- `model_id` = `eos<digit><3 alnum>` (regex `eos\d[A-Za-z0-9]{3}`)
- `version` = `v\d+`
- `ext` ∈ `{csv, h5}`
- Chunks directories end in `_chunks` (e.g. `eos4e40_v1_chunks`)
- A leading prefix token is allowed (e.g. `260313_gardp_eos4e40_v1.csv`, `example_eos4e40_v1.csv`) — `_STEM_RE` matches `<model_id>_<version>` at the end of the stem.

`is_valid_name` / `parse_name` gate most write paths. `get_model_id_from_path` is the looser helper — it finds a model ID anywhere in the basename and is what `read_csv`/`read_h5` use (inputs don't need `_vN`). The `split` CLI command is the one exception that accepts any CSV — it does not require the naming convention on its input.

### Layered module structure

```
cli.py             # Thin Click wrappers. Translates EosframesError → ClickException.
ops.py             # File-level operations: split, convert, stack, append, dedupe.
                   # Handles read→validate→write flow and delegates to read/write modules.
scale.py           # Standard scaler: fit_file saves a scaler JSON, transform_file applies it.
hub.py             # Fetches metadata.json / run_columns.csv from github.com/ersilia-os/<model_id>.
                   # Version "v1" resolves to git tag "v1.0.0" with fallback to "main".
read/read.py       # CSV / H5 / chunked-CSV readers.
write/write.py     # CSV / H5 / chunked-CSV / XLSX writers. XLSX hits GitHub for a legend sheet.
manipulate/stack.py# hstack / vstack (DataFrame-level counterparts of stack_files / append_files).
naming.py          # Convention parsing (see above).
exceptions.py      # EosframesError (sole custom exception).
logger.py          # Singleton logger using rich.RichHandler when available.
```

Keep the layering: CLI → ops → read/write modules. Business logic lives in `ops.py` and `scale.py`; `cli.py` should stay a thin adapter.

### H5 file layout

```
<model_id>_<version>.h5
├── key      (N,)   UTF-8 string   (optional on read, always written if present)
├── input    (N,)   UTF-8 string
├── features (F,)   UTF-8 string   (feature column names)
└── values   (N, F) float32        (model outputs)
```

`values` is always written as `float32`. When round-tripping non-float columns through H5, expect precision loss.

### Write-side safety model

Every write path (CSV, H5, chunks dir, scaler JSON) refuses to overwrite existing files/directories — raises `EosframesError` with "Remove it first." Callers must delete beforehand or choose new paths. Tests rely on this and use `tmp_path` fixtures.

### Tests

Static reference data lives in `data/` (`example_eos4e40_v1.csv`, `example_eos7m30_v1.csv`) and is intentionally committed (see `.gitignore` — `*.csv` is not globally excluded). `tests/test_eosframes.py` uses these as fixtures (`df4e40`, `df7m30`). Tests that write files use pytest `tmp_path`.

### Scaling — `fit_file` + `transform_file`

Two separate file-level functions (mirrored by `eosframes fit` and `eosframes transform`):

| Function         | Behaviour                                                          |
|------------------|--------------------------------------------------------------------|
| `fit_file(input, scaler)` | Fit on `input`, save parameters to `scaler` (must not exist). |
| `transform_file(input, scaler)` | Load parameters from `scaler`, apply to `input`.      |

Scaler files must follow `[prefix_]<model_id>_<version>_transformer.json`. `fit_file` cross-validates the model ID and version encoded in the scaler filename against the input file before writing. `transform_file` cross-validates from the JSON contents. Columns with >25 % missing values are auto-skipped during fitting and recorded in `skipped_columns`.
