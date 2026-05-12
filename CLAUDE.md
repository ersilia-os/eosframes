# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"             # install with dev deps (ruff, pytest)
ruff check src/ tests/              # lint (CI scope)
pytest tests/ -v                    # run full test suite
pytest tests/test_cli.py::test_name -v          # run a single test
eosframes --help                    # CLI entrypoint (installed via poetry script)
```

Python `>=3.8` is supported; ruff is pinned to `target-version = "py38"`, so don't use newer-version-only syntax (`X | Y` unions, `type` built-in as annotation).

## Working conventions

Keep `README.md` in sync as improvements land. When you change CLI flags, Python API surface, naming rules, file formats, or workflows, update the corresponding section of `README.md` in the same change — don't let it drift.

## Architecture

### The `model_id` / `version` attribute contract

Every DataFrame that flows through the library carries two loose attributes: `df.model_id` (always) and `df.version` (when the filename encodes one). Both are set via pandas' loose attribute assignment. This is the load-bearing invariant of the codebase:

- Readers (`read_csv`, `read_h5`, `read_chunked_csvs` in `src/eosframes/read.py`) parse the model ID and version out of the filename (via `get_model_id_from_path` / `get_version_from_path`) and attach both. `df.version` may be `None` if the input lacks a `_v<N>` token.
- Writers (`write_csv`, `write_h5` in `src/eosframes/write.py`) require `df.model_id` **and** cross-validate it against the model ID encoded in the destination path — mismatches raise `EosframesError`.
- `hstack` (in `src/eosframes/stack.py`) requires **both** `model_id` and `version` on every input frame — version is used to suffix columns in `eosmix` mode and to validate the explicit-mode filename. `vstack` requires only `model_id` (version is intentionally not propagated, since vertical concatenation across versions of the same model is allowed).
- Ops that accept arbitrary CSV/H5 files (`ops._read_file`, `ops.append_files`, `ops.dedupe_file`, `ops.unstack_file`) parse the output path with `parse_name` and compare against the incoming `df.model_id`.

When editing or adding I/O paths, preserve this contract: after any transformation, re-set `df.model_id` (and `df.version` where applicable) before writing — pandas drops loose attributes across `concat`/`drop_duplicates` (see `ops.py` for the pattern).

### Naming convention (enforced, not advisory)

`src/eosframes/naming.py` is the single source of truth. Recognised patterns (a leading prefix token is allowed throughout — e.g. `example_`, `260313_gardp_`):

| Pattern                                                  | Purpose                              |
|----------------------------------------------------------|--------------------------------------|
| `[prefix_]<model_id>_<version>.<ext>`                    | Data file (CSV / H5)                 |
| `[prefix_]<model_id>_<version>_chunks/`                  | Folder of chunk CSVs                 |
| `[prefix_]<model_id>_<version>_info.csv`                 | Sidecar — model metadata             |
| `[prefix_]<model_id>_<version>_columns.csv`              | Sidecar — feature column definitions |
| `[prefix_]<model_id>_<version>_summary.csv`              | Sidecar — per-feature stats          |
| `[prefix_]<model_id>_<version>_transformer.json`         | Saved scaler                         |
| `[prefix_]eosmix.csv`                                    | Stack output, Mode A                 |
| `[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv` (N ≥ 2)           | Stack output, Mode B                 |

with:

- `model_id` = `eos<digit><3 alnum>` (regex `eos\d[A-Za-z0-9]{3}`)
- `version` = `v\d+`
- `ext` ∈ `{csv, h5}`
- `prefix` = alphanumeric tokens joined by underscores (e.g. `260313_gardp_eos4e40_v1.csv`)

Sidecar regexes (`_INFO_STEM_RE`, `_COLUMNS_STEM_RE`, `_SUMMARY_STEM_RE`, `_TRANSFORMER_STEM_RE`) are checked **before** the generic data-file pattern in `parse_name`, so the trailing `_info` / `_columns` / `_summary` token is not swallowed as part of a data-file prefix. Stack outputs have dedicated regexes (`_EOSMIX_STEM_RE`, `_MODEL_VER_PAIR_RE`) and are not folded into `parse_name` — Mode B's stem overlaps syntactically with a long-prefix regular data file. Each pattern has paired `is_valid_*_name` / `parse_*_name` / `make_*_name` helpers; everything is re-exported from `eosframes/__init__.py`.

`is_valid_name` is data-file-only — sidecars, transformers, and stack outputs are explicitly rejected by it; use the dedicated helpers for those. `get_model_id_from_path` is the looser helper — it finds a model ID anywhere in the basename and is what `read_csv` / `read_h5` use (inputs don't need `_vN`). The `split` CLI command is the one full exception — it accepts any CSV and does not require the naming convention on its input.

### Two stack modes (`stack_files` / `hstack`)

`stack_files` (CLI: `eosframes stack`) resolves its mode from the **output filename**, not a flag:

- **Mode A — `eosmix`** (`[prefix_]eosmix.csv`): feature columns get a `_<model_id>_<version>` suffix, so column names carry the provenance. The filename does not embed the model list.
- **Mode B — `explicit`** (`[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv`): feature columns stay bare. The output filename must list every `(model_id, version)` in the same order as the inputs — `stack_files` validates this and rejects on mismatch. Duplicate `(model_id, version)` pairs are rejected in both modes.

`unstack_file` (CLI: `eosframes unstack`) is the inverse. In Mode A it groups by the column suffix. In Mode B it fetches each model's `run_columns.csv` via `hub.fetch_columns` and assigns columns by name — ambiguous, unmatched, or missing columns all raise `EosframesError`.

`hstack` accepts an explicit `mode: {"eosmix", "explicit"}` argument; the file-level `stack_files` is the one that infers it from the output path.

### Layered module structure

```
cli.py             # Thin Click wrappers. Translates EosframesError → ClickException.
                   # Commands: split, convert, stack, unstack, append, dedupe,
                   #           summary, info, columns, fit, transform.
ops.py             # File-level operations: split_csv, convert_file, stack_files,
                   # unstack_file, append_files, dedupe_file. Handles
                   # read→validate→write flow and delegates to read/write modules.
scale.py           # Type-aware robust scaler (method "robust_typed"; scaler JSONs are pinned to the running eosframes.__version__).
                   # fit_file saves a scaler JSON, transform_file applies it.
hub.py             # Fetches metadata.json / run_columns.csv from
                   # github.com/ersilia-os/<model_id>. fetch_columns resolves
                   # "v1" → tag "v1.0.0" with fallback to "main"; fetch_metadata
                   # goes straight to "main".
read.py            # CSV / H5 / chunked-CSV readers. Attach df.model_id + df.version.
write.py           # CSV / H5 / chunked-CSV writers.
stack.py           # hstack (mode="eosmix" | "explicit") / vstack — DataFrame-level
                   # counterparts of stack_files / append_files.
naming.py          # Convention parsing (see above).
utils.py           # `chunker(df, chunksize)` — the only helper, used by split_csv
                   # and write_chunked_csvs.
exceptions.py      # EosframesError (sole custom exception).
logger.py          # Singleton logger using rich.RichHandler when available.
```

Keep the layering: CLI → ops → read/write modules. Business logic lives in `ops.py` and `scale.py`; `cli.py` should stay a thin adapter that also handles sidecar-output naming validation (see `_resolve_sidecar_output` for the pattern used by `summary`, `info`, `columns`).

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

The test suite is split by layer:

- `tests/test_cli.py` — CLI integration (happy-paths through `click.testing.CliRunner`).
- `tests/test_naming.py` — direct unit coverage of `naming.py` regex helpers and the sidecar-vs-data-file precedence in `parse_name`.
- `tests/test_scale.py` — direct unit coverage of `scale.py`, parameterized over the deterministic fixtures in `data/scaler_tests/`.
- `tests/test_io_and_stack.py` — direct unit coverage of `read.py`, `write.py`, `stack.py`, and the error paths of `ops.py`.

Static reference data lives in `data/`:

- `example_eos4e40_v1.csv`, `example_eos7m30_v1.csv` — shared fixtures (intentionally committed; `.gitignore` does not globally exclude `*.csv`).
- `data/scaler_tests/*.csv` — 46 synthetic per-branch fixtures driving `test_scale.py`. The three `_distributions*.png` files alongside them (`_distributions.png`, `_distributions_scaled.png`, `_distributions_transforms.png`) are generated artifacts and are git-ignored.

Tests that write files use pytest `tmp_path`. Tests that hit `hub.fetch_metadata` / `hub.fetch_columns` use `monkeypatch` — no network access.

### Scaling — `fit_file` + `transform_file`

Two separate file-level functions (mirrored by `eosframes fit` and `eosframes transform`):

| Function         | Behaviour                                                          |
|------------------|--------------------------------------------------------------------|
| `fit_file(input, scaler, output_path=None, output_dtype="float32", impute=False)` | Fit on `input`, save dtype-agnostic parameters to `scaler` (must not exist). `output_dtype` / `impute` only affect the inline-transform output when `output_path` is given. |
| `transform_file(input, scaler, output, output_dtype="float32", impute=False)` | Load parameters from `scaler`, apply to `input`. `output_dtype="int8"` quantizes the result. `impute=True` fills input NaN with the per-column `impute_value` before dispatch. |

The scaler is **type-aware and robust** (`method: "robust_typed"`). Each numeric feature column is auto-classified at fit time into one of **seven kinds**, the single discriminator the transform path reads. Every output region has max-spread of 1 so distance metrics are commensurable.

| `kind`                  | Transform-time payload (keys in addition to `kind`)         | Output region   | When picked at fit                                              |
|-------------------------|-------------------------------------------------------------|-----------------|-----------------------------------------------------------------|
| `constant`              | (none)                                                      | `[0.0, 0.0]`    | ≤ 1 unique non-NaN value, or all-NaN.                            |
| `binary`                | `low`, `high`                                               | `[0.0, 1.0]`    | Exactly 2 unique non-NaN values.                                 |
| `count_zero_mode`       | `high_anchor`                                               | `[0.0, 1.0]`    | Integer-valued ≥ 0, mode = 0. High anchor = `max(Tukey, p99)`.   |
| `count_shifted`         | `center`, `low_anchor`, `high_anchor`, `body_target`        | `[-1.0, 1.0]`   | Integer-valued ≥ 0, mode ≠ 0. Per-side linear body with `body_target = 0.5` (same body design as `continuous_centered` — bulk sits inside `[-0.5, 0.5]`) + a slope-continuous `tanh` asymptote toward `±1` for past-anchor counts. Tail shape differs from `continuous_centered`'s finite-reach quadratic: counts approach `±1` but never reach it, so distinct outliers stay distinct and there's no flat clip plateau. **Soft cap on per-side extent ratio**: when `max(upper_span, lower_span) / min(upper_span, lower_span) > _COUNT_SHIFTED_EXTENT_RATIO_MAX` (currently 2.0, mirrors `continuous_centered`'s `_CENTERED_BODY_RATIO_MAX`), the wider side's extent is capped at `2 × the narrower` and the affected anchor (`high_anchor` or `low_anchor`) is recomputed — so heavily right- or left-tailed count distributions don't squash distinct discrete values into a narrow body band; the excess flows into the `tanh` tail and each distinct count gets a distinct output in `(±0.5, ±1)`. `fit_notes` records `extent_ratio` (pre-cap) and `extent_capped`. |
| `continuous_right_skew` | `tail`, `low_anchor`, `body_anchor`, [`mid_anchor`,] `high_anchor`, `body_target`, [`mid_target`] | `[0.0, 1.0]`    | Bowley > 0.3 **and** `(p99−median)/(median−p1) > 3`. Two tail variants dispatched on the `tail` field. **`tail = "piecewise"`** (heavy tail, p99 > Tukey): 3-segment design — linear body to `(Tukey, body_target=0.8)`, slower linear middle to `(p99, mid_target)`, then `_quadratic_tail` to `(arr.max, 1)`. `mid_target` is derived for C¹ smoothness at the middle→tail join (`(2·a + b·body_target)/(2·a + b)` with `a = p99 − Tukey`, `b = arr.max − p99`). The body→middle slope change is smoothed by a **cubic Hermite blend** across a window of width `2·_PIECEWISE_BLEND_FRACTION·min(body_span, mid_span)` centred on `body_anchor` (the cubic interpolates the body's value+slope at one edge and the middle's value+slope at the other; monotone for the slope ratios seen in practice). The blend folds what would otherwise be a visible "second hill" at `body_target` into a continuous shoulder; bulk stays visible AND outliers spread cleanly across `[body_target, 1]` — no pile-up at `+1`, no double decay. **`tail = "finite"`** (light tail, p99 ≤ Tukey): linear body to `(min(Tukey, arr.max), body_target_derived)` then `_quadratic_tail` to `(arr.max, 1)`; `body_target` is derived from geometry so the quadratic reaches 1 exactly at `arr.max`. |
| `continuous_left_skew`  | `tail`, `low_anchor`, `body_anchor`, [`mid_anchor`,] `high_anchor`, `body_target`, [`mid_target`] | `[-1.0, 0.0]`   | Bowley < −0.3 **and** `(median−p1)/(p99−median) > 3`. Mirror of right-skewed (same `tail = "piecewise"` / `"finite"` dispatch, magnitude flipped). |
| `continuous_centered`   | `center`, `upper_body_extent`, `lower_body_extent`, `upper_max_extent`, `lower_max_extent` | `[-1.0, 1.0]`   | Otherwise (incl. marginal-Bowley columns with two-sided spread like `half_life_obach`). **Per-side body + finite-reach quadratic tail**: each side uses its own `body_extent = max(scale, min(side-p, side-Tukey))` and `max_extent = max(body_extent, side-arr-distance)`. Body slope = `body_target / side_extent`; the apply path takes `effective_max = max(max_extent, _CENTERED_MAX_EXTENT_RATIO · body_extent)` (currently `3·b`), then `body_target = min(_light_tail_body_target(body_extent, effective_max), _CENTERED_BODY_TARGET)` (currently `0.5`). The **cap (`0.5`) and floor (`3·b`) are mathematically locked**: the body→tail join is C¹-smooth precisely when `max_extent = 3·body_extent` for `body_target = 0.5` (body slope `0.5/b` equals quadratic-tail slope `1/(a−b)`). Both activate together when `max_extent < 3·body_extent` (Gaussian-ish / bimodal / bounded U-shape data); past that threshold the natural derivation gives `body_target < 0.5` already and both cap and floor are no-ops. Net effect: the bulk visually lives in `[-0.5, 0.5]` (matching `count_shifted`, commensurable with skewed `[0, 0.8]` / `[-0.8, 0]`); bounded distributions don't compress the tail into a near-zero raw band (so no spurious spray past `±0.5`); OOD inputs past `±effective_max` clip to `±1`. **Safety gate**: when `max(upper, lower)/min(upper, lower) > _CENTERED_BODY_RATIO_MAX` (currently 2.0) — typically a one-sided bounded distribution like `clintox` — the fit falls back to a symmetric body `max(2·scale, min(upper, lower))` AND a symmetric `max_extent = max(upper, lower)` on both sides, so both sides share the same effective max / body_target and the centre has no density step. `fit_notes` records `body_ratio` and `symmetric_fallback`. |

`_OUTPUT_REGIONS` and `_APPLY_DISPATCH` in `scale.py` are the two parallel single-source-of-truth dicts keyed by `kind`. Output region is **never stored** in the JSON — quantization reads it from `_OUTPUT_REGIONS[kind]` at runtime, so the region of a kind can't drift away from its scaling math. Sparse mode-0 counts (mostly zeros, only a handful of distinct output values) get a `fit_notes.degenerate: True` flag and emit a warning — advisory only, transform behaviour is unchanged.

For the full per-kind math (anchor formulas, Hermite smoothstep blend window, the dual `_CENTERED_BODY_TARGET = 0.5` cap and `_CENTERED_MAX_EXTENT_RATIO = 3.0` floor that lock the centered body→tail join to C¹ smoothness, the per-side asymmetric piecewise design), see the module docstring at the top of `scale.py` and the `_fit_*` / `_apply_*` functions below it.

#### Scaler JSON schema

```json
{
  "eosframes_version": "0.1.0",
  "method": "robust_typed",
  "model_id": "eos9000",
  "model_version": "v1",
  "fitted_at": "2026-05-11T20:30:00",
  "n_rows": 1000,
  "columns": {
    "<feature_name>": {
      "transform": { "kind": "<kind>", ...kind-specific params from the table above... },
      "impute_value": <number>,
      "fit_notes": { ... }    // optional; omitted entirely when empty
    },
    ...
  }
}
```

Three things to know:

- **`columns` is a JSON object** keyed by feature name. Fit-time column order is the key-iteration order (Python `json` and every modern parser preserve insertion order). There is no separate `feature_columns` array.
- **`impute_value`** is a peer of `transform`, not inside it. The dispatch math doesn't read it; `transform(..., impute=True)` substitutes it for input NaN *before* dispatch. Integer-valued training columns round the median to `int`; float columns keep it as `float`; all-NaN columns get `0.0`.
- **`fit_notes`** holds fit-time diagnostics the transform path never reads (`bowley` and `tail_asymmetry` for continuous kinds — both must agree to commit to `continuous_right_skew` / `continuous_left_skew`; `scale` / `scale_kind` / `body_ratio` / `symmetric_fallback` for `continuous_centered`; `degenerate` / `mode_fraction` / `n_distinct_output` for degenerate `count_zero_mode`). Omitted entirely when empty.

**Quantization is a transform-time choice, not a fit-time one.** The scaler JSON does not record `output_dtype` — `transform` / `transform_file` accept it as a kwarg, and the CLI exposes it as `--quantize` on `eosframes transform`. `eosframes fit --quantize` is allowed but only meaningful with `-o` (it controls the inline-transform output); without `-o` the CLI emits a warning. `output_dtype="int8"` uses a single trivial map for every column: `int8 = round(x · 127)`, clipped to `[-127, 127]`, with sentinel `-128` reserved for missing. One-sided columns (binary, count mode 0, right- / left-skewed) naturally inhabit only the matching half of the int8 range — a binary 0 stays `0`, a binary 1 is `127` — because the float values themselves only inhabit a half of `[-1, 1]`. This is intentional: simpler than the previous per-column stretch and keeps the int8 representation literally proportional to the float.

#### Versioning gate

Scaler files must follow `[prefix_]<model_id>_<version>_transformer.json`. `fit_file` cross-validates the model ID and version encoded in the scaler filename against the input file before writing. `transform_file` rejects any scaler whose recorded `eosframes_version` doesn't share the **major** component with the running `eosframes.__version__` (and whose `method` isn't `"robust_typed"`). The major-only rule means the whole `0.x.y` line is mutually compatible — only a bump to `1.0.0` invalidates older `0.x` scalers. The running version comes from `pyproject.toml` via `importlib.metadata`. `model_id` / `model_version` in the JSON are cross-validated against the input filename at transform time, so a scaler fit on `eos9000_v1.csv` can never silently apply to `eos9000_v2.csv` outputs.

Every numeric column is fitted at fit time (no >25 %-missing skip threshold); all-NaN columns fall back to `kind: "constant"` with `impute_value: 0.0`.

#### Extending

Adding a new column kind: pick a snake-case `kind` string, add an entry to `_OUTPUT_REGIONS` in `scale.py` (so the int8 quantizer knows the region), write a `_fit_<kind>(series)` returning `{"transform": {"kind": "<kind>", ...}, "impute_value": ..., "fit_notes": {...}?}`, write the matching `_apply_<kind>(series, transform, output_dtype)`, and register it in `_APPLY_DISPATCH`. Structural format breaks (renaming a transform key, removing a kind) are handled automatically by the major-version gate — bump `pyproject.toml` to `1.0.0` to invalidate `0.x` scalers when the change ships.
