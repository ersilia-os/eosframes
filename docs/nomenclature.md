# Nomenclature in eosframes

A reference for the file and directory naming convention enforced by
`src/eosframes/naming.py`. This is the doc to land on when you need to
check whether a filename is valid, what the difference between Mode A and
Mode B stack outputs is, or which helper builds a given canonical name.

> The regexes in `naming.py` are the source of truth. Examples here are
> illustrative; helper function names are stable.

**Contents**

1. [TL;DR](#tldr)
2. [The patterns](#the-patterns)
3. [The components](#the-components)
4. [Strict vs lenient](#strict-vs-lenient)
5. [The two stack modes](#the-two-stack-modes)
6. [What the strict gates refuse](#what-the-strict-gates-refuse)
7. [Edge cases worth knowing](#edge-cases-worth-knowing)
8. [Where to look in the code](#where-to-look-in-the-code)

## TL;DR

Every file and directory eosframes touches encodes a **model ID** and
(usually) a **version** in its name. The write side is strict: writers
refuse non-conforming destinations and refuse to overwrite anything that
already exists. The read side is lenient: readers only need a recognisable
model ID somewhere in the basename. The convention exists so that parsing
is uniform, model-ID / version cross-checks are automatic, and you can't
accidentally write one model's output over another's.

## The patterns

| Pattern | Purpose | Example |
|---|---|---|
| `[prefix_]<model_id>_<version>.<ext>` | Data file (CSV or H5) | `eos4e40_v1.csv`, `example_eos4e40_v1.h5` |
| `[prefix_]<model_id>_<version>_chunks/` | Chunks directory | `eos4e40_v1_chunks`, `260313_gardp_eos4e40_v1_chunks` |
| `[prefix_]<model_id>_<version>_info.csv` | Sidecar — model metadata | `eos4e40_v1_info.csv` |
| `[prefix_]<model_id>_<version>_columns.csv` | Sidecar — feature definitions | `eos4e40_v1_columns.csv` |
| `[prefix_]<model_id>_<version>_summary.csv` | Sidecar — per-feature stats | `eos4e40_v1_summary.csv` |
| `[prefix_]<model_id>_<version>_transformer.json` | Saved scaler | `eos4e40_v1_transformer.json` |
| `[prefix_]eosmix.csv` | Stack output, Mode A | `eosmix.csv`, `project_eosmix.csv` |
| `[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv` (N ≥ 2) | Stack output, Mode B | `eos4e40_v1_eos7m30_v1.csv` |

**Precedence rule.** The sidecar regexes (`_info`, `_columns`, `_summary`)
and the transformer regex are checked **before** the generic data-file
pattern in `parse_name`. Without that ordering,
`example_eos4e40_v1_info.csv` would misparse as a data file with the
prefix `example_eos4e40_v1_info` rather than a sidecar. The stack-output
patterns have **dedicated** helpers (`parse_stack_mix_name`,
`parse_stack_explicit_name`) and are intentionally **not** folded into
`parse_name` — a Mode B stem overlaps syntactically with a long-prefix
regular data file, so the dispatch happens explicitly at the call site
(`stack_files` / `unstack_file`).

## The components

- **`model_id`** — Ersilia identifier: `eos` + 1 digit + 3 alphanumeric
  characters. Regex `eos\d[A-Za-z0-9]{3}`. Examples: `eos4e40`, `eos7m30`,
  `eos2af0`.
- **`version`** — the letter `v` followed by one or more digits. Examples:
  `v1`, `v2`, `v42`.
- **`ext`** — one of `csv` or `h5` for data files. JSON is reserved for the
  transformer pattern.
- **`prefix`** — optional. Alphanumeric tokens joined by underscores. Regex
  `[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*`. Examples: `example`, `260313_gardp`.

## Strict vs lenient

**The write side is strict.** Writers (`write_csv`, `write_h5`,
`write_chunked_csvs`, and every file-level op in `ops.py` —
`stack_files`, `append_files`, `dedupe_file`, …) refuse a destination
path that doesn't match the convention, refuse to overwrite an existing
path, and cross-validate the path-encoded model ID against
`df.model_id`. Mismatches raise `EosframesError` with a suggestion of
what to rename to.

**The read side is lenient.** Readers (`read_csv`, `read_h5`,
`read_chunked_csvs`) only need a recognisable model ID *somewhere* in
the basename — this is the `get_model_id_from_path` helper, which scans
the basename for any `eos\d[A-Za-z0-9]{3}` substring. The `_v<N>` token
is optional on inputs; if absent, `df.version` is `None` (writers can
still operate as long as the destination encodes a version).

The single full exception to both halves is `eosframes split` (CLI) /
`split_csv` (Python). It takes any CSV — raw user-supplied input, with
no model ID expected — because its job is to slice data *before* a model
run.

## The two stack modes

`stack_files` picks its column-naming strategy from the output filename,
not from a flag.

**Mode A — `[prefix_]eosmix.csv`.** Feature columns are suffixed with
`_<model_id>_<version>` so column names carry the provenance. The
filename itself doesn't list the stacked models. Example:
`project_eosmix.csv` containing columns like `logp_eos4e40_v1`,
`logp_eos7m30_v1`.

**Mode B — `[prefix_]<m1>_<v1>_..._<mN>_<vN>.csv` (N ≥ 2).** Feature
columns stay bare. The filename lists every `(model_id, version)` that
was stacked, in the **same order** as the inputs passed to
`stack_files`. The validator rejects a mismatch and the error message
suggests the canonical filename. Example: `eos4e40_v1_eos7m30_v1.csv`.

Both modes reject duplicate `(model_id, version)` pairs across inputs —
they'd collide in Mode A and produce an ambiguous filename in Mode B.

## What the strict gates refuse

The most common rejection reasons, paraphrased from `naming.py` and the
writers / ops:

- Destination filename or directory name doesn't match any recognised
  pattern.
- Destination already exists ("Remove it first." — every writer).
- Path-encoded model ID doesn't match `df.model_id`.
- Mode B output filename lists the wrong models or wrong order (relative
  to the inputs to `stack_files`).
- Sidecar destination's model ID or version doesn't match the input file
  (via `_resolve_sidecar_output` in `cli.py`).

All of these surface as `EosframesError` with a specific message;
filename-shape errors include a "Try: …" suggestion built from the
canonical-name helpers.

## Edge cases worth knowing

- **`eosframes split` is the only entry point that accepts a
  non-conforming input filename.** It pre-processes raw inputs before a
  model run, so there's no model ID to validate against yet.
- **`vstack` drops `df.version`.** Vertical concatenation across versions
  of the same model is allowed, so the result no longer corresponds to a
  single version. `df.model_id` is preserved (and required to match
  across all inputs).
- **Reading a file with no `_v<N>` token works.** The resulting
  DataFrame has `df.version is None`. Writing, by contrast, requires a
  full `<model_id>_<version>` stem in the destination.
- **`is_valid_name` is data-file-only.** It explicitly rejects sidecars,
  transformer JSONs, and stack outputs. Use the dedicated
  `is_valid_*_name` helper for those.
- **Stack-output filenames are *not* recognised by `parse_name`.** It only
  understands data files, chunks directories, and sidecars (the six
  `name_type` values: `csv`, `h5`, `chunks_dir`, `info`, `columns`,
  `summary`). For stack outputs use the Mode-A / Mode-B `parse_*`
  helpers.

## Where to look in the code

All paths are relative to `src/eosframes/naming.py`. Function names are
stable; line numbers will drift.

| Concern                            | Helpers                                                                                                |
|------------------------------------|--------------------------------------------------------------------------------------------------------|
| Model ID validation                | `is_model_id_valid`                                                                                    |
| Data files & chunks                | `parse_name`, `make_output_name`, `make_chunks_dir_name`, `is_valid_name`, `get_version_from_path`     |
| Sidecars (info / columns / summary)| `is_valid_{info,columns,summary}_name`, `make_{info,columns,summary}_name`, `_make_sidecar_name`       |
| Transformer JSON                   | `is_valid_transformer_name`, `parse_transformer_name`, `make_transformer_name`                         |
| Stack Mode A (eosmix)              | `is_valid_stack_mix_name`, `parse_stack_mix_name`, `make_stack_mix_name`                               |
| Stack Mode B (explicit)            | `is_valid_stack_explicit_name`, `parse_stack_explicit_name`, `make_stack_explicit_name`                |
| Lenient model-ID scan              | `get_model_id_from_path`                                                                               |
| Prefix grammar gate                | `_validate_prefix`                                                                                     |

All public helpers above are re-exported from `eosframes/__init__.py` and
can be imported directly as `from eosframes import …`.
