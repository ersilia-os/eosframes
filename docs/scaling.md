# Scaling in eosframes

A reference for the type-aware robust scaler in `src/eosframes/scale.py`.
This is the doc to land on when you need to understand what a column went
through, why it was classified the way it was, or what the saved scaler
JSON means.

> Math is deliberately omitted — see the code for the exact formulas. The
> goal here is the mental model.

**Contents**

1. [TL;DR](#tldr)
2. [The seven kinds](#the-seven-kinds)
3. [How a kind gets picked](#how-a-kind-gets-picked)
4. [The scaler JSON](#the-scaler-json)
5. [Quantization and imputation](#quantization-and-imputation)
6. [What `transform` refuses](#what-transform-refuses)
7. [Edge cases worth knowing](#edge-cases-worth-knowing)
8. [Where to look in the code](#where-to-look-in-the-code)

## TL;DR

The scaler classifies each numeric feature column into one of **seven kinds**
(constant, binary, two count flavours, three continuous flavours) and applies
a kind-specific transform. Every output region has max-spread 1 so distance
metrics are commensurable across columns. The fit is **robust** (medians,
Bowley skew, Tukey whiskers — no means/std), the JSON is **dtype-agnostic**
(float vs int8 is chosen at transform time), and re-running `transform`
against a fresh batch never silently lets a wrong model or a stale schema
through.

It is **not** a generic min-max or z-score scaler. The kind is auto-picked
per column.

## The seven kinds

**`constant`** — single-value or all-NaN columns. Everything non-NaN maps to
**0**; NaN propagates. This is also the fallback for any column that the
other kinds can't make sense of.

**`binary`** — exactly two unique non-NaN values. Output is `{0, 1}`, mapping
the smaller fitted value to 0 and the larger to 1. At transform time, unseen
values **snap to the nearer of the two** (ties go to the low value).

**`count_zero_mode`** — non-negative integers whose mode is 0 (think
fingerprints, sparse count vectors). Output is `[0, 1]`, with the high
anchor placed at **`max(Tukey whisker, p99)`** so a heavy-zero distribution
doesn't collapse the body at Q3+1.5·IQR and a denser column still gets
Tukey's larger anchor. Values above the high anchor clip to 1.

**`count_shifted`** — non-negative integers whose mode isn't 0. Output is
`[-1, 1]`, **centred on the mode** with separate upper and lower anchors
(`max(Tukey, p99)` and `min(-Tukey, p1)`, each capped by the data extent).
The body is **per-side linear with `body_target = 0.5`** — same body design
as `continuous_centered`, so the bulk sits inside `[-0.5, 0.5]` and distinct
counts stay distinct. Past the anchors a slope-continuous `tanh` carries
outliers smoothly toward `±1` without ever reaching them (so no flat clip
plateau forms; the tail shape differs from `continuous_centered`'s
finite-reach quadratic, which does land on `±1` at the data edge). The
body is asymmetric whenever the mode isn't the median.

**Soft cap on per-side extent ratio.** When one side of the mode reaches
far further than the other — typical of right- or left-tailed count
distributions — the wider side's discrete values would otherwise be
compressed into a narrow scaled band. So when
`max(upper_span, lower_span) / min(upper_span, lower_span) >
_COUNT_SHIFTED_EXTENT_RATIO_MAX` (currently 2.0, mirroring
`continuous_centered`'s `_CENTERED_BODY_RATIO_MAX`), the wider extent is
**capped** at `2 × the narrower`, and the affected anchor (`high_anchor`
or `low_anchor`) is recomputed so the stored transform reflects the
capped body. The narrower side is unchanged. The excess raw range past
the new anchor flows into the slope-continuous `tanh` tail, where
distinct outlier counts get distinct outputs in `(±0.5, ±1)` instead of
being squashed into the body's `[0, ±0.5]`. The cap is softer than
`continuous_centered`'s symmetric-collapse fallback because the tanh
asymptote already absorbs the excess gracefully. `fit_notes` records the
pre-cap `extent_ratio` and the boolean `extent_capped` for audit.

**`continuous_right_skew`** — bulk-skew **and** tail-asymmetry both indicate
a right-heavy distribution. Output is `[0, 1]`. Two tail variants, dispatched
on the `tail` field of the transform:

- `tail = "piecewise"` (heavy tail, `p99 > Tukey`): 3-segment design — a
  linear body keeps the bulk visible (Q3 lands near `0.35`), a slower linear
  "middle" segment spreads the upper-decile outliers, and a finite quadratic
  tail reaches `1` exactly at `arr.max` with zero slope. `mid_target` is
  derived for C¹ smoothness at the middle→tail join. The histogram shows
  the bulk in `[0, 0.8]` and a separate smaller "tail bump" in `[0.8, 1]` —
  no pile-up at `+1`.
- `tail = "finite"` (light tail, `p99 ≤ Tukey`): single linear body plus a
  quadratic tail to `(arr.max, 1)`; `body_target` is derived from geometry
  so the quadratic reaches 1 exactly at the data edge.

**`continuous_left_skew`** — mirror of right-skew (same `tail =
"piecewise"` / `"finite"` dispatch). Output region is `[-1, 0]`.

**`continuous_centered`** — the default for "well-behaved" continuous
columns. Output is `[-1, 1]` around a robust centre, but the **bulk
visually lives in `[-0.5, 0.5]`**: each side's `body_target` is derived
from `(body_extent, max_extent)` and then capped at `0.5`, matching
`count_shifted` and keeping centered columns commensurable with the
one-sided skewed kinds (whose body sits in `[0, 0.8]` / `[-0.8, 0]`).
Each side of the median has its own **body extent**
(`max(scale, min(side-p, side-Tukey))`) and **max extent**
(`max(body_extent, side-arr-distance)`). The apply path then derives
an `effective_max = max(max_extent, 3·body_extent)` and the quadratic
tail reaches `±1` at `±effective_max` — so distinct outlier values
spread monotonically across `[body_target, 1]` with no asymptotic
saturation and no visual pile-up at `±1`.

The cap (`body_target ≤ 0.5`) and floor (`effective_max ≥ 3·b`) are
**dual constraints locked together by the C¹-smoothness condition**:
for `body_target = 0.5` the body slope `0.5/b` equals the
quadratic-tail slope `1/(a−b)` precisely when `a = 3b`. So whenever
the natural derivation `2b/(b+a)` would push `body_target` past `0.5`
(short-tail regime — typical Gaussian-ish, bimodal, or bounded
distributions like U-shapes and truncated normals), both the cap
**and** the floor activate together and the body→tail join stays
C¹-smooth; past the threshold the geometric derivation gives
`body_target < 0.5` already and both are no-ops. Without the floor,
bounded distributions where `max_extent ≈ body_extent` would compress
the tail's raw domain to near zero and spray a few near-edge points
across the entire `(0.5, 1]` scaled band; the floor collapses those
points to a thin pile-up just past `±0.5` instead, while keeping OOD
inputs past `±3·body_extent` reaching `±1` cleanly.

**Safety gate.** The per-side trick is only safe when both sides are
similarly populated. When `max(upper, lower) / min(upper, lower) >
_CENTERED_BODY_RATIO_MAX` (currently 2.0) the distribution is
effectively one-sided against a hard bound — typical of probability
outputs piled near 0 (`clintox`, `nr_aromatase`, `ames`). In that case
the fit falls back to a **symmetric** body (`max(2·scale, min(upper,
lower))`, the IQR-floored smaller side) AND a **symmetric** `max_extent`
(the larger of the two sides) so both sides share the same derived
`body_target` and the centre has no density step. `fit_notes` records
`body_ratio` and `symmetric_fallback`.

## How a kind gets picked

Classification walks a decision ladder per column. **Constant** wins first
(≤ 1 unique non-NaN value or all-NaN). Then **binary** (exactly 2 unique
non-NaN). Then **count** — non-negative integers, mode-0 vs mode-shifted
chosen by the mode. Everything else is **continuous**, where the
right/centred/left choice requires **both** a Bowley-skew signal **and** a
tail-asymmetry signal to agree before committing to a one-sided branch.
Marginal-Bowley columns with a disagreeing tail stay centred — this avoids
collapsing one half of the data when only one statistic is suggestive.

## The scaler JSON

The on-disk envelope is a flat object plus a per-column dict. Stripped to
two example columns:

```json
{
  "eosframes_version": "0.1.0",
  "method": "robust_typed",
  "model_id": "eos9000",
  "model_version": "v1",
  "fitted_at": "2026-05-12T08:30:00",
  "n_rows": 1000,
  "columns": {
    "logp": {
      "transform": { "kind": "continuous_centered", "center": 2.31, "upper_body_extent": 3.05, "lower_body_extent": 2.80, "upper_max_extent": 4.10, "lower_max_extent": 3.50 },
      "impute_value": 2.31
    },
    "n_aromatic_rings": {
      "transform": { "kind": "count_shifted", "center": 2, "low_anchor": 0, "high_anchor": 5 },
      "impute_value": 2,
      "fit_notes": { "scale": 1.4, "scale_kind": "half_iqr" }
    }
  }
}
```

Every column carries a `transform` (the kind plus its kind-specific
parameters) and an `impute_value` (the fit-time median, rounded to int if
the column was integer at fit). `fit_notes` is optional and **never read at
transform time** — it's there so you can post-hoc inspect routing decisions
(Bowley value, tail asymmetry, degenerate flag, robust-scale source, etc.).
For the exhaustive list of per-kind `transform` keys, see the matching
`_fit_*` function in `scale.py` — the dict that function returns is what
ends up in the JSON.

## Quantization and imputation

**Quantization** (`output_dtype="int8"` or `--quantize`) maps every value
through the trivial formula **`int8 = round(x · 127)`**, clipped to
`[-127, 127]`, with **`-128` as the sentinel** for NaN. Per-column ranges
aren't stored in the JSON — they're derived from the kind at quantize
time via the `_OUTPUT_REGIONS` table in `scale.py`, so they can't drift.
One-sided regions (`binary`, `count_zero_mode`, right- and left-skew) only
inhabit half of int8 by design; that's intentional and lets distance
metrics across mixed-kind columns stay commensurable.

**Imputation** (`impute=True` or `--impute`) replaces every input NaN with
the column's `impute_value` *before* the kind-specific transform runs. The
output has no NaN (and under int8, no `-128` sentinels). Default behaviour
is NaN-pass-through: NaN in → NaN out (or `-128` int8 sentinel out).

Both flags are no-ops at fit time unless `-o` is also passed; the CLI emits
a warning if you forget that.

## What `transform` refuses

`transform_file` refuses to run when:

- **Version mismatch.** The scaler's `eosframes_version` major component
  must equal the running package's. (`0.1.0` and `0.4.2` are compatible;
  `0.x.y` and `1.0.0` are not.) Mismatch → "re-fit the scaler".
- **Method mismatch.** `method` must be `"robust_typed"`.
- **Identity mismatch.** The input file's `model_id` and `model_version`
  must match the scaler's recorded values.
- **Column mismatch.** The input file's feature columns must exactly match
  the scaler's recorded column list (same set, same order). No silent
  ignore of missing columns.
- **Bad output dtype.** Only `"float32"` and `"int8"` are accepted.

All of these surface as `EosframesError` with a message that names the
specific drift.

## Edge cases worth knowing

- **Sparse mode-0 counts** that collapse to a handful of distinct outputs
  get a `degenerate: True` flag in `fit_notes` and a warning at fit. The
  transform behaviour is unchanged — the flag is advisory.
- **All-NaN columns** fall back to `kind: constant` with `impute_value:
  0.0`. Scaling produces 0 for any non-NaN input and NaN for NaN.
- **Binary `transform` with unseen values** snaps each input to the nearer
  of the two fitted values; ties go to the low value (→ 0).
- **Bounded continuous columns** (e.g. uniform integers in `[7, 13]`) have
  their anchors capped at the data extent so the full output range is used
  rather than left unreachable by Tukey over-extension.
- **High-missing columns are not skipped.** Every numeric column is fitted;
  if you want to drop columns by missingness, do it upstream.

## Where to look in the code

All paths are relative to `src/eosframes/scale.py`. Line numbers drift —
function names are stable.

| Concern                       | Function / constant                         |
|-------------------------------|---------------------------------------------|
| Output region per kind        | `_OUTPUT_REGIONS`                           |
| Dispatch table (apply)        | `_APPLY_DISPATCH`                           |
| Kind classification           | `_classify_type`                            |
| Impute value (median + dtype) | `_compute_impute_value`                     |
| Robust scale cascade          | `_compute_robust_scale`                     |
| Bowley skew                   | `_compute_bowley`                           |
| Fit — constant                | `_fit_constant`                             |
| Fit — binary                  | `_fit_binary`                               |
| Fit — count (both flavours)   | `_fit_count`                                |
| Fit — continuous centered     | `_fit_continuous_centered`                  |
| Fit — continuous right/left   | `_fit_continuous`                           |
| Apply — per kind              | `_apply_constant` … `_apply_continuous_*`   |
| Float → int8                  | `_quantize_to_int8`                         |
| Top-level fit                 | `fit` (DataFrame), `fit_file` (path)        |
| Top-level transform           | `transform`, `transform_file`               |

For threshold constants (Bowley cutoff, tail-asymmetry ratio, degenerate
distinct-count and mode-fraction limits, the int8 NaN sentinel) see the
named module-level constants near the top of `scale.py`.
