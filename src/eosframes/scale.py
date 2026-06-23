"""Type-aware robust scaler for Ersilia model outputs.

Each numeric feature column is auto-classified at fit time as one of
``constant``, ``binary``, ``count``, or ``continuous`` (with a further
``right`` / ``left`` / ``centered`` sub-case for continuous and count
columns that have a non-zero mode), and a type-specific transform is
recorded. Output values land in a per-type region whose maximum spread
is 1, so columns are commensurable for distance calculations:

* ``constant``                       → output 0.
* ``binary``                         → ``{0, 1}`` (lower → 0, higher → 1).
* ``count`` (mode 0)                 → ``[0, 1]``, with 0 pinned at 0.
  Sparse mode-0 columns whose output collapses to a few distinct
  values get a ``degenerate: True`` advisory flag (warn-only).
* ``count`` (mode ≠ 0)               → per-side linear body + ``tanh``
  asymptote, same shape as ``continuous_centered``. Anchors:
  ``upper = min(max(Q3 + 1.5·IQR, p99), arr.max())``,
  ``lower = max(min(Q1 − 1.5·IQR, p1), arr.min())``. The
  ``max(Tukey, p99) / min(Tukey, p1)`` part keeps the original
  outlier-robust anchor (whichever side is wider), and the
  ``arr.max() / arr.min()`` cap prevents over-extension past the
  data range. Mode → 0; the linear body maps ``[mode, high_anchor]``
  → ``[0, body_target]`` (and the mirror on the lower side) with
  ``body_target = _CENTERED_BODY_TARGET = 0.5`` — so the bulk of
  count data sits inside ``[-0.5, 0.5]`` and counts past the anchors
  asymptote toward ``±1`` via the slope-continuous ``tanh``. **Soft
  cap on per-side extent ratio**: when
  ``max(upper_span, lower_span) / min(upper_span, lower_span) >
  _COUNT_SHIFTED_EXTENT_RATIO_MAX`` (currently ``2.0``), the wider
  side's extent is capped at ``2 × the narrower`` and the affected
  anchor is recomputed — so heavily right- or left-tailed count
  distributions don't squash their distinct discrete values into the
  body. The excess raw range flows into the ``tanh`` tail past the
  anchor, where each distinct count gets a distinct output in
  ``(±0.5, ±1)``. ``fit_notes`` records ``extent_ratio`` and
  ``extent_capped``. Distinct outliers always get distinct outputs
  (no flat clip plateau); distinct integer counts inside the body
  keep distinct outputs too (no compression there).
* ``continuous`` right-skewed        → triggered when **both**
  ``bowley > 0.3`` *and* the tail-asymmetry ratio
  ``(p99 − median) / (median − p1) > 3``. Bowley alone is a bulk-only
  asymmetry measure and trips on borderline columns that have real
  spread on both sides. Requiring the tails to agree with the
  bulk keeps those columns in the centered branch where their
  bell-ish shape is preserved. The fit itself is a linear body
  ``[arr.min, body_anchor] → [0, body_target]`` plus a smooth tail:

  * **Heavy tail** (``p99 > Tukey``): 3-segment piecewise (``tail =
    "piecewise"``). Linear body ``[arr.min, Tukey] → [0, body_target
    = 0.8]`` keeps the bulk visible (Q3 lands at ``≈ 0.35``); a
    slower linear "middle" segment ``[Tukey, p99] → [body_target,
    mid_target]`` spreads the upper-decile outliers; a finite
    ``_quadratic_tail`` reaches ``(arr.max, 1)`` with zero slope at
    ``arr.max``. ``mid_target`` is derived for ``C¹`` smoothness at
    the middle→tail join:
    ``mid_target = (2·a + b·body_target) / (2·a + b)``
    with ``a = p99 − Tukey`` and ``b = arr.max − p99``. The
    histogram shows the bulk in ``[0, body_target]`` and a separate
    smaller "tail bump" in ``[body_target, 1]`` — no spike at ``+1``.
  * **Light tail** (``p99 ≤ Tukey``): finite quadratic from
    ``(body_anchor, body_target)`` to ``(arr.max, 1)`` with zero slope
    at ``arr.max`` and slope continuity at the body boundary.
    ``body_anchor`` is capped at ``arr.max`` so bounded distributions
    fill the full output range, and
    ``body_target`` is derived from the geometry so the quadratic
    reaches exactly ``1`` at ``arr.max``. Only OOD inputs past
    ``arr.max`` clip at ``1``.
* ``continuous`` left-skewed         → mirror of right.
* ``continuous`` centered             → per-side linear body +
  finite-reach quadratic tail in ``[-1, 1]``. The per-side body
  extents and max extents adapt independently:

  ``upper_body_extent = max(scale, min(p99 − center, tukey_upper − center))``
  ``lower_body_extent = max(scale, min(center − p1,  center − tukey_lower))``
  ``upper_max_extent  = max(upper_body_extent, arr.max − center)``
  ``lower_max_extent  = max(lower_body_extent, center − arr.min)``

  Tukey caps ``p99`` / ``p1`` to stay outlier-robust;
  ``scale = half-IQR`` floors the body so tight clean data still has a
  usable body. ``body_target`` is derived per side at apply time by
  :func:`_light_tail_body_target` and then **capped at**
  :data:`_CENTERED_BODY_TARGET` (``0.5``), so the bulk of the density
  visually lives in ``[-0.5, 0.5]`` to match ``count_shifted`` and stay
  commensurable with the one-sided skewed kinds. **Dually**, the
  effective ``max_extent`` used for the tail is floored at
  :data:`_CENTERED_MAX_EXTENT_RATIO` × ``body_extent`` (``3·b``); the
  two constants are locked together because the body→tail join is
  ``C¹``-smooth precisely when ``max_extent = 3·body_extent`` given
  ``body_target = 0.5`` (body slope ``0.5/b`` equals quadratic-tail
  slope ``1/(a−b)``). The cap activates when ``max_extent <
  3·body_extent`` (typical Gaussian-ish or bimodal data with little
  tail past the body); past that threshold the natural derivation
  produces ``body_target < 0.5`` already and both the cap and the
  floor are no-ops. Past the body, :func:`_quadratic_tail` reaches
  ``±1`` exactly at ``±effective_max`` (= ``±3·body_extent`` when
  the floor is active, otherwise the actual data edge on that side);
  inputs past that clip to ``±1``. Distinct outlier values spread
  monotonically across ``[body_target, 1]`` — no visual pile-up at
  ``±1`` and no slope kink at the body→tail join.

  **Safety gate.** The per-side design is only safe when both sides
  are similarly populated. When ``max(upper, lower) / min(upper,
  lower) > _CENTERED_BODY_RATIO_MAX`` (currently 2.0) the
  distribution is effectively one-sided against a hard bound (a
  probability output piled at 0, say). The tight side would
  otherwise get a steep body slope that distorts the histogram. The
  fit collapses to a **symmetric** body — both sides equal to
  ``max(2·scale, min(upper, lower))``, the IQR-floored smaller side
  — AND a **symmetric** ``max_extent`` (the larger of the two sides)
  so both sides share the same derived ``body_target`` and the
  centre has no density step. ``fit_notes`` records ``body_ratio``
  and ``symmetric_fallback`` so the gate can be audited after the
  fact.

Output dtype is a **transform-time choice**, not a fit-time one. The
scaler JSON contains only dtype-agnostic parameters; pass
``output_dtype`` to :func:`transform` / :func:`transform_file` (or use
``--quantize`` on the CLI) to select between ``float32`` (default,
NaN-preserving) and ``int8``. The int8 mapping is per-column: every
column's documented output region maps linearly to ``[-127, 127]`` so
asymmetric regions (binary, count, right, left) use the full int8
range, not just half of it. Sentinel ``-128`` is reserved for NaN.
:func:`fit_file` accepts ``output_dtype`` only as a convenience for
the inline fit-then-transform path it offers via its ``output_path``
argument.

Every numeric feature column is fitted — there is no missing-value
skip threshold. Columns that have zero non-NaN values fit as
``type: constant, value: 0.0`` (every row transforms to 0 for non-NaN
inputs and NaN for NaN inputs).

Each per-column params dict also carries an ``impute_value`` recorded
at fit time: the median of the column's non-NaN training values,
rounded to ``int`` if the column was integer-valued and kept as
``float`` otherwise (all-NaN columns get ``0.0``). The transform
ignores it by default — NaN propagates through to the output, just as
before. Passing ``impute=True`` (or ``--impute`` on the CLI) replaces
every input NaN with the recorded ``impute_value`` before dispatch,
so the output column has no NaN (and, under ``--quantize``, no
``-128`` sentinels).

Schema
------

A fitted scaler is written as a flat JSON envelope plus a ``columns``
dict keyed by feature name. Each column entry has at most three keys:

* ``transform`` — the transform-time payload. Always carries a
  ``kind`` string discriminator; the rest of the keys depend on the
  kind (see ``_OUTPUT_REGIONS`` for the seven kinds and their
  per-kind documented output region).
* ``impute_value`` — the median-with-dtype-rounding fill for
  ``transform(..., impute=True)``. Peer of ``transform`` so the
  imputation logic stays separable from the dispatch math.
* ``fit_notes`` — *optional*. Diagnostics never read at transform
  time (e.g. ``bowley`` for continuous, ``degenerate`` /
  ``mode_fraction`` for sparse zero-mode counts, ``scale`` /
  ``scale_kind`` for centered continuous). Omitted entirely when
  empty.

Output region is **derived** at quantize time from the ``kind``, not
stored — drift-free.

The envelope also records ``eosframes_version`` (the full running
``eosframes.__version__`` that produced the JSON), plus ``method``,
``model_id``, ``model_version``, ``fitted_at``, and ``n_rows``.
:func:`transform_file` rejects any scaler whose recorded
``eosframes_version`` doesn't share the **major** component with the
running version (so the whole ``0.x.y`` line is mutually compatible
and only a bump to ``1.0.0`` invalidates older scalers). There is no
hand-maintained schema number.
"""

import json
import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from . import __version__ as _PACKAGE_VERSION
from .exceptions import EosframesError
from .logger import get_logger
from .naming import (
    is_valid_name,
    is_valid_transformer_name,
    parse_name,
    parse_transformer_name,
)

_META_COLS = {"key", "input"}

_METHOD_NAME = "robust_typed"

_VALID_OUTPUT_DTYPES = ("float32", "int8")
_DEFAULT_OUTPUT_DTYPE = "float32"

_INT8_NAN_SENTINEL = -128
_INT8_MAX_VAL = 127

_BOWLEY_THRESHOLD = 0.3

# A column counts as right- or left-skewed only when *both* the bulk
# (Bowley) and the tails (p1, p99 relative to median) agree on the
# direction. Bowley alone is a bulk-only asymmetry measure and trips
# on borderline columns whose tails happen to have comparable extent
# on both sides; requiring the tail-asymmetry ratio to exceed this
# threshold sends those into the centered branch where their
# bell-ish shape is preserved.
_TAIL_ASYMMETRY_THRESHOLD = 3.0
_TUKEY_THRESHOLD = 3.0
_LINEAR_CLIP_DIVISOR_MULTIPLIER = 2.0

# F4: a mode-0 count is flagged degenerate when it is mostly the mode and
# its scaled output collapses to a handful of distinct values. The flag is
# advisory — transform behavior is unchanged.
_DEGENERATE_DISTINCT_MAX = 4
_DEGENERATE_MODE_FRACTION_MIN = 0.4

# Mode-0 count columns anchor at the larger of the Tukey whisker
# (`q3 + 1.5·IQR`) and the 99th percentile of the data. On heavy-zero
# distributions Tukey sits inside the realistic tail (Q3 and IQR collapse
# toward 0), so the p99 lifts the anchor; on denser counts Tukey is the
# more generous of the two and it wins. Both are robust to a single
# rogue outlier (unlike `arr.max()`).
_COUNT_HIGH_PERCENTILE = 0.99

# Continuous fits use a linear body + smooth tail mapping. The body is
# a single linear segment that ends at ``±body_target``; past the body,
# the tail is either an asymptotic tanh (heavy tail — data extends past
# the body anchor) or a finite quadratic ending at ``±1`` with zero
# slope (light tail — data has a real ceiling near the body anchor).
# Both tails preserve slope continuity at the body boundary, so the
# overall transform has no kinks and distinct inputs in the tail get
# distinct outputs — no collisions at a clip plateau.
_CONTINUOUS_BODY_TARGET = 0.8

# Two-sided body target. ``continuous_centered`` and ``count_shifted``
# both produce output in ``[-1, 1]`` symmetrically around 0, so the
# natural visual split is body 50% / each tail 25%. A reader sees the
# bulk inside ``[-0.5, 0.5]`` and anything past ``±0.5`` reads
# unambiguously as tail / outlier. ``count_shifted`` uses this as a
# **hard target** (every fit lands the body at ``±0.5``); for
# ``continuous_centered`` it is a **cap** on the per-side geometry-
# derived target, so heavy-tailed distributions can still pick a
# smaller body_target while light-tailed and bimodal fits collapse to
# the shared 50%-body convention rather than sprawling across
# ``[-1, 1]``. The ``tanh`` (count_shifted) or quadratic (centered)
# tail past the body stretches outliers across the remaining half, so
# int8-quantized outliers get plenty of distinct levels and centered
# columns stay commensurable with the one-sided skewed kinds for
# distance metrics.
_CENTERED_BODY_TARGET = 0.5

# Minimum ``max_extent / body_extent`` ratio for ``continuous_centered``.
# Paired with :data:`_CENTERED_BODY_TARGET` = 0.5: the body→tail join
# is ``C¹``-smooth precisely when ``max_extent = 3·body_extent`` (body
# slope ``0.5/b`` equals quadratic-tail slope ``1/(a−b)``). When the
# data provides a smaller ratio — bounded distributions like a U-shape
# beta, truncated normal, narrow bimodal, or any column where the
# Tukey reach already touches the data edge — the effective
# ``max_extent`` is floored at ``3·body_extent`` at apply time so the
# quadratic tail has enough raw domain to be gentle. Without the
# floor, the few real points sitting in the tiny ``[body_extent,
# max_extent]`` raw band would get sprayed across ``(0.5, 1.0]``;
# with it, they collapse to a thin pile-up just past ``±0.5`` and OOD
# inputs still reach ``±1`` only past ``±3·body_extent`` from the
# centre. The constant is **mathematically locked** to the cap value:
# ``_CENTERED_MAX_EXTENT_RATIO = 2/_CENTERED_BODY_TARGET − 1``. If
# the cap ever moves, this constant must move with it.
_CENTERED_MAX_EXTENT_RATIO = 3.0

# Inside ``continuous_centered``: when the per-side body extents differ
# by more than this factor, the fit falls back to a symmetric body
# (both sides equal to the smaller extent). Per-side asymmetry past
# this point comes from a one-sided bounded distribution (a probability
# output piled near 0, for instance), where the tight side ends up with
# a steep, distorting body slope at the centre. The cap of 2.0 keeps
# genuinely asymmetric bells per-side while folding pile-up columns
# back to a symmetric fit.
_CENTERED_BODY_RATIO_MAX = 2.0

# Max allowed ratio between the per-side body extents in
# ``count_shifted`` (mode-nonzero count fit). When one side's extent
# exceeds ``_COUNT_SHIFTED_EXTENT_RATIO_MAX × the other``, the wider
# side is **capped** at that ratio (a soft cap — the lower side stays
# unchanged in the typical right-tailed case). The excess raw range
# on the wider side then flows into the existing slope-continuous
# ``tanh`` tail past the anchor, so distinct outlier counts get
# distinct outputs in ``(0.5, ~1)`` instead of being squashed into
# the body's ``[0, 0.5]``. The value mirrors
# :data:`_CENTERED_BODY_RATIO_MAX` for design parity between the two
# centered-style kinds; this is a softer enforcement than centered's
# symmetric-collapse fallback because ``count_shifted`` already has
# an asymptotic tail to absorb the excess.
_COUNT_SHIFTED_EXTENT_RATIO_MAX = 2.0

# Half-width of the Hermite smoothstep blend window at the body→middle
# junction of the 3-segment piecewise right/left-skew fit, as a fraction
# of ``min(body_span, mid_span)``. The slope on either side of the
# junction differs by ~20× for heavy-tailed columns; without smoothing,
# the density jump at the junction shows up as a visible "second hill"
# in the scaled histogram. The blend window covers half of each
# neighbouring segment by default — wide enough to spread the slope
# change across most of the segment-2 row mass, so the visible bump
# softens into a continuous shoulder.
_PIECEWISE_BLEND_FRACTION = 0.5


def _tanh_tail(mag: np.ndarray, body_extent: float, body_target: float) -> np.ndarray:
    """Asymptotic tail for ``|x - center| > body_extent``.

    Returns ``y = body_target + (1 − body_target)·tanh(c·u)`` for
    ``u = mag − body_extent ≥ 0``, where
    ``c = body_target / (body_extent · (1 − body_target))`` makes the
    tail slope at ``u = 0`` exactly equal to the body slope
    ``body_target / body_extent`` — so the body→tail join is
    ``C¹``-smooth, no kink, no smoothstep blend needed. As ``u → ∞``,
    ``y → 1`` asymptotically; distinct inputs always get distinct
    outputs.
    """
    if body_target >= 1.0 or body_extent <= 0:
        return np.full_like(mag, body_target, dtype=float)
    c = body_target / (body_extent * (1.0 - body_target))
    return body_target + (1.0 - body_target) * np.tanh(c * (mag - body_extent))


def _quadratic_tail(
    mag: np.ndarray, body_extent: float, body_target: float, max_extent: float
) -> np.ndarray:
    """Finite-reach tail for ``|x - center|`` in ``[body_extent, max_extent]``.

    Reaches ``y = 1`` exactly at ``mag = max_extent`` with zero slope, and
    matches ``(body_extent, body_target)`` with body slope at the body
    boundary. The slope-continuity + zero-slope-at-end constraints fix
    a unique quadratic; the caller is responsible for picking
    ``body_target`` so the quadratic is monotone (in practice
    ``body_target = 2·body_extent / (body_extent + max_extent)``).

    Values past ``max_extent`` clip to ``1``.
    """
    if body_extent <= 0 or max_extent <= body_extent:
        return np.full_like(mag, 1.0, dtype=float)
    alpha = (1.0 - body_target) / (max_extent - body_extent) ** 2
    y = 1.0 - alpha * (max_extent - mag) ** 2
    return np.clip(y, body_target, 1.0)


def _light_tail_body_target(body_extent: float, max_extent: float) -> float:
    """Derive ``body_target`` for a finite (quadratic) tail.

    Picked so the body→tail join is ``C¹``-smooth: the linear body
    slope ``body_target / body_extent`` equals the quadratic's slope at
    its left edge. Solving gives ``body_target = 2·b / (b + a)`` where
    ``b = body_extent`` and ``a = max_extent``. ``b = a`` (no tail)
    yields ``body_target = 1`` and the quadratic degenerates to a
    constant ``1`` past the body — same as a hard clip in that
    degenerate case.
    """
    if max_extent <= 0:
        return 1.0
    return 2.0 * body_extent / (body_extent + max_extent)


# Per-kind documented output region. Read by :func:`_quantize_to_int8`
# to map each column's region linearly into the int8 range
# ``[-127, 127]`` (sentinel ``-128`` reserved for NaN). Single source of
# truth — never stored in the scaler JSON, derived at runtime from
# ``entry["transform"]["kind"]``.
_OUTPUT_REGIONS: Dict[str, Tuple[float, float]] = {
    "constant": (0.0, 0.0),
    "binary": (0.0, 1.0),
    "count_zero_mode": (0.0, 1.0),
    "count_shifted": (-1.0, 1.0),
    "continuous_right_skew": (0.0, 1.0),
    "continuous_left_skew": (-1.0, 0.0),
    "continuous_centered": (-1.0, 1.0),
}


def _major(v: str) -> str:
    """Return the major component of a PEP 440-style version string.

    ``"0.1.0"`` → ``"0"``, ``"1.2.3a1"`` → ``"1"``, ``"unknown"`` →
    ``"unknown"`` (anything without a ``.`` is returned verbatim).
    Used by :func:`transform_file` so scalers fitted by any
    ``eosframes`` release in the same major line stay valid.
    """
    return v.split(".", 1)[0] if v else v


# ---------------------------------------------------------------------------
# Type classification and per-type fit
# ---------------------------------------------------------------------------


def _is_integer_valued(series: pd.Series) -> bool:
    arr = series.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)))


def _mode_value(series: pd.Series) -> float:
    """Most-common non-NaN value, ties broken by the lowest value."""
    counts = series.dropna().value_counts(sort=False)
    if counts.empty:
        raise EosframesError("Cannot compute mode of an all-NaN column.")
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    return float(min(candidates))


def _classify_type(series: pd.Series) -> str:
    non_nan = series.dropna()
    n_unique = non_nan.nunique()
    if n_unique <= 1:
        return "constant"
    if n_unique == 2:
        return "binary"
    if _is_integer_valued(series) and float(non_nan.min()) >= 0.0:
        return "count"
    return "continuous"


def _compute_impute_value(series: pd.Series):
    """Median of *series* with dtype-respecting rounding.

    Integer-valued training columns return an ``int`` (median rounded
    to the nearest integer) so a later ``impute=True`` transform never
    writes fractional values into an originally-int column. Float
    columns return a ``float``. All-NaN columns return ``0.0``.
    """
    non_nan = series.dropna()
    if non_nan.empty:
        return 0.0
    median = float(non_nan.median())
    if _is_integer_valued(series):
        return int(round(median))
    return median


def _compute_robust_scale(
    series: pd.Series,
) -> Tuple[float, float, str]:
    """Median + scale via half-IQR / MAD / range cascade.

    Returns ``(center, scale, scale_kind)``.  ``scale_kind`` is one of
    ``"half_iqr"``, ``"mad"``, ``"range"``.  Raises ``EosframesError`` if
    the column is effectively constant (all three scales are 0).
    """
    arr = series.dropna().to_numpy(dtype=float)
    median = float(np.median(arr))
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    half_iqr = float((q3 - q1) / 2.0)
    if half_iqr > 0:
        return median, half_iqr, "half_iqr"
    mad = float(np.median(np.abs(arr - median)))
    mad_scale = mad * 1.4826
    if mad_scale > 0:
        return median, mad_scale, "mad"
    half_range = float((arr.max() - arr.min()) / 2.0)
    if half_range > 0:
        return median, half_range, "range"
    raise EosframesError("Column is effectively constant; route to constant branch.")


def _compute_bowley(series: pd.Series, scale_kind: str) -> float:
    """Bowley skewness ``((Q3 + Q1) - 2·median) / IQR``.

    Returns 0.0 when the half-IQR cascade did not apply (skew is
    ill-defined for those degenerate cases).
    """
    if scale_kind != "half_iqr":
        return 0.0
    arr = series.dropna().to_numpy(dtype=float)
    q1, q2, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    return float(((q3 + q1) - 2.0 * q2) / iqr)


def _fit_constant(series: pd.Series) -> dict:
    # ``constant`` columns always emit 0 for non-NaN at transform time;
    # the original training value lives in ``impute_value`` if anyone
    # needs to recover it. The transform payload itself is empty.
    return {
        "transform": {"kind": "constant"},
        "impute_value": _compute_impute_value(series),
    }


def _fit_binary(series: pd.Series) -> dict:
    uniques = sorted(float(v) for v in series.dropna().unique())
    return {
        "transform": {"kind": "binary", "low": uniques[0], "high": uniques[1]},
        "impute_value": _compute_impute_value(series),
    }


def _fit_continuous_centered(series: pd.Series) -> dict:
    """Fit a centered continuous column with per-side body + quadratic tail.

    Linear body to ``±body_target`` on each side, then a finite-reach
    ``_quadratic_tail`` to ``±1`` at each side's actual data edge — so
    outliers spread monotonically across ``[body_target, 1]`` instead
    of saturating asymptotically and visually piling at ``±1``.

    * Per-side body extent: ``upper_body_extent =
      max(scale, min(p99 − center, tukey_upper − center))`` and the
      mirror for the lower side. Tukey caps p99 / p1 when outliers
      infiltrate them; ``scale = half-IQR`` floors the body so tight
      clean data still has a usable body.
    * Per-side ``max_extent`` is the actual data extreme on that side
      (``arr.max − center`` for upper, ``center − arr.min`` for lower),
      floored at ``body_extent`` so the quadratic tail is always
      well-defined.
    * ``body_target`` is derived per side at apply time by
      :func:`_light_tail_body_target` so the body→tail join is
      ``C¹``-smooth. It's not stored — derivable from
      ``(body_extent, max_extent)`` alone.
    * **Safety gate.** When the body-extent ratio ``max / min``
      exceeds ``_CENTERED_BODY_RATIO_MAX``, the distribution is
      effectively one-sided against a hard bound (e.g. ``clintox``).
      The fit collapses to a **symmetric** body (smaller side, floored
      at the IQR) AND a **symmetric** ``max_extent`` (max of both
      sides), so both sides share the same derived ``body_target`` and
      there's no density step at the centre.
    """
    center, scale, scale_kind = _compute_robust_scale(series)
    arr = series.dropna().to_numpy(dtype=float)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q3 - q1)
    tukey_upper = float(q3 + 1.5 * iqr)
    tukey_lower = float(q1 - 1.5 * iqr)
    p_upper = float(np.quantile(arr, _COUNT_HIGH_PERCENTILE))
    p_lower = float(np.quantile(arr, 1.0 - _COUNT_HIGH_PERCENTILE))

    upper_body_extent = max(scale, min(p_upper - center, tukey_upper - center))
    lower_body_extent = max(scale, min(center - p_lower, center - tukey_lower))

    upper_max_extent = max(upper_body_extent, float(arr.max()) - center)
    lower_max_extent = max(lower_body_extent, center - float(arr.min()))

    denom = max(min(upper_body_extent, lower_body_extent), 1e-12)
    body_ratio = max(upper_body_extent, lower_body_extent) / denom
    symmetric_fallback = body_ratio > _CENTERED_BODY_RATIO_MAX
    if symmetric_fallback:
        # Symmetric body (smaller side, IQR-floored) and symmetric
        # max_extent (larger side's reach) — same derived body_target
        # on both sides, no centre density step.
        sym_body = max(2.0 * scale, min(upper_body_extent, lower_body_extent))
        sym_max = max(upper_max_extent, lower_max_extent, sym_body)
        upper_body_extent = lower_body_extent = sym_body
        upper_max_extent = lower_max_extent = sym_max

    return {
        "transform": {
            "kind": "continuous_centered",
            "center": float(center),
            "upper_body_extent": float(upper_body_extent),
            "lower_body_extent": float(lower_body_extent),
            "upper_max_extent": float(upper_max_extent),
            "lower_max_extent": float(lower_max_extent),
        },
        "impute_value": _compute_impute_value(series),
        "fit_notes": {
            "scale": float(scale),
            "scale_kind": scale_kind,
            "body_ratio": float(body_ratio),
            "symmetric_fallback": bool(symmetric_fallback),
        },
    }


def _fit_continuous(series: pd.Series) -> dict:
    center, scale, scale_kind = _compute_robust_scale(series)
    bowley = _compute_bowley(series, scale_kind)
    arr = series.dropna().to_numpy(dtype=float)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q3 - q1)
    # Tail asymmetry: right side vs left side relative to the median.
    # Both Bowley (bulk) and tail asymmetry must agree on the direction
    # to commit to the right/left-skew branch — otherwise centered.
    median_val = float(np.quantile(arr, 0.5))
    p1_val = float(np.quantile(arr, 1.0 - _COUNT_HIGH_PERCENTILE))
    p99_val = float(np.quantile(arr, _COUNT_HIGH_PERCENTILE))
    _eps = 1e-12
    right_span = max(p99_val - median_val, 0.0)
    left_span = max(median_val - p1_val, 0.0)
    tail_asymmetry_right = right_span / max(left_span, _eps)
    tail_asymmetry_left = left_span / max(right_span, _eps)
    is_right_skew = (
        bowley > _BOWLEY_THRESHOLD and tail_asymmetry_right > _TAIL_ASYMMETRY_THRESHOLD
    )
    is_left_skew = (
        bowley < -_BOWLEY_THRESHOLD and tail_asymmetry_left > _TAIL_ASYMMETRY_THRESHOLD
    )

    if is_right_skew:
        tukey_upper = float(q3 + 1.5 * iqr)
        p99_upper = float(np.quantile(arr, _COUNT_HIGH_PERCENTILE))
        low = float(arr.min())
        arr_max = float(arr.max())
        if p99_upper > tukey_upper:
            # Heavy tail: 3-segment piecewise — linear body to
            # (body_anchor=Tukey, body_target=0.8), linear "middle" to
            # (mid_anchor=p99, mid_target), then quadratic tail to
            # (high_anchor=arr.max, 1). mid_target is derived from a
            # C¹-smoothness constraint at the middle→tail join so the
            # quadratic stays monotone. The bulk lives in segment 1
            # (visible), the outliers spread linearly across segments
            # 2-3 instead of saturating at +1.
            body_anchor = float(tukey_upper)
            mid_anchor = float(p99_upper)
            high_anchor = float(arr_max)
            body_target = float(_CONTINUOUS_BODY_TARGET)
            a = mid_anchor - body_anchor
            b = high_anchor - mid_anchor
            if b <= 0.0:
                mid_target = 1.0
            else:
                mid_target = (2.0 * a + b * body_target) / (2.0 * a + b)
            transform = {
                "kind": "continuous_right_skew",
                "tail": "piecewise",
                "low_anchor": low,
                "body_anchor": body_anchor,
                "mid_anchor": mid_anchor,
                "high_anchor": high_anchor,
                "body_target": body_target,
                "mid_target": float(mid_target),
            }
        else:
            # Light tail: finite quadratic from (body_anchor, body_target)
            # to (arr.max, 1) with zero slope at the right edge. The
            # body anchor sits at min(Tukey, arr.max) so bounded
            # distributions don't end up squished. body_target is
            # derived from the geometry to keep slope continuity at the
            # body anchor.
            body_anchor = min(tukey_upper, arr_max)
            body_extent = body_anchor - low
            max_extent = arr_max - low
            body_target = _light_tail_body_target(body_extent, max_extent)
            transform = {
                "kind": "continuous_right_skew",
                "tail": "finite",
                "low_anchor": low,
                "body_anchor": float(body_anchor),
                "high_anchor": float(arr_max),
                "body_target": float(body_target),
            }
        return {
            "transform": transform,
            "impute_value": _compute_impute_value(series),
            "fit_notes": {
                "bowley": float(bowley),
                "tail_asymmetry": float(tail_asymmetry_right),
            },
        }

    if is_left_skew:
        tukey_lower = float(q1 - 1.5 * iqr)
        p1_lower = float(np.quantile(arr, 1.0 - _COUNT_HIGH_PERCENTILE))
        high = float(arr.max())
        arr_min = float(arr.min())
        if p1_lower < tukey_lower:
            # Heavy tail: 3-segment piecewise (mirror of right). Magnitude
            # runs from high (mag=0) down through body_anchor (Tukey),
            # mid_anchor (p1), to low_anchor (arr.min, mag=1 on the
            # negated axis). See the right-skew block for the
            # mid_target C¹-smoothness derivation.
            body_anchor = float(tukey_lower)
            mid_anchor = float(p1_lower)
            low_anchor = float(arr_min)
            body_target = float(_CONTINUOUS_BODY_TARGET)
            a = body_anchor - mid_anchor
            b = mid_anchor - low_anchor
            if b <= 0.0:
                mid_target = 1.0
            else:
                mid_target = (2.0 * a + b * body_target) / (2.0 * a + b)
            transform = {
                "kind": "continuous_left_skew",
                "tail": "piecewise",
                "low_anchor": low_anchor,
                "body_anchor": body_anchor,
                "mid_anchor": mid_anchor,
                "high_anchor": high,
                "body_target": body_target,
                "mid_target": float(mid_target),
            }
        else:
            # Light tail (mirror of right-skewed light).
            body_anchor = max(tukey_lower, arr_min)
            body_extent = high - body_anchor
            max_extent = high - arr_min
            body_target = _light_tail_body_target(body_extent, max_extent)
            transform = {
                "kind": "continuous_left_skew",
                "tail": "finite",
                "low_anchor": float(arr_min),
                "body_anchor": float(body_anchor),
                "high_anchor": high,
                "body_target": float(body_target),
            }
        return {
            "transform": transform,
            "impute_value": _compute_impute_value(series),
            "fit_notes": {
                "bowley": float(bowley),
                "tail_asymmetry": float(tail_asymmetry_left),
            },
        }

    # Centered: delegate to _fit_continuous_centered (already in new
    # shape) and just enrich its fit_notes with the bowley reading that
    # routed us here.
    entry = _fit_continuous_centered(series)
    entry["fit_notes"]["bowley"] = float(bowley)
    # Tail-asymmetry is recorded in the direction the bulk-Bowley
    # already points so it's easy to compare against the threshold
    # when wondering "why did this column not go to right/left?".
    if bowley >= 0:
        entry["fit_notes"]["tail_asymmetry"] = float(tail_asymmetry_right)
    else:
        entry["fit_notes"]["tail_asymmetry"] = float(tail_asymmetry_left)
    return entry


def _fit_count(series: pd.Series) -> dict:
    mode = _mode_value(series)
    if mode == 0.0:
        arr = series.dropna().to_numpy(dtype=float)
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        tukey_high = float(q3 + 1.5 * (q3 - q1))
        p99_high = float(np.quantile(arr, _COUNT_HIGH_PERCENTILE))
        high_anchor = max(tukey_high, p99_high)
        if high_anchor <= 0.0:
            high_anchor = float(arr.max())
        if high_anchor <= 0.0:
            return _fit_constant(series)

        entry = {
            "transform": {"kind": "count_zero_mode", "high_anchor": float(high_anchor)},
            "impute_value": _compute_impute_value(series),
        }

        # Flag near-degenerate sparse counts where most rows are 0 and
        # the scaler can only produce a handful of distinct values. Goes
        # into fit_notes — advisory only, never read by the transform.
        # No per-column log here: sparse fingerprints (e.g. Morgan counts)
        # flag hundreds of columns and would flood the log. fit() emits a
        # single aggregated summary instead.
        scaled = np.clip(arr / high_anchor, 0.0, 1.0)
        n_distinct = int(np.unique(scaled).size)
        mode_fraction = float((arr == 0.0).sum()) / float(arr.size)
        if (
            n_distinct <= _DEGENERATE_DISTINCT_MAX
            and mode_fraction >= _DEGENERATE_MODE_FRACTION_MIN
        ):
            entry["fit_notes"] = {
                "degenerate": True,
                "mode_fraction": mode_fraction,
                "n_distinct_output": n_distinct,
            }
        return entry

    # Count with a non-zero mode. Linear+clip on each side of the mode,
    # with independent upper and lower anchors. No tanh — distinct
    # integer counts in the body and moderate tail keep distinct
    # outputs, which is what users expect from count data.
    #
    # Anchor rule: take the wider of (Tukey whisker, p99) on each side
    # to keep the original outlier-robust property, then cap at the
    # observed data extent so the anchor never reaches past where real
    # data lives — fixes bounded distributions (uniform integers) whose
    # Tukey extends 1.5·IQR past Q3 even though no real data does. The
    # cap doesn't bind for heavy-tail data (where arr.max ≥ Tukey/p99),
    # so outliers past the anchor still clip cleanly.
    arr = series.dropna().to_numpy(dtype=float)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = float(q3 - q1)
    tukey_upper = float(q3 + 1.5 * iqr)
    tukey_lower = float(q1 - 1.5 * iqr)
    p99_upper = float(np.quantile(arr, _COUNT_HIGH_PERCENTILE))
    p1_lower = float(np.quantile(arr, 1.0 - _COUNT_HIGH_PERCENTILE))
    arr_max = float(arr.max())
    arr_min = float(arr.min())
    high_anchor = min(max(tukey_upper, p99_upper), arr_max)
    low_anchor = max(min(tukey_lower, p1_lower), arr_min)
    upper_span = max(high_anchor - mode, 0.0)
    lower_span = max(mode - low_anchor, 0.0)
    if upper_span == 0.0 and lower_span == 0.0:
        # Mode is pressed against both edges of the data; nothing to scale.
        return _fit_constant(series)

    # Soft cap on per-side extent ratio. Distance metrics expect
    # commensurable spread per side; when one side's raw extent is far
    # wider than the other (right- or left-tailed count), without the
    # cap the wider side's distinct discrete values get squashed into
    # the narrow scaled body. Capping the wider extent at
    # ``_COUNT_SHIFTED_EXTENT_RATIO_MAX × the narrower`` sends the
    # excess raw range into the slope-continuous ``tanh`` tail past
    # the anchor, where distinct counts spread distinctly in
    # ``(±0.5, ±1)``. The narrower side is unchanged.
    extent_capped = False
    extent_ratio: Optional[float] = None
    if upper_span > 0.0 and lower_span > 0.0:
        extent_ratio = max(upper_span, lower_span) / min(upper_span, lower_span)
        if extent_ratio > _COUNT_SHIFTED_EXTENT_RATIO_MAX:
            cap = _COUNT_SHIFTED_EXTENT_RATIO_MAX * min(upper_span, lower_span)
            if upper_span > cap:
                upper_span = cap
                high_anchor = mode + upper_span
            if lower_span > cap:
                lower_span = cap
                low_anchor = mode - lower_span
            extent_capped = True

    entry: dict = {
        "transform": {
            "kind": "count_shifted",
            "center": float(mode),
            "low_anchor": float(low_anchor),
            "high_anchor": float(high_anchor),
            "body_target": float(_CENTERED_BODY_TARGET),
        },
        "impute_value": _compute_impute_value(series),
    }
    if extent_ratio is not None:
        entry["fit_notes"] = {
            "extent_ratio": float(extent_ratio),
            "extent_capped": bool(extent_capped),
        }
    return entry


# ---------------------------------------------------------------------------
# Apply — per-type transform implementations
# ---------------------------------------------------------------------------


def _to_output_array(float_arr: np.ndarray, output_dtype: str) -> np.ndarray:
    if output_dtype == "float32":
        return float_arr.astype(np.float32)
    if output_dtype == "int8":
        return _quantize_to_int8(float_arr)
    raise EosframesError(f"Unsupported output_dtype '{output_dtype}'.")


def _quantize_to_int8(float_arr: np.ndarray) -> np.ndarray:
    """Trivially map float ``[-1, 1]`` linearly to int8 ``[-127, 127]``.

    The mapping is the same for every column: ``int8 = round(x · 127)``,
    clipped to ``[-127, 127]``. Columns whose output region is one-sided
    (binary, right- / left-skewed, count mode 0) naturally inhabit only
    the matching half of the int8 range — a binary 0 stays 0, a binary
    1 becomes 127. The sentinel ``-128`` is reserved for NaN.
    """
    out = np.full(float_arr.shape, _INT8_NAN_SENTINEL, dtype=np.int8)
    mask = ~np.isnan(float_arr)
    if not mask.any():
        return out
    scaled = np.round(float_arr[mask] * _INT8_MAX_VAL)
    scaled = np.clip(scaled, -_INT8_MAX_VAL, _INT8_MAX_VAL)
    out[mask] = scaled.astype(np.int8)
    return out


def _apply_constant(
    series: pd.Series, _transform: dict, output_dtype: str
) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    out = np.where(np.isnan(arr), np.nan, 0.0)
    return _to_output_array(out, output_dtype)


def _apply_binary(series: pd.Series, transform: dict, output_dtype: str) -> np.ndarray:
    low = float(transform["low"])
    high = float(transform["high"])
    arr = series.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    non_nan = ~np.isnan(arr)
    # Snap every non-NaN value to whichever of {low, high} is closer; ties
    # break to low (a value exactly at the midpoint maps to 0). NaN passes
    # through as NaN (float32) or the int8 sentinel.
    closer_to_high = np.abs(arr - high) < np.abs(arr - low)
    out[non_nan & closer_to_high] = 1.0
    out[non_nan & ~closer_to_high] = 0.0
    return _to_output_array(out, output_dtype)


def _apply_count_zero_mode(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    high_anchor = float(transform["high_anchor"])
    arr = series.to_numpy(dtype=float)
    out = np.where(np.isnan(arr), np.nan, np.clip(arr / high_anchor, 0.0, 1.0))
    return _to_output_array(out, output_dtype)


def _apply_count_shifted(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    """Linear body to ``±body_target`` on each side, ``tanh`` tail toward ``±1``.

    Mirrors :func:`_apply_continuous_centered`: per-side linear body
    inside ``[mode, anchor]`` (mapping to ``[0, ±body_target]``) plus a
    slope-continuous ``tanh`` asymptote past each anchor. With the
    default ``_CENTERED_BODY_TARGET = 0.5`` the bulk of count data sits
    in ``[-0.5, 0.5]`` and extreme counts stretch asymptotically toward
    ``±1`` rather than piling on a flat clip plateau — distinct
    outliers always get distinct outputs.
    """
    center = float(transform["center"])
    high = float(transform["high_anchor"])
    low = float(transform["low_anchor"])
    body_target = float(transform["body_target"])
    upper_extent = high - center
    lower_extent = center - low
    arr = series.to_numpy(dtype=float)
    nan_mask = np.isnan(arr)
    above = ~nan_mask & (arr >= center)
    below = ~nan_mask & ~above
    out = np.full(arr.shape, np.nan, dtype=float)

    if above.any() and upper_extent > 0:
        mag = arr[above] - center
        in_body = mag <= upper_extent
        y_up = np.empty_like(mag)
        y_up[in_body] = mag[in_body] / upper_extent * body_target
        if (~in_body).any():
            y_up[~in_body] = _tanh_tail(mag[~in_body], upper_extent, body_target)
        out[above] = y_up
    elif above.any():
        out[above] = 0.0

    if below.any() and lower_extent > 0:
        mag = center - arr[below]
        in_body = mag <= lower_extent
        y_lo = np.empty_like(mag)
        y_lo[in_body] = mag[in_body] / lower_extent * body_target
        if (~in_body).any():
            y_lo[~in_body] = _tanh_tail(mag[~in_body], lower_extent, body_target)
        out[below] = -y_lo
    elif below.any():
        out[below] = 0.0
    return _to_output_array(out, output_dtype)


def _apply_continuous_right(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    """Linear body + tail dispatch (3-segment piecewise or finite quadratic).

    Heavy tail (``tail == "piecewise"``): 3-segment design. Linear
    body to ``(body_anchor, body_target)``, then a slower linear
    "middle" segment to ``(mid_anchor, mid_target)``, then a
    finite quadratic to ``(high_anchor, 1)`` with C¹ slope at the
    middle→tail join. Outliers spread visibly across
    ``[body_target, 1]`` instead of saturating asymptotically.

    Light tail (``tail == "finite"``): quadratic from
    ``(body_anchor, body_target)`` to ``(high_anchor, 1)`` with
    slope-continuous join at the body and zero slope at
    ``high_anchor``. ``body_target`` is derived from the geometry so
    the quadratic reaches exactly 1 at ``high_anchor``; values past
    ``high_anchor`` clip to 1 (only happens for OOD inputs).
    """
    low = float(transform["low_anchor"])
    body = float(transform["body_anchor"])
    body_target = float(transform["body_target"])
    tail_kind = transform.get("tail", "piecewise")
    body_span = body - low
    arr = series.to_numpy(dtype=float)
    nan = np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)

    body_mask = ~nan & (arr <= body)
    tail_mask = ~nan & (arr > body)

    if body_span > 0:
        out[body_mask] = np.clip(
            (arr[body_mask] - low) / body_span * body_target,
            0.0,
            body_target,
        )
    else:
        out[body_mask] = 0.0

    if tail_mask.any():
        mag = arr[tail_mask] - low  # distance from low_anchor, ≥ body_span
        if tail_kind == "piecewise":
            mid = float(transform["mid_anchor"])
            high = float(transform["high_anchor"])
            mid_target = float(transform["mid_target"])
            mid_end = mid - low
            max_end = high - low
            seg2 = mag <= mid_end
            seg3 = (mag > mid_end) & (mag <= max_end)
            over = mag > max_end
            y = np.empty_like(mag)
            mid_span = mid_end - body_span
            if mid_span > 0 and seg2.any():
                y[seg2] = body_target + (mag[seg2] - body_span) / mid_span * (
                    mid_target - body_target
                )
            else:
                y[seg2] = body_target
            if seg3.any():
                y[seg3] = _quadratic_tail(mag[seg3], mid_end, mid_target, max_end)
            y[over] = 1.0
            out[tail_mask] = y
        else:  # "finite"
            high = float(transform["high_anchor"])
            max_extent = high - low
            out[tail_mask] = _quadratic_tail(mag, body_span, body_target, max_extent)

    # Cubic Hermite blend across the body→middle junction (piecewise
    # tail only). Smooths the steep-to-gentle slope change so the
    # scaled histogram doesn't show a sharp "double decay" hump at
    # body_target. The cubic interpolates between (body-δ, body slope)
    # and (body+δ, mid slope) with prescribed endpoint values; for the
    # slope ratios in heavy-tail right skew the cubic is monotone
    # (Fritsch-Carlson condition holds for `body_target ≤ 1` cases).
    if tail_kind == "piecewise" and body_span > 0:
        mid = float(transform["mid_anchor"])
        high = float(transform["high_anchor"])
        mid_target = float(transform["mid_target"])
        mid_end = mid - low
        mid_span = mid_end - body_span
        if mid_span > 0:
            delta = _PIECEWISE_BLEND_FRACTION * min(body_span, mid_span)
            blend_lo = low + body_span - delta
            blend_hi = low + body_span + delta
            blend_mask = ~nan & (arr > blend_lo) & (arr < blend_hi)
            if blend_mask.any():
                mag_b = arr[blend_mask] - low
                # Endpoint values & slopes for the cubic.
                p0 = (body_span - delta) / body_span * body_target
                p1 = body_target + delta / mid_span * (mid_target - body_target)
                m_body = body_target / body_span
                m_mid = (mid_target - body_target) / mid_span
                dx = 2.0 * delta
                t = (mag_b - (body_span - delta)) / dx
                t2 = t * t
                t3 = t2 * t
                h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                h10 = t3 - 2.0 * t2 + t
                h01 = -2.0 * t3 + 3.0 * t2
                h11 = t3 - t2
                out[blend_mask] = (
                    h00 * p0 + h10 * dx * m_body + h01 * p1 + h11 * dx * m_mid
                )

    return _to_output_array(out, output_dtype)


def _apply_continuous_left(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    """Mirror of right-skew. Output region is ``[-1, 0]``."""
    high = float(transform["high_anchor"])
    body = float(transform["body_anchor"])
    body_target = float(transform["body_target"])
    tail_kind = transform.get("tail", "piecewise")
    body_span = high - body
    arr = series.to_numpy(dtype=float)
    nan = np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)

    body_mask = ~nan & (arr >= body)
    tail_mask = ~nan & (arr < body)

    if body_span > 0:
        out[body_mask] = np.clip(
            -body_target + (arr[body_mask] - body) / body_span * body_target,
            -body_target,
            0.0,
        )
    else:
        out[body_mask] = 0.0

    if tail_mask.any():
        mag = high - arr[tail_mask]  # distance toward the left tail, ≥ body_span
        if tail_kind == "piecewise":
            mid = float(transform["mid_anchor"])
            low = float(transform["low_anchor"])
            mid_target = float(transform["mid_target"])
            mid_end = high - mid
            max_end = high - low
            seg2 = mag <= mid_end
            seg3 = (mag > mid_end) & (mag <= max_end)
            over = mag > max_end
            y = np.empty_like(mag)
            mid_span = mid_end - body_span
            if mid_span > 0 and seg2.any():
                y[seg2] = body_target + (mag[seg2] - body_span) / mid_span * (
                    mid_target - body_target
                )
            else:
                y[seg2] = body_target
            if seg3.any():
                y[seg3] = _quadratic_tail(mag[seg3], mid_end, mid_target, max_end)
            y[over] = 1.0
            out[tail_mask] = -y
        else:  # "finite"
            low = float(transform["low_anchor"])
            max_extent = high - low
            out[tail_mask] = -_quadratic_tail(mag, body_span, body_target, max_extent)

    # Cubic Hermite blend across the body→middle junction (mirror of
    # the right-skew blend).
    if tail_kind == "piecewise" and body_span > 0:
        mid = float(transform["mid_anchor"])
        low = float(transform["low_anchor"])
        mid_target = float(transform["mid_target"])
        mid_end = high - mid
        mid_span = mid_end - body_span
        if mid_span > 0:
            delta = _PIECEWISE_BLEND_FRACTION * min(body_span, mid_span)
            blend_lo = body - delta
            blend_hi = body + delta
            blend_mask = ~nan & (arr > blend_lo) & (arr < blend_hi)
            if blend_mask.any():
                mag_b = high - arr[blend_mask]
                p0 = (body_span - delta) / body_span * body_target
                p1 = body_target + delta / mid_span * (mid_target - body_target)
                m_body = body_target / body_span
                m_mid = (mid_target - body_target) / mid_span
                dx = 2.0 * delta
                t = (mag_b - (body_span - delta)) / dx
                t2 = t * t
                t3 = t2 * t
                h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                h10 = t3 - 2.0 * t2 + t
                h01 = -2.0 * t3 + 3.0 * t2
                h11 = t3 - t2
                out[blend_mask] = -(
                    h00 * p0 + h10 * dx * m_body + h01 * p1 + h11 * dx * m_mid
                )

    return _to_output_array(out, output_dtype)


def _apply_continuous_centered(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    """Per-side linear body + finite-reach quadratic tail.

    For each side independently: linear from 0 at the centre to
    ``±body_target`` at ``center ± *_body_extent``, then
    :func:`_quadratic_tail` to ``±1`` at ``center ± *_effective_max``,
    where ``effective_max = max(*_max_extent,
    _CENTERED_MAX_EXTENT_RATIO · body_extent)``. ``body_target`` is
    derived per side from ``(body_extent, effective_max)`` and
    **capped at** :data:`_CENTERED_BODY_TARGET` (``0.5``) so the bulk
    of the density visually lives in ``[-0.5, 0.5]``, matching
    ``count_shifted``. The dual cap (``0.5``) and floor (``3·b``) are
    mathematically locked: the body→tail join is C¹-smooth precisely
    at ``max_extent = 3·body_extent`` for ``body_target = 0.5``, so
    flooring the effective max at ``3·b`` whenever the data doesn't
    naturally provide it both eliminates the slope kink and prevents
    bounded distributions (U-shape, narrow bimodal, truncated normal)
    from spraying their few near-edge points across the entire tail
    region. Inputs past ``effective_max`` clip to exactly ``±1``.
    """
    center = float(transform["center"])
    upper_body = float(transform["upper_body_extent"])
    lower_body = float(transform["lower_body_extent"])
    upper_max = float(transform["upper_max_extent"])
    lower_max = float(transform["lower_max_extent"])

    arr = series.to_numpy(dtype=float)
    nan = np.isnan(arr)
    out = np.full(arr.shape, np.nan, dtype=float)

    def _side(
        mask: np.ndarray, body_extent: float, max_extent: float, sign: float
    ) -> None:
        if not mask.any():
            return
        if body_extent <= 0:
            out[mask] = 0.0
            return
        effective_max = max(max_extent, _CENTERED_MAX_EXTENT_RATIO * body_extent)
        body_target = min(
            _light_tail_body_target(body_extent, effective_max),
            _CENTERED_BODY_TARGET,
        )
        mag = sign * (arr[mask] - center)
        body_part = mag <= body_extent
        tail_part = ~body_part
        y = np.empty_like(mag)
        y[body_part] = mag[body_part] / body_extent * body_target
        if tail_part.any():
            y[tail_part] = _quadratic_tail(
                mag[tail_part], body_extent, body_target, effective_max
            )
        out[mask] = sign * y

    _side(~nan & (arr >= center), upper_body, upper_max, +1.0)
    _side(~nan & (arr < center), lower_body, lower_max, -1.0)
    return _to_output_array(out, output_dtype)


# Single-string dispatch: each ``kind`` maps to the apply function that
# reads its specific ``transform`` payload. Keep in sync with
# ``_OUTPUT_REGIONS`` — the two tables share the same key set.
_APPLY_DISPATCH: Dict[str, Callable[[pd.Series, dict, str], np.ndarray]] = {
    "constant": _apply_constant,
    "binary": _apply_binary,
    "count_zero_mode": _apply_count_zero_mode,
    "count_shifted": _apply_count_shifted,
    "continuous_right_skew": _apply_continuous_right,
    "continuous_left_skew": _apply_continuous_left,
    "continuous_centered": _apply_continuous_centered,
}


def _dispatch_apply(
    series: pd.Series, transform: dict, output_dtype: str
) -> np.ndarray:
    kind = transform["kind"]
    apply_fn = _APPLY_DISPATCH.get(kind)
    if apply_fn is None:
        raise EosframesError(f"Unknown transform kind '{kind}'.")
    return apply_fn(series, transform, output_dtype)


# ---------------------------------------------------------------------------
# Low-level DataFrame API
# ---------------------------------------------------------------------------


def fit(df: pd.DataFrame) -> dict:
    """Fit a type-aware robust scaler on the numeric feature columns.

    Each numeric feature column is classified into one of the seven
    transform kinds (``constant`` / ``binary`` / ``count_zero_mode`` /
    ``count_shifted`` / ``continuous_right_skew`` /
    ``continuous_left_skew`` / ``continuous_centered``) and a per-column
    entry is recorded. Every numeric column is fitted; all-NaN columns
    fall back to ``kind: "constant"``. The ``key`` and ``input``
    columns and any non-numeric feature columns are ignored entirely.

    Output dtype is a **transform-time** choice, not a fit-time one —
    pass ``output_dtype`` to :func:`transform` (or ``--quantize`` to
    the CLI).

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame. ``key`` and ``input`` columns are ignored.

    Returns
    -------
    dict
        Dtype-agnostic parameters with keys:

        * ``method``  — always ``"robust_typed"``.
        * ``columns`` — ``{column_name: entry}`` for every fitted
          column, in fit-time order. Each ``entry`` has ``transform``
          (the kind + transform-time params), ``impute_value`` (median
          fill for ``impute=True``), and an optional ``fit_notes``
          (provenance diagnostics, never read at transform time).

    Raises
    ------
    EosframesError
        If no numeric columns exist.
    """
    logger = get_logger()
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        raise EosframesError("No numeric feature columns found to fit the scaler.")

    columns: dict = {}
    kind_counts: dict = {}

    for col in numeric_cols:
        series = df[col]

        if series.dropna().empty:
            # All-NaN column: fit as a constant. The dispatch at
            # transform time maps non-NaN inputs to 0 and propagates NaN.
            entry = _fit_constant(series)
        else:
            type_ = _classify_type(series)
            if type_ == "constant":
                entry = _fit_constant(series)
            elif type_ == "binary":
                entry = _fit_binary(series)
            elif type_ == "count":
                entry = _fit_count(series)
            else:
                try:
                    entry = _fit_continuous(series)
                except EosframesError:
                    # Column slipped past _classify_type but has no usable scale.
                    entry = _fit_constant(series)

        columns[col] = entry
        kind = entry["transform"]["kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    kind_breakdown = ", ".join(
        f"{kind}={count}" for kind, count in sorted(kind_counts.items())
    )
    logger.info(
        "Fitted %d / %d numeric columns (%s)",
        len(numeric_cols),
        len(numeric_cols),
        kind_breakdown,
    )

    # Single aggregated advisory for near-degenerate count columns. The
    # per-column flag lives in each entry's fit_notes; here we just
    # summarise how many tripped it so sparse fingerprints don't flood
    # the log with one warning per bit.
    degenerate_cols = [
        col
        for col, entry in columns.items()
        if entry.get("fit_notes", {}).get("degenerate")
    ]
    if degenerate_cols:
        preview = ", ".join(str(c) for c in degenerate_cols[:5])
        if len(degenerate_cols) > 5:
            preview += ", …"
        logger.warning(
            "%d of %d count columns are near-degenerate (e.g. %s): output "
            "collapses to a handful of values. Common for sparse "
            "fingerprints; see each column's fit_notes for details. "
            "Consider dropping them or revisiting upstream featurization.",
            len(degenerate_cols),
            len(numeric_cols),
            preview,
        )

    return {"method": _METHOD_NAME, "columns": columns}


def transform(
    df: pd.DataFrame,
    params: dict,
    output_dtype: str = _DEFAULT_OUTPUT_DTYPE,
    impute: bool = False,
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame.

    The ``key`` / ``input`` columns pass through unchanged. The fitted
    feature columns in *df* must match exactly — same set and same
    order — the ``feature_columns`` recorded in *params*.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to transform. Must have the same numeric feature columns
        as the frame the scaler was fitted on.
    params : dict
        Dtype-agnostic parameters as returned by :func:`fit` or loaded
        from a scaler JSON.
    output_dtype : {"float32", "int8"}, default ``"float32"``
        ``"float32"`` preserves NaN for missing values. ``"int8"``
        quantizes scaled values to ``[-127, 127]`` with sentinel
        ``-128`` for missing — useful for compact storage of fingerprint-
        like outputs.
    impute : bool, default ``False``
        When ``True``, replace every input NaN with the column's
        recorded ``impute_value`` *before* dispatch. The output column
        will have no NaN entries (and, under ``output_dtype="int8"``,
        no ``-128`` sentinels). The substituted value is the median of
        the column's training data, rounded to ``int`` if the column
        was integer-valued at fit time.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with scaled feature columns in the requested
        dtype. The returned frame does **not** carry ``model_id`` /
        ``version`` attributes — re-attach them before writing.

    Raises
    ------
    EosframesError
        On column mismatch, invalid ``output_dtype``, or unknown
        column type in *params*.
    """
    if output_dtype not in _VALID_OUTPUT_DTYPES:
        raise EosframesError(
            f"Unknown output_dtype '{output_dtype}'. Supported: {_VALID_OUTPUT_DTYPES}"
        )

    expected_feature_cols = list(params["columns"].keys())
    df_feature_cols = [c for c in df.columns if c not in _META_COLS]

    if df_feature_cols != expected_feature_cols:
        raise EosframesError(
            f"Column mismatch: input has feature columns {df_feature_cols} "
            f"but transformer was fitted on {expected_feature_cols}."
        )

    columns = params["columns"]
    result = df.copy()
    for col in expected_feature_cols:
        entry = columns[col]
        series = df[col]
        if impute:
            series = series.fillna(entry.get("impute_value", 0.0))
        result[col] = _dispatch_apply(series, entry["transform"], output_dtype)
    return result


# ---------------------------------------------------------------------------
# File-level API
# ---------------------------------------------------------------------------


def _values_dtype_for(output_dtype: str) -> np.dtype:
    if output_dtype == "int8":
        return np.int8
    return np.float32


def _write_df(
    df: pd.DataFrame, output_path: str, values_dtype: np.dtype = np.float32
) -> None:
    """Write a DataFrame to CSV or H5, bypassing the naming convention check.

    *values_dtype* controls the H5 ``values`` dataset dtype; CSV ignores it.
    """
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".h5":
        feat_cols = [c for c in df.columns if c not in _META_COLS]
        with h5py.File(output_path, "w") as f:
            dt = h5py.string_dtype(encoding="utf-8")
            if "key" in df.columns:
                f.create_dataset("key", data=df["key"].astype(str).tolist(), dtype=dt)
            if "input" in df.columns:
                f.create_dataset(
                    "input", data=df["input"].astype(str).tolist(), dtype=dt
                )
            f.create_dataset("features", data=feat_cols, dtype=dt)
            f.create_dataset("values", data=df[feat_cols].values, dtype=values_dtype)
    else:
        raise EosframesError(f"Unsupported output format '{ext}'. Expected .csv or .h5")


def fit_file(
    input_path: str,
    scaler_path: str,
    output_path: Optional[str] = None,
    output_dtype: str = _DEFAULT_OUTPUT_DTYPE,
    impute: bool = False,
) -> str:
    """Fit a scaler on an Ersilia output file and save the parameters.

    The scaler JSON written to *scaler_path* is dtype-agnostic — see
    :func:`fit` for the parameter set. When *output_path* is provided
    the scaled data is also written immediately (fit-then-transform in
    one call), and *output_dtype* selects the dtype of that inline
    output. The dtype is **not** recorded in the scaler JSON; later
    calls to :func:`transform_file` choose the dtype independently.

    The scaler filename's encoded model ID and version must match the
    input file's. The transformer JSON records ``eosframes_version``
    (the running package version, via ``importlib.metadata``) and
    ``method`` (``"robust_typed"``); :func:`transform_file` rejects on
    any mismatch with the current ``eosframes.__version__`` so a
    package release automatically forces a re-fit.

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file. Must follow the Ersilia naming
        convention.
    scaler_path : str
        Path where the JSON parameter file will be written. Must not
        already exist. Must follow
        ``[prefix_]<model_id>_<version>_transformer.json`` with model
        ID and version matching *input_path*.
    output_path : str, optional
        If provided, also write the scaled data here (fit-transform).
        The file must not already exist; its extension determines
        whether CSV or H5 is written.
    output_dtype : {"float32", "int8"}, default ``"float32"``
        Only used for the inline transform when *output_path* is
        given. ``"int8"`` quantizes the scaled values into
        ``[-127, 127]`` with sentinel ``-128`` for missing.

    Returns
    -------
    str
        Absolute path of the saved scaler JSON file.

    Raises
    ------
    EosframesError
        On naming convention violations, pre-existing files,
        model-ID / version mismatch between scaler and input, no
        numeric columns to fit, or invalid ``output_dtype``.
    """
    logger = get_logger()

    if not is_valid_name(input_path):
        raise EosframesError(
            f"'{input_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext>"
        )

    if not is_valid_transformer_name(scaler_path):
        raise EosframesError(
            f"'{scaler_path}' does not follow the scaler naming convention. "
            "Expected: [prefix_]<model_id>_<version>_transformer.json"
        )

    parsed = parse_name(input_path)
    scaler_parsed = parse_transformer_name(scaler_path)

    if scaler_parsed["model_id"] != parsed["model_id"]:
        raise EosframesError(
            f"Scaler model ID '{scaler_parsed['model_id']}' does not match "
            f"input model ID '{parsed['model_id']}'. "
            "Scaler filename must encode the same model ID as the input file."
        )
    if scaler_parsed["version"] != parsed["version"]:
        raise EosframesError(
            f"Scaler version '{scaler_parsed['version']}' does not match "
            f"input version '{parsed['version']}'. "
            "Scaler filename must encode the same version as the input file."
        )

    if os.path.exists(scaler_path):
        raise EosframesError(
            f"Scaler file '{scaler_path}' already exists. Remove it first."
        )

    if output_path is not None and os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    from .ops import _read_file

    df = _read_file(input_path)

    if output_dtype not in _VALID_OUTPUT_DTYPES:
        raise EosframesError(
            f"Unknown output_dtype '{output_dtype}'. Supported: {_VALID_OUTPUT_DTYPES}"
        )

    mode_label = "fit + transform" if output_path is not None else "fit only"
    logger.info("%s — reading %d rows from %s", mode_label, len(df), input_path)
    fitted = fit(df)

    transformer = {
        "eosframes_version": _PACKAGE_VERSION,
        "method": fitted["method"],
        "model_id": parsed["model_id"],
        "model_version": parsed["version"],
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": len(df),
        "columns": fitted["columns"],
    }
    with open(scaler_path, "w") as fh:
        json.dump(transformer, fh, indent=2)
    logger.info("Scaler saved to %s", scaler_path)

    if output_path is not None:
        logger.info(
            "Transforming inline (output_dtype=%s, impute=%s)",
            output_dtype,
            impute,
        )
        scaled_df = transform(df, transformer, output_dtype=output_dtype, impute=impute)
        _write_df(
            scaled_df,
            output_path,
            values_dtype=_values_dtype_for(output_dtype),
        )
        logger.info("Scaled output written to %s", output_path)

    return scaler_path


def transform_file(
    input_path: str,
    scaler_path: str,
    output_path: str,
    output_dtype: str = _DEFAULT_OUTPUT_DTYPE,
    impute: bool = False,
) -> str:
    """Apply a saved scaler to an Ersilia output file.

    The scaler's recorded ``eosframes_version`` must exactly match the
    running ``eosframes.__version__``, and ``method`` must still be
    ``"robust_typed"``; any mismatch raises with a clear "re-fit"
    error. The scaler's recorded ``model_id`` and ``version`` must
    match the model ID / version encoded in *input_path* — running a
    scaler against a different model's outputs is never silently
    allowed.

    Parameters
    ----------
    input_path : str
        Input CSV or H5 file. Must follow the Ersilia naming
        convention.
    scaler_path : str
        Path to a JSON scaler file produced by :func:`fit_file`.
    output_path : str
        Where to write the scaled data. Must not already exist; its
        extension determines whether CSV or H5 is written.
    output_dtype : {"float32", "int8"}, default ``"float32"``
        ``"int8"`` quantizes scaled values to ``[-127, 127]`` with
        sentinel ``-128`` for missing.

    Returns
    -------
    str
        Absolute path of the scaled output file.

    Raises
    ------
    EosframesError
        On naming-convention violations, missing scaler file,
        pre-existing output path, ``eosframes_version`` or ``method``
        mismatch on the scaler JSON, model-ID or version mismatch
        between scaler and input, column mismatch between scaler and
        input, or invalid ``output_dtype``.
    """
    logger = get_logger()

    if not is_valid_name(input_path):
        raise EosframesError(
            f"'{input_path}' does not follow the naming convention. "
            "Expected: <model_id>_<version>.<ext>"
        )

    if not is_valid_transformer_name(scaler_path):
        raise EosframesError(
            f"'{scaler_path}' does not follow the scaler naming convention. "
            "Expected: [prefix_]<model_id>_<version>_transformer.json"
        )

    parsed = parse_name(input_path)

    if os.path.exists(output_path):
        raise EosframesError(
            f"Output file '{output_path}' already exists. Remove it first."
        )

    if not os.path.exists(scaler_path):
        raise EosframesError(f"Scaler file '{scaler_path}' not found.")

    with open(scaler_path) as fh:
        transformer = json.load(fh)

    scaler_version = transformer.get("eosframes_version") or ""
    method = transformer.get("method")
    if _major(scaler_version) != _major(_PACKAGE_VERSION) or method != _METHOD_NAME:
        raise EosframesError(
            f"Scaler was fitted with eosframes "
            f"{scaler_version!r} (method={method!r}) but this is eosframes "
            f"{_PACKAGE_VERSION!r} (method '{_METHOD_NAME}'). Re-fit the "
            "scaler — the schema only carries across the same eosframes "
            "major version."
        )

    if output_dtype not in _VALID_OUTPUT_DTYPES:
        raise EosframesError(
            f"Unknown output_dtype '{output_dtype}'. Supported: {_VALID_OUTPUT_DTYPES}"
        )

    t_model_id = transformer.get("model_id")
    t_model_version = transformer.get("model_version")
    f_model_id = parsed["model_id"]
    f_model_version = parsed["version"]

    if f_model_id != t_model_id:
        raise EosframesError(
            f"Model ID mismatch: file has '{f_model_id}' but scaler "
            f"was fitted on '{t_model_id}'."
        )
    if f_model_version != t_model_version:
        raise EosframesError(
            f"Model version mismatch: file has '{f_model_version}' but "
            f"scaler was fitted on '{t_model_version}'."
        )

    from .ops import _read_file

    df = _read_file(input_path)

    logger.info(
        "Applying scaler to %d rows from %s (output_dtype=%s, impute=%s)",
        len(df),
        input_path,
        output_dtype,
        impute,
    )
    scaled_df = transform(df, transformer, output_dtype=output_dtype, impute=impute)

    _write_df(scaled_df, output_path, values_dtype=_values_dtype_for(output_dtype))
    logger.info("Scaled output written to %s", output_path)
    return output_path
