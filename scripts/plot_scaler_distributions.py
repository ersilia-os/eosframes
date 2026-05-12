"""Regenerate the per-feature distribution PNGs for eosframes.

Produces three grid PNGs in two locations:

* ``data/scaler_tests/_distributions{,_scaled,_transforms}.png`` —
  one cell per synthetic per-branch fixture under
  ``data/scaler_tests/`` (each fixture has a single feature column).
* ``data/example_<model>_<version>_distributions{,_scaled,_transforms}.png``
  for each real example CSV (``data/example_eos7m30_v1.csv``,
  ``data/example_eos4e40_v1.csv``) — one cell per feature column.

In each grid:

* ``_distributions.png``            — raw histograms (green).
* ``_distributions_scaled.png``     — scaled histograms, red on the
  lower half / blue on the upper half, with reference lines at
  ``±0.5`` (centered body target) and ``±1`` (output region edge).
* ``_distributions_transforms.png`` — raw-to-scaled transform curves.

Usage: ``python scripts/plot_scaler_distributions.py``.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eosframes.scale import fit_file, transform_file

REPO = Path(__file__).resolve().parent.parent
SCALER_TESTS = REPO / "data" / "scaler_tests"
DATA = REPO / "data"
EXAMPLE_FILES = [DATA / "example_eos7m30_v1.csv", DATA / "example_eos4e40_v1.csv"]
N_COLS = 7
STEM_RE = re.compile(r"(\d+)_(.+)_eos\d[A-Za-z0-9]{3}_v\d+$")


def list_fixtures() -> list[Path]:
    def key(p: Path) -> int:
        m = re.match(r"(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    return sorted(SCALER_TESTS.glob("*.csv"), key=key)


def short_title(stem: str) -> str:
    m = STEM_RE.match(stem)
    if not m:
        return stem
    return f"{m.group(1)} {m.group(2)}"


def fit_many(src: Path) -> list[dict]:
    """Fit and transform every feature column in ``src``.

    Returns one record per feature: ``{"raw", "scaled", "kind", "title"}``.
    """
    df = pd.read_csv(src)
    feat_cols = [c for c in df.columns if c not in ("key", "input")]
    with tempfile.TemporaryDirectory() as td:
        scaler = Path(td) / f"{src.stem}_transformer.json"
        out = Path(td) / src.name
        fit_file(str(src), str(scaler))
        transform_file(str(src), str(scaler), str(out))
        with scaler.open() as f:
            scaler_json = json.load(f)
        scaled_df = pd.read_csv(out)
    records: list[dict] = []
    for feat in feat_cols:
        records.append(
            {
                "raw": df[feat],
                "scaled": scaled_df[feat],
                "kind": scaler_json["columns"][feat]["transform"]["kind"],
                "title": feat,
            }
        )
    return records


def make_grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    n_cols = min(N_COLS, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 1.7), dpi=110, squeeze=False
    )
    return fig, axes.flatten()


def plot_raw(records: list[dict], out_path: Path, header: str) -> None:
    fig, axes = make_grid(len(records))
    for ax, rec in zip(axes, records):
        ax.hist(rec["raw"].dropna(), bins=40, color="#5dac6e", edgecolor="none")
        ax.set_title(f"{rec['title']}\n{rec['kind']}", fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes[len(records):]:
        ax.set_visible(False)
    fig.suptitle(f"{header} — RAW distribution per column", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def plot_scaled(records: list[dict], out_path: Path, header: str) -> None:
    fig, axes = make_grid(len(records))
    bins = np.linspace(-1.0, 1.0, 51)
    for ax, rec in zip(axes, records):
        s = rec["scaled"].dropna().to_numpy()
        if s.size:
            neg = s[s < 0]
            pos = s[s > 0]
            zero = s[s == 0]
            if neg.size:
                ax.hist(neg, bins=bins, color="#d35454", edgecolor="none")
            if pos.size:
                ax.hist(pos, bins=bins, color="#4978bf", edgecolor="none")
            if zero.size:
                ax.hist(zero, bins=bins, color="#888888", edgecolor="none")
        for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
            ax.axvline(x, color="grey", lw=0.4, alpha=0.4,
                       linestyle="--" if abs(x) == 0.5 else "-")
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{rec['title']}\n{rec['kind']}", fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes[len(records):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{header} — SCALED output per column\n"
        "red = lower half | blue = upper half | dashed ±0.5 = body target",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_transforms(records: list[dict], out_path: Path, header: str) -> None:
    fig, axes = make_grid(len(records))
    for ax, rec in zip(axes, records):
        raw = rec["raw"].to_numpy()
        scaled = rec["scaled"].to_numpy()
        mask = ~(np.isnan(raw) | np.isnan(scaled))
        x, y = raw[mask], scaled[mask]
        order = np.argsort(x)
        ax.plot(x[order], y[order], color="black", lw=0.8)
        for h in (-1.0, -0.5, 0.0, 0.5, 1.0):
            ax.axhline(h, color="grey", lw=0.4, alpha=0.4,
                       linestyle="--" if abs(h) == 0.5 else "-")
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"{rec['title']}\n{rec['kind']}", fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes[len(records):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{header} — TRANSFORM curve per column\n"
        "x = raw input | y = scaled output | dashed ±0.5 = body target",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def render_set(records: list[dict], base_dir: Path, base_stem: str, header: str) -> None:
    """Write the three PNGs ``<base_stem>{,_scaled,_transforms}.png`` to ``base_dir``."""
    plot_raw(records, base_dir / f"{base_stem}.png", header)
    plot_scaled(records, base_dir / f"{base_stem}_scaled.png", header)
    plot_transforms(records, base_dir / f"{base_stem}_transforms.png", header)
    print(
        f"wrote {base_stem}.png, {base_stem}_scaled.png, {base_stem}_transforms.png "
        f"to {base_dir} ({len(records)} cells)"
    )


def main() -> None:
    fixtures = list_fixtures()
    print(f"scaler_tests: {len(fixtures)} fixtures in {SCALER_TESTS}")
    fixture_records: list[dict] = []
    for src in fixtures:
        rec = fit_many(src)[0]
        rec["title"] = short_title(src.stem)
        fixture_records.append(rec)
    render_set(
        fixture_records,
        SCALER_TESTS,
        "_distributions",
        "eosframes scaler test fixtures",
    )

    for src in EXAMPLE_FILES:
        if not src.exists():
            print(f"example file missing, skipping: {src}")
            continue
        records = fit_many(src)
        print(f"{src.name}: {len(records)} feature columns")
        render_set(
            records,
            src.parent,
            f"{src.stem}_distributions",
            src.stem,
        )


if __name__ == "__main__":
    main()
