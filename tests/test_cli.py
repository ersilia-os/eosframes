"""
Tests for the eosframes CLI and naming convention module.

Run with:
    pytest tests/test_cli.py -v
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from eosframes.cli import main
from eosframes.naming import (
    is_valid_name,
    make_chunks_dir_name,
    make_output_name,
    parse_name,
    get_version_from_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp(tmp_path):
    """Provide a temporary directory as a pathlib Path."""
    return tmp_path


def _write_csv(path: str, n_rows: int = 5, model_id: str = "eos4e40") -> None:
    """Write a minimal Ersilia-format CSV to *path*."""
    df = pd.DataFrame({
        "key":   [f"k{i}" for i in range(n_rows)],
        "input": [f"mol{i}" for i in range(n_rows)],
        "score": [round(i * 0.1, 2) for i in range(n_rows)],
        "prob":  [round(1 - i * 0.1, 2) for i in range(n_rows)],
    })
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# naming.py
# ---------------------------------------------------------------------------

class TestNaming:

    def test_parse_csv(self):
        r = parse_name("eos4e40_v1.csv")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

    def test_parse_h5(self):
        r = parse_name("eos4e40_v1.h5")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "h5", "name_type": "h5"}

    def test_parse_chunks_dir(self):
        r = parse_name("eos4e40_v1_chunks")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": None, "name_type": "chunks_dir"}

    def test_parse_chunks_dir_trailing_slash(self):
        assert parse_name("eos4e40_v1_chunks/")["name_type"] == "chunks_dir"

    def test_parse_full_path(self):
        r = parse_name("/some/dir/eos4e40_v1.csv")
        assert r["model_id"] == "eos4e40"

    def test_parse_with_prefix(self):
        r = parse_name("260313_gardp_eos4e40_v1.csv")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

    def test_parse_with_prefix_chunks(self):
        r = parse_name("260313_gardp_eos4e40_v1_chunks")
        assert r["model_id"] == "eos4e40"
        assert r["name_type"] == "chunks_dir"

    def test_parse_no_version(self):
        assert parse_name("eos4e40.csv") is None

    def test_parse_extra_token(self):
        assert parse_name("eos4e40_v1_extra.csv") is None

    def test_parse_arbitrary_filename(self):
        assert parse_name("output.csv") is None

    def test_is_valid_name_true(self):
        assert is_valid_name("eos4e40_v1.csv") is True
        assert is_valid_name("eos3804_v2.h5") is True
        assert is_valid_name("eos4e40_v1_chunks") is True

    def test_is_valid_name_false(self):
        assert is_valid_name("results.csv") is False
        assert is_valid_name("eos4e40.csv") is False

    def test_make_output_name(self):
        assert make_output_name("eos4e40", "v1", "csv") == "eos4e40_v1.csv"
        assert make_output_name("eos3804", "v3", "h5") == "eos3804_v3.h5"

    def test_make_output_name_bad_model_id(self):
        with pytest.raises(ValueError):
            make_output_name("badid", "v1", "csv")

    def test_make_output_name_bad_version(self):
        with pytest.raises(ValueError):
            make_output_name("eos4e40", "1", "csv")

    def test_make_output_name_bad_ext(self):
        with pytest.raises(ValueError):
            make_output_name("eos4e40", "v1", "xlsx")

    def test_make_chunks_dir_name(self):
        assert make_chunks_dir_name("eos4e40", "v1") == "eos4e40_v1_chunks"

    def test_get_version_from_path(self):
        assert get_version_from_path("eos4e40_v2.csv") == "v2"
        assert get_version_from_path("output.csv") is None


# ---------------------------------------------------------------------------
# CLI: split
# ---------------------------------------------------------------------------

class TestSplit:

    def test_basic(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=10)
        out = str(tmp / "chunks")

        result = CliRunner().invoke(main, ["split", src, out, "--chunksize", "3"])
        assert result.exit_code == 0, result.output
        files = sorted(os.listdir(out))
        assert len(files) == 4   # ceil(10/3) = 4
        assert files[0] == "chunk_000.csv"

    def test_zfill_3_digit(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=5)
        out = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", src, out, "--chunksize", "1"])
        files = os.listdir(out)
        # 5 chunks → 3-digit padding
        assert all(f.startswith("chunk_") and len(f) == 13 for f in files)  # chunk_000.csv

    def test_output_folder_exists_error(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src)
        out = str(tmp / "exists")
        os.makedirs(out)
        result = CliRunner().invoke(main, ["split", src, out])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_row_count_preserved(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=7)
        out = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", src, out, "--chunksize", "3"])
        total = sum(
            len(pd.read_csv(os.path.join(out, f))) for f in os.listdir(out)
        )
        assert total == 7


# ---------------------------------------------------------------------------
# CLI: convert
# ---------------------------------------------------------------------------

class TestConvert:

    def test_csv_to_h5(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        dst = str(tmp / "eos4e40_v1.h5")
        _write_csv(src)
        result = CliRunner().invoke(main, ["convert", src, dst])
        assert result.exit_code == 0, result.output
        assert os.path.exists(dst)

    def test_h5_to_csv(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        h5 = str(tmp / "eos4e40_v1.h5")
        CliRunner().invoke(main, ["convert", src, h5])
        dst = str(tmp / "eos4e40_v2.csv")
        result = CliRunner().invoke(main, ["convert", h5, dst])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(dst)
        assert list(df["key"]) == [f"k{i}" for i in range(5)]

    def test_chunks_to_csv(self, tmp):
        raw = str(tmp / "input.csv")
        _write_csv(raw, n_rows=6)
        chunks = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", raw, chunks, "--chunksize", "2"])
        dst = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["convert", chunks, dst])
        assert result.exit_code == 0, result.output
        assert len(pd.read_csv(dst)) == 6

    def test_bad_output_name_error(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["convert", src, str(tmp / "output.h5")])
        assert result.exit_code != 0
        assert "naming convention" in result.output


# ---------------------------------------------------------------------------
# CLI: stack
# ---------------------------------------------------------------------------

class TestStack:

    def _two_files(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        f2 = str(tmp / "eos3804_v1.csv")
        for path, col in [(f1, "score"), (f2, "activity")]:
            pd.DataFrame({
                "key":   ["k0", "k1", "k2"],
                "input": ["mol0", "mol1", "mol2"],
                col:     [0.1, 0.2, 0.3],
            }).to_csv(path, index=False)
        return f1, f2

    def test_basic_with_suffix(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "stacked.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert "score.eos4e40" in df.columns
        assert "activity.eos3804" in df.columns
        assert df.columns.tolist().count("input") == 1

    def test_no_suffix(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "stacked.csv")
        CliRunner().invoke(main, ["stack", f1, f2, "-o", out, "--no-suffix"])
        df = pd.read_csv(out)
        assert "score" in df.columns
        assert "activity" in df.columns

    def test_duplicate_model_error(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        _write_csv(f1)
        out = str(tmp / "stacked.csv")
        result = CliRunner().invoke(main, ["stack", f1, f1, "-o", out])
        assert result.exit_code != 0
        assert "more than once" in result.output

    def test_input_mismatch_error(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        f2 = str(tmp / "eos3804_v1.csv")
        pd.DataFrame({"key": ["k0"], "input": ["mol0"], "score": [0.1]}).to_csv(f1, index=False)
        pd.DataFrame({"key": ["k1"], "input": ["mol1"], "score": [0.2]}).to_csv(f2, index=False)
        out = str(tmp / "stacked.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code != 0
        assert "mismatch" in result.output


# ---------------------------------------------------------------------------
# CLI: append
# ---------------------------------------------------------------------------

class TestAppend:

    def test_basic(self, tmp):
        b1 = str(tmp / "eos4e40_v1_batch1.csv")
        b2 = str(tmp / "eos4e40_v1_batch2.csv")
        _write_csv(b1, n_rows=3)
        _write_csv(b2, n_rows=4)
        out = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["append", b1, b2, "-o", out])
        assert result.exit_code == 0, result.output
        assert len(pd.read_csv(out)) == 7

    def test_order_preserved(self, tmp):
        b1 = str(tmp / "eos4e40_v1_b1.csv")
        b2 = str(tmp / "eos4e40_v1_b2.csv")
        pd.DataFrame({"key": ["k0", "k1"], "input": ["m0", "m1"], "score": [1.0, 2.0]}).to_csv(b1, index=False)
        pd.DataFrame({"key": ["k2", "k3"], "input": ["m2", "m3"], "score": [3.0, 4.0]}).to_csv(b2, index=False)
        out = str(tmp / "eos4e40_v1.csv")
        CliRunner().invoke(main, ["append", b1, b2, "-o", out])
        df = pd.read_csv(out)
        assert df["score"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_mixed_model_error(self, tmp):
        b1 = str(tmp / "eos4e40_v1_b1.csv")
        b2 = str(tmp / "eos3804_v1_b1.csv")
        _write_csv(b1, n_rows=2)
        _write_csv(b2, n_rows=2, model_id="eos3804")
        out = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["append", b1, b2, "-o", out])
        assert result.exit_code != 0
        assert "mismatch" in result.output

    def test_column_mismatch_error(self, tmp):
        b1 = str(tmp / "eos4e40_v1_b1.csv")
        b2 = str(tmp / "eos4e40_v1_b2.csv")
        pd.DataFrame({"key": ["k0"], "input": ["m0"], "score": [1.0]}).to_csv(b1, index=False)
        pd.DataFrame({"key": ["k1"], "input": ["m1"], "other": [2.0]}).to_csv(b2, index=False)
        out = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["append", b1, b2, "-o", out])
        assert result.exit_code != 0
        assert "mismatch" in result.output


# ---------------------------------------------------------------------------
# CLI: dedupe
# ---------------------------------------------------------------------------

class TestDedupe:

    def test_removes_duplicates(self, tmp):
        src = str(tmp / "eos4e40_v1_raw.csv")
        pd.DataFrame({
            "key":   ["k0", "k1", "k0", "k2", "k1"],
            "input": ["m0", "m1", "m0", "m2", "m1"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }).to_csv(src, index=False)
        dst = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["dedupe", src, dst])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(dst)
        assert len(df) == 3
        # First occurrence of k0 has score 1.0, k1 has 2.0
        assert df[df["key"] == "k0"]["score"].iloc[0] == 1.0
        assert df[df["key"] == "k1"]["score"].iloc[0] == 2.0

    def test_no_duplicates_unchanged(self, tmp):
        src = str(tmp / "eos4e40_v1_clean.csv")
        _write_csv(src, n_rows=4)
        dst = str(tmp / "eos4e40_v1.csv")
        CliRunner().invoke(main, ["dedupe", src, dst])
        assert len(pd.read_csv(dst)) == 4

    def test_bad_output_name_error(self, tmp):
        src = str(tmp / "eos4e40_v1_raw.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["dedupe", src, str(tmp / "out.csv")])
        assert result.exit_code != 0
        assert "naming convention" in result.output
