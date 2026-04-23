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
    get_version_from_path,
    is_valid_columns_name,
    is_valid_info_name,
    is_valid_name,
    is_valid_summary_name,
    make_chunks_dir_name,
    make_columns_name,
    make_info_name,
    make_output_name,
    make_summary_name,
    parse_name,
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

    # --- sidecar (_info.csv / _columns.csv) ---

    def test_parse_info_csv(self):
        r = parse_name("eos4e40_v1_info.csv")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "info"}

    def test_parse_info_csv_with_prefix(self):
        r = parse_name("example_eos4e40_v1_info.csv")
        assert r["model_id"] == "eos4e40"
        assert r["name_type"] == "info"

    def test_parse_columns_csv(self):
        r = parse_name("eos7m30_v2_columns.csv")
        assert r == {"model_id": "eos7m30", "version": "v2", "extension": "csv", "name_type": "columns"}

    def test_parse_columns_csv_with_prefix(self):
        assert parse_name("260313_gardp_eos4e40_v1_columns.csv")["name_type"] == "columns"

    def test_parse_info_h5_rejected(self):
        # Sidecar convention is csv-only.
        assert parse_name("eos4e40_v1_info.h5") is None

    def test_parse_info_without_version_rejected(self):
        assert parse_name("eos4e40_info.csv") is None

    def test_is_valid_name_rejects_sidecar(self):
        # is_valid_name covers data files only; sidecars must not sneak through.
        assert is_valid_name("eos4e40_v1_info.csv") is False
        assert is_valid_name("eos4e40_v1_columns.csv") is False

    def test_is_valid_info_name(self):
        assert is_valid_info_name("eos4e40_v1_info.csv") is True
        assert is_valid_info_name("example_eos4e40_v1_info.csv") is True
        assert is_valid_info_name("eos4e40_v1.csv") is False
        assert is_valid_info_name("eos4e40_v1_columns.csv") is False
        assert is_valid_info_name("bogus.csv") is False

    def test_is_valid_columns_name(self):
        assert is_valid_columns_name("eos4e40_v1_columns.csv") is True
        assert is_valid_columns_name("example_eos4e40_v1_columns.csv") is True
        assert is_valid_columns_name("eos4e40_v1.csv") is False
        assert is_valid_columns_name("eos4e40_v1_info.csv") is False

    def test_parse_summary_csv(self):
        r = parse_name("eos4e40_v1_summary.csv")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "summary"}

    def test_parse_summary_with_prefix(self):
        assert parse_name("example_eos4e40_v1_summary.csv")["name_type"] == "summary"

    def test_is_valid_summary_name(self):
        assert is_valid_summary_name("eos4e40_v1_summary.csv") is True
        assert is_valid_summary_name("example_eos4e40_v1_summary.csv") is True
        assert is_valid_summary_name("eos4e40_v1.csv") is False
        assert is_valid_summary_name("eos4e40_v1_info.csv") is False
        assert is_valid_summary_name("eos4e40_v1_columns.csv") is False

    def test_is_valid_name_rejects_summary_sidecar(self):
        assert is_valid_name("eos4e40_v1_summary.csv") is False

    def test_make_summary_name_no_prefix(self):
        assert make_summary_name("eos4e40", "v1") == "eos4e40_v1_summary.csv"

    def test_make_summary_name_with_prefix(self):
        assert (
            make_summary_name("eos4e40", "v1", prefix="example")
            == "example_eos4e40_v1_summary.csv"
        )

    def test_make_info_name_no_prefix(self):
        assert make_info_name("eos4e40", "v1") == "eos4e40_v1_info.csv"

    def test_make_info_name_with_prefix(self):
        assert make_info_name("eos4e40", "v1", prefix="example") == "example_eos4e40_v1_info.csv"

    def test_make_columns_name_no_prefix(self):
        assert make_columns_name("eos7m30", "v2") == "eos7m30_v2_columns.csv"

    def test_make_columns_name_with_prefix(self):
        assert (
            make_columns_name("eos4e40", "v1", prefix="260313_gardp")
            == "260313_gardp_eos4e40_v1_columns.csv"
        )

    def test_make_info_name_bad_model_id(self):
        with pytest.raises(ValueError):
            make_info_name("badid", "v1")

    def test_make_info_name_bad_version(self):
        with pytest.raises(ValueError):
            make_info_name("eos4e40", "1")

    def test_make_info_name_bad_prefix(self):
        with pytest.raises(ValueError):
            make_info_name("eos4e40", "v1", prefix="bad prefix!")


# ---------------------------------------------------------------------------
# CLI: info
# ---------------------------------------------------------------------------

class TestInfoCLI:

    @pytest.fixture()
    def fake_metadata(self, monkeypatch):
        """Stub hub.fetch_metadata to return a deterministic payload."""
        payload = {
            "Identifier": "eos4e40",
            "Slug": "chemprop-antibiotic",
            "Title": "Broad spectrum antibiotic activity",
            "Tags": ["Antibiotic", "E.coli"],
            "Description": None,
        }
        monkeypatch.setattr("eosframes.hub.fetch_metadata", lambda model_id: payload)
        return payload

    def test_print_only(self, tmp, fake_metadata):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["info", src])
        assert result.exit_code == 0, result.output
        # Pretty table shows the stubbed fields
        assert "chemprop-antibiotic" in result.output
        assert "Antibiotic | E.coli" in result.output  # list flattened
        # No sidecar file produced
        assert not os.path.exists(str(tmp / "eos4e40_v1_info.csv"))

    def test_write_sidecar(self, tmp, fake_metadata):
        src = str(tmp / "example_eos4e40_v1.csv")
        _write_csv(src)
        out = str(tmp / "example_eos4e40_v1_info.csv")
        result = CliRunner().invoke(main, ["info", src, "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert set(df.columns) == {"field", "value"}
        assert "Identifier" in df["field"].tolist()

    def test_bad_input_name_has_helpful_error(self, tmp, fake_metadata):
        src = str(tmp / "no_convention.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["info", src])
        assert result.exit_code != 0
        # The error should mention the convention and give an example.
        assert "naming convention" in result.output
        assert "eos4e40_v1" in result.output

    def test_bad_output_suffix_error(self, tmp, fake_metadata):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["info", src, "-o", str(tmp / "bogus.csv")])
        assert result.exit_code != 0
        assert "_info.csv" in result.output  # suggested filename referenced
        assert "eos4e40_v1_info.csv" in result.output

    def test_output_model_id_mismatch_error(self, tmp, fake_metadata):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        wrong = str(tmp / "eos7m30_v1_info.csv")
        result = CliRunner().invoke(main, ["info", src, "-o", wrong])
        assert result.exit_code != 0
        assert "Model ID mismatch" in result.output
        assert "eos4e40_v1_info.csv" in result.output

    def test_output_version_mismatch_error(self, tmp, fake_metadata):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        wrong = str(tmp / "eos4e40_v2_info.csv")
        result = CliRunner().invoke(main, ["info", src, "-o", wrong])
        assert result.exit_code != 0
        assert "Version mismatch" in result.output

    def test_output_exists_error(self, tmp, fake_metadata):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        out = str(tmp / "eos4e40_v1_info.csv")
        # Pre-create the output to trigger the refuse-overwrite guard.
        with open(out, "w") as fh:
            fh.write("existing\n")
        result = CliRunner().invoke(main, ["info", src, "-o", out])
        assert result.exit_code != 0
        assert "already exists" in result.output


# ---------------------------------------------------------------------------
# CLI: columns
# ---------------------------------------------------------------------------

class TestColumnsCLI:

    @pytest.fixture()
    def fake_columns(self, monkeypatch):
        """Stub hub.fetch_columns to return a small DataFrame."""
        df = pd.DataFrame({
            "name":        ["inhibition_50um"],
            "type":        ["float"],
            "direction":   ["high"],
            "description": ["Probability of inhibiting the growth of E.coli at 50 uM"],
        })
        monkeypatch.setattr(
            "eosframes.hub.fetch_columns",
            lambda model_id, version: df.copy(),
        )
        return df

    def test_print_only(self, tmp, fake_columns):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["columns", src])
        assert result.exit_code == 0, result.output
        assert "inhibition_50um" in result.output
        assert not os.path.exists(str(tmp / "eos4e40_v1_columns.csv"))

    def test_write_sidecar(self, tmp, fake_columns):
        src = str(tmp / "example_eos4e40_v1.csv")
        _write_csv(src)
        out = str(tmp / "example_eos4e40_v1_columns.csv")
        result = CliRunner().invoke(main, ["columns", src, "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert "name" in df.columns
        assert df.iloc[0]["name"] == "inhibition_50um"

    def test_bad_output_suffix_error(self, tmp, fake_columns):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["columns", src, "-o", str(tmp / "bad.csv")])
        assert result.exit_code != 0
        assert "eos4e40_v1_columns.csv" in result.output

    def test_output_model_id_mismatch_error(self, tmp, fake_columns):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        wrong = str(tmp / "eos3804_v1_columns.csv")
        result = CliRunner().invoke(main, ["columns", src, "-o", wrong])
        assert result.exit_code != 0
        assert "Model ID mismatch" in result.output

    def test_bad_input_name_error(self, tmp, fake_columns):
        src = str(tmp / "no_convention.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["columns", src])
        assert result.exit_code != 0
        assert "naming convention" in result.output


# ---------------------------------------------------------------------------
# CLI: split
# ---------------------------------------------------------------------------

class TestSplit:

    def test_basic(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=10)
        out = str(tmp / "chunks")

        result = CliRunner().invoke(main, ["split", src, "-o", out, "--chunksize", "3"])
        assert result.exit_code == 0, result.output
        files = sorted(os.listdir(out))
        assert len(files) == 4   # ceil(10/3) = 4
        assert files[0] == "chunk_000.csv"

    def test_zfill_3_digit(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=5)
        out = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", src, "-o", out, "--chunksize", "1"])
        files = os.listdir(out)
        # 5 chunks → 3-digit padding
        assert all(f.startswith("chunk_") and len(f) == 13 for f in files)  # chunk_000.csv

    def test_output_folder_exists_error(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src)
        out = str(tmp / "exists")
        os.makedirs(out)
        result = CliRunner().invoke(main, ["split", src, "-o", out])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_row_count_preserved(self, tmp):
        src = str(tmp / "input.csv")
        _write_csv(src, n_rows=7)
        out = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", src, "-o", out, "--chunksize", "3"])
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
        result = CliRunner().invoke(main, ["convert", src, "-o", dst])
        assert result.exit_code == 0, result.output
        assert os.path.exists(dst)

    def test_h5_to_csv(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        h5 = str(tmp / "eos4e40_v1.h5")
        CliRunner().invoke(main, ["convert", src, "-o", h5])
        dst = str(tmp / "eos4e40_v2.csv")
        result = CliRunner().invoke(main, ["convert", h5, "-o", dst])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(dst)
        assert list(df["key"]) == [f"k{i}" for i in range(5)]

    def test_chunks_to_csv(self, tmp):
        raw = str(tmp / "input.csv")
        _write_csv(raw, n_rows=6)
        chunks = str(tmp / "chunks")
        CliRunner().invoke(main, ["split", raw, "-o", chunks, "--chunksize", "2"])
        dst = str(tmp / "eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["convert", chunks, "-o", dst])
        assert result.exit_code == 0, result.output
        assert len(pd.read_csv(dst)) == 6

    def test_bad_output_name_error(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["convert", src, "-o", str(tmp / "output.h5")])
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

    # --- Mode A (eosmix) ---

    def test_mode_a_eosmix(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "project_eosmix.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        # Columns suffixed with _<model_id>_<version>
        assert "score_eos4e40_v1" in df.columns
        assert "activity_eos3804_v1" in df.columns
        assert df.columns.tolist().count("input") == 1

    def test_mode_a_no_prefix(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "eosmix.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code == 0, result.output

    # --- Mode B (explicit) ---

    def test_mode_b_explicit(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "eos4e40_v1_eos3804_v1.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        # Column names stay bare (provenance is in the filename)
        assert "score" in df.columns
        assert "activity" in df.columns

    def test_mode_b_with_prefix(self, tmp):
        f1, f2 = self._two_files(tmp)
        out = str(tmp / "project_eos4e40_v1_eos3804_v1.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code == 0, result.output

    def test_mode_b_order_mismatch(self, tmp):
        f1, f2 = self._two_files(tmp)
        # -i order is (eos4e40, eos3804) but filename swaps them
        out = str(tmp / "eos3804_v1_eos4e40_v1.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code != 0
        assert "Model order mismatch" in result.output

    # --- Invalid output names / other errors ---

    def test_bad_output_name_suggests_both_modes(self, tmp):
        f1, f2 = self._two_files(tmp)
        result = CliRunner().invoke(
            main, ["stack", f1, f2, "-o", str(tmp / "stacked.csv")]
        )
        assert result.exit_code != 0
        # Both mode names surfaced in the error
        assert "eosmix" in result.output
        assert "Mode A" in result.output and "Mode B" in result.output

    def test_duplicate_model_version_error(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        _write_csv(f1)
        # Same (model_id, version) twice — should be rejected at hstack layer.
        out = str(tmp / "project_eosmix.csv")
        result = CliRunner().invoke(main, ["stack", f1, f1, "-o", out])
        assert result.exit_code != 0
        assert "Duplicate" in result.output or "appears more than once" in result.output.lower()

    def test_input_mismatch_error(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        f2 = str(tmp / "eos3804_v1.csv")
        pd.DataFrame({"key": ["k0"], "input": ["mol0"], "score": [0.1]}).to_csv(f1, index=False)
        pd.DataFrame({"key": ["k1"], "input": ["mol1"], "score": [0.2]}).to_csv(f2, index=False)
        out = str(tmp / "eosmix.csv")
        result = CliRunner().invoke(main, ["stack", f1, f2, "-o", out])
        assert result.exit_code != 0
        assert "mismatch" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: unstack
# ---------------------------------------------------------------------------

class TestUnstack:

    def _two_model_files(self, tmp):
        f1 = str(tmp / "eos4e40_v1.csv")
        f2 = str(tmp / "eos3804_v1.csv")
        pd.DataFrame({
            "key":   ["k0", "k1", "k2"],
            "input": ["mol0", "mol1", "mol2"],
            "score": [0.1, 0.2, 0.3],
        }).to_csv(f1, index=False)
        pd.DataFrame({
            "key":      ["k0", "k1", "k2"],
            "input":    ["mol0", "mol1", "mol2"],
            "activity": [1.0, 2.0, 3.0],
        }).to_csv(f2, index=False)
        return f1, f2

    def test_mode_a_round_trip(self, tmp):
        f1, f2 = self._two_model_files(tmp)
        stacked = str(tmp / "project_eosmix.csv")
        r = CliRunner().invoke(main, ["stack", f1, f2, "-o", stacked])
        assert r.exit_code == 0, r.output

        out_folder = str(tmp / "split_a")
        r = CliRunner().invoke(main, ["unstack", stacked, "-o", out_folder])
        assert r.exit_code == 0, r.output

        # Prefix inherited from the input; one file per model.
        produced = sorted(os.listdir(out_folder))
        assert produced == ["project_eos3804_v1.csv", "project_eos4e40_v1.csv"]
        df4 = pd.read_csv(os.path.join(out_folder, "project_eos4e40_v1.csv"))
        assert list(df4.columns) == ["key", "input", "score"]
        assert df4["score"].tolist() == [0.1, 0.2, 0.3]

    def test_mode_a_no_prefix(self, tmp):
        f1, f2 = self._two_model_files(tmp)
        stacked = str(tmp / "eosmix.csv")
        CliRunner().invoke(main, ["stack", f1, f2, "-o", stacked])

        out_folder = str(tmp / "split_no_prefix")
        r = CliRunner().invoke(main, ["unstack", stacked, "-o", out_folder])
        assert r.exit_code == 0, r.output
        # No prefix in the inputs → outputs are bare canonical names.
        produced = sorted(os.listdir(out_folder))
        assert produced == ["eos3804_v1.csv", "eos4e40_v1.csv"]

    def test_mode_b_uses_hub_run_columns(self, tmp, monkeypatch):
        f1, f2 = self._two_model_files(tmp)
        stacked = str(tmp / "proj_eos4e40_v1_eos3804_v1.csv")
        r = CliRunner().invoke(main, ["stack", f1, f2, "-o", stacked])
        assert r.exit_code == 0, r.output

        # Mock the hub so unstack doesn't hit the network.
        def fake_fetch(model_id, version):
            cols = {"eos4e40": ["score"], "eos3804": ["activity"]}[model_id]
            return pd.DataFrame({"name": cols})
        monkeypatch.setattr("eosframes.hub.fetch_columns", fake_fetch)

        out_folder = str(tmp / "split_b")
        r = CliRunner().invoke(main, ["unstack", stacked, "-o", out_folder])
        assert r.exit_code == 0, r.output
        produced = sorted(os.listdir(out_folder))
        assert produced == ["proj_eos3804_v1.csv", "proj_eos4e40_v1.csv"]
        df4 = pd.read_csv(os.path.join(out_folder, "proj_eos4e40_v1.csv"))
        assert list(df4.columns) == ["key", "input", "score"]

    def test_mode_b_ambiguous_column(self, tmp, monkeypatch):
        f1, f2 = self._two_model_files(tmp)
        stacked = str(tmp / "eos4e40_v1_eos3804_v1.csv")
        CliRunner().invoke(main, ["stack", f1, f2, "-o", stacked])

        # Both run_columns include a column named 'score' — ambiguous since
        # the stacked file has a 'score' column that can't be assigned.
        def fake_fetch(model_id, version):
            return pd.DataFrame({"name": ["score", "activity"]})
        monkeypatch.setattr("eosframes.hub.fetch_columns", fake_fetch)

        out_folder = str(tmp / "split_ambig")
        r = CliRunner().invoke(main, ["unstack", stacked, "-o", out_folder])
        assert r.exit_code != 0
        assert "Ambiguous" in r.output

    def test_invalid_input_name(self, tmp):
        bogus = str(tmp / "not_a_stack.csv")
        pd.DataFrame({"key": ["k0"], "input": ["m0"], "x": [1]}).to_csv(bogus, index=False)
        out_folder = str(tmp / "split_bad")
        r = CliRunner().invoke(main, ["unstack", bogus, "-o", out_folder])
        assert r.exit_code != 0
        assert "stack naming convention" in r.output

    def test_existing_output_folder_error(self, tmp):
        f1, f2 = self._two_model_files(tmp)
        stacked = str(tmp / "project_eosmix.csv")
        CliRunner().invoke(main, ["stack", f1, f2, "-o", stacked])
        existing = str(tmp / "already_there")
        os.makedirs(existing)
        r = CliRunner().invoke(main, ["unstack", stacked, "-o", existing])
        assert r.exit_code != 0
        assert "already exists" in r.output


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
        result = CliRunner().invoke(main, ["dedupe", src, "-o", dst])
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
        CliRunner().invoke(main, ["dedupe", src, "-o", dst])
        assert len(pd.read_csv(dst)) == 4

    def test_bad_output_name_error(self, tmp):
        src = str(tmp / "eos4e40_v1_raw.csv")
        _write_csv(src)
        result = CliRunner().invoke(main, ["dedupe", src, "-o", str(tmp / "out.csv")])
        assert result.exit_code != 0
        assert "naming convention" in result.output


# ---------------------------------------------------------------------------
# CLI: summary
# ---------------------------------------------------------------------------

class TestSummary:

    def _write_eos4e40(self, path: str, keys, scores=None, with_key: bool = True) -> None:
        """Write a minimal Ersilia-format CSV with configurable keys."""
        n = len(keys)
        data = {"input": [f"mol{i}" for i in range(n)], "score": scores or [0.1 * i for i in range(n)]}
        if with_key:
            data = {"key": list(keys), **data}
        pd.DataFrame(data).to_csv(path, index=False)

    def test_print_only_no_duplicates(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        self._write_eos4e40(src, keys=[f"k{i}" for i in range(5)])
        result = CliRunner().invoke(main, ["summary", src])
        assert result.exit_code == 0, result.output
        # Header fields are present
        assert "Columns:" in result.output
        assert "2 meta + 1 features" in result.output  # key + input + score
        assert "Unique keys:" in result.output
        assert "Duplicates:" in result.output
        # No duplicates path
        assert "no" in result.output  # green "no" on Duplicates line
        assert "Missing data:" in result.output

    def test_print_duplicates_detected(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        self._write_eos4e40(
            src, keys=["k0", "k1", "k0", "k2"], scores=[1.0, 2.0, 3.0, 4.0]
        )
        result = CliRunner().invoke(main, ["summary", src])
        assert result.exit_code == 0, result.output
        # Unique count < total, and yes line with count of 1 duplicate.
        assert "Unique keys:" in result.output
        assert "yes (1" in result.output

    def test_missing_data_flag(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        pd.DataFrame({
            "key":   ["k0", "k1", "k2"],
            "input": ["m0", "m1", "m2"],
            "score": [1.0, None, 3.0],
        }).to_csv(src, index=False)
        result = CliRunner().invoke(main, ["summary", src])
        assert result.exit_code == 0, result.output
        assert "Missing data:" in result.output
        assert "yes" in result.output  # flagged

    def test_write_sidecar_csv(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        self._write_eos4e40(src, keys=[f"k{i}" for i in range(3)])
        out = str(tmp / "eos4e40_v1_summary.csv")
        result = CliRunner().invoke(main, ["summary", src, "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert set(df.columns) == {"column", "dtype", "missing", "min", "mean", "max"}
        assert df["column"].tolist() == ["score"]
        assert df.iloc[0]["missing"] == 0

    def test_write_sidecar_bad_suffix(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        self._write_eos4e40(src, keys=["k0", "k1"])
        result = CliRunner().invoke(main, ["summary", src, "-o", str(tmp / "bogus.csv")])
        assert result.exit_code != 0
        assert "_summary" in result.output
        assert "eos4e40_v1_summary.csv" in result.output

    def test_write_sidecar_model_id_mismatch(self, tmp):
        src = str(tmp / "eos4e40_v1.csv")
        self._write_eos4e40(src, keys=["k0", "k1"])
        wrong = str(tmp / "eos7m30_v1_summary.csv")
        result = CliRunner().invoke(main, ["summary", src, "-o", wrong])
        assert result.exit_code != 0
        assert "Model ID mismatch" in result.output
