"""
Exhaustive tests for eosframes.

Uses static reference data in data/ for realistic end-to-end coverage:
  data/example_eos4e40_v1.csv  — 100 rows, 1 feature column (inhibition_50um)
  data/example_eos7m30_v1.csv  — 100 rows, 49 ADMET feature columns

Run with:
    pytest tests/test_eosframes.py -v
"""

import json
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import eosframes
from eosframes import (
    EosframesError,
    append_files,
    apply_scaler,
    convert_file,
    dedupe_file,
    fetch_columns,
    fetch_metadata,
    fit_scaler,
    transform_file,
    hstack,
    is_valid_name,
    make_chunks_dir_name,
    make_output_name,
    parse_name,
    read_csv,
    read_h5,
    split_csv,
    stack_files,
    vstack,
    write_csv,
    write_h5,
)
from eosframes.cli import main
from eosframes.naming import get_version_from_path

# ---------------------------------------------------------------------------
# Paths to static reference data
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EOS4E40_CSV = os.path.join(DATA_DIR, "example_eos4e40_v1.csv")
EOS7M30_CSV = os.path.join(DATA_DIR, "example_eos7m30_v1.csv")

EOS4E40_ROWS = 100
EOS4E40_FEATURES = ["inhibition_50um"]
EOS7M30_ROWS = 100
EOS7M30_N_FEATURES = 49


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def df4e40():
    """eos4e40 reference DataFrame with model_id set."""
    df = pd.read_csv(EOS4E40_CSV)
    df.model_id = "eos4e40"
    return df


@pytest.fixture()
def df7m30():
    """eos7m30 reference DataFrame with model_id set."""
    df = pd.read_csv(EOS7M30_CSV)
    df.model_id = "eos7m30"
    return df


def _runner():
    return CliRunner()


# ===========================================================================
# 1. Naming convention
# ===========================================================================

class TestNaming:

    def test_parse_canonical_csv(self):
        r = parse_name("eos4e40_v1.csv")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "csv", "name_type": "csv"}

    def test_parse_canonical_h5(self):
        r = parse_name("eos4e40_v1.h5")
        assert r == {"model_id": "eos4e40", "version": "v1", "extension": "h5", "name_type": "h5"}

    def test_parse_chunks_dir(self):
        r = parse_name("eos4e40_v1_chunks")
        assert r["name_type"] == "chunks_dir"
        assert r["extension"] is None

    def test_parse_chunks_dir_trailing_slash(self):
        assert parse_name("eos4e40_v1_chunks/")["name_type"] == "chunks_dir"

    def test_parse_with_prefix(self):
        r = parse_name("260313_gardp_eos4e40_v1.csv")
        assert r["model_id"] == "eos4e40"
        assert r["version"] == "v1"

    def test_parse_with_prefix_chunks(self):
        r = parse_name("project_eos7m30_v2_chunks")
        assert r["model_id"] == "eos7m30"
        assert r["version"] == "v2"
        assert r["name_type"] == "chunks_dir"

    def test_parse_full_path(self):
        r = parse_name("/some/dir/eos7m30_v1.h5")
        assert r["model_id"] == "eos7m30"

    def test_parse_no_version_returns_none(self):
        assert parse_name("eos4e40.csv") is None

    def test_parse_extra_token_returns_none(self):
        assert parse_name("eos4e40_v1_extra.csv") is None

    def test_parse_arbitrary_name_returns_none(self):
        assert parse_name("output.csv") is None

    def test_is_valid_true(self):
        for name in ("eos4e40_v1.csv", "eos3804_v2.h5", "eos4e40_v1_chunks",
                     "date_eos4e40_v1.csv"):
            assert is_valid_name(name), f"expected valid: {name}"

    def test_is_valid_false(self):
        for name in ("results.csv", "eos4e40.csv", "eos4e40_v1_extra.csv"):
            assert not is_valid_name(name), f"expected invalid: {name}"

    def test_make_output_name(self):
        assert make_output_name("eos4e40", "v1", "csv") == "eos4e40_v1.csv"
        assert make_output_name("eos7m30", "v3", "h5") == "eos7m30_v3.h5"

    def test_make_output_name_bad_id(self):
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

    def test_static_data_names_are_valid(self):
        assert is_valid_name(EOS4E40_CSV)
        assert is_valid_name(EOS7M30_CSV)


# ===========================================================================
# 2. Read / write
# ===========================================================================

class TestReadWrite:

    def test_read_csv_eos4e40(self):
        df = read_csv(EOS4E40_CSV)
        assert len(df) == EOS4E40_ROWS
        assert list(df.columns[:2]) == ["key", "input"]
        assert df.model_id == "eos4e40"
        assert EOS4E40_FEATURES[0] in df.columns

    def test_read_csv_eos7m30(self):
        df = read_csv(EOS7M30_CSV)
        assert len(df) == EOS7M30_ROWS
        assert df.model_id == "eos7m30"
        feat_cols = [c for c in df.columns if c not in {"key", "input"}]
        assert len(feat_cols) == EOS7M30_N_FEATURES

    def test_round_trip_csv(self, tmp, df4e40):
        path = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, path)
        df2 = read_csv(path)
        pd.testing.assert_frame_equal(df4e40.reset_index(drop=True), df2.reset_index(drop=True))

    def test_round_trip_h5_eos4e40(self, tmp, df4e40):
        path = str(tmp / "eos4e40_v1.h5")
        write_h5(df4e40, path, dtype=np.float32)
        df2 = read_h5(path)
        assert df2.model_id == "eos4e40"
        assert len(df2) == EOS4E40_ROWS
        assert list(df2.columns) == list(df4e40.columns)
        # values should be close (float32 precision)
        np.testing.assert_allclose(
            df2["inhibition_50um"].values,
            df4e40["inhibition_50um"].values,
            rtol=1e-4,
        )

    def test_round_trip_h5_eos7m30(self, tmp, df7m30):
        path = str(tmp / "eos7m30_v1.h5")
        write_h5(df7m30, path, dtype=np.float32)
        df2 = read_h5(path)
        assert len(df2) == EOS7M30_ROWS
        feat_cols = [c for c in df2.columns if c not in {"key", "input"}]
        assert len(feat_cols) == EOS7M30_N_FEATURES

    def test_h5_has_correct_datasets(self, tmp, df4e40):
        path = str(tmp / "eos4e40_v1.h5")
        write_h5(df4e40, path, dtype=np.float32)
        with h5py.File(path, "r") as f:
            assert set(f.keys()) >= {"key", "input", "features", "values"}
            assert f["values"].shape == (EOS4E40_ROWS, len(EOS4E40_FEATURES))
            assert list(f["features"][:]) == [b"inhibition_50um"]

    def test_write_csv_wrong_model_id_raises(self, tmp, df4e40):
        path = str(tmp / "eos7m30_v1.csv")
        with pytest.raises(Exception, match="model"):
            write_csv(df4e40, path)

    def test_write_csv_no_naming_convention_raises(self, tmp, df4e40):
        path = str(tmp / "output.csv")
        with pytest.raises(Exception):
            write_csv(df4e40, path)


# ===========================================================================
# 3. Split
# ===========================================================================

class TestSplit:

    def test_split_eos4e40(self, tmp):
        out = str(tmp / "chunks")
        n = split_csv(EOS4E40_CSV, out, chunksize=10)
        assert n == 10  # 100 rows / 10
        files = sorted(os.listdir(out))
        assert len(files) == 10
        assert files[0] == "chunk_000.csv"

    def test_split_preserves_header(self, tmp):
        out = str(tmp / "chunks")
        split_csv(EOS4E40_CSV, out, chunksize=10)
        for fname in os.listdir(out):
            df = pd.read_csv(os.path.join(out, fname))
            assert list(df.columns[:2]) == ["key", "input"]

    def test_split_row_count_preserved(self, tmp):
        out = str(tmp / "chunks")
        split_csv(EOS4E40_CSV, out, chunksize=7)
        total = sum(len(pd.read_csv(os.path.join(out, f))) for f in os.listdir(out))
        assert total == EOS4E40_ROWS

    def test_split_eos7m30_large_chunksize(self, tmp):
        out = str(tmp / "chunks")
        n = split_csv(EOS7M30_CSV, out, chunksize=100)
        assert n == 1

    def test_split_existing_folder_raises(self, tmp):
        out = str(tmp / "chunks")
        os.makedirs(out)
        with pytest.raises(EosframesError, match="already exists"):
            split_csv(EOS4E40_CSV, out)

    def test_split_6digit_padding(self, tmp):
        # manufacture a file with enough rows to trigger 6-digit padding (>999 chunks)
        # we fake it by using chunksize=1 and a 1000-row file
        big = str(tmp / "big.csv")
        pd.DataFrame({"key": range(1000), "input": range(1000), "v": range(1000)}).to_csv(big, index=False)
        out = str(tmp / "chunks")
        split_csv(big, out, chunksize=1)
        files = os.listdir(out)
        assert all(len(f) == len("chunk_000000.csv") for f in files)

    def test_cli_split(self, tmp):
        out = str(tmp / "chunks")
        result = _runner().invoke(main, ["split", EOS4E40_CSV, out, "--chunksize", "10"])
        assert result.exit_code == 0, result.output
        assert len(os.listdir(out)) == 10


# ===========================================================================
# 4. Convert
# ===========================================================================

class TestConvert:

    def test_csv_to_h5(self, tmp):
        dst = str(tmp / "eos4e40_v1.h5")
        convert_file(EOS4E40_CSV, dst)
        df = read_h5(dst)
        assert len(df) == EOS4E40_ROWS
        assert df.model_id == "eos4e40"

    def test_h5_to_csv(self, tmp):
        h5 = str(tmp / "eos4e40_v1.h5")
        convert_file(EOS4E40_CSV, h5)
        dst = str(tmp / "out_eos4e40_v1.csv")
        convert_file(h5, dst)
        df = read_csv(dst)
        assert len(df) == EOS4E40_ROWS
        assert list(df["key"]) == list(pd.read_csv(EOS4E40_CSV)["key"])

    def test_chunks_folder_to_csv(self, tmp):
        chunks = str(tmp / "chunks")
        split_csv(EOS4E40_CSV, chunks, chunksize=10)
        dst = str(tmp / "eos4e40_v1.csv")
        convert_file(chunks, dst)
        assert len(pd.read_csv(dst)) == EOS4E40_ROWS

    def test_chunks_folder_to_h5(self, tmp):
        chunks = str(tmp / "chunks")
        split_csv(EOS7M30_CSV, chunks, chunksize=10)
        dst = str(tmp / "eos7m30_v1.h5")
        convert_file(chunks, dst)
        df = read_h5(dst)
        assert len(df) == EOS7M30_ROWS

    def test_bad_output_name_raises(self, tmp):
        with pytest.raises(EosframesError, match="naming convention"):
            convert_file(EOS4E40_CSV, str(tmp / "output.h5"))

    def test_cli_csv_to_h5(self, tmp):
        dst = str(tmp / "eos4e40_v1.h5")
        result = _runner().invoke(main, ["convert", EOS4E40_CSV, dst])
        assert result.exit_code == 0, result.output
        assert os.path.exists(dst)


# ===========================================================================
# 5. Stack (horizontal)
# ===========================================================================

class TestStack:

    def test_stack_two_models(self, tmp, df4e40, df7m30):
        # write both to tmp
        p4 = str(tmp / "eos4e40_v1.csv")
        p7 = str(tmp / "eos7m30_v1.csv")
        write_csv(df4e40, p4)
        write_csv(df7m30, p7)
        out = str(tmp / "stacked.csv")
        stack_files([p4, p7], out, suffix=True)
        df = pd.read_csv(out)
        # key and input once each
        assert df.columns.tolist().count("key") == 1
        assert df.columns.tolist().count("input") == 1
        # suffixed columns
        assert "inhibition_50um.eos4e40" in df.columns
        assert "molecular_weight.eos7m30" in df.columns
        assert len(df) == EOS4E40_ROWS

    def test_stack_no_suffix(self, tmp, df4e40, df7m30):
        p4 = str(tmp / "eos4e40_v1.csv")
        p7 = str(tmp / "eos7m30_v1.csv")
        write_csv(df4e40, p4)
        write_csv(df7m30, p7)
        out = str(tmp / "stacked.csv")
        stack_files([p4, p7], out, suffix=False)
        df = pd.read_csv(out)
        assert "inhibition_50um" in df.columns
        assert "molecular_weight" in df.columns

    def test_stack_total_feature_count(self, tmp, df4e40, df7m30):
        p4 = str(tmp / "eos4e40_v1.csv")
        p7 = str(tmp / "eos7m30_v1.csv")
        write_csv(df4e40, p4)
        write_csv(df7m30, p7)
        out = str(tmp / "stacked.csv")
        stack_files([p4, p7], out)
        df = pd.read_csv(out)
        n_feat = len(df.columns) - 2  # minus key, input
        assert n_feat == 1 + EOS7M30_N_FEATURES

    def test_stack_duplicate_model_raises(self, tmp, df4e40):
        p = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, p)
        with pytest.raises(EosframesError, match="more than once"):
            stack_files([p, p], str(tmp / "out.csv"))

    def test_stack_input_mismatch_raises(self, tmp, df4e40):
        p4a = str(tmp / "eos4e40_v1.csv")
        p4b = str(tmp / "eos7m30_v1.csv")
        write_csv(df4e40, p4a)
        df_diff = df4e40.copy()
        df_diff["input"] = df_diff["input"].str.upper()
        df_diff.model_id = "eos7m30"
        write_csv(df_diff, p4b)
        with pytest.raises(EosframesError, match="mismatch"):
            stack_files([p4a, p4b], str(tmp / "out.csv"))

    def test_hstack_api(self, df4e40, df7m30):
        result = hstack([df4e40, df7m30])
        assert "inhibition_50um.eos4e40" in result.columns
        assert len(result) == EOS4E40_ROWS

    def test_cli_stack(self, tmp, df4e40, df7m30):
        p4 = str(tmp / "eos4e40_v1.csv")
        p7 = str(tmp / "eos7m30_v1.csv")
        write_csv(df4e40, p4)
        write_csv(df7m30, p7)
        out = str(tmp / "stacked.csv")
        result = _runner().invoke(main, ["stack", p4, p7, "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)


# ===========================================================================
# 6. Append (vertical)
# ===========================================================================

class TestAppend:

    def _split_half(self, tmp, src_path, model_id):
        """Split src_path into two halves, return their paths."""
        df = pd.read_csv(src_path)
        half = len(df) // 2
        p1 = str(tmp / f"{model_id}_v1_b1.csv")
        p2 = str(tmp / f"{model_id}_v1_b2.csv")
        df.iloc[:half].to_csv(p1, index=False)
        df.iloc[half:].to_csv(p2, index=False)
        return p1, p2

    def test_append_eos4e40(self, tmp):
        p1, p2 = self._split_half(tmp, EOS4E40_CSV, "eos4e40")
        out = str(tmp / "eos4e40_v1.csv")
        append_files([p1, p2], out)
        df = pd.read_csv(out)
        assert len(df) == EOS4E40_ROWS

    def test_append_eos7m30(self, tmp):
        p1, p2 = self._split_half(tmp, EOS7M30_CSV, "eos7m30")
        out = str(tmp / "eos7m30_v1.csv")
        append_files([p1, p2], out)
        df = pd.read_csv(out)
        assert len(df) == EOS7M30_ROWS
        feat_cols = [c for c in df.columns if c not in {"key", "input"}]
        assert len(feat_cols) == EOS7M30_N_FEATURES

    def test_append_preserves_order(self, tmp):
        p1, p2 = self._split_half(tmp, EOS4E40_CSV, "eos4e40")
        out = str(tmp / "eos4e40_v1.csv")
        append_files([p1, p2], out)
        original_keys = list(pd.read_csv(EOS4E40_CSV)["key"])
        result_keys = list(pd.read_csv(out)["key"])
        assert result_keys == original_keys

    def test_append_model_mismatch_raises(self, tmp):
        p1 = str(tmp / "eos4e40_v1_b1.csv")
        p2 = str(tmp / "eos7m30_v1_b1.csv")
        pd.read_csv(EOS4E40_CSV).to_csv(p1, index=False)
        pd.read_csv(EOS7M30_CSV).to_csv(p2, index=False)
        with pytest.raises(EosframesError, match="mismatch"):
            append_files([p1, p2], str(tmp / "eos4e40_v1.csv"))

    def test_append_column_mismatch_raises(self, tmp):
        p1 = str(tmp / "eos4e40_v1_b1.csv")
        p2 = str(tmp / "eos4e40_v1_b2.csv")
        pd.read_csv(EOS4E40_CSV).to_csv(p1, index=False)
        df2 = pd.read_csv(EOS4E40_CSV).rename(columns={"inhibition_50um": "score"})
        df2.to_csv(p2, index=False)
        with pytest.raises(EosframesError, match="mismatch"):
            append_files([p1, p2], str(tmp / "eos4e40_v1.csv"))

    def test_vstack_api(self, df4e40):
        half = len(df4e40) // 2
        a, b = df4e40.iloc[:half].copy(), df4e40.iloc[half:].copy()
        a.model_id = b.model_id = "eos4e40"
        result = vstack([a, b])
        assert len(result) == EOS4E40_ROWS


# ===========================================================================
# 7. Dedupe
# ===========================================================================

class TestDedupe:

    def test_dedupe_removes_duplicates(self, tmp, df4e40):
        # introduce duplicates
        df_dup = pd.concat([df4e40, df4e40.iloc[:5]], ignore_index=True)
        src = str(tmp / "eos4e40_v1_raw.csv")
        df_dup.to_csv(src, index=False)
        dst = str(tmp / "eos4e40_v1.csv")
        before, after = dedupe_file(src, dst)
        assert before == EOS4E40_ROWS + 5
        assert after == EOS4E40_ROWS
        assert len(pd.read_csv(dst)) == EOS4E40_ROWS

    def test_dedupe_no_duplicates_unchanged(self, tmp):
        dst = str(tmp / "eos4e40_v1.csv")
        before, after = dedupe_file(EOS4E40_CSV, dst)
        assert before == after == EOS4E40_ROWS

    def test_dedupe_keeps_first_occurrence(self, tmp, df4e40):
        df_dup = pd.concat([df4e40, df4e40.iloc[:3].assign(inhibition_50um=99.0)], ignore_index=True)
        src = str(tmp / "eos4e40_v1_raw.csv")
        df_dup.to_csv(src, index=False)
        dst = str(tmp / "eos4e40_v1.csv")
        dedupe_file(src, dst)
        result = pd.read_csv(dst)
        # first three rows should retain original values, not 99.0
        assert (result.iloc[:3]["inhibition_50um"] != 99.0).all()

    def test_dedupe_bad_output_name_raises(self, tmp):
        with pytest.raises(EosframesError, match="naming convention"):
            dedupe_file(EOS4E40_CSV, str(tmp / "output.csv"))

    def test_cli_dedupe(self, tmp, df4e40):
        df_dup = pd.concat([df4e40, df4e40.iloc[:3]], ignore_index=True)
        src = str(tmp / "eos4e40_v1_raw.csv")
        df_dup.to_csv(src, index=False)
        dst = str(tmp / "eos4e40_v1.csv")
        result = _runner().invoke(main, ["dedupe", src, dst])
        assert result.exit_code == 0, result.output
        assert len(pd.read_csv(dst)) == EOS4E40_ROWS


# ===========================================================================
# 8. Scale — fit_scaler / apply_scaler (DataFrame API)
# ===========================================================================

class TestScalerDataFrame:

    def test_fit_scaler_returns_correct_keys(self, df4e40):
        params = fit_scaler(df4e40)
        assert set(params.keys()) == {"method", "columns", "skipped_columns", "parameters"}
        assert params["method"] == "standard"
        assert params["columns"] == EOS4E40_FEATURES
        assert params["skipped_columns"] == []

    def test_fit_scaler_mean_std(self, df4e40):
        params = fit_scaler(df4e40)
        p = params["parameters"]["inhibition_50um"]
        expected_mean = float(df4e40["inhibition_50um"].mean())
        expected_std  = float(df4e40["inhibition_50um"].std(ddof=0))
        assert abs(p["mean"] - expected_mean) < 1e-6
        assert abs(p["std"] - expected_std) < 1e-6

    def test_apply_scaler_zero_mean(self, df4e40):
        params = fit_scaler(df4e40)
        scaled = apply_scaler(df4e40, params)
        assert abs(scaled["inhibition_50um"].mean()) < 1e-6

    def test_apply_scaler_unit_std(self, df4e40):
        params = fit_scaler(df4e40)
        scaled = apply_scaler(df4e40, params)
        assert abs(scaled["inhibition_50um"].std(ddof=0) - 1.0) < 1e-6

    def test_apply_scaler_preserves_key_input(self, df4e40):
        params = fit_scaler(df4e40)
        scaled = apply_scaler(df4e40, params)
        pd.testing.assert_series_equal(df4e40["key"], scaled["key"])
        pd.testing.assert_series_equal(df4e40["input"], scaled["input"])

    def test_apply_scaler_many_features(self, df7m30):
        params = fit_scaler(df7m30)
        scaled = apply_scaler(df7m30, params)
        # all fitted columns should be roughly zero-mean
        for col in params["columns"]:
            assert abs(scaled[col].mean()) < 1e-4, f"{col} mean not near 0"

    def test_apply_scaler_column_mismatch_raises(self, df4e40, df7m30):
        params = fit_scaler(df4e40)
        with pytest.raises(EosframesError, match="Column mismatch"):
            apply_scaler(df7m30, params)

    def test_fit_scaler_skips_mostly_missing(self):
        df = pd.DataFrame({
            "key": ["k0", "k1", "k2", "k3"],
            "input": ["m0", "m1", "m2", "m3"],
            "good": [1.0, 2.0, 3.0, 4.0],
            "bad":  [np.nan, np.nan, np.nan, 1.0],  # 75% missing → skipped
        })
        df.model_id = "eos4e40"
        params = fit_scaler(df)
        assert "bad" in params["skipped_columns"]
        assert "good" in params["columns"]

    def test_fit_scaler_constant_column(self):
        df = pd.DataFrame({
            "key": ["k0", "k1"],
            "input": ["m0", "m1"],
            "flat": [1.0, 1.0],
        })
        df.model_id = "eos4e40"
        params = fit_scaler(df)
        scaled = apply_scaler(df, params)
        # constant column → std=0 → all zeros after scaling
        assert (scaled["flat"] == 0.0).all()


# ===========================================================================
# 9. Scale — transform_file (file API)
# ===========================================================================

class TestScalerFile:

    def test_transform_creates_csv(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        out = transform_file(src)
        assert os.path.exists(out)
        assert out.endswith(".csv")

    def test_transform_default_output_name(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        out = transform_file(src)
        assert out == str(tmp / "eos4e40_v1_scaled.csv")

    def test_transform_save_params_creates_json(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        t = json.load(open(json_path))
        assert t["model_id"] == "eos4e40"
        assert t["version"] == "v1"
        assert t["n_rows"] == EOS4E40_ROWS
        assert t["method"] == "standard"
        assert "fitted_at" in t
        assert "inhibition_50um" in t["parameters"]

    def test_transform_scaled_values_correct(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        out = transform_file(src)
        scaled = pd.read_csv(out)
        assert abs(scaled["inhibition_50um"].mean()) < 1e-6
        assert abs(scaled["inhibition_50um"].std(ddof=0) - 1.0) < 1e-6

    def test_transform_eos7m30(self, tmp, df7m30):
        src = str(tmp / "eos7m30_v1.csv")
        write_csv(df7m30, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        t = json.load(open(json_path))
        assert t["model_id"] == "eos7m30"
        assert t["n_rows"] == EOS7M30_ROWS
        assert len(t["columns"]) == EOS7M30_N_FEATURES

    def test_transform_existing_params_raises(self, tmp):
        json_path = str(tmp / "scaler.json")
        open(json_path, "w").close()
        with pytest.raises(EosframesError, match="already exists"):
            transform_file(EOS4E40_CSV, params=json_path, fit=True)

    def test_transform_bad_naming_raises(self, tmp):
        bad = str(tmp / "output.csv")
        pd.read_csv(EOS4E40_CSV).to_csv(bad, index=False)
        with pytest.raises(EosframesError, match="naming convention"):
            transform_file(bad)

    def test_transform_forward_pass_produces_correct_values(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        fit_out = transform_file(src, params=json_path, fit=True)
        applied_out = str(tmp / "applied.csv")
        transform_file(src, output_path=applied_out, params=json_path)
        s_fit = pd.read_csv(fit_out)["inhibition_50um"].values
        s_applied = pd.read_csv(applied_out)["inhibition_50um"].values
        np.testing.assert_allclose(s_fit, s_applied)

    def test_transform_forward_pass_model_id_mismatch_raises(self, tmp, df4e40, df7m30):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        eos7_src = str(tmp / "eos7m30_v1.csv")
        write_csv(df7m30, eos7_src)
        with pytest.raises(EosframesError, match="Model ID mismatch"):
            transform_file(eos7_src, output_path=str(tmp / "out.csv"), params=json_path)

    def test_transform_forward_pass_version_mismatch_raises(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        v2 = str(tmp / "eos4e40_v2.csv")
        write_csv(df4e40, v2)
        with pytest.raises(EosframesError, match="Version mismatch"):
            transform_file(v2, output_path=str(tmp / "out.csv"), params=json_path)

    def test_transform_forward_pass_column_mismatch_raises(self, tmp, df4e40, df7m30):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        df_bad = df7m30.copy()
        df_bad.model_id = "eos4e40"
        bad_named = str(tmp / "bad_eos4e40_v1.csv")
        df_bad.to_csv(bad_named, index=False)
        with pytest.raises(EosframesError, match="Column mismatch"):
            transform_file(bad_named, output_path=str(tmp / "out.csv"), params=json_path)

    def test_transform_forward_pass_output_to_h5(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        transform_file(src, params=json_path, fit=True)
        out = str(tmp / "scaled.h5")
        transform_file(src, output_path=out, params=json_path)
        assert os.path.exists(out)
        with h5py.File(out, "r") as f:
            assert "values" in f

    def test_cli_transform_fit(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        out = str(tmp / "scaled.csv")
        result = _runner().invoke(main, ["transform", src, "--params", json_path, "--fit", "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(json_path)
        t = json.load(open(json_path))
        assert t["model_id"] == "eos4e40"

    def test_cli_transform_forward_pass(self, tmp, df4e40):
        src = str(tmp / "eos4e40_v1.csv")
        write_csv(df4e40, src)
        json_path = str(tmp / "scaler.json")
        fit_out = str(tmp / "scaled_fit.csv")
        _runner().invoke(main, ["transform", src, "--params", json_path, "--fit", "-o", fit_out])
        out = str(tmp / "scaled_apply.csv")
        result = _runner().invoke(main, ["transform", src, "--params", json_path, "-o", out])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out)
