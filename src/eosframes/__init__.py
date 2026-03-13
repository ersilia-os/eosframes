from .exceptions import EosframesError
from .hub import fetch_columns, fetch_metadata
from .logger import get_logger
from .manipulate.stack import hstack, vstack
from .naming import (
    get_version_from_path,
    is_valid_name,
    make_chunks_dir_name,
    make_output_name,
    parse_name,
)
from .ops import append_files, convert_file, dedupe_file, split_csv, stack_files
from .read.read import read_chunked_csvs, read_csv, read_h5
from .scale import apply_scaler, apply_scaler_file, fit_scaler, fit_scaler_file
from .write.write import write_chunked_csvs, write_csv, write_h5, write_xlsx

__all__ = [
    # Low-level I/O
    "read_csv",
    "read_h5",
    "read_chunked_csvs",
    "write_csv",
    "write_h5",
    "write_chunked_csvs",
    "write_xlsx",
    # DataFrame operations
    "hstack",
    "vstack",
    # File-level operations
    "split_csv",
    "convert_file",
    "stack_files",
    "append_files",
    "dedupe_file",
    # GitHub / hub
    "fetch_metadata",
    "fetch_columns",
    # Scaling
    "fit_scaler",
    "apply_scaler",
    "fit_scaler_file",
    "apply_scaler_file",
    # Naming utilities
    "parse_name",
    "make_output_name",
    "make_chunks_dir_name",
    "get_version_from_path",
    "is_valid_name",
    # Misc
    "get_logger",
    "EosframesError",
]
