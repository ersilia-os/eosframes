from .exceptions import EosframesError
from .hub import fetch_columns, fetch_metadata
from .logger import get_logger
from .manipulate.stack import hstack, vstack
from .naming import (
    get_model_id_from_path,
    get_version_from_path,
    is_model_id_valid,
    is_valid_columns_name,
    is_valid_info_name,
    is_valid_name,
    is_valid_stack_explicit_name,
    is_valid_stack_mix_name,
    is_valid_summary_name,
    is_valid_transformer_name,
    make_chunks_dir_name,
    make_columns_name,
    make_info_name,
    make_output_name,
    make_stack_explicit_name,
    make_stack_mix_name,
    make_summary_name,
    make_transformer_name,
    parse_name,
    parse_stack_explicit_name,
    parse_stack_mix_name,
    parse_transformer_name,
)
from .ops import (
    append_files,
    convert_file,
    dedupe_file,
    split_csv,
    stack_files,
    unstack_file,
)
from .read.read import read_chunked_csvs, read_csv, read_h5
from .scale import apply_scaler, fit_file, fit_scaler, transform_file
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
    "unstack_file",
    "append_files",
    "dedupe_file",
    # GitHub / hub
    "fetch_metadata",
    "fetch_columns",
    # Scaling
    "fit_scaler",
    "apply_scaler",
    "fit_file",
    "transform_file",
    # Naming utilities
    "parse_name",
    "make_output_name",
    "make_chunks_dir_name",
    "make_info_name",
    "make_columns_name",
    "make_summary_name",
    "make_transformer_name",
    "make_stack_mix_name",
    "make_stack_explicit_name",
    "parse_stack_mix_name",
    "parse_stack_explicit_name",
    "parse_transformer_name",
    "get_version_from_path",
    "get_model_id_from_path",
    "is_valid_name",
    "is_valid_info_name",
    "is_valid_columns_name",
    "is_valid_summary_name",
    "is_valid_stack_mix_name",
    "is_valid_stack_explicit_name",
    "is_valid_transformer_name",
    "is_model_id_valid",
    # Misc
    "get_logger",
    "EosframesError",
]
