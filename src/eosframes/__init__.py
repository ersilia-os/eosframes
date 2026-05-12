"""``eosframes`` — utilities for Ersilia model output dataframes.

Split, assemble, convert, stack, scale, and summarise tabular outputs from
the `Ersilia Model Hub <https://github.com/ersilia-os/ersilia>`_. The CLI
entry point is ``eosframes`` (registered as a ``poetry`` script); the same
operations are available as Python functions imported from this package.

The architecture is layered: ``cli`` → ``ops`` / ``scale`` →
``read`` / ``write`` / ``stack`` / ``naming`` / ``hub``. Every
DataFrame that flows through the library carries two loose attributes —
``df.model_id`` (always) and ``df.version`` (when the filename encodes
it) — and write paths cross-validate them against the destination
filename. See ``CLAUDE.md`` for the full attribute and naming contracts.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("eosframes")
except PackageNotFoundError:
    # Not installed (e.g. running from a source checkout without
    # `pip install -e .`). Keep the package importable but signal the
    # situation — scaler files will fail the version-match check.
    __version__ = "0.0.0+unknown"

from .exceptions import EosframesError
from .hub import fetch_columns, fetch_metadata
from .logger import get_logger, set_verbosity
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
from .read import read_chunked_csvs, read_csv, read_h5
from .scale import fit, fit_file, transform, transform_file
from .stack import hstack, vstack
from .write import write_chunked_csvs, write_csv, write_h5

__all__ = [
    "__version__",
    # Low-level I/O
    "read_csv",
    "read_h5",
    "read_chunked_csvs",
    "write_csv",
    "write_h5",
    "write_chunked_csvs",
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
    "fit",
    "transform",
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
    "set_verbosity",
    "EosframesError",
]
