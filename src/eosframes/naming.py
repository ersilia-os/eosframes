import re
import os
from typing import Optional, Dict

from .utils.utils import is_model_id_valid

# Matches eos<digit><3 alphanumeric>_v<digits> anywhere in the stem (allows leading prefix)
_STEM_RE = re.compile(r'(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)$')

VALID_EXTENSIONS = {"csv", "h5"}


def parse_name(filename: str) -> Optional[Dict]:
    """
    Parse a filename or directory name and return structured components.

    Recognizes:
      - eos4e40_v1.csv       → name_type="csv"
      - eos4e40_v1.h5        → name_type="h5"
      - eos4e40_v1_chunks    → name_type="chunks_dir"
      - eos4e40_v1_chunks/   → name_type="chunks_dir"

    Parameters
    ----------
    filename : str
        Basename or full path. Only the basename is used for matching.

    Returns
    -------
    dict with keys: model_id, version, extension, name_type
    Returns None if the filename does not match the convention.
    """
    basename = os.path.basename(filename.rstrip("/"))

    # Check for chunks directory
    if basename.endswith("_chunks"):
        stem = basename[: -len("_chunks")]
        m = _STEM_RE.search(stem)
        if m and is_model_id_valid(m.group(1)):
            return {
                "model_id": m.group(1),
                "version": m.group(2),
                "extension": None,
                "name_type": "chunks_dir",
            }
        return None

    # Check for file with extension
    if "." not in basename:
        return None
    stem, ext = basename.rsplit(".", 1)
    if ext not in VALID_EXTENSIONS:
        return None
    m = _STEM_RE.search(stem)
    if m and is_model_id_valid(m.group(1)):
        return {
            "model_id": m.group(1),
            "version": m.group(2),
            "extension": ext,
            "name_type": ext,
        }
    return None


def make_output_name(model_id: str, version: str, ext: str) -> str:
    """
    Build a canonical output filename.

    Parameters
    ----------
    model_id : str
        e.g. "eos4e40"
    version : str
        e.g. "v1"
    ext : str
        "csv" or "h5" (without leading dot)

    Returns
    -------
    str
        e.g. "eos4e40_v1.csv"

    Raises
    ------
    ValueError
    """
    if not is_model_id_valid(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    if not re.match(r'^v\d+$', version):
        raise ValueError(f"Invalid version: {version!r}. Expected format: v1, v2, ...")
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {ext!r}. Must be one of {VALID_EXTENSIONS}")
    return f"{model_id}_{version}.{ext}"


def make_chunks_dir_name(model_id: str, version: str) -> str:
    """
    Build a canonical chunks directory name.

    Returns
    -------
    str
        e.g. "eos4e40_v1_chunks"
    """
    if not is_model_id_valid(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    if not re.match(r'^v\d+$', version):
        raise ValueError(f"Invalid version: {version!r}. Expected format: v1, v2, ...")
    return f"{model_id}_{version}_chunks"


def get_version_from_path(path: str) -> Optional[str]:
    """
    Extract the version token (e.g. "v1") from a filename or path.

    Parameters
    ----------
    path : str

    Returns
    -------
    str or None
    """
    parsed = parse_name(path)
    if parsed is None:
        return None
    return parsed["version"]


def is_valid_name(path: str) -> bool:
    """
    Return True if the basename of `path` follows the naming convention
    for any recognized output type (csv, h5, or chunks_dir).
    """
    return parse_name(path) is not None
