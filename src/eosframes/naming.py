"""Naming convention parsing and validation for Ersilia output files."""

import os
import re
from typing import Dict, Optional

# Matches the eos<digit><3 alphanumeric> pattern anywhere in a string
_MODEL_ID_RE = re.compile(r"(?<![A-Za-z0-9])eos\d[A-Za-z0-9]{3}(?![A-Za-z0-9])")

# Matches <model_id>_<version> at the end of a stem (allows leading prefix tokens)
_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)$")

VALID_EXTENSIONS = {"csv", "h5"}


def is_model_id_valid(model_id: str) -> bool:
    """Return True if *model_id* matches the Ersilia pattern ``eos<digit><3 alphanumeric>``.

    Parameters
    ----------
    model_id : str
        Candidate model identifier, e.g. ``"eos4e40"``.

    Returns
    -------
    bool
    """
    return bool(re.fullmatch(r"eos\d[A-Za-z0-9]{3}", model_id))


def get_model_id_from_path(path: str) -> Optional[str]:
    """Extract a model ID from a file or directory path.

    Unlike :func:`parse_name`, no version suffix is required — the function
    finds the first Ersilia model identifier anywhere in the basename.

    Parameters
    ----------
    path : str
        File path, directory path, or bare filename.

    Returns
    -------
    str or None
        The model identifier if found, otherwise ``None``.
    """
    basename = os.path.basename(path.rstrip("/\\"))
    m = _MODEL_ID_RE.search(basename)
    return m.group() if m else None


def parse_name(filename: str) -> Optional[Dict]:
    """Parse a filename or directory name and return structured components.

    Recognizes:

    * ``eos4e40_v1.csv``       → ``name_type="csv"``
    * ``eos4e40_v1.h5``        → ``name_type="h5"``
    * ``eos4e40_v1_chunks``    → ``name_type="chunks_dir"``
    * ``260313_gardp_eos4e40_v1.csv``  → prefix allowed before model_id

    Parameters
    ----------
    filename : str
        Basename or full path; only the basename is used for matching.

    Returns
    -------
    dict or None
        ``{"model_id", "version", "extension", "name_type"}`` on success,
        ``None`` if the filename does not match the convention.
    """
    basename = os.path.basename(filename.rstrip("/\\"))

    # Chunks directory
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

    # File with extension
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
    """Build a canonical output filename.

    Parameters
    ----------
    model_id : str
        e.g. ``"eos4e40"``
    version : str
        e.g. ``"v1"``
    ext : str
        ``"csv"`` or ``"h5"`` (without leading dot)

    Returns
    -------
    str
        e.g. ``"eos4e40_v1.csv"``

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if not is_model_id_valid(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    if not re.match(r"^v\d+$", version):
        raise ValueError(f"Invalid version: {version!r}. Expected format: v1, v2, ...")
    if ext not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {ext!r}. Must be one of {VALID_EXTENSIONS}"
        )
    return f"{model_id}_{version}.{ext}"


def make_chunks_dir_name(model_id: str, version: str) -> str:
    """Build a canonical chunks directory name.

    Returns
    -------
    str
        e.g. ``"eos4e40_v1_chunks"``

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    if not is_model_id_valid(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    if not re.match(r"^v\d+$", version):
        raise ValueError(f"Invalid version: {version!r}. Expected format: v1, v2, ...")
    return f"{model_id}_{version}_chunks"


def get_version_from_path(path: str) -> Optional[str]:
    """Extract the version token (e.g. ``"v1"``) from a filename or path.

    Parameters
    ----------
    path : str

    Returns
    -------
    str or None
    """
    parsed = parse_name(path)
    return parsed["version"] if parsed is not None else None


def is_valid_name(path: str) -> bool:
    """Return ``True`` if *path* follows the Ersilia naming convention.

    Accepts csv files, h5 files, and chunks directories.
    """
    return parse_name(path) is not None
