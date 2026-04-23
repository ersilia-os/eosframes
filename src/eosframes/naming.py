"""Naming convention parsing and validation for Ersilia output files."""

import os
import re
from typing import Dict, List, Optional, Tuple

# Matches the eos<digit><3 alphanumeric> pattern anywhere in a string
_MODEL_ID_RE = re.compile(r"(?<![A-Za-z0-9])eos\d[A-Za-z0-9]{3}(?![A-Za-z0-9])")

# Matches <model_id>_<version> at the end of a stem (allows leading prefix tokens)
_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)$")

# Matches <model_id>_<version>_info / _columns / _summary stems
_INFO_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)_info$")
_COLUMNS_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)_columns$")
_SUMMARY_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)_summary$")

# Stack outputs come in two flavours:
#   Mode A (eosmix):   [prefix]_eosmix.csv — feature cols get _<model_id>_<version>
#   Mode B (explicit): [prefix]_<m1>_<v1>_..._<mN>_<vN>.csv (N>=2) — bare cols
# We match them with dedicated regexes rather than folding into parse_name,
# since Mode B overlaps syntactically with a regular data file that has a
# long prefix (see tests).
_EOSMIX_STEM_RE = re.compile(r"(?:^|_)eosmix$")
# One (model_id, version) pair — matched repeatedly to walk a stack_explicit stem.
_MODEL_VER_PAIR_RE = re.compile(r"(eos\d[A-Za-z0-9]{3})_(v\d+)")

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

    * ``eos4e40_v1.csv``              → ``name_type="csv"``
    * ``eos4e40_v1.h5``               → ``name_type="h5"``
    * ``eos4e40_v1_chunks``           → ``name_type="chunks_dir"``
    * ``eos4e40_v1_info.csv``         → ``name_type="info"``
    * ``eos4e40_v1_columns.csv``      → ``name_type="columns"``
    * ``eos4e40_v1_summary.csv``      → ``name_type="summary"``
    * ``260313_gardp_eos4e40_v1.csv`` → prefix allowed before model_id

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

    # Sidecar CSV files (_info.csv, _columns.csv, _summary.csv) are checked
    # before the generic data-file pattern so the trailing token is not
    # swallowed as part of a data-file prefix.
    if ext == "csv":
        for regex, name_type in (
            (_INFO_STEM_RE, "info"),
            (_COLUMNS_STEM_RE, "columns"),
            (_SUMMARY_STEM_RE, "summary"),
        ):
            m = regex.search(stem)
            if m and is_model_id_valid(m.group(1)):
                return {
                    "model_id": m.group(1),
                    "version": m.group(2),
                    "extension": ext,
                    "name_type": name_type,
                }

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
    """Return ``True`` if *path* is a valid Ersilia data file or directory.

    Accepts csv files, h5 files, and chunks directories. Sidecar files
    (``_info.csv``, ``_columns.csv``) are **not** considered valid data
    names and are rejected here — use :func:`is_valid_info_name` /
    :func:`is_valid_columns_name` for those.
    """
    parsed = parse_name(path)
    return parsed is not None and parsed["name_type"] in {"csv", "h5", "chunks_dir"}


def is_valid_info_name(path: str) -> bool:
    """Return ``True`` if *path* follows the info-sidecar convention.

    Matches ``[prefix]_<model_id>_<version>_info.csv``.
    """
    parsed = parse_name(path)
    return parsed is not None and parsed["name_type"] == "info"


def is_valid_columns_name(path: str) -> bool:
    """Return ``True`` if *path* follows the columns-sidecar convention.

    Matches ``[prefix]_<model_id>_<version>_columns.csv``.
    """
    parsed = parse_name(path)
    return parsed is not None and parsed["name_type"] == "columns"


def is_valid_summary_name(path: str) -> bool:
    """Return ``True`` if *path* follows the summary-sidecar convention.

    Matches ``[prefix]_<model_id>_<version>_summary.csv``.
    """
    parsed = parse_name(path)
    return parsed is not None and parsed["name_type"] == "summary"


def _make_sidecar_name(
    model_id: str, version: str, kind: str, prefix: Optional[str] = None
) -> str:
    """Build a canonical sidecar filename (``_info.csv`` or ``_columns.csv``)."""
    if not is_model_id_valid(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    if not re.match(r"^v\d+$", version):
        raise ValueError(f"Invalid version: {version!r}. Expected format: v1, v2, ...")
    if prefix is not None and not re.fullmatch(
        r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*", prefix
    ):
        raise ValueError(
            f"Invalid prefix: {prefix!r}. Must be alphanumeric tokens joined by underscores."
        )
    stem = f"{model_id}_{version}_{kind}"
    return f"{prefix}_{stem}.csv" if prefix else f"{stem}.csv"


def make_info_name(model_id: str, version: str, prefix: Optional[str] = None) -> str:
    """Build a canonical info-sidecar filename.

    Returns
    -------
    str
        e.g. ``"eos4e40_v1_info.csv"`` or ``"example_eos4e40_v1_info.csv"``.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    return _make_sidecar_name(model_id, version, "info", prefix)


def make_columns_name(model_id: str, version: str, prefix: Optional[str] = None) -> str:
    """Build a canonical columns-sidecar filename.

    Returns
    -------
    str
        e.g. ``"eos4e40_v1_columns.csv"`` or ``"example_eos4e40_v1_columns.csv"``.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    return _make_sidecar_name(model_id, version, "columns", prefix)


def make_summary_name(model_id: str, version: str, prefix: Optional[str] = None) -> str:
    """Build a canonical summary-sidecar filename.

    Returns
    -------
    str
        e.g. ``"eos4e40_v1_summary.csv"`` or ``"example_eos4e40_v1_summary.csv"``.

    Raises
    ------
    ValueError
        If any argument is invalid.
    """
    return _make_sidecar_name(model_id, version, "summary", prefix)


# ---------------------------------------------------------------------------
# Horizontal-stack outputs (two modes, mutually exclusive)
# ---------------------------------------------------------------------------


def _validate_prefix(prefix: Optional[str]) -> None:
    if prefix is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*", prefix):
        raise ValueError(
            f"Invalid prefix: {prefix!r}. Must be alphanumeric tokens joined by underscores."
        )


def parse_stack_mix_name(path: str) -> Optional[Dict]:
    """Parse a Mode A stack filename and return its prefix.

    Mode A filenames look like ``[prefix]_eosmix.csv``. The mixture itself
    has no version and no model id — column names carry the provenance.

    Returns
    -------
    dict or None
        ``{"prefix": <str or None>}`` on success, ``None`` if *path* does
        not follow Mode A. Prefix is ``None`` for the bare ``eosmix.csv``.
    """
    basename = os.path.basename(path.rstrip("/\\"))
    if "." not in basename:
        return None
    stem, ext = basename.rsplit(".", 1)
    if ext != "csv":
        return None
    m = _EOSMIX_STEM_RE.search(stem)
    if not m:
        return None
    # m.start() is 0 when the stem IS "eosmix"; otherwise it's the index of
    # the leading "_" before "eosmix".
    prefix = stem[: m.start()] if m.start() > 0 else ""
    return {"prefix": prefix or None}


def is_valid_stack_mix_name(path: str) -> bool:
    """Return True if *path* follows the Mode A stack convention."""
    return parse_stack_mix_name(path) is not None


def make_stack_mix_name(prefix: Optional[str] = None) -> str:
    """Build a canonical Mode A (``eosmix``) stack filename.

    Returns
    -------
    str
        ``"eosmix.csv"`` or ``"<prefix>_eosmix.csv"``.
    """
    _validate_prefix(prefix)
    return f"{prefix}_eosmix.csv" if prefix else "eosmix.csv"


def parse_stack_explicit_name(path: str) -> Optional[Dict]:
    """Parse a Mode B stack filename: prefix + ordered model list.

    A Mode B filename looks like ``[prefix]_<m1>_<v1>_<m2>_<v2>...<mN>_<vN>.csv``
    with N >= 2. The trailing sequence of ``<model_id>_<version>`` pairs must
    cover the stem; anything preceding the first pair (and its trailing ``_``)
    is the prefix.

    Returns
    -------
    dict or None
        ``{"prefix": <str or None>, "models": [(model_id, version), ...]}``
        on success; the model list has at least 2 entries. Returns ``None``
        if *path* does not follow Mode B (including the single-model case,
        which is a regular data file).
    """
    basename = os.path.basename(path.rstrip("/\\"))
    if "." not in basename:
        return None
    stem, ext = basename.rsplit(".", 1)
    if ext != "csv":
        return None

    # Walk the stem right-to-left collecting trailing model_id_version pairs.
    pairs: List[Tuple[str, str]] = []
    remaining = stem
    prefix = ""
    while True:
        m = re.search(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)$", remaining)
        if not m:
            # Whatever is left is the prefix (may be "").
            prefix = remaining
            break
        pairs.append((m.group(1), m.group(2)))
        end = m.start()
        if end == 0:
            prefix = ""
            break
        remaining = remaining[:end]

    if len(pairs) < 2:
        return None

    pairs.reverse()
    return {"prefix": prefix or None, "models": pairs}


def is_valid_stack_explicit_name(path: str) -> bool:
    """Return True if *path* follows the Mode B stack convention (N>=2 models)."""
    return parse_stack_explicit_name(path) is not None


def make_stack_explicit_name(
    model_versions: List[Tuple[str, str]], prefix: Optional[str] = None
) -> str:
    """Build a canonical Mode B stack filename from an ordered list of models.

    Parameters
    ----------
    model_versions : list of (model_id, version)
        At least two pairs, in the order the models were stacked.
    prefix : str, optional
        Optional prefix (alphanumeric tokens joined by underscores).

    Returns
    -------
    str
        e.g. ``"eos4e40_v1_eos7m30_v1.csv"`` or
        ``"example_eos4e40_v1_eos7m30_v1.csv"``.

    Raises
    ------
    ValueError
        If fewer than two pairs are given, any ``model_id`` / ``version`` is
        invalid, or the prefix is malformed.
    """
    if len(model_versions) < 2:
        raise ValueError(
            "Mode B stack filenames require at least 2 (model_id, version) pairs."
        )
    tokens = []
    for model_id, version in model_versions:
        if not is_model_id_valid(model_id):
            raise ValueError(f"Invalid model_id: {model_id!r}")
        if not re.match(r"^v\d+$", version):
            raise ValueError(
                f"Invalid version: {version!r}. Expected format: v1, v2, ..."
            )
        tokens.append(f"{model_id}_{version}")
    _validate_prefix(prefix)
    stem = "_".join(tokens)
    return f"{prefix}_{stem}.csv" if prefix else f"{stem}.csv"
