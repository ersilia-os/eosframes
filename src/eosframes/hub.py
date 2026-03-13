"""Functions for fetching Ersilia model information from GitHub."""

import io
import json
import re

import pandas as pd
import requests

from .exceptions import EosframesError

_GITHUB_RAW = "https://raw.githubusercontent.com/ersilia-os/{model_id}/{ref}/{filename}"
_METADATA_CANDIDATES = ["metadata.json", "metadata.yml", "metadata.yaml"]
_RUN_COLUMNS_PATH = "model/framework/columns/run_columns.csv"


def _version_to_ref(version: str) -> str:
    """Map a short version string like 'v1' to its semver git tag 'v1.0.0'."""
    m = re.match(r"^v(\d+)$", version)
    if m:
        return f"v{m.group(1)}.0.0"
    return version


def fetch_metadata(model_id: str) -> dict:
    """Fetch metadata for an Ersilia model from GitHub.

    Tries metadata.json first, then metadata.yml/yaml.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier, e.g. ``"eos4e40"``.

    Returns
    -------
    dict
        Raw metadata as a dictionary.

    Raises
    ------
    EosframesError
        If the metadata cannot be fetched.
    """
    for filename in _METADATA_CANDIDATES:
        url = _GITHUB_RAW.format(model_id=model_id, ref="main", filename=filename)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            if filename.endswith(".json"):
                return json.loads(resp.text)
            try:
                import yaml

                return yaml.safe_load(resp.text)
            except ImportError as exc:
                raise EosframesError(
                    f"Model '{model_id}' has a YAML metadata file but 'pyyaml' is not "
                    "installed. Install it with: pip install pyyaml"
                ) from exc
    raise EosframesError(
        f"Could not fetch metadata for model '{model_id}'. "
        f"Make sure the repo exists at https://github.com/ersilia-os/{model_id}"
    )


def fetch_columns(model_id: str, version: str) -> pd.DataFrame:
    """Fetch the run_columns.csv for an Ersilia model version from GitHub.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier, e.g. ``"eos4e40"``.
    version : str
        Version string, e.g. ``"v1"``. Resolved to a git tag (``v1.0.0``)
        with fallback to ``main``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: name, type, direction, description.

    Raises
    ------
    EosframesError
        If the file cannot be fetched.
    """
    refs_to_try = dict.fromkeys([_version_to_ref(version), "main"])  # ordered, unique
    for ref in refs_to_try:
        url = _GITHUB_RAW.format(model_id=model_id, ref=ref, filename=_RUN_COLUMNS_PATH)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text))
    raise EosframesError(
        f"Could not fetch run_columns.csv for model '{model_id}'. "
        f"Make sure the repo exists at https://github.com/ersilia-os/{model_id}"
    )
