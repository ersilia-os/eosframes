"""Fetch Ersilia model information from GitHub.

This module reaches out to ``raw.githubusercontent.com/ersilia-os/<model_id>``
to retrieve metadata (``fetch_metadata``) and per-version column
definitions (``fetch_columns``). ``fetch_columns`` first tries the
semver tag for *version* (``v<N>`` → ``v<N>.0.0``), then falls back to
``main``; ``fetch_metadata`` fetches from ``main`` directly.

All HTTP calls have a 15 s timeout. Failures are surfaced as
:class:`~eosframes.EosframesError` with a hint pointing at the
``ersilia-os`` repo URL.
"""

import io
import json
import re

import pandas as pd
import requests

from .exceptions import EosframesError
from .logger import get_logger

_GITHUB_RAW = "https://raw.githubusercontent.com/ersilia-os/{model_id}/{ref}/{filename}"
_METADATA_CANDIDATES = ["metadata.json", "metadata.yml", "metadata.yaml"]
_RUN_COLUMNS_PATH = "model/framework/columns/run_columns.csv"


def _version_to_ref(version: str) -> str:
    """Map a short version string ``v<N>`` to its semver git tag ``v<N>.0.0``.

    Anything that doesn't match the short form is returned unchanged,
    so callers can pass already-resolved refs (full SHAs, branch
    names) without surprise rewriting.
    """
    m = re.match(r"^v(\d+)$", version)
    if m:
        return f"v{m.group(1)}.0.0"
    return version


def _raw_url(model_id: str, ref: str, filename: str) -> str:
    """Build a ``raw.githubusercontent.com`` URL for *filename* at *ref*."""
    return _GITHUB_RAW.format(model_id=model_id, ref=ref, filename=filename)


def fetch_metadata(model_id: str) -> dict:
    """Fetch metadata for an Ersilia model from GitHub.

    Probes ``main`` for ``metadata.json``, ``metadata.yml`` and
    ``metadata.yaml`` in that order, returning the first one that
    responds with HTTP 200. YAML files require ``pyyaml`` to be
    installed.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier.

    Returns
    -------
    dict
        Raw metadata as a dictionary.

    Raises
    ------
    EosframesError
        If none of the candidate metadata files exist on the model's
        ``main`` branch, or if a YAML metadata file is found but
        ``pyyaml`` is not installed.
    """
    logger = get_logger()
    for filename in _METADATA_CANDIDATES:
        url = _raw_url(model_id, "main", filename)
        logger.debug("GET %s", url)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            logger.info("Fetched metadata for %s from %s", model_id, filename)
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
        logger.debug(
            "Metadata candidate %s returned HTTP %d for %s",
            filename,
            resp.status_code,
            model_id,
        )
    raise EosframesError(
        f"Could not fetch metadata for model '{model_id}'. "
        f"Make sure the repo exists at https://github.com/ersilia-os/{model_id}"
    )


def fetch_columns(model_id: str, version: str) -> pd.DataFrame:
    """Fetch the ``run_columns.csv`` for an Ersilia model version from GitHub.

    Tries the semver tag for *version* first (``v<N>`` → ``v<N>.0.0``),
    then falls back to ``main``. The first ref returning HTTP 200 wins.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier.
    version : str
        Version string matching ``v\\d+``. Resolved by :func:`_version_to_ref`.

    Returns
    -------
    pandas.DataFrame
        Parsed ``run_columns.csv``, typically with columns ``name``,
        ``type``, ``direction``, ``description``.

    Raises
    ------
    EosframesError
        If neither the tagged ref nor ``main`` returns the file.
    """
    logger = get_logger()
    refs_to_try = list(
        dict.fromkeys([_version_to_ref(version), "main"])
    )  # ordered, unique
    for ref in refs_to_try:
        url = _raw_url(model_id, ref, _RUN_COLUMNS_PATH)
        logger.debug("GET %s", url)
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            logger.info(
                "Fetched run_columns.csv for %s @ %s (resolved from version=%s)",
                model_id,
                ref,
                version,
            )
            return pd.read_csv(io.StringIO(resp.text))
        logger.debug(
            "run_columns.csv at %s returned HTTP %d for %s",
            ref,
            resp.status_code,
            model_id,
        )
    raise EosframesError(
        f"Could not fetch run_columns.csv for model '{model_id}' at any of "
        f"the candidate refs {refs_to_try}. "
        f"Make sure the repo exists at https://github.com/ersilia-os/{model_id}"
    )
