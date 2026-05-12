"""Singleton logger for the ``eosframes`` library.

The logger is named ``eosframes`` and configured once on first access. It
prefers ``rich.logging.RichHandler`` for human-friendly CLI output, falling
back to a plain stderr ``StreamHandler`` with a compact ``HH:MM:SS LEVEL
message`` format. Output goes to ``stderr`` so it stays out of piped stdout.

The default level is ``INFO`` — operational progress messages from
:mod:`eosframes.ops`, :mod:`eosframes.read`, and :mod:`eosframes.write` are
visible immediately. Two ways to change it:

* **Environment** — ``EOSFRAMES_LOG_LEVEL=DEBUG`` (or ``WARNING`` /
  ``ERROR`` / ``CRITICAL``) before the process starts.
* **Programmatic** — :func:`set_verbosity` toggles between ``DEBUG`` and
  ``INFO``. Useful in notebooks and library callers.

Level colour styling matches ``ersilia-os/lazy-qsar`` (debug=cyan,
info=blue, warning=yellow, error=red, critical=white-on-red).
"""

import logging
import os
import sys

_logger = None  # singleton

_LEVEL_THEME = {
    "logging.level.debug": "bold cyan",
    "logging.level.info": "bold blue",
    "logging.level.warning": "bold yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
}


def get_logger() -> logging.Logger:
    """Return the singleton ``eosframes`` logger.

    The logger is configured on first call and reused thereafter. Subsequent
    calls return the same instance without re-installing handlers.

    Behaviour
    ---------
    * Uses ``rich.logging.RichHandler`` (on a stderr ``Console``) when
      ``rich`` is importable, falling back to ``logging.StreamHandler`` on
      stderr.
    * Default level is ``INFO``. Override with the ``EOSFRAMES_LOG_LEVEL``
      environment variable (case-insensitive: ``DEBUG`` / ``INFO`` /
      ``WARNING`` / ``ERROR`` / ``CRITICAL``) or with :func:`set_verbosity`
      at runtime. Invalid env values fall back to ``INFO``.
    * ``propagate = False`` so library records don't bubble up to the root
      logger and produce duplicate output in host applications.

    Returns
    -------
    logging.Logger
        The shared ``eosframes`` logger instance.
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("eosframes")
    logger.setLevel(_resolve_level())
    # Don't double-log via the root logger when callers configure their own
    # handlers (e.g. a Flask/Django app importing eosframes).
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(_build_handler())

    _logger = logger
    return _logger


def set_verbosity(verbose: bool) -> None:
    """Toggle verbose (``DEBUG``) logging on or off.

    ``verbose=True`` lowers the level to ``DEBUG`` (everything visible);
    ``verbose=False`` restores it to ``INFO`` — the default operational
    level, where progress messages from readers, writers, and ops are still
    shown but per-HTTP-probe / per-chunk traces are hidden. Warnings and
    errors are emitted at both settings.

    Parameters
    ----------
    verbose : bool
    """
    logger = get_logger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def _build_handler() -> logging.Handler:
    """Build the singleton stderr handler, preferring rich's ``RichHandler``."""
    try:
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.theme import Theme

        console = Console(stderr=True, theme=Theme(_LEVEL_THEME))
        handler: logging.Handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            markup=False,
            log_time_format="%H:%M:%S",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
            )
        )
        return handler


def _resolve_level() -> int:
    """Resolve the logging level from ``EOSFRAMES_LOG_LEVEL``, defaulting to INFO."""
    raw = os.environ.get("EOSFRAMES_LOG_LEVEL", "").strip().upper()
    if not raw:
        return logging.INFO
    level = logging.getLevelName(raw)
    return level if isinstance(level, int) else logging.INFO
