import logging
import sys

_logger = None  # singleton


def get_logger() -> logging.Logger:
    """
    Return the singleton eosframes logger.

    Uses RichHandler (from the ``rich`` library) for console output if
    available, otherwise falls back to a plain StreamHandler.
    Log level defaults to INFO.
    """
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger("eosframes")
    _logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the module is reloaded
    if _logger.handlers:
        return _logger

    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )

    _logger.addHandler(handler)
    return _logger
