"""General-purpose utilities."""

import logging
import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

import pandas as pd

from .logger import get_logger

_T = TypeVar("_T")


def chunker(df: pd.DataFrame, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
    """Yield successive non-overlapping chunks of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    chunksize : int
        Number of rows per chunk (default 10 000).

    Yields
    ------
    pd.DataFrame
    """
    for start in range(0, len(df), chunksize):
        yield df.iloc[start : start + chunksize]


def progress(
    iterable: Iterable[_T],
    total: Optional[int] = None,
    desc: str = "",
    log_every: Optional[int] = None,
) -> Iterator[_T]:
    """Wrap *iterable* with progress feedback that adapts to the environment.

    Three modes, picked automatically (all gated on the ``eosframes`` logger
    being at ``INFO`` or more verbose — a quieted logger gets a silent
    pass-through):

    * **Interactive (TTY)** — an in-place per-item bar on stderr.
    * **Non-interactive with** *log_every* — periodic ``INFO`` log lines
      (every *log_every* items, plus a final line). This is the path that
      keeps long streaming jobs legible when stderr is a file or pipe:
      ``nohup``, CI, ``build_scaler.sh``, etc. Without *log_every* a
      non-interactive run stays silent.
    * **Otherwise** — transparent pass-through, no output.

    Both visible modes share stderr with the logger, so nothing ever leaks
    onto piped stdout.

    Parameters
    ----------
    iterable : Iterable
        The items to iterate over.
    total : int, optional
        Expected number of items. Inferred via ``len(iterable)`` when omitted;
        if neither is available the bar is disabled (percentages are dropped
        from log lines but counts still flow).
    desc : str
        Short label shown to the left of the bar / log line.
    log_every : int, optional
        Emit an ``INFO`` log line every this many items when no bar is drawn.
        Leave unset for the in-memory paths that should stay quiet off-TTY.

    Yields
    ------
    The items of *iterable*, unchanged.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    logger = get_logger()
    info_on = logger.isEnabledFor(logging.INFO)
    bar_on = total is not None and total > 1 and sys.stderr.isatty() and info_on

    if bar_on:
        width = 30
        last_draw = -1.0  # force an immediate first draw
        count = 0
        for count, item in enumerate(iterable, 1):
            yield item
            now = time.monotonic()
            # Throttle redraws to ~10 fps, but always paint the final item.
            if count < total and now - last_draw < 0.1:
                continue
            last_draw = now
            frac = count / total
            filled = int(width * frac)
            bar = "█" * filled + "·" * (width - filled)
            label = f"{desc} " if desc else ""
            sys.stderr.write(f"\r{label}|{bar}| {count}/{total}")
            sys.stderr.flush()
        if count:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return

    if log_every and info_on:
        label = desc or "progress"
        count = 0
        for count, item in enumerate(iterable, 1):
            yield item
            if count % log_every == 0:
                if total:
                    logger.info(
                        "%s: %d/%d (%d%%)", label, count, total, count * 100 // total
                    )
                else:
                    logger.info("%s: %d done", label, count)
        # Final line unless the last item already landed on a logged boundary.
        if count and count % log_every != 0:
            if total:
                logger.info("%s: %d/%d (100%%)", label, count, total)
            else:
                logger.info("%s: %d done", label, count)
        return

    yield from iterable
