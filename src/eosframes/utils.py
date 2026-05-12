"""General-purpose utilities."""

from typing import Iterator

import pandas as pd


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
