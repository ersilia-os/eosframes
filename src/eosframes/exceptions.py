"""Custom exceptions for the ``eosframes`` library."""


class EosframesError(Exception):
    """Sole exception type raised by ``eosframes``.

    Every validation failure inside the library — naming-convention
    violations, model-ID / version mismatches, attempts to overwrite an
    existing output, malformed scaler JSONs, HTTP failures from the model
    hub — surfaces as ``EosframesError``. Users only need to catch this one
    exception class to handle any library-level problem.

    Examples
    --------
    >>> from eosframes import EosframesError, write_csv
    >>> try:
    ...     write_csv(df, "results.csv")  # missing model ID in path
    ... except EosframesError as e:
    ...     print(f"eosframes refused the write: {e}")
    """
