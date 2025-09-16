import re
import pandas as pd
import requests


def chunker(df: pd.DataFrame, chunksize: int = 10000):
    """
    Generator that yields chunks of the input DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be chunked.
    chunksize: int
        The number of rows per chunk (default=10000).

    Yields
    ------
    pd.DataFrame
        A chunk of the original DataFrame.
    """
    for start in range(0, len(df), chunksize):
        yield df.iloc[start:start + chunksize]


def get_model_id_from_path(path: str) -> str:
    """
    Given a file name, extract the model identifier if it exists.
    
    The model identifier is assumed to follow the pattern 'eosXYYY' where X is a digit and YYY are alphanumeric characters.
    Example valid identifiers: eos42ez, eos4e40, eos3804

    Parameters
    ----------
    path : str
        The name of the file or folder from which to extract the model identifier
    
    Returns
    -------
    str
        The extracted model identifier if found, otherwise None
    """
    text = path.split("/")[-1]
    pattern = r'(?<![A-Za-z0-9])eos\d[A-Za-z0-9]{3}(?![A-Za-z0-9])'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None


def is_model_id_valid(model_id: str) -> bool:
    """
    Simple function to determine if a model identifier is valid or not.

    Parameters
    ----------
    model_id : str
        The model identifier to validate (e.g. "eos4e40").

    Returns
    -------
    bool
        True if the model identifier is valid, False otherwise.
    """
    if len(model_id) != 7:
        return False
    if not model_id.startswith("eos"):
        return False
    return True


def get_run_columns(model_id: str) -> pd.DataFrame:
    """
    Fetch run_columns.csv from a given repository under ersilia-os.

    Parameters
    ----------
    model_id : str
        Repository name inside the ersilia-os org (e.g. "eos4e40").

    Returns
    -------
    pd.DataFrame
        The CSV contents as a pandas DataFrame.
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/model/framework/columns/run_columns.csv"
    return pd.read_csv(url)


def get_model_slug(model_id: str) -> str:
    """
    Get the model slug from the GitHub README.md file in ersilia-os/{model_id}.
    Assumes README.md contains a line like: **Slug**: `some-slug`

    Parameters
    ----------
    model_id : str
        Repository name inside the ersilia-os org (e.g. "eos4e40")

    Returns
    -------
    str
        The model slug extracted from the README.md file
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/README.md"
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    try:
        slug = text.split("**Slug:** `")[1].split("`")[0].strip()
        return slug
    except IndexError:
        raise ValueError(f"No slug found in README.md for {model_id}")


def get_model_title(model_id: str) -> str:
    """
    Get the model title from the GitHub README.md file in ersilia-os/{model_id}.
    Assumes the README contains a line like: "# Title".

    Parameters
    ----------
    model_id : str
        Repository name inside the ersilia-os org (e.g. "eos4e40")
    
    Returns
    -------
    str
        The model title extracted from the README.md file
    """
    repo = model_id
    branch = "main"
    url = f"https://raw.githubusercontent.com/ersilia-os/{repo}/{branch}/README.md"
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    try:
        title = text.split("# ")[1].split("\n")[0].strip()
        return title
    except IndexError:
        raise ValueError(f"No title found in README.md for {model_id}")
