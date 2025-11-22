import pandas as pd
from typing import Optional


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file and drop fully empty rows.

    Parameters:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = pd.read_csv(path)
    return df.dropna(how="all")


def compute_headline_lengths(
    df: pd.DataFrame, headline_col: str = "headline"
) -> pd.Series:
    """
    Compute the character length of each headline.

    Parameters:
        df (pd.DataFrame): Dataframe containing news data.
        headline_col (str): Column name for headlines.

    Returns:
        pd.Series: Series of headline lengths.
    """
    return df[headline_col].astype(str).apply(len)


def publisher_counts(df: pd.DataFrame, publisher_col: str = "publisher") -> pd.Series:
    """
    Count number of articles per publisher.

    Parameters:
        df (pd.DataFrame)
        publisher_col (str)

    Returns:
        pd.Series: Publisher counts sorted descending.
    """
    return df[publisher_col].fillna("unknown").value_counts()


def parse_dates(df: pd.DataFrame, date_col: str = "published_at") -> pd.DataFrame:
    """
    Convert a date column to datetime format.

    Parameters:
        df (pd.DataFrame)
        date_col (str)

    Returns:
        pd.DataFrame: New dataframe with parsed dates.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df
