import pandas as pd
from src.data_processing import compute_headline_lengths, publisher_counts, parse_dates


def test_compute_headline_lengths():
    df = pd.DataFrame({"headline": ["hello world", "test"]})
    lengths = compute_headline_lengths(df, "headline")
    assert lengths.tolist() == [11, 4]


def test_publisher_counts():
    df = pd.DataFrame({"publisher": ["A", "A", "B"]})
    counts = publisher_counts(df, "publisher")
    assert counts["A"] == 2
    assert counts["B"] == 1


def test_parse_dates():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
    df = parse_dates(df, "date")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
