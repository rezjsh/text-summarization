import pandas as pd

from text_summarization_project.preprocessing.strategies import (
    DropDuplicatesStrategy,
    DropNAStrategy,
    LengthFilterStrategy,
    TextCleaningStrategy,
)


def _sample_df():
    return pd.DataFrame({
        "article": [
            "(CNN) -- This   is a   test article with enough characters to pass filters. " * 3,
            None,
            "short",
            "This   is a   test article with enough characters to pass filters. " * 3,
        ],
        "highlights": [
            "This is a summary of the test article.",
            "Missing article row.",
            "too short article above",
            "This is a summary of the test article.",
        ],
    })


def test_drop_na():
    df = _sample_df()
    out = DropNAStrategy().apply(df, "article", "highlights")
    assert out["article"].isna().sum() == 0
    assert len(out) == 3


def test_drop_duplicates():
    df = _sample_df().dropna()
    out = DropDuplicatesStrategy().apply(df, "article", "highlights")
    assert len(out) < len(df)


def test_text_cleaning_strips_cnn_tag_and_whitespace():
    df = _sample_df().dropna()
    out = TextCleaningStrategy(lowercase=False).apply(df, "article", "highlights")
    assert not out["article"].iloc[0].startswith("(CNN)")
    assert "   " not in out["article"].iloc[0]


def test_length_filter_drops_short_rows():
    df = _sample_df().dropna()
    out = LengthFilterStrategy(
        min_article_chars=50, max_article_chars=10000,
        min_summary_chars=5, max_summary_chars=1000,
    ).apply(df, "article", "highlights")
    assert (out["article"].str.len() >= 50).all()
