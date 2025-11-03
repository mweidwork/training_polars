import polars as pl
import pytest

@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    schema = {"a": pl.Int64, "b": pl.Utf8}
    data = [
        {"a": 1, "b": "foo"},
        {"a": 2, "b": "bar"},
        {"a": 3, "b": "baz"},
    ]
    return pl.DataFrame(data, schema)

#######################################################
# Test DataFrame Series fold
#######################################################
def test_dataframe_series_fold(sample_dataframe) -> None:
    df: pl.DataFrame = sample_dataframe
    
    series_a = df["a"]
    series_b = df["b"]
    
    # Fold Series into a single string per row
    folded_series: pl.Series = df.fold(
        lambda series_a, series_b: series_a.cast(pl.Utf8) + "-" + series_b,
    )
    
    expected_list = ["1-foo", "2-bar", "3-baz"]
    assert folded_series.to_list() == expected_list


#######################################################
# Test DataFrame row hashing
#######################################################
def test_dataframe_row_hashing(sample_dataframe) -> None:
    df: pl.DataFrame = sample_dataframe

    hashed_series: pl.Series = df.hash_rows(42)

    assert len(hashed_series.unique()) == df.height