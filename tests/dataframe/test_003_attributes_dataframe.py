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

def test_dataframe_attributes(sample_dataframe) -> None:
    df: pl.DataFrame = sample_dataframe
    
    # Test DataFrame shape
    assert df.shape == (3, 2)
    
    # Test DataFrame height
    assert df.height == 3
    
    # Test DataFrame width
    assert df.width == 2
    
    # Test DataFrame dtypes
    assert df.dtypes == [pl.Int64, pl.Utf8]
    
    # Test DataFrame columns
    assert df.columns == ["a", "b"]
    
    # Test DataFrame schema
    expected_schema = {"a": pl.Int64, "b": pl.Utf8}
    assert df.schema == expected_schema
    
    assert df.flags == {'a': {'SORTED_ASC': False, 'SORTED_DESC': False}, 'b': {'SORTED_ASC': False, 'SORTED_DESC': False}}