# class polars.DataFrame(

#     data: FrameInitTypes | None = None,
#     schema: SchemaDefinition | None = None,
#     *,
#     schema_overrides: SchemaDict | None = None,
#     strict: bool = True,
#     orient: Orientation | None = None,
#     infer_schema_length: int | None = 100,
#     nan_to_null: bool = False,

# )
# Source: https://docs.pola.rs/api/python/stable/reference/dataframe/index.html

import polars as pl
import numpy as np
import pytest


#######################################################
# Fixtures
#######################################################

@pytest.fixture
def sample_data() -> list[dict[str, object]]:
    return [
        {"a": 1, "b": "foo"},
        {"a": 2, "b": "bar"},
        {"a": 3, "b": "baz"},
    ]
    
@pytest.fixture
def sample_dataframe(sample_data) -> pl.DataFrame:
    schema = {"a": pl.Int64, "b": pl.Utf8}
    return pl.DataFrame(sample_data, schema)

@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    return np.array([[1, "foo"], [2, "bar"], [3, "baz"]])

#######################################################
# Test create DataFrame
#######################################################

def test_create_dataframe_from_dicts(sample_data) -> None:

    df = pl.DataFrame(sample_data)
    assert isinstance(df, pl.DataFrame)

def test_create_dataframe_with_schema(sample_data) -> None:

    # Separately define the schema
    schema = {"a": pl.Int64, "b": pl.Utf8}
    
    df1 = pl.DataFrame(sample_data, schema=schema)
    assert isinstance(df1, pl.DataFrame)
    
    # Combine data and schema overrides
    data = [
        pl.Series("a", [1, 2, 3], dtype=pl.Int64),
        pl.Series("b", ["foo", "bar", "baz"], dtype=pl.Utf8),
    ]
    
    df2 = pl.DataFrame(data)
    assert isinstance(df2, pl.DataFrame)
    
def test_override_schema(sample_dataframe) -> None:

    schema_overrides = {"a": pl.Float64}
    
    df = pl.DataFrame(sample_dataframe, schema_overrides=schema_overrides)
    assert isinstance(df, pl.DataFrame)
    assert df["a"].dtype == pl.Float64

def test_create_dataframe_from_numpy_array(sample_numpy_array) -> None:

    df = pl.DataFrame(sample_numpy_array, schema={"a": pl.Int64, "b": pl.Utf8})
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)