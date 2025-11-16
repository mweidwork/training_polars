import polars as pl
import pytest

@pytest.fixture
def sample_dataframe():
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "city": ["New York", "Los Angeles", "Chicago", "Houston"]
    }
    return pl.DataFrame(data)

#######################################################
# Test DataFrame select
#######################################################

@pytest.mark.parametrize("column_name, expected_values", [
    ("name", ["Alice", "Bob", "Charlie", "David"]),
    ("age", [25, 30, 35, 40]),  
    ("city", ["New York", "Los Angeles", "Chicago", "Houston"])     
])
def test_column_selection(sample_dataframe, column_name, expected_values):
    selected_column = sample_dataframe.select(pl.col(column_name))
    assert selected_column[column_name].to_list() == expected_values
    

@pytest.mark.parametrize("invalid_column", [
    "invalid_name", 
    "invalid_age",
    "invalid_city"
])
def test_invalid_column_selection(sample_dataframe, invalid_column):
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        sample_dataframe.select(pl.col(invalid_column))


@pytest.mark.parametrize("column_name, expected_dtype", [
    ("name", pl.Utf8),  
    ("age", pl.Int64),   
    ("city", pl.Utf8)
])
def test_column_dtype(sample_dataframe, column_name, expected_dtype):
    selected_column = sample_dataframe.select(pl.col(column_name))
    assert selected_column[column_name].dtype == expected_dtype
    

@pytest.mark.parametrize("column_names", [
    ("name",),
    ("name", "age"),
    ("age", "city")
])
def test_column_selector_by_name(sample_dataframe, column_names):
    selected = sample_dataframe.select(pl.selectors.by_name(column_names))
    assert list(selected.columns) == list(column_names)
    
def test_column_selector_threshold(sample_dataframe):
    selected = sample_dataframe.with_columns(
        pl.when(pl.col("age") >= 31)
            .then(pl.col("age"))
            .otherwise(None)
            .alias("age")
    )

    assert selected["age"].to_list() == [None, None, 35, 40]

#######################################################
# Test DataFrame with_columns
#######################################################

def test_column_selector_threshold_shrinked(sample_dataframe):
    selected = (
        sample_dataframe
        .with_columns(
            pl.when(pl.col("age") >= 31)
                .then(pl.col("age"))
                .otherwise(None)
                .alias("age")
        )
        .drop_nulls("age")
    )

    assert selected["age"].to_list() == [35, 40]


#######################################################
# Test DataFrame filter
#######################################################

def test_column_selector_filtered(sample_dataframe):
    selected = sample_dataframe.filter(pl.col("age") >= 31)

    assert selected["age"].to_list() == [35, 40]

def test_multiple_conditions(sample_dataframe):
    selected = sample_dataframe.filter(
        (pl.col("age") >= 30) & (pl.col("city").str.contains("o"))
    )
    assert selected["name"].to_list() == ["Bob", "Charlie", "David"]


# #######################################################
# Test DataFrame row selection & slicing
# #######################################################

def test_row_slicing(sample_dataframe):
    assert sample_dataframe[:2]["name"].to_list() == ["Alice", "Bob"]
    assert sample_dataframe[1:3]["name"].to_list() == ["Bob", "Charlie"]