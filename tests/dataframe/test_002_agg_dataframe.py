import polars as pl
import pytest

import training_polars.polars.agg as agg

#######################################################
# Test aggregate DataFrame
#######################################################

@pytest.mark.parametrize("df,expected_dict",
[
    (pl.DataFrame({"col1": [1, 2, 3]}), {"col1": 0}),
    (pl.DataFrame({"col1": [0, None, None, None]}), {"col1": 3}),
    (pl.DataFrame({"col1": [0, None, None, None], "col2": [4, 5, None, None]}), {"col1": 3, "col2": 2}),
    (pl.DataFrame({"col1": [1, 2], "col2": [3, 4]}), {"col1": 0, "col2": 0}),
    (pl.DataFrame({"col1": [], "col2": []}), {"col1": 0, "col2": 0}),
])
def test_count_aggregation_dataframe(df, expected_dict) -> None:
    
    # Return the number of non-null values in each column
    # Source: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.count.html#polars.DataFrame.count
    
    result = agg.count_dataframe(df)
    assert result.to_dicts() == [expected_dict]

@pytest.mark.parametrize("df,expected_dict",[
    (pl.DataFrame({"col1": [1, 2, 3]}), {"col1": 3}),
    (pl.DataFrame({"col1": [3, None, None, None]}), {"col1": 3}),
    (pl.DataFrame({"col1": [0, None, None, None], "col2": [4, 5, None, None]}), {"col1": 0, "col2": 5}),
    (pl.DataFrame({"col1": ["a","b"], "col2": [3, 4]}), {"col1": "b", "col2": 4}),
])

def test_max_aggregation_dataframe(df, expected_dict) -> None:
    
    # Return the maximum value in each column
    # Source: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.max.html#polars.DataFrame.max
    
    result = agg.max_dataframe(df)
    assert result.to_dicts() == [expected_dict]


@pytest.mark.parametrize("df,expected_list",[
    (pl.DataFrame({"col1": [1, 2, 3]}), [1, 2, 3]),
    (pl.DataFrame({"col1": [3, None, 1, 2]}), [3, None, 1, 2]),
    (pl.DataFrame({"col1": [0, None, None, None], "col2": [4, 5, None, None]}), [4, 5, None, None]),
    (pl.DataFrame({"col1": ["a","b"], "col2": [3, 4]}), ["a", "b"]),
]) 
def test_max_horizontal_aggregation_dataframe(df, expected_list) -> None:
    
    # Return the maximum value across each row
    # Source: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.max_horizontal.html#polars.DataFrame.max_horizontal
    
    result = agg.max_horizontal_dataframe(df)
    assert result.to_series().to_list() == expected_list