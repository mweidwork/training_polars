import pytest
import polars as pl

#######################################################
# Test DataFrame Fixtures
#######################################################
@pytest.fixture
def left_df():
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [10.5, 20.0, 15.5],
        "passed": [True, False, True],
        "tags": [["math", "science"], ["history"], ["math"]],
    })


@pytest.fixture
def right_df():
    return pl.DataFrame({
        "id": [2, 3, 4],
        "name": ["Bob", "Charlie", "David"],
        "score": [21.0, 16.0, 18.0],
        "passed": [False, True, True],
        "tags": [["history"], ["math", "art"], ["art"]],
    })

#######################################################
# Helper function for assertion
#######################################################
def assert_df_equal(actual: pl.DataFrame, expected: pl.DataFrame):
    """
    Compare two DataFrames by converting to list of dicts.
    Handles mixed types, lists, and nulls reliably.
    """
    actual_list = actual.to_dicts()
    expected_list = expected.to_dicts()
    assert actual_list == expected_list, f"\nActual: {actual_list}\nExpected: {expected_list}"

#######################################################
# Test Inner Join
#######################################################
def test_inner_join(left_df, right_df):
    """
    Inner join:
    - Keeps only rows with matching 'id' in both DataFrames
    - Conflicting columns from right_df get '_right' suffix
    - Result columns: [left columns, right columns with suffix]
    """
    out = left_df.join(right_df, on="id", how="inner", suffix="_right")

    expected = pl.DataFrame({
        "id": [2, 3],
        "name": ["Bob", "Charlie"],
        "score": [20.0, 15.5],
        "passed": [False, True],
        "tags": [["history"], ["math"]],
        "name_right": ["Bob", "Charlie"],
        "score_right": [21.0, 16.0],
        "passed_right": [False, True],
        "tags_right": [["history"], ["math", "art"]],
    })

    assert_df_equal(out, expected)

#######################################################
# Test Left Outer Join
#######################################################
def test_left_join(left_df, right_df):
    """
    Left join:
    - Keeps all rows from left_df
    - Matching rows from right_df included, else right columns null
    - Conflicting right columns get '_right' suffix
    - Result columns: [left columns, right columns with suffix]
    """
    out = left_df.join(right_df, on="id", how="left", suffix="_right")

    expected = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [10.5, 20.0, 15.5],
        "passed": [True, False, True],
        "tags": [["math", "science"], ["history"], ["math"]],
        "name_right": [None, "Bob", "Charlie"],
        "score_right": [None, 21.0, 16.0],
        "passed_right": [None, False, True],
        "tags_right": [None, ["history"], ["math", "art"]],
    })

    assert_df_equal(out, expected)

#######################################################
# Test Right Outer Join
#######################################################
def test_right_join(left_df, right_df):
    """
    Right join:
    - Keeps all rows from right_df
    - Matching rows from left_df included, else left columns null
    - Conflicting left columns get '_left' suffix
    - Result columns: [right columns, left columns with suffix]
    """
    # Swap left and right so the expected output aligns with test
    out = right_df.join(left_df, on="id", how="left", suffix="_left")

    expected = pl.DataFrame({
        "id": [2, 3, 4],
        "name": ["Bob", "Charlie", "David"],
        "score": [21.0, 16.0, 18.0],
        "passed": [False, True, True],
        "tags": [["history"], ["math", "art"], ["art"]],
        "name_left": ["Bob", "Charlie", None],
        "score_left": [20.0, 15.5, None],
        "passed_left": [False, True, None],
        "tags_left": [["history"], ["math"], None],
    })

    assert_df_equal(out, expected)



def test_full_outer_join(left_df, right_df):
    """
    Full outer join:
    - Keeps all rows from both left_df and right_df
    - Missing values filled with nulls
    - Conflicting right columns get '_right' suffix
    - Result columns: [left columns, right columns with suffix]
    """
    # Full outer join
    out = left_df.join(right_df, on="id", how="outer", suffix="_right")

    # Polars may create duplicate 'id_right' column
    if "id_right" in out.columns:
        # Use id from left if exists, else from right
        out = out.with_columns(
            pl.coalesce(["id", "id_right"]).alias("id")
        ).drop("id_right")

    expected = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", None],
        "score": [10.5, 20.0, 15.5, None],
        "passed": [True, False, True, None],
        "tags": [["math", "science"], ["history"], ["math"], None],
        "name_right": [None, "Bob", "Charlie", "David"],
        "score_right": [None, 21.0, 16.0, 18.0],
        "passed_right": [None, False, True, True],
        "tags_right": [None, ["history"], ["math", "art"], ["art"]],
    })

    # Sort by id for deterministic comparison
    out_sorted = out.sort("id", nulls_last=True)
    expected_sorted = expected.sort("id", nulls_last=True)

    assert_df_equal(out_sorted, expected_sorted)
