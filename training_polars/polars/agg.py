from typing import Any
import polars as pl

def count_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with the count of non-null values in each column.
    
    Parameters:
    df (pl.DataFrame): The input DataFrame.
    
    Returns:
    pl.DataFrame: A DataFrame containing the count of non-null values for each column.
    """
    
    col_details: dict[str, int] = {}
    
    for col in df.get_columns():
        
        null_counter = 0
        for i in range(df.height):
            if col[i] is None:
                null_counter += 1
                
        col_details[col.name] = null_counter
        
    return pl.DataFrame(col_details, schema={k: pl.Int64 for k in col_details.keys()})

def max_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with the maximum value in each column.
    
    Parameters:
    df (pl.DataFrame): The input DataFrame.
    
    Returns:
    pl.DataFrame: A DataFrame containing the maximum value for each column.
    """
    
    col_details: dict[str, int] = {}
    
    for col in df.get_columns():
        
        max_value: Any | None = None
        for i in range(df.height):
            if col[i] is not None:
                if max_value is None or col[i] > max_value:
                    max_value = col[i]
                    
        col_details[col.name] = max_value
        
    return pl.DataFrame(col_details)

def max_horizontal_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with the maximum value across each row.
    
    Parameters:
    df (pl.DataFrame): The input DataFrame.
    
    Returns:
    pl.DataFrame: A DataFrame containing the maximum value for each row.
    """
    
    max_values: list[Any | None] = []
    
    for i in range(df.height):
        max_value: Any | None = None
        for col in df.get_columns():
            if col[i] is not None:
                if max_value is None or str(col[i]).encode("utf-8").hex() > str(max_value).encode("utf-8").hex():
                    max_value = col[i]
        max_values.append(max_value)
        
    return pl.DataFrame({"max_value": max_values})

def mean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a DataFrame with the mean value in each column.
    
    Parameters:
    df (pl.DataFrame): The input DataFrame.
    
    Returns:
    pl.DataFrame: A DataFrame containing the mean value for each column.
    """
    
    col_details: dict[str, float] = {}
    
    for col in df.get_columns():
        
        total = 0.0
        count = 0
        for i in range(df.height):
            if col[i] is not None:
                total += float(col[i])
                count += 1
                
        mean_value = total / count if count > 0 else None
        col_details[col.name] = mean_value
        
    return pl.DataFrame(col_details)

def mean_horizontal_dataframe(df: pl.DataFrame) -> pl.Series:
    """
    Returns a DataFrame with the mean value across each row.
    
    Parameters:
    df (pl.DataFrame): The input DataFrame.
    
    Returns:
    pl.DataFrame: A DataFrame containing the mean value for each row.
    """
    
    mean_values: list[float | None] = []
    
    for i in range(df.height):
        total = 0.0
        count = 0
        for col in df.get_columns():
            if col[i] is not None:
                total += float(col[i])
                count += 1
                
        mean_value = total / count if count > 0 else None
        mean_values.append(mean_value)
        
    return pl.Series(name="mean", values=mean_values, dtype=pl.Float64)