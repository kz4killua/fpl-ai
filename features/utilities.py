"""Helper functions for feature engineering."""

import pandas as pd


def simple_rolling(series, window, operation='mean', fill_value=0):
    """
    Perform simple rolling window calculations.

    Missing values will be ignored and ffilled.
    """

    # Remove missing values to avoid gaps
    values = series.dropna()

    # Compute the rolling window
    values = values.rolling(window=window, min_periods=1)

    # Apply the specified operation
    if operation == 'mean':
        values = values.mean()
    elif operation == 'sum':
        values = values.sum()
    elif operation == 'median':
        values = values.median()
    elif operation == 'std':
        values = values.std()
    elif operation == 'var':
        values = values.var()
    else:
        raise Exception('Invalid operation!')

    # Restore removed indices
    values = values.reindex(series.index)

    # Do not use any current (unknown) data
    values = values.shift(1)

    # Fill in values for removed indices
    values = values.fillna(method='ffill')

    # Fill in other missing values
    values = values.fillna(fill_value)

    return values


def exponential_rolling(series, alpha=None, times=None, halflife=None, shift=1, fill_value=0):
    """
    Perform exponential weighted average calculations.
    """

    if not times.is_monotonic_increasing:
        raise Exception("Values in series must be monotonically increasing.")
    
    values = series.ewm(alpha=alpha, times=times, halflife=halflife)

    # Apply the mean operation
    values = values.mean()

    # Shift previous values to the next row
    values = values.shift(shift)

    # Fill in missing values
    values = values.fillna(fill_value)

    return values


def iter_column(df, column):
    """Iterate over each unique value in column of a dataframe of players."""
    for value in df[column].unique():
        subset = df[df[column] == value]

        yield subset


def iter_players(players):
    """Shortcut for iterating over players."""
    return iter_column(players, 'code')


def iter_teams(teams):
    """Shortcut for iterating over teams."""
    return iter_column(teams, 'fpl_code')