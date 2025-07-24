import polars as pl


def compute_one_hot_minutes_category(df: pl.DataFrame):
    """Compute one-hot encoded categories for minute categories."""
    return df.with_columns(df.select("minutes_category").to_dummies())
