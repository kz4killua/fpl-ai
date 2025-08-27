from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from prediction.utils import FeatureSelector


def make_minutes_predictor():
    columns = [
        # General attributes
        "value",
        "element_type",
        "availability",
        "record_count",
        # Means
        "availability_rolling_mean_1",
        "availability_rolling_mean_3",
        "availability_rolling_mean_5",
        "availability_rolling_mean_10",
        "availability_mean_last_season",
        "minutes_rolling_mean_1",
        "minutes_rolling_mean_3",
        "minutes_rolling_mean_5",
        "minutes_rolling_mean_10",
        "minutes_mean_last_season",
        "minutes_category_1_to_59_minutes_rolling_mean_1",
        "minutes_category_1_to_59_minutes_rolling_mean_3",
        "minutes_category_1_to_59_minutes_rolling_mean_5",
        "minutes_category_1_to_59_minutes_rolling_mean_10",
        "minutes_category_1_to_59_minutes_mean_last_season",
        "minutes_category_60_plus_minutes_rolling_mean_1",
        "minutes_category_60_plus_minutes_rolling_mean_3",
        "minutes_category_60_plus_minutes_rolling_mean_5",
        "minutes_category_60_plus_minutes_rolling_mean_10",
        "minutes_category_60_plus_minutes_mean_last_season",
        # Standard deviations
        "minutes_rolling_std_3",
        "minutes_rolling_std_5",
        "minutes_rolling_std_10",
        "minutes_std_last_season",
        "minutes_category_1_to_59_minutes_rolling_std_3",
        "minutes_category_1_to_59_minutes_rolling_std_5",
        "minutes_category_1_to_59_minutes_rolling_std_10",
        "minutes_category_1_to_59_minutes_std_last_season",
        "minutes_category_60_plus_minutes_rolling_std_3",
        "minutes_category_60_plus_minutes_rolling_std_5",
        "minutes_category_60_plus_minutes_rolling_std_10",
        "minutes_category_60_plus_minutes_std_last_season",
        # Means over available matches
        "minutes_rolling_mean_1_when_available",
        "minutes_rolling_mean_3_when_available",
        "minutes_rolling_mean_5_when_available",
        "minutes_rolling_mean_10_when_available",
        "minutes_mean_last_season_when_available",
        "minutes_category_1_to_59_minutes_rolling_mean_1_when_available",
        "minutes_category_1_to_59_minutes_rolling_mean_3_when_available",
        "minutes_category_1_to_59_minutes_rolling_mean_5_when_available",
        "minutes_category_1_to_59_minutes_rolling_mean_10_when_available",
        "minutes_category_1_to_59_minutes_mean_last_season_when_available",
        "minutes_category_60_plus_minutes_rolling_mean_1_when_available",
        "minutes_category_60_plus_minutes_rolling_mean_3_when_available",
        "minutes_category_60_plus_minutes_rolling_mean_5_when_available",
        "minutes_category_60_plus_minutes_rolling_mean_10_when_available",
        "minutes_category_60_plus_minutes_mean_last_season_when_available",
        # Standard deviations over available matches
        "minutes_rolling_std_3_when_available",
        "minutes_rolling_std_5_when_available",
        "minutes_rolling_std_10_when_available",
        "minutes_std_last_season_when_available",
        "minutes_category_1_to_59_minutes_rolling_std_3_when_available",
        "minutes_category_1_to_59_minutes_rolling_std_5_when_available",
        "minutes_category_1_to_59_minutes_rolling_std_10_when_available",
        "minutes_category_1_to_59_minutes_std_last_season_when_available",
        "minutes_category_60_plus_minutes_rolling_std_3_when_available",
        "minutes_category_60_plus_minutes_rolling_std_5_when_available",
        "minutes_category_60_plus_minutes_rolling_std_10_when_available",
        "minutes_category_60_plus_minutes_std_last_season_when_available",
        # Team depth ranks and (un)availability
        "depth_rank_value",
        "adjusted_depth_rank_value",
        "depth_rank_change_value",
        "depth_unavailability_value",
        "depth_rank_minutes_rolling_mean_38",
        "adjusted_depth_rank_minutes_rolling_mean_38",
        "depth_rank_change_minutes_rolling_mean_38",
        "depth_unavailability_minutes_rolling_mean_38",
        # Fatigue
        "minutes_sum_5_days",
        "minutes_sum_7_days",
        "minutes_sum_10_days",
        "minutes_sum_14_days",
    ]
    model = LGBMClassifier(
        # Problem parameters
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        # Learning parameters
        min_child_samples=128,
        num_leaves=15,
        reg_lambda=100.0,
        # Training parameters
        verbosity=-1,
        # Reproducibility parameters
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("selector", FeatureSelector(columns)),
            ("predictor", model),
        ]
    )
    return pipeline
