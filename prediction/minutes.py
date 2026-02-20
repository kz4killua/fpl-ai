from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline

from prediction.utils import FeatureSelector


def make_minutes_category_predictor():
    columns = [
        "availability",
        "record_count",
        "element_type",
        "value",
        "availability_rolling_mean_1",
        "availability_rolling_mean_3",
        "availability_rolling_mean_5",
        "availability_rolling_mean_10",
        "availability_rolling_mean_20",
        "availability_mean_last_season",
        "minutes_rolling_mean_1",
        "minutes_rolling_mean_3",
        "minutes_rolling_mean_5",
        "minutes_rolling_mean_10",
        "minutes_rolling_mean_20",
        "minutes_mean_last_season",
        "minutes_rolling_mean_1_when_available",
        "minutes_rolling_mean_3_when_available",
        "minutes_rolling_mean_5_when_available",
        "minutes_rolling_mean_10_when_available",
        "minutes_rolling_mean_20_when_available",
        "minutes_mean_last_season_when_available",
        "minutes_category_0_minutes_rolling_mean_1",
        "minutes_category_0_minutes_rolling_mean_3",
        "minutes_category_0_minutes_rolling_mean_5",
        "minutes_category_0_minutes_rolling_mean_10",
        "minutes_category_0_minutes_rolling_mean_20",
        "minutes_category_0_minutes_mean_last_season",
        "minutes_category_1_to_59_minutes_rolling_mean_1",
        "minutes_category_1_to_59_minutes_rolling_mean_3",
        "minutes_category_1_to_59_minutes_rolling_mean_5",
        "minutes_category_1_to_59_minutes_rolling_mean_10",
        "minutes_category_1_to_59_minutes_rolling_mean_20",
        "minutes_category_1_to_59_minutes_mean_last_season",
        "minutes_rolling_std_3",
        "minutes_rolling_std_5",
        "minutes_rolling_std_10",
        "minutes_rolling_std_20",
        "minutes_std_last_season",
        "minutes_sum_5_days",
        "minutes_sum_7_days",
        "minutes_sum_10_days",
        "minutes_sum_14_days",
        "depth_rank_value",
        "adjusted_depth_rank_value",
        "depth_rank_change_value",
        "depth_unavailability_value",
        "depth_rank_minutes_rolling_mean_38",
        "adjusted_depth_rank_minutes_rolling_mean_38",
        "depth_rank_change_minutes_rolling_mean_38",
        "depth_unavailability_minutes_rolling_mean_38",
    ]
    categorical_feature_names = [
        "element_type",
    ]
    categorical_feature_indices = [
        columns.index(name) for name in categorical_feature_names
    ]
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=10_000,
        early_stopping=True,
        learning_rate=0.1,
        max_leaf_nodes=15,
        min_samples_leaf=256,
        l2_regularization=100.0,
        categorical_features=categorical_feature_indices,
        verbose=0,
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("selector", FeatureSelector(columns)),
            ("predictor", model),
        ]
    )
    return pipeline
