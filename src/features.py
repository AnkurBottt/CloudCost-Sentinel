"""
src/features.py

Small, readable feature-engineering helpers for CloudCost Sentinel.

This file only does:
1. Add a few time/context features.
2. Split chronologically into train / validation / test.
3. Convert categorical columns into numbers with pd.get_dummies().
"""

import numpy as np
import pandas as pd


def add_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # CHANGED: explicitly sort by context + date before historical features.
    # This makes "previous" unambiguously mean the previous observation
    # for the same provider/service/region.
    group_cols = ["ProviderName", "ServiceName", "RegionName"]
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    # Calendar information
    df["day_of_week"] = df["date"].dt.dayofweek

    # Compare each service/region with its own recent history
    grouped = df.groupby(group_cols, dropna=False) # a grouped view/instruction over the dataframe

    # Previous cost for the same context
    df["previous_cost"] = grouped["daily_cost"].shift(1) # move_evry_value_down_byNrows(N)

    # Previous 7 observations only.
    # shift(1) means today's cost is NOT included in today's baseline.
    df["rolling_7d_avg"] = grouped["daily_cost"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )

    # How large is today's cost compared with its recent average?
    # np.where(condition, value_if_true, value_if_false)
    df["cost_vs_7d_avg"] = np.where(
        df["rolling_7d_avg"] > 0,
        df["daily_cost"] / df["rolling_7d_avg"],
        1.0,
    )

    # CHANGED:
    # Add a rolling MEDIAN signal as well as the rolling mean.
    # A mean can be pulled upward by a previous cost spike.
    # The median is more robust, so it gives XGBoost another way
    # to decide whether today's value is unusual.
    df["rolling_7d_median"] = grouped["daily_cost"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).median()
    )

    df["cost_vs_7d_median"] = np.where(
        df["rolling_7d_median"] > 0,
        df["daily_cost"] / df["rolling_7d_median"],
        1.0,
    )

    # New groups have no history yet, so use simple defaults.
    # data.py now avoids injecting synthetic anomalies into these
    # history-less rows, so these defaults are mainly for normal rows.
    df["previous_cost"] = df["previous_cost"].fillna(df["daily_cost"])
    df["rolling_7d_avg"] = df["rolling_7d_avg"].fillna(df["daily_cost"])
    df["rolling_7d_median"] = df["rolling_7d_median"].fillna(df["daily_cost"])

    for ratio_col in ["cost_vs_7d_avg", "cost_vs_7d_median"]:
        df[ratio_col] = (
            df[ratio_col]
            .replace([np.inf, -np.inf], 1.0)
            .fillna(1.0)
        )

    return df


def chronological_split(df):
    """70% earliest dates -> train, next 15% -> validation, last 15% -> test."""
    dates = sorted(df["date"].unique())

    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )

    return train_df, val_df, test_df


def prepare_xy(train_df, val_df, test_df):
    """
    Models like Logistic Regression, XGBoost, neural networks cannot directly interpret: -as they r strings
    So one-hot encoding converts categories into numeric indicator columns.
    Select a small feature set and convert text columns using pd.get_dummies().(binary codes-one hot endcodes(1or0))

    Training columns are created first.
    Validation/test are then reindexed to match them.
    """
    feature_columns = [
        "daily_cost",
        "usage_quantity",
        "resource_count",
        "previous_cost",
        "rolling_7d_avg",
        "cost_vs_7d_avg",
        "rolling_7d_median",
        "cost_vs_7d_median",
        "day_of_week",
        "ProviderName",
        "ServiceName",
        "RegionName",
    ]

    feature_columns = [
        col for col in feature_columns
        if col in train_df.columns
    ]

    X_train = pd.get_dummies(train_df[feature_columns], dtype=int)
    X_val = pd.get_dummies(val_df[feature_columns], dtype=int)
    X_test = pd.get_dummies(test_df[feature_columns], dtype=int)

    # NOTE:
    # Reindexing keeps validation/test columns identical to training.
    # A category never seen during training becomes all-zero across the
    # known dummy columns. That is acceptable for this simple V1, though
    # a fitted OneHotEncoder(handle_unknown="ignore") is cleaner later.
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df["is_anomaly"]
    y_val = val_df["is_anomaly"]
    y_test = test_df["is_anomaly"]

    return X_train, X_val, X_test, y_train, y_val, y_test
