"""Build the small CloudCost Sentinel V1 dataset from the public FOCUS sample."""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/"
    "FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/"
    "main/FOCUS-1.0/focus_sample_100000.csv.gz"
)
RAW_PATH = "data/raw_focus_large.csv.gz"
OUTPUT_PATH = "data/cloud_cost.csv"
RANDOM_SEED = 42
ANOMALY_FRACTION = 0.04


def download_data():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(RAW_PATH):
        print("Raw dataset already exists.")
        return

    print("Downloading FOCUS sample data...")
    urlretrieve(RAW_URL, RAW_PATH)
    print("Download complete.")


def build_dataset():
    df = pd.read_csv(RAW_PATH)
    print(f"Raw row count: {len(df):,}")
    df["date"] = pd.to_datetime(df["ChargePeriodStart"]).dt.date

    wanted_columns = [
        "date", "ProviderName", "ServiceName", "RegionName",
        "EffectiveCost", "ConsumedQuantity", "ResourceId",
    ]
    df = df[[col for col in wanted_columns if col in df.columns]].copy()

    for col in ["ProviderName", "ServiceName", "RegionName"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # one row roughly means: what did this provider/service/region cost on this day?
    daily = (
        df.groupby(["date", "ProviderName", "ServiceName", "RegionName"], dropna=False)
        .agg(
            daily_cost=("EffectiveCost", "sum"),
            usage_quantity=("ConsumedQuantity", "sum"),
            resource_count=("ResourceId", "nunique"),
        )
        .reset_index()
    )

    # DATA QUALITY FILTER
    # Filter out structurally sparse periods from the 100K FOCUS dataset.
    # Pre-collapse period (Sept 1-19) consistently shows 20-39 positive-cost contexts per day (median ~30).
    # Post-collapse tail (Sept 20+) drops to 2-7. A threshold of <10 safely isolates the corrupted tail.
    print(f"\nAggregated rows before filtering: {len(daily):,}")
    
    pos_counts_per_date = daily[daily["daily_cost"] > 0].groupby("date").size()
    valid_dates = pos_counts_per_date[pos_counts_per_date >= 10].index
    
    daily = daily[daily["date"].isin(valid_dates)].copy()
    
    print(f"Retained {len(valid_dates)} valid dates out of {df['date'].nunique()} total dates.")
    print(f"Aggregated rows after filtering: {len(daily):,}")

    # sort first because anomaly injection uses earlier observations from same context
    group_cols = ["ProviderName", "ServiceName", "RegionName"]
    daily = daily.sort_values(group_cols + ["date"]).reset_index(drop=True)

    # FOCUS has no anomaly label, so V1 creates controlled synthetic spikes.
    # only use rows with a real positive cost and enough previous history; this avoids
    # the old issue where a 0-cost row could be labelled anomaly even after 0*x = 0
    rng = np.random.RandomState(RANDOM_SEED)
    daily["original_daily_cost"] = daily["daily_cost"]

    grouped_cost = daily.groupby(group_cols, dropna=False)["original_daily_cost"]
    history_count = daily.groupby(group_cols, dropna=False).cumcount()
    historical_median = grouped_cost.transform(
        lambda s: s.shift(1).rolling(7, min_periods=3).median()
    )

    eligible_mask = (
        (daily["original_daily_cost"] > 0)
        & (historical_median > 0)
        & (history_count >= 3)
    )
    eligible_indices = daily.index[eligible_mask].to_numpy()

    daily["is_anomaly"] = 0
    daily["anomaly_multiplier"] = 1.0

    target_anomalies = max(1, int(len(daily) * ANOMALY_FRACTION))
    n_anomalies = min(target_anomalies, len(eligible_indices))

    if n_anomalies == 0:
        raise ValueError("No rows are eligible for synthetic anomaly injection.")

    anomaly_indices = rng.choice(eligible_indices, size=n_anomalies, replace=False)
    multipliers = rng.uniform(2.0, 4.0, size=n_anomalies)

    # make the spike unusual relative to the row's own recent context
    spike_base = np.maximum(
        daily.loc[anomaly_indices, "original_daily_cost"].to_numpy(),
        historical_median.loc[anomaly_indices].to_numpy(),
    )
    daily.loc[anomaly_indices, "daily_cost"] = spike_base * multipliers
    daily.loc[anomaly_indices, "anomaly_multiplier"] = multipliers
    daily.loc[anomaly_indices, "is_anomaly"] = 1

    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv(OUTPUT_PATH, index=False)

    print(f"Eligible rows for anomaly injection: {len(eligible_indices):,}/{len(daily):,}")
    print(f"Aggregated row count: {len(daily):,}")
    print(f"Saved {len(daily):,} rows to {OUTPUT_PATH}")
    print(f"Anomalies: {daily['is_anomaly'].sum():,}")
    print(f"Anomaly rate: {daily['is_anomaly'].mean():.2%}")


if __name__ == "__main__":
    download_data()
    build_dataset()
