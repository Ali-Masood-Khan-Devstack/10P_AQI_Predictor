import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CITIES

def engineer_features(df):
    """Engineers ML features across 9 pollutants + weather indicators."""
    df = df.sort_values("datetime").copy()
    
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(timezone.utc)

    # 1. Log Transformation for PM2.5 & PM10
    df["pm25_log"] = np.log1p(np.maximum(df["pm2_5"], 0))
    if "pm10" in df.columns:
        df["pm10_log"] = np.log1p(np.maximum(df["pm10"], 0))

    # 2. Time Lags (1h, 2h, 3h, 6h, 12h, 24h, 48h)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pm25_lag_{lag}h"] = df["pm25_log"].shift(lag)

    # 3. Rolling Statistics (3h, 6h, 12h, 24h)
    for window in [3, 6, 12, 24]:
        df[f"pm25_roll_mean_{window}h"] = df["pm25_log"].rolling(window).mean()
        df[f"pm25_roll_std_{window}h"] = df["pm25_log"].rolling(window).std()

    # 4. Multi-Pollutant Lag Interactions
    for col in ["pm10", "no2", "so2", "co", "o3", "dust", "aod", "uv_index"]:
        if col in df.columns:
            df[f"{col}_lag_1h"] = df[col].shift(1)
            df[f"{col}_lag_6h"] = df[col].shift(6)

    # 5. Wind Vector Transformations
    if "wind_speed" in df.columns and "wind_dir" in df.columns:
        df["wind_x"] = df["wind_speed"] * np.cos(np.deg2rad(df["wind_dir"]))
        df["wind_y"] = df["wind_speed"] * np.sin(np.deg2rad(df["wind_dir"]))

    # 6. Cyclical Time Features (24h Daily + 12m Seasonal + 365d Day of Year)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofyear / 365.25)

    # 7. Stagnation & Photochemical Smog Proxy (AOD * UV Index / Wind Speed)
    if "wind_speed" in df.columns:
        df["stagnation_index"] = df["pm25_log"] / (df["wind_speed"] + 1)
    if "aod" in df.columns and "uv_index" in df.columns and "wind_speed" in df.columns:
        df["smog_potential"] = (df["aod"] * df["uv_index"]) / (df["wind_speed"] + 1)

    df["is_weekend"] = df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    # 8. Target Horizons (24h, 48h, 72h Ahead)
    df["target_h24"] = df["pm2_5"].shift(-24)
    df["target_h48"] = df["pm2_5"].shift(-48)
    df["target_h72"] = df["pm2_5"].shift(-72)

    target_cols = ["target_h24", "target_h48", "target_h72"]
    exclude_cols = target_cols + ["datetime", "city"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df[feature_cols] = df[feature_cols].ffill()
    df.dropna(subset=["pm25_lag_48h"], inplace=True)
    df[feature_cols] = df[feature_cols].bfill()

    return df
