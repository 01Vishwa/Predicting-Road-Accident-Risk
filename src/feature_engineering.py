# src/feature_engineering.py
import numpy as np
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic feature engineering used for training and inference.
    Keeps implementation identical between train and inference.
    """
    df = df.copy()
    df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
    df['danger_score'] = (df['speed_limit'] / 100) * (df['curvature'] ** 2)
    df['road_complexity'] = df['curvature'] * (df['speed_limit'] / 100)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
    df['accident_density'] = df['accidents_per_lane'] * df['speed_x_curvature']
    return df
