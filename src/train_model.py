# src/train_model.py
"""
Train script (production-ready). Usage from project root:
  python -m src.train_model --train-csv data/train.csv --models-dir models

Outputs:
  - models/pipeline_xgb_<ts>.pkl       (sklearn Pipeline: fe -> selector -> encoder -> xgb)
  - models/xgb_booster_<ts>.json       (XGBoost booster for fast inference)
  - models/target_encoder_<ts>.joblib  (category_encoders.TargetEncoder instance)
  - models/metadata_<ts>.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# robust import for feature_engineering
try:
    from .feature_engineering import feature_engineering
except Exception:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.feature_engineering import feature_engineering

# CONFIG
RANDOM_STATE = 42
DEFAULT_SPLITS = 10

cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
engineered_cols = [
    'speed_x_curvature', 'danger_score', 'road_complexity', 'accidents_per_lane',
    'accidents_log', 'curvature_squared', 'speed_per_lane', 'accident_density'
]
target_col = 'accident_risk'

DEFAULT_PARAMS = {
    'n_estimators': 2217,
    'max_depth': 11,
    'learning_rate': 0.045238118744298804,
    'subsample': 0.7384590439930039,
    'gamma': 0.01440523822277898,
    'colsample_bytree': 0.8165153235195506,
    'min_child_weight': 5,
    'reg_alpha': 0.08351735648195949,
    'reg_lambda': 0.0183512761721743,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'verbosity': 0,
    'random_state': RANDOM_STATE
}

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_model")

# TRANSFORMERS
class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineerTransformer expects a pandas DataFrame")
        X_copy = feature_engineering(X)
        for c in bool_cols:
            if c in X_copy.columns:
                X_copy[c] = X_copy[c].astype(int)
        return X_copy

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, bool_cols, num_cols, engineered_cols):
        self.cols = list(cat_cols) + list(bool_cols) + list(num_cols) + list(engineered_cols)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ColumnSelector expects a pandas DataFrame")
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns after feature engineering: {missing}")
        return X[self.cols].copy()

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10.0):
        self.cols = list(cols)
        self.smoothing = smoothing
        self.encoder = None
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TargetEncoderWrapper.fit expects a DataFrame")
        self.encoder = TargetEncoder(cols=self.cols, smoothing=self.smoothing)
        self.encoder.fit(X[self.cols], y)
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TargetEncoderWrapper.transform expects a DataFrame")
        Xt = X.copy()
        Xt[self.cols] = self.encoder.transform(Xt[self.cols]).astype(float)
        return Xt

# helpers
def build_pipeline(params):
    fe = FeatureEngineerTransformer()
    selector = ColumnSelector(cat_cols, bool_cols, num_cols, engineered_cols)
    te = TargetEncoderWrapper(cols=cat_cols, smoothing=10.0)
    xgb = XGBRegressor(**params)
    pipeline = Pipeline([
        ("feature_engineer", fe),
        ("selector", selector),
        ("target_encoder", te),
        ("xgb", xgb)
    ])
    return pipeline

# training and save
def train_and_save(train_df: pd.DataFrame, models_dir: Path, params: dict, n_splits: int):
    if target_col not in train_df.columns:
        raise ValueError(f"Training dataframe missing target column '{target_col}'")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    pipeline = build_pipeline(params)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_list = []

    logger.info("Starting KFold CV (%d splits)", n_splits)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[tr_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
        y_tr, y_val = y.iloc[tr_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        rmse_list.append(rmse)
        logger.info("Fold %d RMSE: %.6f", fold, rmse)

    mean_rmse = float(np.mean(rmse_list))
    logger.info("Mean CV RMSE: %.6f", mean_rmse)

    logger.info("Fitting final pipeline on full dataset")
    pipeline.fit(X, y)

    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pipeline_path = models_dir / f"pipeline_xgb_{ts}.pkl"
    booster_path = models_dir / f"xgb_booster_{ts}.json"
    encoder_path = models_dir / f"target_encoder_{ts}.joblib"
    meta_path = models_dir / f"metadata_{ts}.json"

    # save pipeline (convenience single object)
    joblib.dump(pipeline, pipeline_path)

    # extract encoder and booster for fast inference
    te_wrapper = pipeline.named_steps.get("target_encoder")
    xgb_reg = pipeline.named_steps.get("xgb")
    encoder_obj = getattr(te_wrapper, "encoder", None)
    if encoder_obj is None:
        raise RuntimeError("Encoder object not found in pipeline (unexpected)")

    # save encoder and booster separately
    joblib.dump(encoder_obj, encoder_path)
    booster = xgb_reg.get_booster()
    booster.save_model(str(booster_path))

    metadata = {
        "timestamp": ts,
        "cv_rmse": mean_rmse,
        "params": params,
        "categorical_columns": cat_cols,
        "boolean_columns": bool_cols,
        "numerical_columns": num_cols,
        "engineered_columns": engineered_cols,
        "target_column": target_col,
        "pipeline_path": str(pipeline_path),
        "booster_path": str(booster_path),
        "encoder_path": str(encoder_path),
        "random_state": RANDOM_STATE
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved pipeline -> %s", pipeline_path)
    logger.info("Saved encoder -> %s", encoder_path)
    logger.info("Saved xgb booster -> %s", booster_path)
    logger.info("Saved metadata -> %s", meta_path)
    return pipeline_path, booster_path, encoder_path, meta_path, mean_rmse

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Train and save XGBoost accident-risk pipeline")
    parser.add_argument("--train-csv", default="data/train.csv", help="Path to training CSV (default: data/train.csv)")
    parser.add_argument("--models-dir", default="models", help="Directory to save artifacts")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS, help="KFold splits")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (hist)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU (gpu_hist)")
    parser.add_argument("--quick", action="store_true", help="Quick training mode (fewer trees)")
    return parser.parse_args()

def main():
    args = parse_args()
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found at {train_csv.resolve()}")

    params = DEFAULT_PARAMS.copy()
    if args.gpu:
        params["tree_method"] = "gpu_hist"
    if args.cpu:
        params["tree_method"] = "hist"
    if args.quick:
        params["n_estimators"] = min(400, params["n_estimators"])
        params["subsample"] = 0.7

    logger.info("Config: tree_method=%s n_estimators=%s", params["tree_method"], params["n_estimators"])

    df = pd.read_csv(train_csv)
    logger.info("Loaded training data: %d rows x %d cols", df.shape[0], df.shape[1])

    pipeline_path, booster_path, encoder_path, meta_path, cv_rmse = train_and_save(
        df, Path(args.models_dir), params, args.n_splits
    )

    logger.info("Training complete. Model: %s | CV RMSE: %.6f", pipeline_path, cv_rmse)

if __name__ == "__main__":
    main()
