"""
Production-ready Streamlit app for Road Accident Risk Prediction (single-file).
Requirements:
    pip install streamlit xgboost scikit-learn pandas numpy category_encoders joblib
Run (from project root):
    streamlit run app/app.py
"""

import sys
from pathlib import Path
import os
import logging
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Streamlit is required. Install with 'pip install streamlit'")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("streamlit_app")

# -------------------------
# Custom Transformers (inline for joblib unpickle)
# -------------------------
BOOL_COLS = ['road_signs_present', 'public_road', 'holiday', 'school_season']

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineerTransformer expects a pandas DataFrame")
        df = X.copy()
        df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
        df['danger_score'] = (df['speed_limit']/100)*(df['curvature']**2)
        df['road_complexity'] = df['curvature']*(df['speed_limit']/100)
        df['accidents_per_lane'] = df['num_reported_accidents']/(df['num_lanes']+1)
        df['accidents_log'] = np.log1p(df['num_reported_accidents'])
        df['curvature_squared'] = df['curvature']**2
        df['speed_per_lane'] = df['speed_limit']/(df['num_lanes']+1)
        df['accident_density'] = df['accidents_per_lane'] * df['speed_x_curvature']
        for c in BOOL_COLS:
            if c in df.columns:
                df[c] = df[c].astype(int)
        return df

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, bool_cols, num_cols, engineered_cols):
        self.cols = list(cat_cols)+list(bool_cols)+list(num_cols)+list(engineered_cols)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.cols].copy()

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10.0):
        from category_encoders import TargetEncoder
        self.cols = list(cols)
        self.smoothing = smoothing
        self.encoder = None
    def fit(self, X, y=None):
        from category_encoders import TargetEncoder
        self.encoder = TargetEncoder(cols=self.cols, smoothing=self.smoothing)
        self.encoder.fit(X[self.cols], y)
        return self
    def transform(self, X):
        Xt = X.copy()
        Xt[self.cols] = self.encoder.transform(Xt[self.cols]).astype(float)
        return Xt

# -------------------------
# Constants
# -------------------------
CAT_COLS = ['road_type', 'lighting', 'weather', 'time_of_day']
NUM_COLS = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
ENGINEERED_COLS = [
    'speed_x_curvature', 'danger_score', 'road_complexity', 'accidents_per_lane',
    'accidents_log', 'curvature_squared', 'speed_per_lane', 'accident_density'
]

ROAD_TYPE_OPTIONS = ["highway", "urban", "rural"]
LIGHTING_OPTIONS = ["daylight", "dim", "night"]
WEATHER_OPTIONS = ["clear", "rainy", "foggy"]
TIME_OF_DAY_OPTIONS = ["morning", "afternoon", "evening"]

# -------------------------
# Feature engineering (inline)
# -------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineerTransformer().transform(df)

# -------------------------
# Pipeline loading
# -------------------------
@st.cache_resource
def load_pipeline(pipeline_path: str):
    # Ensure training-time classes are importable for joblib unpickle
    try:
        import src.train_model  # noqa: F401
    except Exception:
        # add project root if needed
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            import src.train_model  # noqa: F401
        except Exception:
            pass
    return joblib.load(pipeline_path)

def list_pipeline_files(models_dir: Path):
    files = list(models_dir.glob("pipeline_xgb_*.pkl"))
    return sorted(files, key=os.path.getctime, reverse=True)

def find_pipeline(models_dir: Path):
    files = list_pipeline_files(models_dir)
    if not files:
        return None
    return files[0]

# -------------------------
# Prediction helper
# -------------------------
def predict_with_pipeline(pipeline, X: pd.DataFrame) -> float:
    for c in BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].astype(int)
    preds = pipeline.predict(X)
    return float(preds[0]) if hasattr(preds, "__len__") else float(preds)

# -------------------------
# Streamlit App
# -------------------------
def app():
    st.set_page_config(page_title="Road Accident Risk Predictor", layout="wide")
    st.title("🚗 Road Accident Risk Predictor")
    st.markdown("Predict accident risk (higher = more risky).")

    # Resolve project root and models dir (always project_root/models)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    models_dir = project_root / "models"

    pipeline_path = find_pipeline(models_dir)
    if not pipeline_path:
        st.error("No trained pipeline found. Run training first.")
        st.info("Tip: Train via `python -m src.train_model` so custom classes are importable as `src.train_model.*`.")
        st.stop()
    # Try load pipeline; if it fails, attempt to reconstruct from encoder+booster
    try:
        pipeline = load_pipeline(str(pipeline_path))
    except Exception as e:
        logger.exception("Pipeline load failed, attempting reconstruction")
        ts = pipeline_path.stem.split("_")[-1]
        encoder_path = models_dir / f"target_encoder_{ts}.joblib"
        booster_path = models_dir / f"xgb_booster_{ts}.json"
        if not (encoder_path.exists() and booster_path.exists()):
            st.error(f"Failed to load pipeline: {e}")
            st.info(f"Missing fallback artifacts: {encoder_path.name if not encoder_path.exists() else ''} {booster_path.name if not booster_path.exists() else ''}")
            st.stop()
        # Rebuild inference pipeline
        fe = FeatureEngineerTransformer()
        selector = ColumnSelector(CAT_COLS, BOOL_COLS, NUM_COLS, ENGINEERED_COLS)
        te = TargetEncoderWrapper(cols=CAT_COLS, smoothing=10.0)
        te.encoder = joblib.load(encoder_path)
        xgb = XGBRegressor()
        xgb.load_model(str(booster_path))
        from sklearn.pipeline import Pipeline as SKPipeline
        pipeline = SKPipeline([
            ("feature_engineer", fe),
            ("selector", selector),
            ("target_encoder", te),
            ("xgb", xgb),
        ])

    # Input form
    st.subheader("Enter Road Conditions")
    with st.form("input_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Categorical")
            road_type = st.selectbox("🛣️ Road Type", ROAD_TYPE_OPTIONS)
            lighting = st.selectbox("💡 Lighting", LIGHTING_OPTIONS)
            weather = st.selectbox("🌦️ Weather", WEATHER_OPTIONS)
            time_of_day = st.selectbox("🕒 Time of Day", TIME_OF_DAY_OPTIONS)

        with col_right:
            st.markdown("### Boolean")
            road_signs_present = st.checkbox("🚸 Road Signs Present", value=True)
            public_road = st.checkbox("🛣️ Public Road", value=True)
            holiday = st.checkbox("🎉 Holiday", value=False)
            school_season = st.checkbox("🏫 School Season", value=True)

            st.markdown("### Numerical")
            num_lanes = st.number_input("🛤️ Number of Lanes", min_value=1, max_value=10, value=2)
            curvature = st.number_input("🔄 Road Curvature (degrees)", min_value=0.0, max_value=180.0, value=30.0)
            speed_limit = st.number_input("🚗💨 Speed Limit (km/h)", min_value=20, max_value=200, value=60)
            num_reported_accidents = st.number_input("📊 Historical Accidents", min_value=0, max_value=1000, value=5)

        # Right-aligned submit button
        spacer, btn_col = st.columns([5, 1])
        with btn_col:
            submit_button = st.form_submit_button("Predict Risk")

    if submit_button:
        input_data = {
            'road_type': [road_type],
            'lighting': [lighting],
            'weather': [weather],
            'time_of_day': [time_of_day],
            'road_signs_present': [int(road_signs_present)],
            'public_road': [int(public_road)],
            'holiday': [int(holiday)],
            'school_season': [int(school_season)],
            'num_lanes': [num_lanes],
            'curvature': [curvature],
            'speed_limit': [speed_limit],
            'num_reported_accidents': [num_reported_accidents]
        }
        X_input = pd.DataFrame(input_data)

    # Prediction
        risk_score = predict_with_pipeline(pipeline, X_input)
        st.subheader("Prediction")

        # Two-column layout with icon (no scrolling tables)
        col_icon, col_metric = st.columns([1, 3])

        # Determine risk level and icon
        if risk_score < 0.3:
            level_text = "Low Risk — conditions look safe."
            level_icon = "🟢"
        elif risk_score < 0.7:
            level_text = "Medium Risk — exercise caution."
            level_icon = "🟡"
        else:
            level_text = "High Risk — take extra precautions or avoid."
            level_icon = "🔴"

        with col_icon:
            st.markdown(f"## {level_icon}")
            st.caption(level_text)
        with col_metric:
            st.metric("Accident Risk Score", f"{risk_score:.4f}")
            st.progress(min(max(int(risk_score*100), 0), 100))

        # Optional: compact input summary and model metadata for organization
        with st.expander("Details", expanded=False):
            # Input snapshot
            st.markdown("**Input Summary**")
            st.table(X_input)

            # Model metadata (if available)
            ts = pipeline_path.stem.split("_")[-1]
            meta_path = models_dir / f"metadata_{ts}.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    cv = meta.get("cv_rmse", "N/A")
                    saved = meta.get("timestamp", ts)
                    st.markdown(f"**Model CV RMSE:** {cv}")
                    st.markdown(f"**Saved at (UTC):** {saved}")
                except Exception:
                    st.caption("Metadata could not be read.")

    st.markdown("---")
    st.markdown("Built with Streamlit & XGBoost. Ensure trained pipeline exists in `models/`.")

if __name__ == "__main__":
    app()
