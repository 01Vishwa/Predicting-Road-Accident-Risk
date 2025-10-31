# Predicting Road Accident Risk

End-to-end project to train an XGBoost regression model for road-accident risk and serve it via a Streamlit app.

Key components:

- `src/feature_engineering.py` — feature construction used for both training and inference.
- `src/train_model.py` — CLI training script with KFold CV; saves model and target encoder to `models/`.
- `app/app.py` — interactive UI to input road/context features and predict risk using saved artifacts.
- `data/` — training and optional test CSVs.
- `models/` — saved artifacts created after training (created automatically).

## Setup (Windows cmd.exe)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

Run with CPU by default:

```
python -m src.train_model --cpu --models-dir models
```

Faster smoke test (fewer trees):

```
python -m src.train_model --cpu --quick
```

If you have a CUDA-enabled GPU and GPU-enabled XGBoost installed:

```
python -m src.train_model --gpu
```

Artifacts produced (timestamps in UTC):

- `models/pipeline_xgb_<timestamp>.pkl` — full sklearn Pipeline (FE -> selector -> target encoder -> XGBoost)
- `models/xgb_booster_<timestamp>.json` — raw XGBoost booster for fast/lightweight inference if needed
- `models/target_encoder_<timestamp>.joblib` — fitted TargetEncoder object
- `models/metadata_<timestamp>.json` — training metadata (CV RMSE, params, etc.)

## Run the Streamlit app

Make sure artifacts exist (train first), then run:

```
streamlit run app/app.py
```

The app loads artifacts from the repository root `models/` folder regardless of where it's launched from.

## Data schema

The training and app expect at least the following columns in `data/train.csv`:

- Categorical: `road_type`, `lighting`, `weather`, `time_of_day`
- Boolean (0/1): `road_signs_present`, `public_road`, `holiday`, `school_season`
- Numeric: `num_lanes`, `curvature`, `speed_limit`, `num_reported_accidents`
- Target: `accident_risk`

Notes:
- In the provided sample data, categorical values are lowercase (e.g., `highway`, `urban`). The app’s dropdowns use title case by default; unseen categories are handled safely by target encoding but may slightly affect scores.

## Notes

- Target encoding handles unseen categories at inference by reverting to the global prior.
- Default training uses CPU (`tree_method=hist`). Use `--gpu` to enable `gpu_hist` when supported.
- For reproducibility, random seeds are set for CV shuffling; XGBoost may still have minor nondeterminism across hardware/backends.