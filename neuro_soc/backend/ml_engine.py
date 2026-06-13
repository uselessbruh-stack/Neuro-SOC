"""
===============================================================================
  Neuro-SOC  |  ML Engine — Isolation Forest + SHAP Explainability
  --------------------------------------------------------------------------
  Module      : backend/ml_engine.py
  Purpose     : Train an Isolation Forest anomaly detector on the enriched
                feature set and provide per-event SHAP explanations for
                any flagged anomaly.
  Author      : Neuro-SOC Engineering Team
  Created     : 2026-06-13
  Python      : 3.11+
  Dependencies: scikit-learn, shap, pandas, numpy, joblib
===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuro_soc.ml_engine")

# ---------------------------------------------------------------------------
#  Path Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ENRICHED_CSV = PROJECT_ROOT / "data_pipeline" / "processed_data" / "enriched_features.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

# ---------------------------------------------------------------------------
#  Feature Configuration
#  These are the NUMERIC columns we feed into the Isolation Forest.
#  Non-numeric / identifier columns (user_id, timestamp, hire_date, etc.)
#  are excluded.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "days_inactive",
    "tenure_months",
    "high_risk_flag",              # bool → treated as 0/1
    "Tenure_Risk_Modifier",
    "Equipment_Mismatch_Score",
    "Temporal_Velocity",
    "rowcount_deviation",
    "After_Hours_High_Sensitivity",
    "Failed_Action_Flag",
    "systems_access_count",
    "Privilege_Sensitivity_Mismatch",
]

# ---------------------------------------------------------------------------
#  IsolationForest Hyperparameters
# ---------------------------------------------------------------------------
IF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,        # Number of trees in the forest
    "contamination": 0.05,      # Expected fraction of anomalies (5%)
    "max_samples": "auto",      # Subsample size for each tree
    "random_state": 42,         # Reproducibility
    "n_jobs": -1,               # Use all CPU cores
}


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_model(force_retrain: bool = False) -> tuple[IsolationForest, StandardScaler, pd.DataFrame]:
    """
    Train (or load) the Isolation Forest model.

    Steps:
      1. Load enriched_features.csv
      2. Select only the numeric feature columns
      3. Standardise features (IsolationForest works better with scaled data)
      4. Fit the Isolation Forest
      5. Persist model + scaler to disk for fast reload

    Parameters
    ----------
    force_retrain : bool
        If True, retrain even if a saved model exists on disk.

    Returns
    -------
    (model, scaler, training_data) — the fitted model, scaler, and the
    feature matrix used for SHAP background.
    """
    # --- Try to load a previously trained model ---
    if not force_retrain and MODEL_PATH.exists() and SCALER_PATH.exists():
        logger.info("Loading pre-trained model from disk …")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        training_data = _load_feature_matrix()
        logger.info("  ✓ Model loaded successfully.")
        return model, scaler, training_data

    # --- Load and prepare training data ---
    logger.info("Training new Isolation Forest model …")
    training_data = _load_feature_matrix()

    logger.info(f"  Feature matrix shape: {training_data.shape}")
    logger.info(f"  Features: {list(training_data.columns)}")

    # --- Standardise ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(training_data)

    # --- Train Isolation Forest ---
    model = IsolationForest(**IF_PARAMS)
    model.fit(X_scaled)

    # Count anomalies detected in training data
    predictions = model.predict(X_scaled)
    n_anomalies = (predictions == -1).sum()
    logger.info(
        f"  ✓ Model trained. Anomalies in training set: "
        f"{n_anomalies}/{len(predictions)} "
        f"({n_anomalies / len(predictions) * 100:.1f}%)"
    )

    # --- Persist to disk ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"  ✓ Model saved to {MODEL_PATH}")

    return model, scaler, training_data


def _load_feature_matrix() -> pd.DataFrame:
    """
    Load enriched_features.csv and return only the numeric feature columns.
    Coerces boolean columns to int and fills any NaN with 0.
    """
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(
            f"Enriched features not found at {ENRICHED_CSV}. "
            f"Run the data_pipeline/enrichment.py script first."
        )

    df = pd.read_csv(ENRICHED_CSV)

    # Select only our defined feature columns
    features = df[FEATURE_COLUMNS].copy()

    # Coerce booleans → int (high_risk_flag may be True/False strings)
    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = features[col].map(
                {"True": 1, "False": 0, "true": 1, "false": 0}
            ).fillna(0).astype(int)
        elif features[col].dtype == "bool":
            features[col] = features[col].astype(int)

    # Fill any remaining NaN with 0
    features = features.fillna(0)

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════
def explain_anomaly(
    model: IsolationForest,
    scaler: StandardScaler,
    training_data: pd.DataFrame,
    event_features: np.ndarray,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Use SHAP TreeExplainer to identify the top ``top_n`` features
    that most contributed to the anomaly score.

    Parameters
    ----------
    model : IsolationForest
        The fitted model.
    scaler : StandardScaler
        The fitted scaler (used to transform the event).
    training_data : pd.DataFrame
        The raw (unscaled) training feature matrix — used for
        SHAP background sampling.
    event_features : np.ndarray
        1-D array of raw (unscaled) feature values for the event.
    top_n : int
        Number of top deviating features to return.

    Returns
    -------
    list[dict]  — Each dict has:
        { "feature": str, "shap_value": float, "event_value": float }
    Sorted by absolute SHAP value descending.
    """
    # Scale the single event
    event_scaled = scaler.transform(event_features.reshape(1, -1))

    # Use TreeExplainer for Isolation Forest
    # We use a subsample of training data as background to keep it fast
    background_sample = training_data.sample(
        n=min(100, len(training_data)),
        random_state=42,
    )
    background_scaled = scaler.transform(background_sample)

    explainer = shap.TreeExplainer(
        model,
        data=background_scaled,
        feature_names=FEATURE_COLUMNS,
    )
    shap_values = explainer.shap_values(event_scaled)

    # shap_values is shape (1, n_features) — extract the single row
    sv = shap_values[0]

    # Pair each SHAP value with its feature name and actual event value
    feature_impacts = []
    for i, fname in enumerate(FEATURE_COLUMNS):
        feature_impacts.append({
            "feature": fname,
            "shap_value": round(float(sv[i]), 6),
            "event_value": round(float(event_features[i]), 4),
        })

    # Sort by absolute SHAP value — highest impact first
    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return feature_impacts[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#  EVENT EVALUATION (public API)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_event(event_data: dict[str, Any]) -> dict[str, Any]:
    """
    The main entry point called by FastAPI's /analyze_event endpoint.

    Takes a single event's data, runs it through the Isolation Forest,
    and returns the anomaly score + SHAP explanation if anomalous.

    Parameters
    ----------
    event_data : dict
        Must contain keys matching FEATURE_COLUMNS (or a superset).
        Example:
        {
            "user_id": "USR00057",
            "days_inactive": 14,
            "tenure_months": 22,
            "high_risk_flag": 1,
            "Tenure_Risk_Modifier": 2.0,
            "Equipment_Mismatch_Score": 0,
            "Temporal_Velocity": 15,
            "rowcount_deviation": 3.5,
            "After_Hours_High_Sensitivity": 1,
            "Failed_Action_Flag": 0,
            "systems_access_count": 4,
            "Privilege_Sensitivity_Mismatch": 0.75,
        }

    Returns
    -------
    dict with keys:
      - "user_id"           : str
      - "is_anomaly"        : bool
      - "anomaly_score"     : float (raw decision_function score;
                              more negative = more anomalous)
      - "prediction"        : int (-1 = anomaly, 1 = normal)
      - "top_shap_features" : list[dict] (only if anomalous, else [])
    """
    # --- Load / initialise model (cached after first call) ---
    model, scaler, training_data = _get_cached_model()

    # --- Extract feature vector in the correct column order ---
    try:
        feature_values = np.array(
            [float(event_data.get(col, 0)) for col in FEATURE_COLUMNS]
        )
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not parse feature values from event_data: {e}"
        ) from e

    # --- Scale and predict ---
    X_scaled = scaler.transform(feature_values.reshape(1, -1))
    prediction = model.predict(X_scaled)[0]            # -1 or 1
    anomaly_score = model.decision_function(X_scaled)[0]  # continuous score

    # --- Build response ---
    result: dict[str, Any] = {
        "user_id": event_data.get("user_id", "unknown"),
        "is_anomaly": prediction == -1,
        "anomaly_score": round(float(anomaly_score), 6),
        "prediction": int(prediction),
        "top_shap_features": [],
    }

    # --- SHAP explanation only for anomalies (saves compute on normals) ---
    if prediction == -1:
        result["top_shap_features"] = explain_anomaly(
            model=model,
            scaler=scaler,
            training_data=training_data,
            event_features=feature_values,
            top_n=3,
        )
        logger.warning(
            f"🚨 ANOMALY detected for user {result['user_id']} "
            f"(score: {result['anomaly_score']:.4f})"
        )
    else:
        logger.info(
            f"✓ Normal event for user {result['user_id']} "
            f"(score: {result['anomaly_score']:.4f})"
        )

    return result


# ---------------------------------------------------------------------------
#  Model Cache (singleton pattern — train once, reuse across requests)
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] | None = None


def _get_cached_model() -> tuple[IsolationForest, StandardScaler, pd.DataFrame]:
    """Return the cached (model, scaler, training_data) tuple."""
    global _model_cache
    if _model_cache is None:
        model, scaler, training_data = train_model()
        _model_cache = {
            "model": model,
            "scaler": scaler,
            "training_data": training_data,
        }
    return (
        _model_cache["model"],
        _model_cache["scaler"],
        _model_cache["training_data"],
    )


def reset_model_cache() -> None:
    """Force the next evaluate_event() call to retrain."""
    global _model_cache
    _model_cache = None
    logger.info("Model cache cleared — will retrain on next call.")


# ---------------------------------------------------------------------------
#  CLI entry point — train and validate
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 72)
    logger.info("  Neuro-SOC  |  ML Engine — Training & Validation")
    logger.info("=" * 72)

    model, scaler, training_data = train_model(force_retrain=True)

    # Quick validation: score all training data and show distribution
    X_scaled = scaler.transform(training_data)
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    n_anomalies = (predictions == -1).sum()
    logger.info(f"\n{'='*50}")
    logger.info(f"  Anomaly Score Distribution:")
    logger.info(f"    min   = {scores.min():.4f}")
    logger.info(f"    max   = {scores.max():.4f}")
    logger.info(f"    mean  = {scores.mean():.4f}")
    logger.info(f"    std   = {scores.std():.4f}")
    logger.info(f"  Anomalies: {n_anomalies} / {len(predictions)}")
    logger.info(f"{'='*50}")

    # Test evaluate_event with the first row
    sample_row = training_data.iloc[0].to_dict()
    sample_row["user_id"] = "USR_TEST"
    result = evaluate_event(sample_row)
    logger.info(f"\n  Sample evaluation result: {result}")

    logger.info("\n  ✅ ML Engine ready.")
