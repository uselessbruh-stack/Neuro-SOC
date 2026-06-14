"""
Neuro-SOC — Full Retrain Script
================================
Run this from neuro_soc/ directory:
    python retrain.py

This script:
  1. Deletes old model files
  2. Runs the enrichment pipeline (v2.0)
  3. Trains IsolationForest on the new 19-feature dataset
  4. Exports to ONNX
  5. Evaluates against ground-truth labels
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure we can import from backend and data_pipeline
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "backend"))
sys.path.insert(0, str(SCRIPT_DIR / "data_pipeline"))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print("=" * 72)
print("  Neuro-SOC  |  Full Retrain Pipeline")
print("=" * 72)

# ─── Step 1: Clear old models ────────────────────────────────────────────
print("\n[1/5] Clearing old model files...")
models_dir = SCRIPT_DIR / "backend" / "models"
models_dir.mkdir(parents=True, exist_ok=True)

for f in models_dir.glob("*"):
    if f.is_file():
        f.unlink()
        print(f"  ✗ Deleted: {f.name}")
print("  ✓ Models directory clean")

# ─── Step 2: Run enrichment pipeline ─────────────────────────────────────
print("\n[2/5] Running enrichment pipeline v2.0...")
from data_pipeline.enrichment import main as run_enrichment
run_enrichment()

# Verify output exists
enriched_path = SCRIPT_DIR / "data_pipeline" / "processed_data" / "enriched_features.csv"
if not enriched_path.exists():
    print("  ✗ ERROR: enriched_features.csv was not created!")
    sys.exit(1)

enriched = pd.read_csv(enriched_path)
print(f"  ✓ Enriched dataset: {enriched.shape[0]} rows × {enriched.shape[1]} cols")

# ─── Step 3: Train IsolationForest ───────────────────────────────────────
print("\n[3/5] Training IsolationForest on 23 features...")
from ml_engine import (
    FEATURE_COLUMNS, IF_PARAMS, MODEL_PATH, SCALER_PATH, ONNX_PATH,
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Prepare features
X = enriched[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
print(f"  Feature matrix: {X.shape}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = IsolationForest(**IF_PARAMS)
model.fit(X_scaled)

# Save sklearn model + scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"  ✓ Model saved: {MODEL_PATH}")
print(f"  ✓ Scaler saved: {SCALER_PATH}")

# ─── Step 4: Export to ONNX ──────────────────────────────────────────────
print("\n[4/5] Exporting to ONNX...")
try:
    from skl2onnx import to_onnx
    import onnxruntime as ort

    onnx_model = to_onnx(model, X_scaled[:1].astype(np.float32),
                          target_opset={'': 17, 'ai.onnx.ml': 3})
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  ✓ ONNX model saved: {ONNX_PATH}")
    onnx_size = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"    Size: {onnx_size:.2f} MB")

    # Validate ONNX vs sklearn
    session = ort.InferenceSession(str(ONNX_PATH))
    input_name = session.get_inputs()[0].name
    onnx_preds = session.run(None, {input_name: X_scaled.astype(np.float32)})

    # Compare predictions
    sklearn_preds = model.predict(X_scaled)
    onnx_pred_labels = onnx_preds[0].flatten()

    match_rate = (sklearn_preds == onnx_pred_labels).mean() * 100
    print(f"  ✓ ONNX validation: {match_rate:.1f}% match with sklearn")
except Exception as e:
    print(f"  ⚠ ONNX export failed: {e}")
    print("    Falling back to sklearn-only inference.")

# ─── Step 5: Evaluate against ground-truth labels ────────────────────────
print("\n[5/5] Evaluating against ground-truth labels...")
labels_path = SCRIPT_DIR / "data_pipeline" / "raw_data" / "data_access_labels.csv"

if not labels_path.exists():
    print("  ⚠ data_access_labels.csv not found — skipping evaluation")
else:
    labels = pd.read_csv(labels_path)
    print(f"  Labels loaded: {labels.shape[0]} rows")

    # Ground truth: is_anomaly column (True/False → 1/0)
    y_true = labels["is_anomaly"].map({True: 1, False: 0, "True": 1, "False": 0}).values

    # Model predictions: IsolationForest returns -1 (anomaly) or 1 (normal)
    sklearn_preds = model.predict(X_scaled)
    y_pred = (sklearn_preds == -1).astype(int)  # Convert: -1 → 1 (anomaly), 1 → 0 (normal)

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │        EVALUATION RESULTS                    │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Precision:  {precision:.4f}  (target > 0.75) {'✅' if precision > 0.75 else '❌'} │")
    print(f"  │  Recall:     {recall:.4f}  (target > 0.70) {'✅' if recall > 0.70 else '❌'} │")
    print(f"  │  F1 Score:   {f1:.4f}  (target > 0.72) {'✅' if f1 > 0.72 else '❌'} │")
    print(f"  └─────────────────────────────────────────────┘")

    # Anomaly count comparison
    true_anomalies = y_true.sum()
    pred_anomalies = y_pred.sum()
    print(f"\n  True anomalies:      {true_anomalies}/{len(y_true)} ({true_anomalies/len(y_true)*100:.1f}%)")
    print(f"  Predicted anomalies: {pred_anomalies}/{len(y_pred)} ({pred_anomalies/len(y_pred)*100:.1f}%)")

    # Detailed breakdown
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    # Per anomaly-type recall
    if "anomaly_type" in labels.columns:
        print("  Per Anomaly-Type Detection Rate:")
        print("  ─" * 30)
        for atype in labels[labels["is_anomaly"] == True]["anomaly_type"].unique():
            if atype == "NORMAL":
                continue
            mask = labels["anomaly_type"] == atype
            atype_indices = labels[mask].index.tolist()
            if atype_indices:
                atype_preds = y_pred[atype_indices]
                atype_recall = atype_preds.mean()
                count = len(atype_indices)
                detected = atype_preds.sum()
                status = "✅" if atype_recall >= 0.70 else "⚠️"
                print(f"    {status} {atype:30s}  {detected:3d}/{count:3d}  ({atype_recall*100:.1f}%)")

print("\n" + "=" * 72)
print("  Neuro-SOC  |  Full Retrain Pipeline — COMPLETE")
print("=" * 72)
