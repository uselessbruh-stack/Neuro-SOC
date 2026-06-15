# Neuro-SOC — Deliverables

> Project documentation, evaluation outputs, and analysis artifacts for the Neuro-SOC Insider Threat Detection System.

---

## Contents

This directory contains all deliverables produced during the development and evaluation of Neuro-SOC.

### 📄 Documentation

| File | Description |
|------|-------------|
| [Project_Documentation.md](./Project_Documentation.md) | **Comprehensive project documentation** — the primary deliverable. Covers the full problem statement, solution architecture (three-tiered "Inversion of Responsibility" pipeline), technology stack, data generation strategy (50,000 synthetic events with 10 anomaly types), feature engineering (23 features), ML engine design (Isolation Forest + SHAP + ONNX), LLM integration (constrained Llama-3), semantic caching (Redis), frontend dashboard, evaluation results (F1: 0.794), false positive mitigation, scaling roadmap, regulatory compliance (GDPR, NIST, SOX), and challenges faced. |
| [Technical_Documentation.md](./Technical_Documentation.md) | **Technical reference** — a focused, engineer-oriented document. Details the ML algorithm selection rationale (Isolation Forest vs. Autoencoder vs. DAGMM), hyperparameter choices (`contamination=0.43`), feature preprocessing (StandardScaler), anomaly scoring formula, SHAP TreeExplainer integration, LLM prompt engineering (temperature 0.1, strict JSON schema), semantic cache architecture (SHA-256 keying, Redis + in-memory fallback, 1-hour TTL), and scaling strategy (Kafka, ONNX, vLLM, Kubernetes). |
| [False_Positive_Analysis.md](./False_Positive_Analysis.md) | **False positive mitigation analysis** — demonstrates how the three-tiered architecture handles three common enterprise FP scenarios: (1) legitimate month-end financial bulk exports, (2) temporary role changes / on-call duty rotations, and (3) seasonal or calendar-driven behavioral shifts. Projects a blended FP rate of <15% vs. the 40–55% industry average for traditional SIEMs. |

### 📊 Exploratory Data Analysis

| File | Description |
|------|-------------|
| [EDA_and_Feature_Importance.ipynb](./EDA_and_Feature_Importance.ipynb) | **Jupyter notebook** — full exploratory data analysis of the 50,000-event dataset. Contains feature distributions, correlation analysis, SHAP global feature importance (beeswarm and bar plots), anomaly score distributions, and model evaluation visualisations. |

### 📈 Visualisation Outputs

| File | Description |
|------|-------------|
| [feature_distributions.png](./feature_distributions.png) | Distribution plots for all 23 engineered features across normal vs. anomalous events |
| [shap_global_importance.png](./shap_global_importance.png) | SHAP global feature importance bar chart — mean absolute SHAP values across all events |
| [shap_summary_beeswarm.png](./shap_summary_beeswarm.png) | SHAP beeswarm plot — per-feature SHAP value distributions showing impact direction and magnitude |
| [shap_waterfall_single.png](./shap_waterfall_single.png) | SHAP waterfall plot for a single high-risk event — shows how each feature pushed the prediction |
| [anomaly_score_distribution.png](./anomaly_score_distribution.png) | Distribution of Isolation Forest anomaly scores across all 50,000 events (normal vs. anomalous) |
| [action_sensitivity.png](./action_sensitivity.png) | Sensitivity analysis of recommended action thresholds (Quarantine / Investigate / Monitor) |
| [temporal_baselining.png](./temporal_baselining.png) | Temporal velocity and after-hours access patterns across departments |

### 📦 Model Outputs

| File | Description |
|------|-------------|
| [flagged_anomalies_output.json](./flagged_anomalies_output.json) | JSON export of the top 50 flagged anomalies with full SHAP explanations, LLM narratives, event context, and ground truth labels — representative output of the `/flagged_events` endpoint |

---

## Key Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Precision** | > 75% | **80.3%** | ✅ Exceeded |
| **Recall** | > 70% | **78.5%** | ✅ Exceeded |
| **F1 Score** | > 0.72 | **0.794** | ✅ Exceeded |
| **False Positive Rate** | < 20% | **< 15% projected** | ✅ On target |
| **Detection Latency** | < 5 min | **< 30 seconds** | ✅ Exceeded |

---

*Part of the [Neuro-SOC](../../README.md) Insider Threat Detection System*
