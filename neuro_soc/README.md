# Neuro-SOC — Insider Threat Detection Platform

> **Problem Statement 04:** Data Access Audit & Insider Threat Detection  
> Detect abnormal data access patterns before sensitive information leaves the building.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Neuro-SOC Architecture                       │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Data Pipeline│───▶│  ML Engine   │───▶│   FastAPI Backend  │  │
│  │ (Pandas)     │    │ IsolationForest│  │   + Redis Cache    │  │
│  │              │    │ + SHAP XAI   │    │   + Llama-3 LLM    │  │
│  └─────────────┘    └──────────────┘    └────────┬───────────┘  │
│                                                   │              │
│                                          ┌────────▼───────────┐  │
│                                          │  Next.js Dashboard  │  │
│                                          │  Human-in-the-Loop  │  │
│                                          └────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Three-Tiered Detection Pipeline:**
1. **ML Baselining** — Isolation Forest trained on 11 engineered features
2. **LLM Translation** — Llama-3-8B generates objective threat narratives (T=0.1)
3. **Human-in-the-Loop** — SOC analyst makes final Quarantine/False Positive decision

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Redis Cloud account (free tier) or local Redis

### 1. Backend Setup

```bash
cd neuro_soc/backend
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
cd neuro_soc/data_pipeline
python enrichment.py
```

Output: `processed_data/enriched_features.csv` (1,200 events × 11 engineered features)

### 3. Start Backend

```bash
cd neuro_soc/backend
export HF_TOKEN="your_huggingface_token"
export REDIS_URL="redis://default:password@host:port"
python main.py
```

Server runs on `http://localhost:8000`. API docs at `/docs`.

### 4. Start Frontend

```bash
cd neuro_soc/frontend
npm install
npm run dev
```

Dashboard runs on `http://localhost:3000`.

---

## Project Structure

```
neuro_soc/
├── backend/
│   ├── main.py                  # FastAPI server + Redis cache + Llama-3 integration
│   ├── ml_engine.py             # Isolation Forest + SHAP explainability
│   ├── requirements.txt         # Python dependencies
│   └── models/                  # Saved model files (.joblib)
│
├── data_pipeline/
│   ├── enrichment.py            # Feature engineering pipeline
│   ├── raw_data/                # Source CSVs (data_access_logs, user_profiles)
│   └── processed_data/          # Enriched features output
│
├── frontend/
│   ├── app/page.tsx             # Next.js SOC dashboard
│   ├── app/layout.tsx           # Root layout with dark mode
│   └── package.json             # Node dependencies
│
└── deliverables/
    ├── EDA_and_Feature_Importance.ipynb   # EDA + SHAP analysis notebook
    ├── flagged_anomalies_output.json      # 22 flagged anomalies with narratives
    ├── False_Positive_Analysis.md         # FP mitigation strategies
    ├── Technical_Documentation.md         # ML approach + scaling
    └── Pitch_Script.md                    # 5-minute presentation script
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data Pipeline | Pandas | Feature engineering (11 features) |
| ML Engine | Scikit-learn, SHAP | Isolation Forest anomaly detection + explainability |
| Backend API | FastAPI, Uvicorn | REST API with async support |
| Semantic Cache | Redis Cloud | SHA-256 hashed cache with 1-hour TTL |
| LLM | Llama-3-8B (HuggingFace) | Constrained threat narrative generation |
| Frontend | Next.js 16, TypeScript, Tailwind CSS | SOC analyst dashboard |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze_event` | Analyze a single access event (ML + SHAP + LLM) |
| `GET` | `/health` | Service health check (Redis, LLM, model status) |
| `DELETE` | `/cache` | Flush semantic cache |

---

## Engineered Features

| Feature | Description | Source |
|---------|-------------|--------|
| `Tenure_Risk_Modifier` | Risk weight based on hire date + high_risk_flag | user_profiles |
| `Equipment_Mismatch_Score` | Unregistered device detection | access_logs |
| `Temporal_Velocity` | Event rate per hour (burst detection) | access_logs |
| `rowcount_deviation` | Ratio of daily access count to 30-day rolling average | access_logs |
| `After_Hours_High_Sensitivity` | Off-hours access to restricted resources | access_logs |
| `Failed_Action_Flag` | Authentication/authorization failures | access_logs |
| `Privilege_Sensitivity_Mismatch` | Access above user's standard clearance | access_logs + user_profiles |
| `systems_access_count` | Number of distinct systems user can access | user_profiles |

---

## Compliance Alignment

- **GDPR Article 32** — Monitoring unauthorized access, detecting exfiltration
- **GDPR Article 22** — Human-in-the-loop prevents automated enforcement decisions
- **NIST IR-4** — Detection → Analysis → Containment mapped to 3-tier architecture
- **SOX 302** — Audit trail of all access events and analyst decisions

---

## Team

Neuro-SOC Engineering Team — Hackathon 2026
