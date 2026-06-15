# Neuro-SOC — Backend (FastAPI + ML Engine)

> REST API layer that receives security events, checks a semantic cache, routes to the ML engine for anomaly detection, computes SHAP explanations, generates LLM-powered threat narratives via Llama-3, and returns fully explained results.

---

## Overview

The backend is the orchestration hub of Neuro-SOC. It combines **Tier 1** (ML Engine) and **Tier 2** (LLM Translator) of the three-tiered detection pipeline into a single FastAPI service. Every request flows through: Cache Check → Isolation Forest → SHAP → Llama-3 → Response.

```
Event ──▶ Semantic Cache ──▶ IsolationForest (ONNX) ──▶ SHAP ──▶ Llama-3 ──▶ JSON Response
              │ HIT                                                               │
              └──────────────── Cached Result (\<1ms) ──────────────────────────────┘
```

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **FastAPI** | ≥ 0.111 | Async REST API framework with automatic OpenAPI docs |
| **Uvicorn** | ≥ 0.30 | ASGI server |
| **Scikit-learn** | ≥ 1.5 | Isolation Forest anomaly detector |
| **SHAP** | ≥ 0.45 | TreeExplainer for per-event feature attribution |
| **ONNX Runtime** | ≥ 1.18 | Optimised model inference (\<1ms per event) |
| **skl2onnx** | ≥ 1.17 | Sklearn → ONNX model conversion |
| **Pandas** | ≥ 2.2 | Data loading and batch processing |
| **NumPy** | ≥ 1.26 | Numerical computation |
| **Redis** | ≥ 5.0 | Semantic cache (Redis Cloud primary, in-memory fallback) |
| **Hugging Face Hub** | ≥ 0.23 | Llama-3-8B-Instruct inference API |
| **Pydantic** | ≥ 2.7 | Request/response schema validation |
| **joblib** | ≥ 1.4 | Model serialisation |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe — returns service status, Redis connectivity, LLM config, and cache size |
| `POST` | `/analyze_event` | Analyse a single security event: cache check → ML scoring → SHAP → LLM narrative |
| `GET` | `/flagged_events?limit=50&refresh=false` | Batch-process all enriched features, score with Isolation Forest, compute SHAP + narratives, return top N flagged events with evaluation metrics |
| `DELETE` | `/cache` | Flush the semantic cache (Redis + in-memory fallback) |
| `GET` | `/docs` | Interactive Swagger/OpenAPI documentation |
| `GET` | `/redoc` | ReDoc API documentation |

---

## Architecture

### Module: `main.py` — API Layer

- **CORS middleware** configured for Next.js frontend on port 3000
- **Semantic Cache** with Redis Cloud (SHA-256 keyed, 1-hour TTL, in-memory fallback)
- **LLM Integration** with "Inversion of Responsibility" — the LLM (Llama-3-8B-Instruct, temperature 0.1) is a **deterministic translator**, NOT a decision maker
- **SOC System Prompt** constrains LLM output to strict JSON: `{threat_narrative, evidence_list, recommended_action}`
- **Rule-based fallback** ensures the API never blocks on LLM availability
- **23-entry human-readable feature dictionary** maps technical feature names to plain-English descriptions for auditor-facing narratives
- **Batch endpoint** (`/flagged_events`) processes the full dataset, computes evaluation metrics (Precision, Recall, F1) against ground truth, and returns the top anomalies with full SHAP + narrative payloads

### Module: `ml_engine.py` — ML Engine

- **Isolation Forest** with 200 trees, `contamination=0.43` (tuned to match 44% anomaly rate)
- **StandardScaler** preprocessing for equalised feature contribution
- **ONNX export** for optimised C++ inference (\<1ms per event)
- **SHAP TreeExplainer** with background sampling (100 events) for per-event feature attribution
- **Singleton cache** — model is trained once on first request, reused across all subsequent calls
- **Dual inference** — ONNX Runtime (fast path) with sklearn fallback

### Models Directory: `models/`

| File | Size | Description |
|------|------|-------------|
| `isolation_forest.joblib` | ~2.7 MB | Serialised sklearn Isolation Forest (200 trees) |
| `isolation_forest.onnx` | ~1.8 MB | ONNX-exported model for fast inference |
| `scaler.joblib` | ~1 KB | Fitted StandardScaler for feature normalisation |

---

## Semantic Cache

```
Request → SHA-256(canonical JSON payload) → Redis lookup
├── HIT  → Return cached result (<1ms, 0 LLM calls)
└── MISS → ML → SHAP → LLM → Cache store (TTL 3600s) → Return (3-5s)
```

| Configuration | Value | Env Variable |
|--------------|-------|-------------|
| Primary store | Redis Cloud | `REDIS_URL` |
| Fallback store | In-memory Python dict | — |
| Key prefix | `neurosoc:cache:` | — |
| TTL | 1 hour | `CACHE_TTL_SECONDS` |
| Key algorithm | SHA-256 of canonical JSON | — |

---

## Directory Structure

```
backend/
├── main.py              # FastAPI application — endpoints, cache, LLM integration
├── ml_engine.py          # Isolation Forest training, ONNX export, SHAP explanation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (HF_TOKEN, REDIS_URL)
├── models/               # Trained model artifacts
│   ├── isolation_forest.joblib
│   ├── isolation_forest.onnx
│   └── scaler.joblib
└── README.md             # ← You are here
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **Data pipeline** must have been run first (`../data_pipeline/enrichment.py`) to generate `enriched_features.csv`

### Environment Variables

Create a `.env` file (or set environment variables):

```env
# Hugging Face API token for Llama-3 inference
HF_TOKEN=hf_your_token_here

# Redis Cloud connection string (optional — falls back to in-memory)
REDIS_URL=redis://default:<password>@<host>:<port>

# Cache TTL in seconds (default: 3600)
CACHE_TTL_SECONDS=3600
```

### Install & Run

```bash
cd neuro_soc/backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Train the model manually
python ml_engine.py

# Start the API server
python main.py
```

The API will be available at [http://localhost:8000](http://localhost:8000).
Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### First Request

On the first `/flagged_events` call, the backend will:
1. Load the enriched CSV (~50,000 rows)
2. Train the Isolation Forest (if not cached on disk)
3. Export to ONNX format
4. Score all events + compute SHAP for the top 50
5. Cache the results in memory

> ⏱ **First call may take 30–60 seconds** due to SHAP computation. Subsequent calls return cached results instantly.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`contamination=0.43`** | Empirically tuned to match the 44% anomaly rate in ground truth — default 5% missed most threats |
| **`n_jobs=1`** | Prevents Windows multiprocessing RAM explosion (fork bomb under IsolationForest) |
| **`reload=False`** | Uvicorn reload spawns duplicate processes, each loading the full 50K dataset into RAM |
| **`gc.collect()` after batch** | Explicit memory freeing after processing — prevents OOM when running alongside the frontend |
| **LLM temperature 0.1** | Near-deterministic output for SOC auditability — same input always yields the same narrative |
| **Rule-based fallback** | System never blocks on LLM — if Llama-3 fails, SHAP values are translated via a deterministic rule engine |

---

*Part of the [Neuro-SOC](../../README.md) Insider Threat Detection System*
