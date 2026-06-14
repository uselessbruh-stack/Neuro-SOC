"""
===============================================================================
  Neuro-SOC  |  FastAPI Backend — Event Analysis, Semantic Cache & LLM
  --------------------------------------------------------------------------
  Module      : backend/main.py
  Purpose     : REST API layer that receives security events, checks a
                local semantic cache, routes to the ML engine for anomaly
                detection, generates LLM-powered threat narratives via
                Hugging Face Llama-3, and returns fully explained results.
  Author      : Neuro-SOC Engineering Team
  Created     : 2026-06-13
  Python      : 3.11+
  Dependencies: fastapi, uvicorn, pydantic, huggingface_hub
===============================================================================

ENDPOINTS:
  GET  /health          → Liveness probe
  POST /analyze_event   → Anomaly detection + LLM narrative (cached)
  DELETE /cache         → Flush semantic cache
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import gc
import os
import re
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
import redis
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score as _precision_score,
    recall_score as _recall_score,
    f1_score as _f1_score,
)

# Local ML engine import
from ml_engine import evaluate_event, FEATURE_COLUMNS, _get_cached_model

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuro_soc.api")

# ---------------------------------------------------------------------------
#  Hugging Face LLM Configuration
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TEMPERATURE = 0.1          # Near-deterministic for SOC auditability
HF_MAX_TOKENS = 512           # Enough for structured JSON, not novels

# Initialise the Inference Client (lazy — won't fail if token is missing
# until an actual LLM call is made)
hf_client = InferenceClient(
    model=HF_MODEL,
    token=HF_TOKEN or None,
)

# ---------------------------------------------------------------------------
#  SOC System Prompt — "Inversion of Responsibility" constraints
#  The LLM is NOT the decision-maker. It is a deterministic translator
#  of pre-computed ML outputs into human-readable SOC narratives.
# ---------------------------------------------------------------------------
SOC_SYSTEM_PROMPT = """You are a deterministic SOC translation agent. You are provided with a confirmed anomaly score, top three deviating features, and a user profile. You must translate this into a plain-English explanation using ONLY the provided data. Do not infer malicious intent; state the deviations objectively.

You MUST respond with a raw, valid JSON object containing exactly these three keys:
- "threat_narrative": A concise 2-3 sentence plain-English summary of why this event is anomalous based solely on the provided SHAP features and anomaly score.
- "evidence_list": An array of exactly 3 strings, each mapping one SHAP feature to its deviation in plain English.
- "recommended_action": Exactly one of: "Quarantine", "Monitor", or "Investigate".

Rules:
- Output ONLY the raw JSON object. No markdown code fences, no backticks, no conversational text before or after.
- Do not add any keys beyond the three specified.
- Base your recommended_action on severity: anomaly_score < -0.3 → "Quarantine", -0.3 to -0.1 → "Investigate", > -0.1 → "Monitor".
- Never speculate about user intent. Only describe measurable deviations."""

# ---------------------------------------------------------------------------
#  FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Neuro-SOC API",
    description=(
        "Insider Threat Detection API for Banking SOC — "
        "Anomaly scoring with IsolationForest + SHAP explainability "
        "+ Llama-3 threat narrative generation."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
#  CORS — allow the Next.js frontend (default dev port 3000) to call this API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
#  Human-Readable Feature Dictionary — for SOC auditor-facing narratives
#  Maps raw ML feature names → plain-English descriptions.
# ---------------------------------------------------------------------------
FEATURE_DISPLAY: dict[str, dict[str, str]] = {
    "tenure_months":               {"name": "Employee Tenure",            "desc": "length of employment at the organisation"},
    "high_risk_flag":              {"name": "HR Risk Flag",               "desc": "employee flagged by HR as high-risk (PIP, termination notice, etc.)"},
    "notice_period_flag":          {"name": "Notice Period",              "desc": "employee is currently serving their notice period"},
    "failed_logins_30d":           {"name": "Failed Logins (30 days)",    "desc": "number of failed authentication attempts in the past 30 days"},
    "stale_account_days":          {"name": "Account Inactivity",         "desc": "number of days the account was dormant before this access"},
    "approved_assets_count":       {"name": "Approved Asset Count",       "desc": "number of data assets the employee is authorised to access"},
    "Tenure_Risk_Modifier":        {"name": "Tenure-Based Risk Weight",   "desc": "risk multiplier based on employee tenure and HR risk status"},
    "Equipment_Risk_Score":        {"name": "Device Risk",                "desc": "risk score based on equipment mismatch or unauthorised device usage"},
    "Access_Tier_Mismatch":        {"name": "Access Level Mismatch",      "desc": "discrepancy between the employee's clearance level and the resource sensitivity"},
    "Cross_Dept_Access_Flag":      {"name": "Cross-Department Access",    "desc": "access to resources belonging to a different department"},
    "Rowcount_Deviation":          {"name": "Unusual Data Volume",        "desc": "volume of records accessed is significantly above the employee's baseline"},
    "Exfiltration_Dest_Score":     {"name": "Destination Risk",           "desc": "data was sent to a high-risk destination (external email, USB, cloud storage)"},
    "Query_Type_Risk":             {"name": "Query Type Risk",            "desc": "the type of database query used carries elevated risk (bulk export, DELETE, etc.)"},
    "Weak_Auth_Flag":              {"name": "Weak Authentication",        "desc": "access was performed using weak or single-factor authentication"},
    "Suspicious_Geo_Flag":         {"name": "Suspicious Location",        "desc": "access originated from an unusual or high-risk geographic location"},
    "VPN_Mismatch":                {"name": "VPN Anomaly",                "desc": "VPN usage pattern is inconsistent with employee's normal behaviour"},
    "Temporal_Velocity":           {"name": "Rapid Access Burst",         "desc": "unusually high number of access events in a short time window"},
    "After_Hours_High_Sensitivity":{"name": "After-Hours Sensitive Access","desc": "high-sensitivity resource accessed outside normal business hours"},
    "Failed_Action_Flag":          {"name": "Failed Action Detected",     "desc": "one or more attempted actions resulted in access denial or failure"},
    "Cross_Dept_Sensitivity":      {"name": "Cross-Dept Sensitive Access","desc": "accessed sensitive resources belonging to another department"},
    "Time_Sensitivity_Risk":       {"name": "Off-Hours Sensitivity Risk", "desc": "combined risk of after-hours access and resource sensitivity level"},
    "Stale_Sensitivity_Risk":      {"name": "Dormant Account Risk",       "desc": "previously dormant account accessed sensitive resources"},
    "Volume_Dest_Compound":        {"name": "Bulk Export to Risky Dest",  "desc": "large data volume sent to a high-risk external destination"},
}


def _friendly_name(feature: str) -> str:
    """Return the auditor-friendly display name for a feature."""
    entry = FEATURE_DISPLAY.get(feature)
    if entry:
        return entry["name"]
    return feature.replace("_", " ").title()


def _friendly_desc(feature: str) -> str:
    """Return the auditor-friendly description for a feature."""
    entry = FEATURE_DISPLAY.get(feature)
    if entry:
        return entry["desc"]
    return f"{feature.replace('_', ' ').lower()} deviated from normal"


def _fmt_name(raw: str) -> str:
    """kavya.dubois → Kavya Dubois, USR_0042 → USR-0042."""
    if "." in raw and not raw.startswith("USR"):
        return " ".join(part.capitalize() for part in raw.split("."))
    return raw


def _fmt_asset(raw: str) -> str:
    """All_Financial → All Financial, HR_Records → HR Records."""
    return raw.replace("_", " ")


def _fmt_ts(raw: str) -> str:
    """2025-10-10 00:08:23 → 10 Oct 2025 at 00:08."""
    try:
        from datetime import datetime as _dt
        dt = _dt.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %b %Y at %H:%M")
    except Exception:
        return str(raw)


def _severity_word(score: int) -> str:
    """Map risk score to a human severity word."""
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 40:
        return "elevated"
    return "low"


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC CACHE — Redis Cloud (with in-memory fallback)
#  --------------------------------------------------------------------------
#  Primary: Redis Cloud for persistent, shared, TTL-enabled caching.
#  Fallback: In-memory dict if Redis is unreachable (app never breaks).
#
#  Cache Key = SHA-256( user_id + sorted(event_feature_values) )
#  Cache TTL = 1 hour (configurable via CACHE_TTL_SECONDS)
#
#  Set REDIS_URL env var to your Redis Cloud connection string:
#    redis://default:<password>@<host>:<port>
# ═══════════════════════════════════════════════════════════════════════════
REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))  # 1 hour
CACHE_KEY_PREFIX = "neurosoc:cache:"  # namespace to avoid key collisions

# ---------------------------------------------------------------------------
#  Data File Paths (for batch processing endpoint)
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_ENRICHED_CSV = _PROJECT_ROOT / "data_pipeline" / "processed_data" / "enriched_features.csv"
_LABELS_CSV = _PROJECT_ROOT / "data_pipeline" / "raw_data" / "data_access_labels.csv"
_LOGS_CSV = _PROJECT_ROOT / "data_pipeline" / "raw_data" / "data_access_logs.csv"

# In-memory fallback cache
_fallback_cache: dict[str, dict[str, Any]] = {}

# Redis client — initialised lazily; None if Redis is unavailable
_redis_client: redis.Redis | None = None


def _init_redis() -> redis.Redis | None:
    """Attempt to connect to Redis Cloud. Returns None on failure."""
    global _redis_client
    if not REDIS_URL:
        logger.info("  ℹ REDIS_URL not set — using in-memory fallback cache.")
        return None
    try:
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        client.ping()
        logger.info(f"  ✓ Connected to Redis Cloud: {REDIS_URL.split('@')[-1]}")
        _redis_client = client
        return client
    except Exception as e:
        logger.warning(f"  ⚠ Redis connection failed: {e} — using in-memory fallback.")
        return None


def _compute_cache_key(event: dict[str, Any]) -> str:
    """
    Generate a deterministic hash from user_id + feature values.

    We sort the keys so that field order doesn't affect the hash.
    Non-feature fields (like timestamp) are included so that
    truly identical events share a cache entry but the same user
    doing something different gets a fresh evaluation.
    """
    canonical = json.dumps(event, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_lookup(key: str) -> dict[str, Any] | None:
    """
    Look up a cache entry. Tries Redis first, falls back to in-memory.
    """
    full_key = CACHE_KEY_PREFIX + key

    # Try Redis
    if _redis_client is not None:
        try:
            cached = _redis_client.get(full_key)
            if cached is not None:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"  ⚠ Redis GET failed: {e} — checking fallback.")

    # Fallback to in-memory
    return _fallback_cache.get(key)


def _cache_store(key: str, result: dict[str, Any]) -> None:
    """
    Store a result in the cache. Writes to both Redis and in-memory.
    """
    full_key = CACHE_KEY_PREFIX + key
    serialized = json.dumps(result, default=str)

    # Write to Redis (with TTL)
    if _redis_client is not None:
        try:
            _redis_client.setex(full_key, CACHE_TTL_SECONDS, serialized)
            logger.info(
                f"  📦 Redis STORE — key={key[:12]}… "
                f"(TTL={CACHE_TTL_SECONDS}s)"
            )
        except Exception as e:
            logger.warning(f"  ⚠ Redis SET failed: {e} — storing in fallback only.")

    # Always write to in-memory fallback too
    _fallback_cache[key] = result


def _cache_clear() -> int:
    """
    Flush all Neuro-SOC cache entries. Returns count of entries removed.
    """
    count = len(_fallback_cache)
    _fallback_cache.clear()

    if _redis_client is not None:
        try:
            # Delete all keys with our prefix
            cursor = 0
            redis_count = 0
            while True:
                cursor, keys = _redis_client.scan(cursor, match=f"{CACHE_KEY_PREFIX}*", count=100)
                if keys:
                    _redis_client.delete(*keys)
                    redis_count += len(keys)
                if cursor == 0:
                    break
            count = max(count, redis_count)
            logger.info(f"  🗑️  Redis cleared — {redis_count} keys removed.")
        except Exception as e:
            logger.warning(f"  ⚠ Redis FLUSH failed: {e}")

    return count


# ═══════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE SCHEMAS (Pydantic v2)
# ═══════════════════════════════════════════════════════════════════════════

class EventRequest(BaseModel):
    """
    Schema for a single security event submitted for analysis.
    All feature fields have defaults so that partial events can still
    be scored (missing features default to 0 / baseline).
    """
    user_id: str = Field(..., description="Employee identifier (e.g. USR00057)")
    days_inactive: float = Field(0, description="Days since last login")
    tenure_months: float = Field(0, description="Months since hire date")
    high_risk_flag: int = Field(0, description="1 if admin/power-user/service-account")
    Tenure_Risk_Modifier: float = Field(1.0, description="Composite tenure-privilege risk weight")
    Equipment_Mismatch_Score: int = Field(0, description="1 if policy-violating device")
    Temporal_Velocity: int = Field(1, description="Events by this user in last 1h")
    rowcount_deviation: float = Field(1.0, description="Daily event count / user's average")
    After_Hours_High_Sensitivity: int = Field(0, description="1 if after-hours + high sensitivity")
    Failed_Action_Flag: int = Field(0, description="1 if the action failed")
    systems_access_count: int = Field(1, description="Number of systems this user can access")
    Privilege_Sensitivity_Mismatch: float = Field(1.0, description="Sensitivity / access breadth ratio")

    model_config = {"json_schema_extra": {
        "examples": [{
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
        }]
    }}


class ShapFeature(BaseModel):
    """A single SHAP feature contribution."""
    feature: str
    shap_value: float
    event_value: float


class LlmNarrative(BaseModel):
    """Structured LLM output — constrained to exactly 3 keys."""
    threat_narrative: str = Field("", description="Plain-English anomaly explanation")
    evidence_list: list[str] = Field(default_factory=list, description="SHAP feature deviations in plain English")
    recommended_action: str = Field("Monitor", description="Quarantine | Investigate | Monitor")


class EventResponse(BaseModel):
    """Response schema for the /analyze_event endpoint."""
    user_id: str
    is_anomaly: bool
    anomaly_score: float
    prediction: int = Field(description="-1 = anomaly, 1 = normal")
    top_shap_features: list[ShapFeature] = []
    llm_narrative: LlmNarrative | None = Field(None, description="LLM-generated threat narrative (anomalies only)")
    cache_hit: bool = Field(False, description="True if result was served from cache")
    analyzed_at: str = Field(description="ISO 8601 timestamp of analysis")


class GroundTruth(BaseModel):
    """Ground truth label from data_access_labels.csv."""
    is_anomaly: bool
    anomaly_type: str
    severity: str
    explanation: str


class FlaggedEvent(BaseModel):
    """A single flagged event with ML analysis and ground truth."""
    access_id: str
    user_id: str
    username: str
    department: str
    timestamp: str
    data_asset: str
    data_sensitivity: str
    query_type: str
    destination: str
    anomaly_score: float
    risk_score: int
    is_anomaly_predicted: bool
    prediction: int
    top_shap_features: list[ShapFeature] = []
    llm_narrative: LlmNarrative | None = None
    ground_truth: GroundTruth


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics against ground truth labels."""
    total_events: int
    total_predicted_anomalies: int
    total_ground_truth_anomalies: int
    precision: float
    recall: float
    f1_score: float


class FlaggedEventsResponse(BaseModel):
    """Response for the /flagged_events endpoint."""
    metrics: EvaluationMetrics
    flagged_events: list[FlaggedEvent]


# ---------------------------------------------------------------------------
#  Utility: score normalisation (same formula as frontend)
# ---------------------------------------------------------------------------
def _normalize_score(raw: float) -> int:
    clamped = max(-0.5, min(0.5, raw))
    return round(((0.5 - clamped) / 1.0) * 100)


# Cache for batch-processed flagged events
_flagged_events_cache: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Liveness probe — returns 200 when the service is running."""
    redis_status = "disconnected"
    if _redis_client is not None:
        try:
            _redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "error"

    return {
        "status": "healthy",
        "service": "neuro-soc-backend",
        "version": "0.3.0",
        "llm_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "cache_backend": "redis" if _redis_client else "in-memory",
        "redis_status": redis_status,
        "fallback_cache_size": len(_fallback_cache),
    }


@app.post("/analyze_event", response_model=EventResponse)
async def analyze_event(event: EventRequest):
    """
    Analyze a single security event for insider-threat anomalies.

    Workflow:
      1. Hash the event to check the semantic cache.
      2. If cache HIT → return the stored result immediately.
      3. If cache MISS → route to ml_engine.evaluate_event().
      4. If anomalous → send SHAP data to Llama-3 for narrative generation.
      5. Parse LLM JSON, store full result in cache, return to client.
    """
    event_dict = event.model_dump()
    now = datetime.utcnow().isoformat() + "Z"

    # ---- Step 1: Semantic Cache Lookup ------------------------------------
    cache_key = _compute_cache_key(event_dict)
    cached_result = _cache_lookup(cache_key)

    if cached_result is not None:
        logger.info(
            f"⚡ Cache HIT for user {event.user_id} — "
            f"key={cache_key[:12]}…"
        )
        return EventResponse(
            **cached_result,
            cache_hit=True,
            analyzed_at=now,
        )

    logger.info(f"🔍 Cache MISS for user {event.user_id} — running ML pipeline …")

    # ---- Step 2: ML Engine Evaluation -------------------------------------
    try:
        ml_result = evaluate_event(event_dict)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"ML model not ready: {e}. "
                f"Run `python ml_engine.py` to train the model first."
            ),
        )
    except Exception as e:
        logger.exception(f"ML engine error for user {event.user_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal ML engine error: {str(e)}",
        )

    # ---- Step 3: LLM Narrative (only for anomalies) -----------------------
    llm_narrative = None

    if ml_result["is_anomaly"]:
        llm_narrative = await _generate_threat_narrative(
            user_id=event.user_id,
            anomaly_score=ml_result["anomaly_score"],
            shap_features=ml_result["top_shap_features"],
            event_data=event_dict,
        )

    # ---- Step 4: Assemble full result & cache -----------------------------
    full_result = {
        **ml_result,
        "llm_narrative": llm_narrative,
    }
    _cache_store(cache_key, full_result)

    # ---- Step 5: Return ---------------------------------------------------
    return EventResponse(
        **full_result,
        cache_hit=False,
        analyzed_at=now,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH FLAGGED EVENTS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/flagged_events", response_model=FlaggedEventsResponse)
async def get_flagged_events(limit: int = 50, refresh: bool = False):
    """
    Batch-process enriched features, score with Isolation Forest,
    compute SHAP explanations, and return the top N flagged events
    with evaluation metrics against ground truth labels.

    Results are cached in memory after first computation.
    Use refresh=true to force recomputation.
    """
    global _flagged_events_cache

    # Return cached results if available
    if _flagged_events_cache is not None and not refresh:
        cached_events = _flagged_events_cache["flagged_events"][:limit]
        return FlaggedEventsResponse(
            metrics=_flagged_events_cache["metrics"],
            flagged_events=cached_events,
        )

    logger.info("📊 Computing flagged events (first call may take 30-60s) ...")

    # ---- Load data files ----
    enriched = pd.read_csv(_ENRICHED_CSV)
    labels = pd.read_csv(_LABELS_CSV)
    logs = pd.read_csv(_LOGS_CSV)
    logger.info(f"  Loaded: {len(enriched)} enriched, {len(labels)} labels, {len(logs)} logs")

    # ---- Get model ----
    model, scaler, training_data, onnx_session = _get_cached_model()

    # ---- Prepare features and predict all events ----
    X = enriched[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    # ---- Compute evaluation metrics ----
    y_true = labels["is_anomaly"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).values
    y_pred = (predictions == -1).astype(int)

    metrics = EvaluationMetrics(
        total_events=len(enriched),
        total_predicted_anomalies=int(y_pred.sum()),
        total_ground_truth_anomalies=int(y_true.sum()),
        precision=round(float(_precision_score(y_true, y_pred)), 4),
        recall=round(float(_recall_score(y_true, y_pred)), 4),
        f1_score=round(float(_f1_score(y_true, y_pred)), 4),
    )
    logger.info(
        f"  Metrics: P={metrics.precision} R={metrics.recall} F1={metrics.f1_score}"
    )

    # ---- Top N anomalies by score (most anomalous first) ----
    anomaly_indices = np.where(predictions == -1)[0]
    anomaly_scores = scores[anomaly_indices]
    sorted_order = np.argsort(anomaly_scores)  # ascending = most negative first
    max_compute = min(len(sorted_order), max(limit, 50))
    top_indices = anomaly_indices[sorted_order[:max_compute]]

    # ---- Create SHAP explainer ONCE for efficiency ----
    logger.info(f"  Creating SHAP explainer for {max_compute} events ...")
    bg = training_data.sample(n=min(50, len(training_data)), random_state=42)
    bg_scaled = scaler.transform(bg)
    explainer = shap.TreeExplainer(
        model, data=bg_scaled, feature_names=FEATURE_COLUMNS
    )

    # ---- Build fast index lookups ----
    labels_lkp = labels.drop_duplicates(subset="access_id").set_index("access_id")
    logs_lkp = logs.drop_duplicates(subset="access_id").set_index("access_id")

    # ---- Process each flagged event ----
    flagged_events: list[FlaggedEvent] = []
    for count, idx in enumerate(top_indices):
        row = enriched.iloc[idx]
        aid = row["access_id"]

        # Ground truth
        if aid in labels_lkp.index:
            lbl = labels_lkp.loc[aid]
            is_anom = lbl["is_anomaly"]
            if isinstance(is_anom, str):
                is_anom = is_anom == "True"
            gt = GroundTruth(
                is_anomaly=bool(is_anom),
                anomaly_type=str(lbl["anomaly_type"]),
                severity=str(lbl.get("severity", "NONE")),
                explanation=str(lbl.get("explanation", "")),
            )
        else:
            gt = GroundTruth(
                is_anomaly=False,
                anomaly_type="UNKNOWN",
                severity="NONE",
                explanation="",
            )

        # Log context
        if aid in logs_lkp.index:
            lg = logs_lkp.loc[aid]
            username = str(lg.get("username", "unknown"))
            department = str(lg.get("department", "unknown"))
            data_asset = str(lg.get("data_asset", "unknown"))
            data_sensitivity = str(lg.get("data_sensitivity", "unknown"))
            query_type = str(lg.get("query_type", "unknown"))
            destination = str(lg.get("destination", "unknown"))
        else:
            username = department = data_asset = "unknown"
            data_sensitivity = query_type = destination = "unknown"

        # SHAP values
        sv = explainer.shap_values(X_scaled[idx : idx + 1])[0]
        impacts = [
            {
                "feature": FEATURE_COLUMNS[i],
                "shap_value": round(float(sv[i]), 6),
                "event_value": round(float(X[idx][i]), 4),
            }
            for i in range(len(FEATURE_COLUMNS))
        ]
        impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_shap = [ShapFeature(**f) for f in impacts[:3]]

        # Narrative (rule-based for batch speed)
        risk_score = _normalize_score(float(scores[idx]))
        raw_score = float(scores[idx])
        action = _derive_fallback_action(raw_score)
        sev_word = _severity_word(risk_score)

        # Clean display values
        disp_name = _fmt_name(username)
        disp_asset = _fmt_asset(data_asset)
        disp_ts = _fmt_ts(str(row['timestamp']))

        # Build auditor-friendly evidence bullets
        evidence = []
        for sf in top_shap:
            fn = _friendly_name(sf.feature)
            fd = _friendly_desc(sf.feature)
            evidence.append(f"{fn}: {fd}.")

        # Build contextual threat summary
        primary = _friendly_name(top_shap[0].feature) if top_shap else "unknown factor"
        primary_desc = _friendly_desc(top_shap[0].feature) if top_shap else ""
        narrative_text = (
            f"A {sev_word}-severity event was detected for {disp_name} "
            f"({department}) on {disp_ts}. "
            f"The employee accessed the {disp_asset} data store "
            f"(classified as {data_sensitivity.lower()}) "
            f"and the system flagged this activity with a risk score of {risk_score} out of 100. "
            f"The primary indicator was {primary} — {primary_desc}."
        )
        if len(top_shap) > 1:
            secondary = _friendly_name(top_shap[1].feature)
            sec_desc = _friendly_desc(top_shap[1].feature)
            narrative_text += f" A secondary concern was {secondary} — {sec_desc}."

        narrative_dict = {
            "threat_narrative": narrative_text,
            "evidence_list": evidence,
            "recommended_action": action,
        }
        narrative = LlmNarrative(**narrative_dict)

        flagged_events.append(
            FlaggedEvent(
                access_id=aid,
                user_id=str(row["user_id"]),
                username=username,
                department=department,
                timestamp=str(row["timestamp"]),
                data_asset=data_asset,
                data_sensitivity=data_sensitivity,
                query_type=query_type,
                destination=destination,
                anomaly_score=round(float(scores[idx]), 6),
                risk_score=_normalize_score(float(scores[idx])),
                is_anomaly_predicted=True,
                prediction=-1,
                top_shap_features=top_shap,
                llm_narrative=narrative,
                ground_truth=gt,
            )
        )

        if (count + 1) % 10 == 0:
            logger.info(f"  Processed {count + 1}/{max_compute} events")

    logger.info(f"  ✓ Computed {len(flagged_events)} flagged events")

    # Free the large DataFrames — they are no longer needed
    del enriched, labels, logs, X, X_scaled, predictions, scores
    del labels_lkp, logs_lkp, bg, bg_scaled, explainer
    gc.collect()
    logger.info("  ✓ Freed temporary DataFrames from memory")

    # Cache for subsequent requests
    _flagged_events_cache = {
        "metrics": metrics,
        "flagged_events": flagged_events,
    }

    return FlaggedEventsResponse(
        metrics=metrics,
        flagged_events=flagged_events[:limit],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  LLM NARRATIVE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

async def _generate_threat_narrative(
    user_id: str,
    anomaly_score: float,
    shap_features: list[dict[str, Any]],
    event_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Send the pre-computed ML outputs to Llama-3 with strict constraints
    and return the parsed JSON narrative.

    This implements the "Inversion of Responsibility" pattern:
    - The ML engine is the DECISION MAKER (anomaly yes/no + score).
    - The LLM is only a TRANSLATOR (math → plain English).
    - The LLM never sees raw data; only the algorithmic outputs.
    """

    # ---- Format the SHAP features into a structured block ----------------
    shap_block = "\n".join(
        f"  {i+1}. Feature: {f['feature']}, "
        f"SHAP Value: {f['shap_value']}, "
        f"Event Value: {f['event_value']}"
        for i, f in enumerate(shap_features)
    )

    # ---- Construct the user prompt with ONLY algorithmic data -------------
    user_prompt = (
        f"Anomaly Detection Result:\n"
        f"  User ID: {user_id}\n"
        f"  Anomaly Score: {anomaly_score}\n"
        f"  Prediction: ANOMALY CONFIRMED\n\n"
        f"Top 3 Deviating Features (SHAP):\n"
        f"{shap_block}\n\n"
        f"User Profile Context:\n"
        f"  Tenure: {event_data.get('tenure_months', 'N/A')} months\n"
        f"  High Risk Flag: {'Yes' if event_data.get('high_risk_flag') else 'No'}\n"
        f"  Days Inactive: {event_data.get('days_inactive', 'N/A')}\n"
        f"  Systems Access Count: {event_data.get('systems_access_count', 'N/A')}\n\n"
        f"Translate the above into the required JSON format."
    )

    logger.info(f"🤖 Sending anomaly for {user_id} to LLM ({HF_MODEL}) …")

    try:
        # ---- Call Hugging Face Inference API with constrained decoding ----
        response = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": SOC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=HF_TEMPERATURE,
            max_tokens=HF_MAX_TOKENS,
            top_p=0.95,
        )

        # Extract the generated text
        raw_text = response.choices[0].message.content.strip()
        logger.info(f"  📝 LLM raw response length: {len(raw_text)} chars")

        # ---- Parse the JSON from the LLM output -------------------------
        narrative = _parse_llm_json(raw_text)

        # Validate the recommended_action is one of the allowed values
        allowed_actions = {"Quarantine", "Monitor", "Investigate"}
        if narrative.get("recommended_action") not in allowed_actions:
            narrative["recommended_action"] = _derive_fallback_action(anomaly_score)

        logger.info(
            f"  ✓ LLM narrative generated — action: {narrative['recommended_action']}"
        )
        return narrative

    except Exception as e:
        logger.warning(
            f"  ⚠ LLM call failed for {user_id}: {e}. "
            f"Falling back to rule-based narrative."
        )
        # Graceful degradation — return a deterministic fallback
        return _build_fallback_narrative(
            anomaly_score=anomaly_score,
            shap_features=shap_features,
        )


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    """
    Parse the LLM output into a valid JSON dict.

    Handles common LLM failure modes:
      1. Clean JSON → direct parse
      2. JSON wrapped in markdown fences → strip fences, then parse
      3. JSON embedded in conversational text → regex extract
      4. Total garbage → raise ValueError
    """
    # Attempt 1: Direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: Regex extract first JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # All attempts failed
    logger.error(f"  ✗ Could not parse LLM JSON. Raw output: {raw_text[:200]}")
    raise ValueError("LLM did not return valid JSON")


def _derive_fallback_action(anomaly_score: float) -> str:
    """Deterministic action based on anomaly score thresholds."""
    if anomaly_score < -0.3:
        return "Quarantine"
    elif anomaly_score < -0.1:
        return "Investigate"
    else:
        return "Monitor"


def _build_fallback_narrative(
    anomaly_score: float,
    shap_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Rule-based fallback when the LLM is unavailable or returns garbage.
    Ensures the API always returns a usable response.
    """
    evidence = [
        f"{_friendly_name(f['feature'])} — {_friendly_desc(f['feature'])} "
        f"(observed value: {f['event_value']})"
        for f in shap_features[:3]
    ]

    primary = _friendly_name(shap_features[0]["feature"]) if shap_features else "unknown"
    primary_desc = _friendly_desc(shap_features[0]["feature"]) if shap_features else ""

    return {
        "threat_narrative": (
            f"Anomaly detected with risk score based on raw value {anomaly_score:.4f}. "
            f"The primary concern is {primary}: {primary_desc}."
            if shap_features
            else f"Anomaly detected with score {anomaly_score:.4f}. "
                 f"No feature-level explanation available."
        ),
        "evidence_list": evidence,
        "recommended_action": _derive_fallback_action(anomaly_score),
    }


@app.delete("/cache")
async def clear_cache():
    """
    Administrative endpoint to flush the semantic cache (Redis + fallback).
    Useful after retraining the model.
    """
    count = _cache_clear()
    logger.info(f"🗑️  Cache cleared — {count} entries removed.")
    return {"status": "cache_cleared", "entries_removed": count}


# ---------------------------------------------------------------------------
#  Startup Event — warm up the model on boot
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_warmup():
    """
    Pre-load the ML model into memory when the server starts,
    so the first /analyze_event request doesn't pay the training cost.
    """
    logger.info("🚀 Neuro-SOC API starting …")

    # ---- Redis connection ----
    _init_redis()

    # ---- ML model warmup ----
    try:
        from ml_engine import _get_cached_model
        _get_cached_model()
        logger.info("  ✓ ML model loaded and ready.")
    except Exception as e:
        logger.warning(
            f"  ⚠ Could not pre-load ML model: {e}. "
            f"Will train on first request."
        )


# ---------------------------------------------------------------------------
#  Entry point — run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,        # DISABLED: reload spawns a 2nd process that doubles RAM usage
        log_level="info",
    )
