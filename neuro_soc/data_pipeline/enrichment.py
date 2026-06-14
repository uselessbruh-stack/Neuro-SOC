"""
===============================================================================
  Neuro-SOC  |  Feature Enrichment Pipeline  v2.0
  --------------------------------------------------------------------------
  Module      : data_pipeline/enrichment.py
  Purpose     : Ingest the NEW raw CSVs (v2 schema), merge, engineer 19
                enterprise insider-threat features, and export an ML-ready
                dataset.
  Author      : Neuro-SOC Engineering Team
  Updated     : 2026-06-14
  Python      : 3.11+
  Dependencies: pandas >= 2.2, numpy >= 1.26
===============================================================================

INPUT SCHEMAS (in ./raw_data/):

  1. data_access_logs.csv  (1,201 events × 22 cols)
     access_id, timestamp, user_id, username, department, data_asset,
     data_sensitivity, data_category, query_type, rowcount, access_method,
     destination, status, source_ip, bytes_transferred, session_duration_min,
     client_application, authentication_method, is_vpn, geo_location,
     time_classification, anomaly_marker

  2. user_profiles.csv  (101 users × 20 cols)
     user_id, username, email, department, job_title, access_tier,
     tenure_months, approved_data_assets, avg_queries_per_day,
     typical_access_hours, avg_rowcount_per_query, high_risk_flag,
     equipment, clearance_level, notice_period, last_performance_rating,
     security_training_current, failed_logins_30d, vpn_user, last_access_date

OUTPUT:
  processed_data/enriched_features.csv  — 19 numeric features per event

19 ENGINEERED FEATURES:
  ┌─────┬───────────────────────────────┬──────────────────────────────┐
  │  #  │ Feature                       │ Anomaly Type Covered         │
  ├─────┼───────────────────────────────┼──────────────────────────────┤
  │  1  │ tenure_months                 │ (baseline)                   │
  │  2  │ high_risk_flag                │ (risk modifier)              │
  │  3  │ notice_period_flag            │ PRE_RESIGNATION_DOWNLOAD     │
  │  4  │ failed_logins_30d             │ FAILED_AUTH_BURST            │
  │  5  │ stale_account_days            │ STALE_ACCOUNT_ACCESS         │
  │  6  │ approved_assets_count         │ (scope baseline)             │
  │  7  │ Tenure_Risk_Modifier          │ PRE_RESIGNATION + baseline   │
  │  8  │ Equipment_Risk_Score          │ DEVICE_ANOMALY               │
  │  9  │ Access_Tier_Mismatch          │ PRIVILEGE_ESCALATION         │
  │ 10  │ Cross_Dept_Access_Flag        │ CROSS_DEPT_ACCESS            │
  │ 11  │ Rowcount_Deviation            │ BULK_EXPORT                  │
  │ 12  │ Exfiltration_Dest_Score       │ EXFILTRATION_RISK            │
  │ 13  │ Query_Type_Risk               │ BULK_EXPORT, EXFILTRATION    │
  │ 14  │ Weak_Auth_Flag                │ NIGHT_BULK_CRITICAL          │
  │ 15  │ Suspicious_Geo_Flag           │ NIGHT_BULK_CRITICAL          │
  │ 16  │ VPN_Mismatch                  │ DEVICE_ANOMALY               │
  │ 17  │ Temporal_Velocity             │ BULK_EXPORT (burst)          │
  │ 18  │ After_Hours_High_Sensitivity  │ AFTER_HOURS_RESTRICTED       │
  │ 19  │ Failed_Action_Flag            │ FAILED_AUTH_BURST            │
  └─────┴───────────────────────────────┴──────────────────────────────┘
===============================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuro_soc.enrichment")

# ---------------------------------------------------------------------------
#  Path Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
PROCESSED_DATA_DIR = SCRIPT_DIR / "processed_data"

ACCESS_LOGS_FILE = RAW_DATA_DIR / "data_access_logs.csv"
USER_PROFILES_FILE = RAW_DATA_DIR / "user_profiles.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "enriched_features.csv"

REFERENCE_DATE = pd.Timestamp("2026-04-20")

# ---------------------------------------------------------------------------
#  Threat Geography — locations associated with nation-state / anonymizing
# ---------------------------------------------------------------------------
SUSPICIOUS_GEO = {
    "pyongyang", "tor_exit_node", "unknown_vpn", "lagos", "caracas",
}

# ---------------------------------------------------------------------------
#  Exfiltration destination risk scores
# ---------------------------------------------------------------------------
DEST_RISK = {
    "usb_drive": 3, "external_email": 3, "personal_device": 3,
    "cloud_storage": 2, "print": 2,
    "shared_drive": 1, "internal_report": 0,
    "internal_server": 0, "local_workstation": 0,
}

# ---------------------------------------------------------------------------
#  Query type risk scores
# ---------------------------------------------------------------------------
QUERY_RISK = {
    "export": 3, "bulk_download": 3,
    "delete": 2,
    "update": 1, "insert": 1,
    "select": 0,
}

# ---------------------------------------------------------------------------
#  Access tier → numeric privilege score  (for mismatch calculation)
# ---------------------------------------------------------------------------
TIER_SCORE = {
    "junior": 1, "contractor": 1,
    "standard": 2,
    "senior": 3,
    "admin": 4,
}

# ---------------------------------------------------------------------------
#  Data sensitivity → numeric score
# ---------------------------------------------------------------------------
SENSITIVITY_SCORE = {
    "low": 1, "medium": 2, "high": 3, "restricted": 4,
}


# ═══════════════════════════════════════════════════════════════════════════
#  1.  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load access logs and user profiles with type parsing."""
    logger.info("Loading raw data …")

    if not ACCESS_LOGS_FILE.exists():
        logger.error(f"Access logs not found: {ACCESS_LOGS_FILE}")
        sys.exit(1)

    access_logs = pd.read_csv(ACCESS_LOGS_FILE, parse_dates=["timestamp"])
    logger.info(
        f"  ✓ data_access_logs.csv  → {access_logs.shape[0]:,} rows × "
        f"{access_logs.shape[1]} cols"
    )

    if not USER_PROFILES_FILE.exists():
        logger.error(f"User profiles not found: {USER_PROFILES_FILE}")
        sys.exit(1)

    user_profiles = pd.read_csv(
        USER_PROFILES_FILE,
        parse_dates=["last_access_date"],
    )
    logger.info(
        f"  ✓ user_profiles.csv     → {user_profiles.shape[0]:,} rows × "
        f"{user_profiles.shape[1]} cols"
    )

    return access_logs, user_profiles


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DERIVE PROFILE FEATURES (before merge)
# ═══════════════════════════════════════════════════════════════════════════
def derive_profile_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Compute features that live on the user profile."""
    logger.info("Deriving profile-level features …")
    df = profiles.copy()

    # ---- notice_period_flag (bool → int) ---------------------------------
    df["notice_period_flag"] = df["notice_period"].astype(int)
    logger.info(
        f"  ✓ notice_period_flag — {df['notice_period_flag'].sum()} users on notice"
    )

    # ---- stale_account_days (days since last access) ---------------------
    df["stale_account_days"] = (
        (REFERENCE_DATE - df["last_access_date"]).dt.days
    ).clip(lower=0).fillna(0).astype(int)
    logger.info(
        f"  ✓ stale_account_days — range [{df['stale_account_days'].min()}, "
        f"{df['stale_account_days'].max()}]"
    )

    # ---- approved_assets_count -------------------------------------------
    df["approved_assets_count"] = (
        df["approved_data_assets"]
        .fillna("")
        .str.split("|")
        .apply(lambda x: len([a for a in x if a.strip()]))
    )
    logger.info(
        f"  ✓ approved_assets_count — range [{df['approved_assets_count'].min()}, "
        f"{df['approved_assets_count'].max()}]"
    )

    # ---- high_risk_flag (ensure int) -------------------------------------
    df["high_risk_flag"] = df["high_risk_flag"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)

    # ---- Tenure_Risk_Modifier --------------------------------------------
    conditions = [
        (df["tenure_months"] < 6) & (df["high_risk_flag"] == 1),
        (df["tenure_months"] < 6) & (df["high_risk_flag"] == 0),
        (df["tenure_months"] >= 6) & (df["high_risk_flag"] == 1),
    ]
    choices = [3.0, 1.5, 2.0]
    df["Tenure_Risk_Modifier"] = np.select(conditions, choices, default=1.0)
    logger.info("  ✓ Tenure_Risk_Modifier — computed")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3.  MERGE DATASETS
# ═══════════════════════════════════════════════════════════════════════════
def merge_datasets(
    access_logs: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join logs ← profiles on user_id."""
    logger.info("Merging access_logs ← user_profiles on 'user_id' …")

    # Avoid duplicate column names
    profile_cols = [c for c in profiles.columns
                    if c not in {"username", "email", "department"}]

    merged = access_logs.merge(
        profiles[profile_cols],
        on="user_id",
        how="left",
    )

    orphans = merged["tenure_months"].isna().sum()
    if orphans > 0:
        logger.warning(f"  ⚠ {orphans} events have NO matching user profile")

    logger.info(
        f"  ✓ Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} cols"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  4.  FEATURE ENGINEERING (per-event)
# ═══════════════════════════════════════════════════════════════════════════

def compute_equipment_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Flag device anomalies:
    - contractor_machine accessing high/restricted data → 2
    - External IP (not 10.x.x.x) → 2 (unregistered device)
    - contractor_machine accessing low/medium data → 1
    - Otherwise → 0
    """
    logger.info("  → Engineering: Equipment_Risk_Score")

    equipment = df["equipment"].fillna("").str.strip().str.lower()
    sensitivity = df["data_sensitivity"].fillna("").str.strip().str.lower()
    source_ip = df["source_ip"].fillna("").astype(str)

    is_contractor = equipment == "contractor_machine"
    is_external_ip = ~source_ip.str.startswith("10.")
    is_high_sens = sensitivity.isin(["high", "restricted"])

    score = pd.Series(0, index=df.index)
    score = score.where(~(is_contractor & is_high_sens), 2)
    score = score.where(~(is_external_ip & ~is_contractor), 2)
    score = score.where(~(is_contractor & ~is_high_sens & (score == 0)), 1)

    return score.rename("Equipment_Risk_Score")


def compute_access_tier_mismatch(df: pd.DataFrame) -> pd.Series:
    """
    Mismatch = max(0, sensitivity_score - tier_score).
    Higher = accessing data above clearance level.
    """
    logger.info("  → Engineering: Access_Tier_Mismatch")

    tier = df["access_tier"].fillna("standard").str.strip().str.lower()
    sensitivity = df["data_sensitivity"].fillna("low").str.strip().str.lower()

    tier_num = tier.map(TIER_SCORE).fillna(2)
    sens_num = sensitivity.map(SENSITIVITY_SCORE).fillna(1)

    mismatch = (sens_num - tier_num).clip(lower=0)
    return mismatch.astype(int).rename("Access_Tier_Mismatch")


def compute_cross_dept_access(df: pd.DataFrame) -> pd.Series:
    """
    Flag: data_asset NOT IN user's approved_data_assets list.
    """
    logger.info("  → Engineering: Cross_Dept_Access_Flag")

    approved = df["approved_data_assets"].fillna("")
    asset = df["data_asset"].fillna("")

    def _check(row_approved, row_asset):
        approved_set = {a.strip() for a in row_approved.split("|") if a.strip()}
        return 0 if row_asset.strip() in approved_set else 1

    flag = pd.Series(
        [_check(a, d) for a, d in zip(approved, asset)],
        index=df.index,
    )
    return flag.rename("Cross_Dept_Access_Flag")


def compute_rowcount_deviation(df: pd.DataFrame) -> pd.Series:
    """
    rowcount / avg_rowcount_per_query — direct from data.
    Ratio >> 1 = bulk export anomaly.
    """
    logger.info("  → Engineering: Rowcount_Deviation")

    rowcount = df["rowcount"].fillna(0).astype(float)
    avg = df["avg_rowcount_per_query"].fillna(1).astype(float).clip(lower=1)

    ratio = (rowcount / avg).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return ratio.round(4).rename("Rowcount_Deviation")


def compute_exfiltration_dest_score(df: pd.DataFrame) -> pd.Series:
    """Map destination to risk score."""
    logger.info("  → Engineering: Exfiltration_Dest_Score")

    dest = df["destination"].fillna("").str.strip().str.lower()
    score = dest.map(DEST_RISK).fillna(0).astype(int)
    return score.rename("Exfiltration_Dest_Score")


def compute_query_type_risk(df: pd.DataFrame) -> pd.Series:
    """Map query_type to risk score."""
    logger.info("  → Engineering: Query_Type_Risk")

    qtype = df["query_type"].fillna("").str.strip().str.lower()
    score = qtype.map(QUERY_RISK).fillna(0).astype(int)
    return score.rename("Query_Type_Risk")


def compute_weak_auth_flag(df: pd.DataFrame) -> pd.Series:
    """Flag weak authentication methods."""
    logger.info("  → Engineering: Weak_Auth_Flag")

    auth = df["authentication_method"].fillna("").str.strip().str.lower()
    flag = auth.isin(["password_only", "api_key"]).astype(int)
    return flag.rename("Weak_Auth_Flag")


def compute_suspicious_geo_flag(df: pd.DataFrame) -> pd.Series:
    """Flag access from suspicious geolocations."""
    logger.info("  → Engineering: Suspicious_Geo_Flag")

    geo = df["geo_location"].fillna("").str.strip().str.lower()
    flag = geo.isin(SUSPICIOUS_GEO).astype(int)
    return flag.rename("Suspicious_Geo_Flag")


def compute_vpn_mismatch(df: pd.DataFrame) -> pd.Series:
    """Flag when is_vpn status doesn't match vpn_user profile."""
    logger.info("  → Engineering: VPN_Mismatch")

    is_vpn = df["is_vpn"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)

    vpn_user = df["vpn_user"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)

    mismatch = (is_vpn != vpn_user).astype(int)
    return mismatch.rename("VPN_Mismatch")


def compute_temporal_velocity(df: pd.DataFrame, window: str = "1h") -> pd.Series:
    """Rolling 1-hour event count per user (burst detection)."""
    logger.info(f"  → Engineering: Temporal_Velocity (window={window})")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_sorted = df[["user_id", "timestamp"]].copy()
    df_sorted = df_sorted.sort_values(["user_id", "timestamp"])
    df_sorted["_event_marker"] = 1

    velocity = (
        df_sorted
        .set_index("timestamp")
        .groupby("user_id")["_event_marker"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    df_sorted["Temporal_Velocity"] = velocity.values
    result = df_sorted["Temporal_Velocity"].astype(int)
    return result.reindex(df.index).rename("Temporal_Velocity")


def compute_after_hours_high_sensitivity(df: pd.DataFrame) -> pd.Series:
    """Flag off-hours access to high/restricted data."""
    logger.info("  → Engineering: After_Hours_High_Sensitivity")

    time_cls = df["time_classification"].fillna("").str.strip().str.lower()
    sensitivity = df["data_sensitivity"].fillna("").str.strip().str.lower()

    flag = (
        time_cls.isin(["night", "weekend", "after_hours", "early_morning"])
        & sensitivity.isin(["high", "restricted"])
    ).astype(int)

    return flag.rename("After_Hours_High_Sensitivity")


def compute_failed_action_flag(df: pd.DataFrame) -> pd.Series:
    """Flag failed access attempts."""
    logger.info("  → Engineering: Failed_Action_Flag")

    flag = (
        df["status"].fillna("").str.strip().str.lower() != "success"
    ).astype(int)
    return flag.rename("Failed_Action_Flag")


# ═══════════════════════════════════════════════════════════════════════════
#  4b. COMPOUND FEATURES (amplify weak anomaly signals)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cross_dept_sensitivity(df: pd.DataFrame) -> pd.Series:
    """
    Compound: Cross_Dept_Access_Flag × data_sensitivity_score.
    Amplifies cross-department access to HIGH/RESTRICTED data.
    CROSS_DEPT_ACCESS to low-sensitivity data = 0, to restricted = 4.
    """
    logger.info("  → Engineering: Cross_Dept_Sensitivity")
    sensitivity = df["data_sensitivity"].fillna("low").str.strip().str.lower()
    sens_num = sensitivity.map(SENSITIVITY_SCORE).fillna(1)
    score = df["Cross_Dept_Access_Flag"] * sens_num
    return score.astype(float).rename("Cross_Dept_Sensitivity")


def compute_time_sensitivity_risk(df: pd.DataFrame) -> pd.Series:
    """
    Compound: time_risk_weight × data_sensitivity_score.
    Night + restricted = 12, after_hours + high = 6, business_hours = 0.
    """
    logger.info("  → Engineering: Time_Sensitivity_Risk")
    time_cls = df["time_classification"].fillna("").str.strip().str.lower()
    sensitivity = df["data_sensitivity"].fillna("low").str.strip().str.lower()

    time_weight = time_cls.map({
        "night": 3, "weekend": 2, "after_hours": 2,
        "early_morning": 1, "business_hours": 0,
    }).fillna(0)
    sens_num = sensitivity.map(SENSITIVITY_SCORE).fillna(1)

    score = time_weight * sens_num
    return score.astype(float).rename("Time_Sensitivity_Risk")


def compute_stale_sensitivity_risk(df: pd.DataFrame) -> pd.Series:
    """
    Compound: log1p(stale_account_days) × data_sensitivity_score.
    Dormant account + restricted data = strong signal.
    """
    logger.info("  → Engineering: Stale_Sensitivity_Risk")
    sensitivity = df["data_sensitivity"].fillna("low").str.strip().str.lower()
    sens_num = sensitivity.map(SENSITIVITY_SCORE).fillna(1)

    stale = df["stale_account_days"].fillna(0).astype(float)
    score = np.log1p(stale) * sens_num
    return score.round(4).rename("Stale_Sensitivity_Risk")


def compute_volume_dest_compound(df: pd.DataFrame) -> pd.Series:
    """
    Compound: log1p(Rowcount_Deviation) × Exfiltration_Dest_Score.
    High-volume export to risky destination = massive signal.
    """
    logger.info("  → Engineering: Volume_Dest_Compound")
    rowcount_dev = df["Rowcount_Deviation"].fillna(0).astype(float)
    dest_score = df["Exfiltration_Dest_Score"].fillna(0).astype(float)

    score = np.log1p(rowcount_dev) * dest_score
    return score.round(4).rename("Volume_Dest_Compound")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  ASSEMBLE & EXPORT
# ═══════════════════════════════════════════════════════════════════════════

FINAL_FEATURE_COLUMNS = [
    # ── User profile features (static per user) ──
    "tenure_months",
    "high_risk_flag",
    "notice_period_flag",
    "failed_logins_30d",
    "stale_account_days",
    "approved_assets_count",

    # ── Computed risk modifiers ──
    "Tenure_Risk_Modifier",
    "Equipment_Risk_Score",
    "Access_Tier_Mismatch",
    "Cross_Dept_Access_Flag",

    # ── Per-event features ──
    "Rowcount_Deviation",
    "Exfiltration_Dest_Score",
    "Query_Type_Risk",
    "Weak_Auth_Flag",
    "Suspicious_Geo_Flag",
    "VPN_Mismatch",

    # ── Temporal features ──
    "Temporal_Velocity",
    "After_Hours_High_Sensitivity",
    "Failed_Action_Flag",

    # ── Compound features (amplify weak signals) ──
    "Cross_Dept_Sensitivity",
    "Time_Sensitivity_Risk",
    "Stale_Sensitivity_Risk",
    "Volume_Dest_Compound",
]

# ID columns to preserve in output (for joining with labels later)
ID_COLUMNS = ["access_id", "user_id", "timestamp"]


def main() -> None:
    """End-to-end enrichment: Load → Derive → Merge → Engineer → Export."""
    logger.info("=" * 72)
    logger.info("  Neuro-SOC  |  Feature Enrichment Pipeline v2.0 — START")
    logger.info("=" * 72)

    # ── Step 1: Load ──
    access_logs, user_profiles = load_raw_data()

    # ── Step 2: Derive profile features ──
    user_profiles = derive_profile_features(user_profiles)

    # ── Step 3: Merge ──
    merged = merge_datasets(access_logs, user_profiles)

    # ── Step 4: Engineer per-event features ──
    logger.info("Engineering 23 enterprise features …")

    merged["Equipment_Risk_Score"] = compute_equipment_risk_score(merged)
    merged["Access_Tier_Mismatch"] = compute_access_tier_mismatch(merged)
    merged["Cross_Dept_Access_Flag"] = compute_cross_dept_access(merged)
    merged["Rowcount_Deviation"] = compute_rowcount_deviation(merged)
    merged["Exfiltration_Dest_Score"] = compute_exfiltration_dest_score(merged)
    merged["Query_Type_Risk"] = compute_query_type_risk(merged)
    merged["Weak_Auth_Flag"] = compute_weak_auth_flag(merged)
    merged["Suspicious_Geo_Flag"] = compute_suspicious_geo_flag(merged)
    merged["VPN_Mismatch"] = compute_vpn_mismatch(merged)
    merged["Temporal_Velocity"] = compute_temporal_velocity(merged)
    merged["After_Hours_High_Sensitivity"] = compute_after_hours_high_sensitivity(merged)
    merged["Failed_Action_Flag"] = compute_failed_action_flag(merged)

    # ── Compound features (boost weak categories) ──
    merged["Cross_Dept_Sensitivity"] = compute_cross_dept_sensitivity(merged)
    merged["Time_Sensitivity_Risk"] = compute_time_sensitivity_risk(merged)
    merged["Stale_Sensitivity_Risk"] = compute_stale_sensitivity_risk(merged)
    merged["Volume_Dest_Compound"] = compute_volume_dest_compound(merged)

    # ── Step 5: Select output columns ──
    # Keep ID columns for label joining + all 19 features
    output_cols = [c for c in ID_COLUMNS if c in merged.columns] + FINAL_FEATURE_COLUMNS
    enriched = merged[output_cols].copy()

    # Fill any remaining NaN
    for col in FINAL_FEATURE_COLUMNS:
        enriched[col] = enriched[col].fillna(0)

    logger.info(f"\n  Final enriched shape: {enriched.shape[0]:,} × {enriched.shape[1]}")
    logger.info(f"  Features: {FINAL_FEATURE_COLUMNS}")

    # ── Step 6: Summary statistics ──
    logger.info("\n  Feature summary:")
    for col in FINAL_FEATURE_COLUMNS:
        vals = enriched[col]
        nonzero = (vals != 0).sum()
        logger.info(
            f"    {col:35s} — min: {vals.min():8.2f}  max: {vals.max():8.2f}  "
            f"nonzero: {nonzero:,}/{len(vals):,}"
        )

    # ── Step 7: Export ──
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_FILE, index=False)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(
        f"\n✅  Enriched dataset exported → {OUTPUT_FILE}  "
        f"({enriched.shape[0]:,} rows × {enriched.shape[1]} cols, "
        f"{file_size_mb:.2f} MB)"
    )

    logger.info("=" * 72)
    logger.info("  Neuro-SOC  |  Feature Enrichment Pipeline v2.0 — COMPLETE")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
