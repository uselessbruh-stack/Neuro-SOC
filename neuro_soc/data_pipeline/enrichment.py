"""
===============================================================================
  Neuro-SOC  |  Feature Enrichment Pipeline
  --------------------------------------------------------------------------
  Module      : data_pipeline/enrichment.py
  Purpose     : Ingest raw CSVs (access logs + user profiles), merge them,
                engineer enterprise insider-threat features, and export
                an ML-ready dataset.
  Author      : Neuro-SOC Engineering Team
  Created     : 2026-06-13
  Python      : 3.11+
  Dependencies: pandas >= 2.2, numpy >= 1.26
===============================================================================

ACTUAL INPUT SCHEMAS (in ./raw_data/):

  1. data_access_logs.csv
     Columns: timestamp, user_id, username, action, resource,
              resource_sensitivity, status, source_ip, time_classification

  2. user_profiles.csv
     Columns: user_id, username, email, department, job_title,
              privilege_level, systems_access, last_login, days_inactive,
              is_active, hire_date

OUTPUT FILE (in ./processed_data/):
  enriched_features.csv — numeric / boolean features ready for model training.

FEATURE MAPPING (spec → actual data):
  ┌──────────────────────────────┬───────────────────────────────────────┐
  │ Spec Feature                 │ Derived From                          │
  ├──────────────────────────────┼───────────────────────────────────────┤
  │ tenure_months                │ COMPUTED: today − hire_date            │
  │ high_risk_flag               │ DERIVED: privilege_level ∈            │
  │                              │   {admin, power-user, service-account}│
  │ data_sensitivity             │ MAPPED: resource_sensitivity column   │
  │ equipment (company_laptop)   │ PROXY: privilege_level != admin →     │
  │                              │   "non-corporate" for restricted data │
  │ rowcount                     │ N/A — substituted with action count   │
  │ avg_rowcount_per_query       │ N/A — substituted with user avg       │
  │ Temporal_Velocity            │ Rolling 1h event count per user       │
  └──────────────────────────────┴───────────────────────────────────────┘
===============================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuro_soc.enrichment")

# ---------------------------------------------------------------------------
#  Path Constants — relative to *this* script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
PROCESSED_DATA_DIR = SCRIPT_DIR / "processed_data"

ACCESS_LOGS_FILE = RAW_DATA_DIR / "data_access_logs.csv"
USER_PROFILES_FILE = RAW_DATA_DIR / "user_profiles.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "enriched_features.csv"

# ---------------------------------------------------------------------------
#  Reference date for tenure calculation — set to the latest event date
#  so the pipeline is reproducible regardless of when it's run.
# ---------------------------------------------------------------------------
REFERENCE_DATE = pd.Timestamp("2026-04-20")


# ═══════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two source CSVs and perform basic sanity checks.

    Returns
    -------
    (access_logs, user_profiles) : tuple of DataFrames
    """
    logger.info("Loading raw data …")

    # --- Access Logs ---------------------------------------------------------
    if not ACCESS_LOGS_FILE.exists():
        logger.error(f"Access logs file not found: {ACCESS_LOGS_FILE}")
        sys.exit(1)

    access_logs = pd.read_csv(ACCESS_LOGS_FILE, parse_dates=["timestamp"])
    logger.info(
        f"  ✓ data_access_logs.csv  → {access_logs.shape[0]:,} rows × "
        f"{access_logs.shape[1]} cols"
    )

    # --- User Profiles -------------------------------------------------------
    if not USER_PROFILES_FILE.exists():
        logger.error(f"User profiles file not found: {USER_PROFILES_FILE}")
        sys.exit(1)

    user_profiles = pd.read_csv(
        USER_PROFILES_FILE,
        parse_dates=["hire_date", "last_login"],
    )
    logger.info(
        f"  ✓ user_profiles.csv     → {user_profiles.shape[0]:,} rows × "
        f"{user_profiles.shape[1]} cols"
    )

    return access_logs, user_profiles


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DERIVE INTERMEDIATE COLUMNS ON USER PROFILES
# ═══════════════════════════════════════════════════════════════════════════
def derive_profile_features(user_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived columns on user_profiles BEFORE the merge, so they
    are available for feature engineering on the merged DataFrame.

    Derived columns:
      • tenure_months  : integer months since hire_date
      • high_risk_flag : True if privilege_level ∈ {admin, power-user, service-account}
    """
    logger.info("Deriving intermediate profile features …")
    df = user_profiles.copy()

    # ---- tenure_months -------------------------------------------------------
    # Calculate months of tenure from hire_date to a fixed reference date.
    # Using a fixed date ensures reproducibility across pipeline runs.
    df["tenure_months"] = (
        (REFERENCE_DATE - df["hire_date"]).dt.days / 30.44
    ).astype(int)

    logger.info(
        f"  ✓ tenure_months — range: [{df['tenure_months'].min()}, "
        f"{df['tenure_months'].max()}] months"
    )

    # ---- high_risk_flag ------------------------------------------------------
    # Banking SOC policy: admin, power-user, and service-accounts are
    # considered high-risk due to elevated privileges.
    HIGH_RISK_PRIVILEGE_LEVELS = {"admin", "power-user", "service-account"}
    df["high_risk_flag"] = df["privilege_level"].str.strip().str.lower().isin(
        HIGH_RISK_PRIVILEGE_LEVELS
    )

    n_high_risk = df["high_risk_flag"].sum()
    logger.info(
        f"  ✓ high_risk_flag — {n_high_risk}/{len(df)} users flagged "
        f"({n_high_risk / len(df) * 100:.1f}%)"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3.  MERGE DATASETS
# ═══════════════════════════════════════════════════════════════════════════
def merge_datasets(
    access_logs: pd.DataFrame,
    user_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join access logs with user profiles on ``user_id``.

    A left join ensures every access event is retained even if the user
    profile is (unexpectedly) missing — that itself is an anomaly signal
    worth keeping.
    """
    logger.info("Merging access_logs ← user_profiles on 'user_id' …")

    # Drop the duplicate 'username' column from user_profiles before merge
    # (both CSVs have 'username'; we keep the one from access_logs)
    profile_cols_to_merge = [
        c for c in user_profiles.columns if c != "username"
    ]

    merged = access_logs.merge(
        user_profiles[profile_cols_to_merge],
        on="user_id",
        how="left",
    )

    # Flag rows where the profile was missing (potential orphan accounts)
    orphan_mask = merged["tenure_months"].isna()
    n_orphans = orphan_mask.sum()
    if n_orphans > 0:
        logger.warning(
            f"  ⚠  {n_orphans:,} access events have NO matching user profile "
            f"(orphan / terminated accounts?)"
        )

    logger.info(
        f"  ✓ Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} cols"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  4.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

# ---- 4a. Tenure Risk Modifier -----------------------------------------------
def compute_tenure_risk_modifier(df: pd.DataFrame) -> pd.Series:
    """
    Combine ``tenure_months`` with ``high_risk_flag`` into a single
    continuous risk multiplier.

    Business Logic
    ──────────────
    • tenure < 6 months  AND  high_risk_flag is True  →  risk × 3.0
      (New employees with elevated privileges — highest exfiltration risk.)
    • tenure < 6 months  AND  high_risk_flag is False →  risk × 1.5
      (New but in a standard role — moderate onboarding risk.)
    • tenure ≥ 6 months  AND  high_risk_flag is True  →  risk × 2.0
      (Established but in a privileged role — sustained elevated risk.)
    • Otherwise                                       →  risk × 1.0
      (Baseline — tenured employee in a standard role.)

    Returns
    -------
    pd.Series[float]  — one modifier per row.
    """
    logger.info("  → Engineering: Tenure_Risk_Modifier")

    conditions = [
        (df["tenure_months"] < 6) & (df["high_risk_flag"] == True),   # noqa: E712
        (df["tenure_months"] < 6) & (df["high_risk_flag"] == False),  # noqa: E712
        (df["tenure_months"] >= 6) & (df["high_risk_flag"] == True),  # noqa: E712
    ]
    choices = [3.0, 1.5, 2.0]

    return pd.Series(
        np.select(conditions, choices, default=1.0),
        index=df.index,
        name="Tenure_Risk_Modifier",
    )


# ---- 4b. Equipment Mismatch Score -------------------------------------------
def compute_equipment_mismatch_score(df: pd.DataFrame) -> pd.Series:
    """
    Flag events where a user accessed **restricted / high-sensitivity**
    data but does NOT hold a corporate-managed privilege level.

    Adaptation to real data
    ───────────────────────
    The original spec checks `equipment != 'company_laptop'`. Since the
    actual dataset has no equipment column, we use a privilege-based proxy:

      • resource_sensitivity == 'high'  AND  privilege_level == 'user'
        → mismatch = 1  (a basic "user" account touching high-sensitivity
          resources is a policy violation / anomaly signal)

    This captures the same insider-threat signal: someone accessing data
    above their expected clearance level.

    Returns
    -------
    pd.Series[int]  — 1 (mismatch) or 0 (compliant).
    """
    logger.info("  → Engineering: Equipment_Mismatch_Score")

    sensitivity = df["resource_sensitivity"].str.strip().str.lower()
    privilege = df["privilege_level"].str.strip().str.lower()

    mismatch = (
        (sensitivity == "high") & (privilege == "user")
    ).astype(int)

    return mismatch.rename("Equipment_Mismatch_Score")


# ---- 4c. Temporal Velocity --------------------------------------------------
def compute_temporal_velocity(
    df: pd.DataFrame,
    window: str = "1h",
) -> pd.Series:
    """
    For each event, count how many actions the **same user** executed in
    the preceding ``window`` (default: 1 hour).

    This captures burst / exfiltration behaviour — a user who suddenly
    fires 200 actions in 60 minutes when their baseline is 5 triggers a
    massive velocity spike.

    Implementation
    ──────────────
    1. Sort by (user_id, timestamp).
    2. Group by user_id.
    3. Within each group, apply a rolling count on the timestamp index
       with a 1-hour lookback.

    Returns
    -------
    pd.Series[int]  — event count within the rolling window (including
                       the current event).
    """
    logger.info(f"  → Engineering: Temporal_Velocity (window={window})")

    # Ensure the timestamp column is a proper datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort chronologically within each user — critical for rolling to work
    df_sorted = df[["user_id", "timestamp"]].copy()
    df_sorted = df_sorted.sort_values(["user_id", "timestamp"])

    # We need the timestamp as the index for a time-based rolling window.
    # Use a helper column '1' that we sum in the window → gives event count.
    df_sorted["_event_marker"] = 1

    velocity = (
        df_sorted
        .set_index("timestamp")                        # make timestamp the index
        .groupby("user_id")["_event_marker"]            # per-user groups
        .rolling(window, min_periods=1)                 # 1-hour lookback
        .sum()                                          # count events in window
        .reset_index(level=0, drop=True)                # drop user_id group level
        .sort_index()                                   # restore chronological order
    )

    # Re-align to the original DataFrame index
    df_sorted["Temporal_Velocity"] = velocity.values
    result = df_sorted.set_index(df_sorted.index)["Temporal_Velocity"].astype(int)

    # Map back to the original row order via the sorted index
    return result.reindex(df.index).rename("Temporal_Velocity")


# ---- 4d. Rowcount Deviation (adapted: Action Frequency Deviation) -----------
def compute_action_frequency_deviation(df: pd.DataFrame) -> pd.Series:
    """
    Adaptation of the ``rowcount_deviation`` spec to real data.

    Since the dataset has no ``rowcount`` or ``avg_rowcount_per_query``
    columns, we compute an analogous metric:

      action_frequency_deviation = (user's daily event count on this day)
                                   / (user's overall average daily event count)

    Interpretation
    ──────────────
    • ratio ≈ 1.0  → normal activity level for this user
    • ratio >> 1.0  → abnormally active day (potential exfiltration burst)
    • ratio << 1.0  → unusually quiet (may indicate recon / compromised idle account)
    • NaN / Inf     → user has zero average (edge case); replaced with 0.0

    Returns
    -------
    pd.Series[float]  — deviation ratio, rounded to 4 decimal places.
    """
    logger.info("  → Engineering: rowcount_deviation (Action Frequency Deviation)")

    # Compute per-user daily event counts
    df_temp = df[["user_id", "timestamp"]].copy()
    df_temp["event_date"] = df_temp["timestamp"].dt.date

    # Daily counts per user
    daily_counts = (
        df_temp.groupby(["user_id", "event_date"])
        .size()
        .reset_index(name="daily_event_count")
    )

    # Average daily count per user (across all their active days)
    user_avg = (
        daily_counts.groupby("user_id")["daily_event_count"]
        .mean()
        .reset_index(name="avg_daily_events")
    )

    # Merge daily counts back to the temp frame
    df_temp = df_temp.merge(daily_counts, on=["user_id", "event_date"], how="left")
    df_temp = df_temp.merge(user_avg, on="user_id", how="left")

    # Compute deviation ratio
    deviation = df_temp["daily_event_count"] / df_temp["avg_daily_events"]

    # Guard against division by zero or NaN averages
    deviation = deviation.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return deviation.round(4).rename("rowcount_deviation")


# ═══════════════════════════════════════════════════════════════════════════
#  5.  BONUS FEATURES (high-value signals from available data)
# ═══════════════════════════════════════════════════════════════════════════
def compute_bonus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features that leverage columns unique to this
    dataset and provide strong insider-threat signals.
    """
    logger.info("  → Engineering: Bonus features …")

    # ---- 5a. After-Hours High-Sensitivity Access ----------------------------
    # Flag: accessing high-sensitivity resources during non-business hours
    time_cls = df["time_classification"].str.strip().str.lower()
    sensitivity = df["resource_sensitivity"].str.strip().str.lower()

    df["After_Hours_High_Sensitivity"] = (
        (time_cls.isin(["night", "unusual_hours", "weekend"]))
        & (sensitivity == "high")
    ).astype(int)

    logger.info(
        f"    ✓ After_Hours_High_Sensitivity — "
        f"{df['After_Hours_High_Sensitivity'].sum():,} events flagged"
    )

    # ---- 5b. Failed Action Flag ---------------------------------------------
    # Binary: did this action fail? Repeated failures = brute-force / probing
    df["Failed_Action_Flag"] = (
        df["status"].str.strip().str.lower() != "success"
    ).astype(int)

    logger.info(
        f"    ✓ Failed_Action_Flag — "
        f"{df['Failed_Action_Flag'].sum():,} failures detected"
    )

    # ---- 5c. Privilege-Sensitivity Mismatch ---------------------------------
    # How many system accesses does this user have? Compare to sensitivity.
    # Users with few system accesses touching high-sensitivity = suspicious.
    df["systems_access_count"] = (
        df["systems_access"]
        .fillna("")
        .str.split("|")
        .apply(len)
    )

    # Normalised score: high sensitivity (mapped to 3) vs access breadth
    sens_map = {"high": 3, "medium": 2, "low": 1}
    df["_sens_numeric"] = sensitivity.map(sens_map).fillna(1)
    df["Privilege_Sensitivity_Mismatch"] = (
        df["_sens_numeric"] / df["systems_access_count"].clip(lower=1)
    ).round(4)

    # Clean up temp column
    df.drop(columns=["_sens_numeric"], inplace=True)

    logger.info(
        f"    ✓ Privilege_Sensitivity_Mismatch — computed"
    )

    # ---- 5d. Days Inactive ---------------------------------------------------
    # Already numeric in the profile — great signal for dormant-account abuse.
    # No transformation needed; just ensure it's preserved.

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  6.  DROP REDUNDANT CATEGORICALS
# ═══════════════════════════════════════════════════════════════════════════
def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove raw categorical / string columns that are NOT directly
    consumable by an ML model.

    We keep:
      • user_id        — needed downstream for grouping / auditing
      • timestamp      — may be converted to numeric features later
      • All engineered numeric features

    We drop:
      • Free-text / categorical columns whose information has already
        been encoded into the engineered features above.
    """
    logger.info("Dropping redundant categorical columns …")

    # Identify all object (string) and categorical dtype columns
    cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preserve user_id (it's a key, not a feature — but critical for audit trail)
    preserve = {"user_id"}
    to_drop = [col for col in cat_columns if col not in preserve]

    logger.info(f"  ✗ Dropping {len(to_drop)} columns: {to_drop}")
    df = df.drop(columns=to_drop, errors="ignore")

    logger.info(
        f"  ✓ Remaining shape: {df.shape[0]:,} rows × {df.shape[1]} cols"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  7.  EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def export_enriched_features(df: pd.DataFrame) -> None:
    """
    Write the final enriched DataFrame to CSV in the processed_data/ folder.
    Creates the output directory if it doesn't already exist.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    logger.info(
        f"✅  Enriched dataset exported → {OUTPUT_FILE}  "
        f"({df.shape[0]:,} rows × {df.shape[1]} cols, {file_size_mb:.2f} MB)"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8.  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    End-to-end enrichment pipeline:
      Load → Derive → Merge → Engineer → Bonus → Clean → Export
    """
    logger.info("=" * 72)
    logger.info("  Neuro-SOC  |  Feature Enrichment Pipeline — START")
    logger.info("=" * 72)

    # ---- Step 1 : Load raw CSVs --------------------------------------------
    access_logs, user_profiles = load_raw_data()

    # ---- Step 2 : Derive intermediate columns on profiles ------------------
    user_profiles = derive_profile_features(user_profiles)

    # ---- Step 3 : Merge on user_id -----------------------------------------
    merged = merge_datasets(access_logs, user_profiles)

    # ---- Step 4 : Engineer the 4 spec-required features --------------------
    logger.info("Engineering enterprise features …")

    merged["Tenure_Risk_Modifier"] = compute_tenure_risk_modifier(merged)
    merged["Equipment_Mismatch_Score"] = compute_equipment_mismatch_score(merged)
    merged["Temporal_Velocity"] = compute_temporal_velocity(merged)
    merged["rowcount_deviation"] = compute_action_frequency_deviation(merged)

    logger.info(
        f"  ✓ 4 core features added. Shape: "
        f"{merged.shape[0]:,} × {merged.shape[1]}"
    )

    # ---- Step 5 : Engineer bonus features ----------------------------------
    merged = compute_bonus_features(merged)

    logger.info(
        f"  ✓ Bonus features added. Shape: "
        f"{merged.shape[0]:,} × {merged.shape[1]}"
    )

    # ---- Step 6 : Drop redundant categoricals ------------------------------
    enriched = drop_redundant_columns(merged)

    # ---- Step 7 : Export ---------------------------------------------------
    export_enriched_features(enriched)

    logger.info("=" * 72)
    logger.info("  Neuro-SOC  |  Feature Enrichment Pipeline — COMPLETE")
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
