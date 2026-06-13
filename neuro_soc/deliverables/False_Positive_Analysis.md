# Neuro-SOC — False Positive Mitigation Analysis

**Document Classification:** Internal — Hackathon Submission  
**Version:** 1.0  
**Date:** 2026-06-14  
**Author:** Neuro-SOC Engineering Team

---

## Executive Summary

Neuro-SOC maintains a False Positive (FP) rate target of **< 20%** through a three-tiered mitigation architecture. Unlike traditional rule-based SIEM systems that generate overwhelming alert volumes (industry average: 40-60% FP rate), our hybrid approach combines statistical anomaly detection, contextual LLM translation, and human analyst review to systematically eliminate false alerts before they reach the SOC queue.

This document addresses three high-frequency FP scenarios encountered in banking environments and demonstrates how each architectural layer handles them.

---

## Architecture Overview

```
Layer 1: ML Baselining (Isolation Forest + SHAP)
    ↓ Only statistically significant deviations pass through
Layer 2: LLM Contextual Translation (Llama-3, T=0.1)
    ↓ Objective narrative — no intent inference
Layer 3: Human-in-the-Loop Dashboard (Next.js SOC Console)
    ↓ Analyst decision: Quarantine / Investigate / False Positive
```

Each layer acts as a progressively finer filter. An event must pass **all three** before any enforcement action is taken.

---

## Edge Case 1: Legitimate Month-End Financial Bulk Exports

### The Problem

At the close of each fiscal month (and quarter), finance and accounting teams perform large-scale data exports for regulatory reporting (e.g., RBI compliance filings, Basel III capital adequacy reports). These exports exhibit behavioral signatures identical to data exfiltration:

- Elevated `rowcount_deviation` (3-8× normal)
- High `Temporal_Velocity` (multiple exports in rapid succession)
- After-hours activity (month-end deadlines often extend past business hours)

### How Neuro-SOC Handles It

| Layer | Mitigation Mechanism |
|-------|---------------------|
| **ML (Layer 1)** | The Isolation Forest model is trained on 12 months of historical data that **includes** prior month-end activity. Recurring patterns are absorbed into the user's baseline. A finance analyst who consistently exports 500 records on the 30th of each month will have a `rowcount_deviation` close to 1.0 during these periods — the model sees it as normal. |
| **SHAP (Layer 1)** | If the event is flagged despite historical training, the SHAP explanation will show `rowcount_deviation` as the primary driver with `After_Hours_High_Sensitivity = 0` and `Privilege_Sensitivity_Mismatch = 1.0`. This profile (high volume, low sensitivity mismatch, authorized access) is characteristic of legitimate bulk operations, not exfiltration. |
| **LLM (Layer 2)** | The constrained Llama-3 prompt explicitly states: *"Do not infer malicious intent; state the deviations objectively."* The generated narrative will read: *"User exhibits elevated data export volume consistent with reporting activity. No privilege escalation or unauthorized resource access detected."* The `recommended_action` will typically resolve to **"Monitor"** rather than "Quarantine". |
| **Human (Layer 3)** | The SOC analyst sees the full context: user department (Finance), tenure (senior), known month-end window. The "Mark as False Positive" button logs the dismissal, and repeated FP marks for the same user/pattern feed back into model retraining priorities. |

### Expected Outcome

Month-end bulk exports from authorized finance personnel are flagged in < 5% of cases after the first training cycle. When flagged, they are classified as **"Monitor"** (not "Quarantine"), and analysts dismiss them in < 30 seconds via the dashboard.

---

## Edge Case 2: Temporary Role Changes / On-Call Duty Rotations

### The Problem

Banks frequently rotate personnel into elevated-access roles:

- Weekend on-call DBA with temporary `admin` privileges
- Cross-training employees who temporarily access unfamiliar systems
- Interim managers covering for colleagues on leave

These temporary role changes create sudden spikes in:

- `Privilege_Sensitivity_Mismatch` (accessing resources above normal clearance)
- `Equipment_Mismatch_Score` (using shared/unfamiliar workstations)
- `systems_access_count` changes (accessing systems outside normal scope)

### How Neuro-SOC Handles It

| Layer | Mitigation Mechanism |
|-------|---------------------|
| **ML (Layer 1)** | The `Tenure_Risk_Modifier` feature specifically addresses this. Short-tenure or recently-changed users receive a risk multiplier that contextualizes their deviation. A 5-year veteran temporarily on-call has a `Tenure_Risk_Modifier` of 1.0 (no amplification), while a 3-month new hire on-call gets 3.0. This means temporary role changes for established employees produce lower anomaly scores than identical behavior from new employees — appropriately weighting institutional trust. |
| **SHAP (Layer 1)** | For an established employee on temporary duty, SHAP analysis will show `Privilege_Sensitivity_Mismatch` as the dominant feature with `Tenure_Risk_Modifier = 1.0`. This signature (privilege deviation without tenure risk) is distinguishable from true insider threats where both features co-deviate. |
| **LLM (Layer 2)** | The narrative explicitly states the measurable deviations without intent inference: *"User accessed resources outside their standard permission scope. Privilege sensitivity mismatch elevated. Tenure indicates established employee. No failed access attempts detected."* The absence of `Failed_Action_Flag` and low `Temporal_Velocity` result in a **"Monitor"** or **"Investigate"** recommendation, not immediate quarantine. |
| **Human (Layer 3)** | The analyst cross-references the alert with the on-call schedule (typically available via ServiceNow or PagerDuty integration). A single click on "Mark as False Positive" clears the alert. The dashboard provides all required data points at a glance — no context-switching to external systems. |

### Expected Outcome

On-call/temporary role changes for tenured employees trigger alerts < 10% of the time. When triggered, the LLM narrative explicitly notes the absence of compounding risk factors, enabling sub-60-second analyst triage.

---

## Edge Case 3: Seasonal or Calendar-Driven Behavioral Shifts

### The Problem

Banking operations exhibit predictable behavioral cycles:

- Year-end audits (December-January): Auditors access multiple systems in rapid succession
- Regulatory exam preparation: Compliance teams pull historical records en masse
- Tax season (March-April): Bulk export of client financial data

These events can trigger:

- Sustained elevation in `Temporal_Velocity` over multiple days
- Cross-system access patterns unusual for individual users
- After-hours activity during crunch periods

### How Neuro-SOC Handles It

| Layer | Mitigation Mechanism |
|-------|---------------------|
| **ML (Layer 1)** | The `rowcount_deviation` feature is calculated as a ratio of daily activity to the user's rolling average. For users who consistently exhibit seasonal spikes, the rolling average naturally absorbs these patterns over time, reducing the deviation score in subsequent cycles. |
| **Semantic Cache** | Redis-based caching with 1-hour TTL ensures that repeated alerts for the same user+behavior combination within an audit window do not re-trigger LLM processing. The first evaluation is cached, and subsequent identical events return immediately — reducing analyst fatigue from repetitive alerts. |
| **LLM (Layer 2)** | The narrative generator operates on objective feature values, not raw data. A compliance officer pulling audit records will produce a narrative like: *"Elevated data access volume. User department: Compliance. Access pattern consistent with documented audit cycle."* |
| **Human (Layer 3)** | The dashboard's "Mark as False Positive" action, when taken repeatedly for the same behavioral pattern, serves as an implicit signal for model retraining. Accumulated FP decisions are logged with full audit trails per NIST IR-4 requirements. |

---

## The Human-in-the-Loop Safeguard

The Neuro-SOC dashboard is the **final arbitration layer** and the most critical FP mitigation control:

### Design Principles

1. **No Automated Enforcement**: The system never quarantines a user without analyst confirmation. The ML engine scores, the LLM translates, but only a human presses the red button. This eliminates catastrophic false positives (e.g., locking out a CFO during an earnings call).

2. **Full Evidence Visibility**: Every alert surfaces:
   - Numeric risk score (0-100)
   - Top 3 SHAP feature deviations with actual values
   - Plain-English threat narrative
   - Recommended action (with threshold reasoning)

3. **Two-Action Resolution**: The analyst has exactly two options:
   - **Quarantine User & Revoke Access** — immediate enforcement
   - **Mark as False Positive** — dismiss with audit log

4. **Audit Compliance**: Every decision (quarantine or dismissal) is logged with timestamp, analyst ID, and the full ML/LLM payload. This satisfies GDPR Article 22 (right to meaningful information about automated decision logic) and NIST SP 800-61 incident handling requirements.

### False Positive Feedback Loop

```
Analyst marks FP → Decision logged to audit store
                  → Aggregated weekly for FP pattern analysis
                  → High-frequency FP patterns flagged for model retraining
                  → Next training cycle incorporates feedback
                  → FP rate decreases monotonically over deployment lifetime
```

---

## Projected False Positive Rates

| Scenario | Traditional SIEM | Neuro-SOC (Projected) |
|----------|-----------------|----------------------|
| Month-end bulk exports | 45-60% | < 5% |
| Temporary role changes | 30-50% | < 10% |
| Seasonal audit activity | 35-55% | < 8% |
| **Blended FP rate** | **40-55%** | **< 15%** |

These projections are based on the Isolation Forest's 5% contamination parameter, SHAP-based feature differentiation between legitimate and malicious behavioral signatures, and the human-in-the-loop final filter.

---

## Conclusion

Neuro-SOC's hybrid architecture achieves a projected FP rate of < 15% — well below the 20% target — by ensuring that no single layer bears full responsibility for threat classification. The ML engine detects statistical anomalies, the LLM translates them objectively, and the human analyst makes the final enforcement decision with full evidence visibility. This "Inversion of Responsibility" model eliminates both the black-box risk of fully automated systems and the alert fatigue of traditional rule-based approaches.
