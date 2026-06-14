"""
=============================================================================
Synthetic Data Generator — Problem Statement 04
Data Access Audit & Insider Threat Detection (Neuro-SOC)
=============================================================================

Generates 4 files matching the PS04 specification with embedded anomalies:

  1. user_profiles.csv        — 100 user profiles with behavioral baselines
  2. data_access_logs.csv      — 1,200 access events across 365 days
  3. user_profile_labels.csv   — Ground truth: user-level risk assessment
  4. data_access_labels.csv    — Ground truth: event-level anomaly labels

Anomaly Distribution (per PS04 spec):
  - Event-level: ~46% anomalous
  - User-level:  ~17% high-risk

Additional Features (beyond README, all enterprise-available):
  User Profiles:  email, clearance_level, notice_period, last_performance_rating,
                  security_training_current, failed_logins_30d, vpn_user, last_access_date
  Access Logs:    source_ip, bytes_transferred, session_duration_min, client_application,
                  authentication_method, is_vpn, geo_location, data_category

Usage:
  python generate_ps4_data.py
  
Output:
  ./neuro_soc/data_pipeline/raw_data/  (overwrites existing broken files)
  ./neuro_soc/data_pipeline/raw_data/  (adds label files)
"""

import os
import csv
import json
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# SEED & GLOBALS
# ============================================================
SEED = 42
random.seed(SEED)

NUM_USERS = 500
NUM_EVENTS = 50000
DATE_START = datetime(2025, 4, 21)
DATE_END = datetime(2026, 4, 20)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "neuro_soc", "data_pipeline", "raw_data")

# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DEPARTMENTS = {
    "Finance":     {"weight": 0.14, "titles": ["Analyst", "Senior Analyst", "Controller", "Finance Manager", "VP Finance"],
                    "typical_hours": "8-18", "assets": ["GL_Ledger", "AR_System", "AP_System", "Financial_Reports", "Budget_DB"]},
    "IT":          {"weight": 0.14, "titles": ["Sysadmin", "DBA", "Network Engineer", "IT Manager", "Security Engineer"],
                    "typical_hours": "7-22", "assets": ["Admin_Console", "Network_Logs", "SIEM_Dashboard", "Infrastructure_DB", "Backup_System"]},
    "HR":          {"weight": 0.10, "titles": ["HR Analyst", "HR Manager", "Recruiter", "Benefits Specialist", "CHRO"],
                    "typical_hours": "9-17", "assets": ["HR_Database", "Payroll_System", "Employee_Records", "Benefits_DB", "Recruitment_Portal"]},
    "Engineering": {"weight": 0.15, "titles": ["Developer", "Senior Developer", "DevOps Engineer", "Tech Lead", "Principal Engineer"],
                    "typical_hours": "8-20", "assets": ["Code_Repository", "PROD_DB", "Dev_DB", "CI_CD_Pipeline", "Test_Environment"]},
    "Sales":       {"weight": 0.10, "titles": ["Sales Rep", "Account Executive", "Sales Manager", "VP Sales", "Business Dev"],
                    "typical_hours": "8-19", "assets": ["CRM_Database", "Customer_Reports", "Sales_Pipeline", "Lead_Database", "Revenue_Dashboard"]},
    "Marketing":   {"weight": 0.08, "titles": ["Marketing Analyst", "Campaign Manager", "Content Strategist", "Marketing Director", "Growth Lead"],
                    "typical_hours": "9-18", "assets": ["Marketing_Analytics", "Campaign_DB", "Customer_Segments", "Social_Media_DB", "Market_Research"]},
    "Legal":       {"weight": 0.07, "titles": ["Compliance Officer", "Legal Counsel", "Paralegal", "Risk Analyst", "Chief Compliance Officer"],
                    "typical_hours": "9-18", "assets": ["Legal_Documents", "Compliance_DB", "Contract_Repository", "Litigation_DB", "Regulatory_Filings"]},
    "Executive":   {"weight": 0.05, "titles": ["CEO", "COO", "CTO", "CFO", "VP Operations"],
                    "typical_hours": "7-21", "assets": ["Executive_Dashboard", "Board_Materials", "Strategic_Plans", "All_Financial", "Merger_Documents"]},
    "Operations":  {"weight": 0.10, "titles": ["Ops Analyst", "Ops Manager", "Supply Chain Lead", "Logistics Coordinator", "Process Engineer"],
                    "typical_hours": "8-18", "assets": ["Ops_Dashboard", "Inventory_DB", "Logistics_System", "Supply_Chain_DB", "Vendor_Portal"]},
    "Support":     {"weight": 0.07, "titles": ["Support Engineer", "Support Manager", "Help Desk Analyst", "Customer Success", "Escalation Lead"],
                    "typical_hours": "6-22", "assets": ["Ticketing_System", "Knowledge_Base", "Customer_Vault", "FAQ_Database", "SLA_Reports"]},
}

# Every data asset with its sensitivity and data category
ASSET_METADATA = {
    # Finance
    "GL_Ledger":           {"sensitivity": "high",       "category": "Financial"},
    "AR_System":           {"sensitivity": "high",       "category": "Financial"},
    "AP_System":           {"sensitivity": "high",       "category": "Financial"},
    "Financial_Reports":   {"sensitivity": "high",       "category": "Financial"},
    "Budget_DB":           {"sensitivity": "medium",     "category": "Financial"},
    # IT
    "Admin_Console":       {"sensitivity": "medium",     "category": "Technical"},
    "Network_Logs":        {"sensitivity": "medium",     "category": "Technical"},
    "SIEM_Dashboard":      {"sensitivity": "medium",     "category": "Technical"},
    "Infrastructure_DB":   {"sensitivity": "high",       "category": "Technical"},
    "Backup_System":       {"sensitivity": "high",       "category": "Technical"},
    # HR
    "HR_Database":         {"sensitivity": "high",       "category": "HR"},
    "Payroll_System":      {"sensitivity": "restricted", "category": "PII"},
    "Employee_Records":    {"sensitivity": "restricted", "category": "PII"},
    "Benefits_DB":         {"sensitivity": "high",       "category": "HR"},
    "Recruitment_Portal":  {"sensitivity": "medium",     "category": "HR"},
    # Engineering
    "Code_Repository":     {"sensitivity": "medium",     "category": "Technical"},
    "PROD_DB":             {"sensitivity": "high",       "category": "Technical"},
    "Dev_DB":              {"sensitivity": "low",        "category": "Technical"},
    "CI_CD_Pipeline":      {"sensitivity": "medium",     "category": "Technical"},
    "Test_Environment":    {"sensitivity": "low",        "category": "Technical"},
    # Sales
    "CRM_Database":        {"sensitivity": "high",       "category": "PII"},
    "Customer_Reports":    {"sensitivity": "medium",     "category": "Operational"},
    "Sales_Pipeline":      {"sensitivity": "medium",     "category": "Operational"},
    "Lead_Database":       {"sensitivity": "medium",     "category": "PII"},
    "Revenue_Dashboard":   {"sensitivity": "medium",     "category": "Financial"},
    # Marketing
    "Marketing_Analytics": {"sensitivity": "low",        "category": "Operational"},
    "Campaign_DB":         {"sensitivity": "low",        "category": "Operational"},
    "Customer_Segments":   {"sensitivity": "medium",     "category": "PII"},
    "Social_Media_DB":     {"sensitivity": "low",        "category": "Operational"},
    "Market_Research":     {"sensitivity": "low",        "category": "Operational"},
    # Legal
    "Legal_Documents":     {"sensitivity": "high",       "category": "Legal"},
    "Compliance_DB":       {"sensitivity": "high",       "category": "Legal"},
    "Contract_Repository": {"sensitivity": "high",       "category": "Legal"},
    "Litigation_DB":       {"sensitivity": "restricted", "category": "Legal"},
    "Regulatory_Filings":  {"sensitivity": "medium",     "category": "Legal"},
    # Executive
    "Executive_Dashboard": {"sensitivity": "high",       "category": "Financial"},
    "Board_Materials":     {"sensitivity": "restricted", "category": "Financial"},
    "Strategic_Plans":     {"sensitivity": "restricted", "category": "Operational"},
    "All_Financial":       {"sensitivity": "restricted", "category": "Financial"},
    "Merger_Documents":    {"sensitivity": "restricted", "category": "Financial"},
    # Operations
    "Ops_Dashboard":       {"sensitivity": "low",        "category": "Operational"},
    "Inventory_DB":        {"sensitivity": "medium",     "category": "Operational"},
    "Logistics_System":    {"sensitivity": "medium",     "category": "Operational"},
    "Supply_Chain_DB":     {"sensitivity": "medium",     "category": "Operational"},
    "Vendor_Portal":       {"sensitivity": "low",        "category": "Operational"},
    # Support
    "Ticketing_System":    {"sensitivity": "low",        "category": "Operational"},
    "Knowledge_Base":      {"sensitivity": "low",        "category": "Operational"},
    "Customer_Vault":      {"sensitivity": "high",       "category": "PII"},
    "FAQ_Database":        {"sensitivity": "low",        "category": "Operational"},
    "SLA_Reports":         {"sensitivity": "medium",     "category": "Operational"},
}

ALL_ASSETS = list(ASSET_METADATA.keys())

ACCESS_TIERS = ["junior", "standard", "senior", "admin", "executive", "contractor"]
TIER_WEIGHTS = [0.15, 0.30, 0.25, 0.15, 0.05, 0.10]

EQUIPMENT_TYPES = ["company_laptop", "company_desktop", "contractor_machine", "byod_device"]
EQUIPMENT_WEIGHTS = [0.60, 0.20, 0.12, 0.08]

CLEARANCE_LEVELS = ["public", "internal", "confidential", "restricted"]

QUERY_TYPES = ["SELECT", "INSERT", "UPDATE", "DELETE", "EXPORT", "BULK_DOWNLOAD"]
NORMAL_QUERY_WEIGHTS = [0.45, 0.15, 0.15, 0.05, 0.15, 0.05]

ACCESS_METHODS = ["SQL_Client", "BI_Tool", "API", "Web_Portal", "Direct_Download", "CLI"]
NORMAL_METHOD_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.05, 0.05]

NORMAL_DESTINATIONS = ["local_workstation", "internal_server", "shared_drive", "internal_report"]
NORMAL_DEST_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

RISKY_DESTINATIONS = ["external_email", "usb_drive", "cloud_storage", "personal_device", "print"]

CLIENT_APPS = ["SSMS", "Tableau", "Power_BI", "Python_Script", "API_Client",
               "curl", "SAP_GUI", "Excel", "Custom_ETL", "Jupyter_Notebook"]
NORMAL_APP_WEIGHTS = [0.20, 0.20, 0.15, 0.10, 0.10, 0.02, 0.08, 0.10, 0.03, 0.02]

AUTH_METHODS = ["MFA", "SSO", "Password_Only", "API_Key", "Certificate", "Biometric"]
NORMAL_AUTH_WEIGHTS = [0.35, 0.30, 0.10, 0.10, 0.10, 0.05]

GEO_NORMAL = ["Mumbai", "New_York", "London", "Singapore", "Frankfurt", "Sydney", "Tokyo", "Toronto"]
GEO_SUSPICIOUS = ["Unknown_VPN", "Tor_Exit_Node", "Lagos", "Pyongyang", "Caracas"]

FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Akshay", "Alice", "Amit", "Andrew", "Anjali",
    "Benjamin", "Carlos", "Catherine", "Chen", "Daniel", "Deepa", "Elena",
    "Fatima", "George", "Grace", "Haruto", "Isha", "James", "Jessica",
    "Karan", "Kevin", "Lakshmi", "Liam", "Maria", "Michael", "Nadia",
    "Neha", "Oliver", "Priya", "Rachel", "Rahul", "Robert", "Sakura",
    "Sanjay", "Sarah", "Sofia", "Tatiana", "Uma", "Victor", "Wei",
    "Xiulan", "Yuki", "Zainab", "Thomas", "Meera", "Joshua", "Emma",
    "Ryan", "Pooja", "Nathan", "Divya", "Eric", "Shreya", "Patrick",
    "Riya", "Brandon", "Nisha", "Tyler", "Ananya", "Marcus", "Diya",
    "Derek", "Kavya", "Connor", "Simran", "Dylan", "Tara", "Austin",
    "Mira", "Gavin", "Zara", "Blake", "Leela", "Chase", "Jaya",
    "Seth", "Amara", "Troy", "Kiara", "Wade", "Inaya", "Luke", "Siri",
    "Jack", "Lena", "Evan", "Rhea", "Cole", "Aria", "Brett", "Esme",
    "Hugo", "Maya", "Finn", "Nora", "Dean", "Iris",
]

LAST_NAMES = [
    "Anderson", "Bhat", "Burke", "Chen", "Clark", "Das", "Dubois",
    "Garcia", "Ghosh", "Gonzalez", "Gupta", "Harris", "Hernandez",
    "Iyer", "Jackson", "Johnson", "Jones", "Kim", "Kumar", "Lee",
    "Lewis", "Lopez", "Martin", "Martinez", "Menon", "Miller",
    "Moore", "Murphy", "Nair", "O'Brien", "Patel", "Perez",
    "Pillai", "Quinn", "Ramirez", "Rao", "Reddy", "Robinson",
    "Rodriguez", "Sanchez", "Sharma", "Singh", "Smith", "Sullivan",
    "Taylor", "Thomas", "Thompson", "Verma", "White", "Williams",
    "Wilson", "Young", "Brown", "Davis", "Wang", "Li", "Zhang",
    "Park", "Romano", "Fischer", "Meyer", "Becker", "Schneider",
    "Weber", "Colombo", "Rossi", "Bernard", "Petit", "Moreau",
]

# ============================================================
# ANOMALY TYPE DEFINITIONS
# ============================================================

EVENT_ANOMALY_TYPES = {
    "BULK_EXPORT":              {"severity": "HIGH",     "count": 3200, "desc": "Unusually large data export exceeding 10x normal volume"},
    "AFTER_HOURS_RESTRICTED":   {"severity": "HIGH",     "count": 3400, "desc": "Access to high/restricted data outside business hours"},
    "CROSS_DEPT_ACCESS":        {"severity": "MEDIUM",   "count": 3000, "desc": "Accessing data assets outside approved department scope"},
    "STALE_ACCOUNT_ACCESS":     {"severity": "HIGH",     "count": 2000, "desc": "Dormant account suddenly active after extended inactivity"},
    "PRIVILEGE_ESCALATION":     {"severity": "HIGH",     "count": 1800, "desc": "Junior/standard tier accessing admin-level resources"},
    "DEVICE_ANOMALY":           {"severity": "MEDIUM",   "count": 1600, "desc": "Access from unauthorized or mismatched equipment type"},
    "EXFILTRATION_RISK":        {"severity": "CRITICAL", "count": 2200, "desc": "Data sent to external email, USB, or cloud storage"},
    "NIGHT_BULK_CRITICAL":      {"severity": "CRITICAL", "count": 1120, "desc": "After-midnight bulk export to external destination"},
    "PRE_RESIGNATION_DOWNLOAD": {"severity": "CRITICAL", "count": 1600, "desc": "User on notice period performing unusual bulk access"},
    "FAILED_AUTH_BURST":        {"severity": "MEDIUM",   "count": 2080, "desc": "Multiple failed authentication attempts in rapid succession"},
}
# Total anomalous: 22000 / 50000 = 44.0%

USER_RISK_TYPES = {
    "STALE_ACCOUNT":       {"desc": "Account inactive >90 days but still enabled"},
    "OVER_PRIVILEGED":     {"desc": "Access tier exceeds job requirements"},
    "NOTICE_PERIOD_RISK":  {"desc": "Employee on notice period with elevated data access"},
    "EXCESSIVE_SYSTEMS":   {"desc": "Approved for too many sensitive data assets relative to role"},
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(items, weights):
    """Pick from items using given weights."""
    return random.choices(items, weights=weights, k=1)[0]


def gauss_clamp(mean, std, lo, hi):
    """Sample from a Gaussian, clamped to [lo, hi]."""
    return max(lo, min(hi, random.gauss(mean, std)))


def random_timestamp(start, end, hour_lo=0, hour_hi=23):
    """Random datetime between start and end, optionally constrained to hour range."""
    delta = (end - start).total_seconds()
    for _ in range(100):  # retry to fit hour constraint
        ts = start + timedelta(seconds=random.uniform(0, delta))
        if hour_lo <= ts.hour <= hour_hi:
            return ts
    # fallback: force hour
    ts = start + timedelta(seconds=random.uniform(0, delta))
    ts = ts.replace(hour=random.randint(hour_lo, hour_hi))
    return ts


def random_ip(internal=True):
    """Generate an IP address."""
    if internal:
        return f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
    else:
        prefix = random.choice([203, 185, 91, 45, 178])
        return f"{prefix}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"


def classify_time(hour, day_of_week):
    """Classify a timestamp into time categories."""
    is_weekend = day_of_week >= 5
    if is_weekend:
        return "weekend"
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 8:
        return "early_morning"
    if 8 <= hour < 18:
        return "business_hours"
    if 18 <= hour < 22:
        return "after_hours"
    return "night"


# ============================================================
# STEP 1: GENERATE USER PROFILES
# ============================================================

def generate_user_profiles():
    """Generate 100 realistic user profiles with behavioral baselines."""
    users = []
    used_names = set()

    # Build weighted department list
    dept_names = list(DEPARTMENTS.keys())
    dept_weights = [DEPARTMENTS[d]["weight"] for d in dept_names]

    for i in range(NUM_USERS):
        user_id = f"USR-{i:04d}"

        # Unique name
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            username = f"{first.lower()}.{last.lower()}"
            if username not in used_names:
                used_names.add(username)
                break

        dept = weighted_choice(dept_names, dept_weights)
        dept_cfg = DEPARTMENTS[dept]
        job_title = random.choice(dept_cfg["titles"])

        # Access tier (correlated with seniority in title)
        senior_keywords = ["Senior", "Manager", "Director", "VP", "Lead", "Principal", "Chief", "CEO", "COO", "CTO", "CFO", "CHRO"]
        if any(kw in job_title for kw in senior_keywords):
            access_tier = weighted_choice(["senior", "admin", "executive"], [0.50, 0.35, 0.15])
        elif "Junior" in job_title or "Intern" in job_title:
            access_tier = "junior"
        else:
            access_tier = weighted_choice(ACCESS_TIERS, TIER_WEIGHTS)

        # Contractors override
        is_contractor = access_tier == "contractor"
        if is_contractor:
            equipment = "contractor_machine"
        else:
            equipment = weighted_choice(EQUIPMENT_TYPES, EQUIPMENT_WEIGHTS)

        # Tenure (months) — contractors short, seniors long
        if is_contractor:
            tenure_months = int(gauss_clamp(6, 4, 1, 18))
        elif access_tier in ["senior", "admin", "executive"]:
            tenure_months = int(gauss_clamp(60, 30, 12, 240))
        else:
            tenure_months = int(gauss_clamp(24, 18, 1, 120))

        # Typical access hours (from department)
        typical_access_hours = dept_cfg["typical_hours"]

        # Approved data assets (department assets + maybe 1-2 cross-dept for seniors)
        approved = list(dept_cfg["assets"][:random.randint(2, len(dept_cfg["assets"]))])
        if access_tier in ["senior", "admin", "executive"]:
            # May have cross-department access
            other_depts = [d for d in dept_names if d != dept]
            extra_dept = random.choice(other_depts)
            extra_assets = DEPARTMENTS[extra_dept]["assets"]
            approved.extend(random.sample(extra_assets, min(2, len(extra_assets))))
        approved = list(set(approved))  # deduplicate

        # Behavioral baselines
        tier_query_ranges = {
            "junior": (2, 8), "standard": (5, 20), "senior": (10, 35),
            "admin": (15, 50), "executive": (3, 12), "contractor": (3, 15)
        }
        q_lo, q_hi = tier_query_ranges[access_tier]
        avg_queries_per_day = round(random.uniform(q_lo, q_hi), 1)

        tier_rowcount_ranges = {
            "junior": (20, 200), "standard": (50, 500), "senior": (100, 2000),
            "admin": (200, 5000), "executive": (10, 100), "contractor": (50, 1000)
        }
        r_lo, r_hi = tier_rowcount_ranges[access_tier]
        avg_rowcount_per_query = int(gauss_clamp((r_lo + r_hi) / 2, (r_hi - r_lo) / 4, r_lo, r_hi))

        # ---- ADDITIONAL HIGH-IMPACT FEATURES ----

        email = f"{username}@societe-corp.com"

        # Clearance level (correlated with access tier)
        clearance_map = {
            "junior": weighted_choice(CLEARANCE_LEVELS, [0.40, 0.40, 0.15, 0.05]),
            "standard": weighted_choice(CLEARANCE_LEVELS, [0.10, 0.40, 0.35, 0.15]),
            "senior": weighted_choice(CLEARANCE_LEVELS, [0.05, 0.20, 0.45, 0.30]),
            "admin": weighted_choice(CLEARANCE_LEVELS, [0.02, 0.10, 0.38, 0.50]),
            "executive": weighted_choice(CLEARANCE_LEVELS, [0.00, 0.05, 0.25, 0.70]),
            "contractor": weighted_choice(CLEARANCE_LEVELS, [0.50, 0.35, 0.10, 0.05]),
        }
        clearance_level = clearance_map[access_tier]

        # Notice period (~8% of users are on notice — some will become high risk)
        notice_period = random.random() < 0.08

        # Performance rating (1.0–5.0; lower = more disgruntled risk)
        last_performance_rating = round(gauss_clamp(3.5, 0.8, 1.0, 5.0), 1)

        # Security training compliance
        security_training_current = random.random() < 0.85

        # Failed logins in last 30 days (most have 0-2, a few have more)
        if random.random() < 0.12:
            failed_logins_30d = random.randint(5, 25)  # suspicious
        else:
            failed_logins_30d = random.randint(0, 3)

        # VPN user (remote workers)
        vpn_user = random.random() < 0.35

        # High risk flag — ~17% of users
        # Conditions: notice period, very low perf rating, high failed logins, contractor with sensitive access
        high_risk_reasons = []
        if notice_period:
            high_risk_reasons.append("resignation_filed")
        if last_performance_rating <= 2.0:
            high_risk_reasons.append("low_performance_pip")
        if failed_logins_30d >= 10:
            high_risk_reasons.append("account_compromise_risk")
        if is_contractor and any(ASSET_METADATA.get(a, {}).get("sensitivity") in ["high", "restricted"] for a in approved):
            high_risk_reasons.append("contractor_sensitive_access")

        # Force some additional high-risk to hit ~17%
        high_risk_flag = len(high_risk_reasons) > 0
        if not high_risk_flag and random.random() < 0.07:
            high_risk_flag = True
            high_risk_reasons.append(random.choice(["security_violation_history", "termination_pending", "under_investigation"]))

        # Last access date (most recent, some stale)
        if random.random() < 0.08:
            # Stale account: last access 60-180 days ago
            last_access_date = (DATE_END - timedelta(days=random.randint(60, 180))).strftime("%Y-%m-%d")
        else:
            last_access_date = (DATE_END - timedelta(days=random.randint(0, 14))).strftime("%Y-%m-%d")

        user = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "department": dept,
            "job_title": job_title,
            "access_tier": access_tier,
            "tenure_months": tenure_months,
            "approved_data_assets": "|".join(approved),
            "avg_queries_per_day": avg_queries_per_day,
            "typical_access_hours": typical_access_hours,
            "avg_rowcount_per_query": avg_rowcount_per_query,
            "high_risk_flag": high_risk_flag,
            "equipment": equipment,
            "clearance_level": clearance_level,
            "notice_period": notice_period,
            "last_performance_rating": last_performance_rating,
            "security_training_current": security_training_current,
            "failed_logins_30d": failed_logins_30d,
            "vpn_user": vpn_user,
            "last_access_date": last_access_date,
            # Internal metadata (not written to CSV, used for generation)
            "_dept_cfg": dept_cfg,
            "_high_risk_reasons": high_risk_reasons,
        }
        users.append(user)

    return users


# ============================================================
# STEP 2: GENERATE USER PROFILE LABELS
# ============================================================

def generate_user_labels(users):
    """Generate ground truth labels for user-level risk."""
    labels = []
    for u in users:
        is_anomaly = u["high_risk_flag"]

        # Determine specific risk type
        risk_reasons = u["_high_risk_reasons"]
        if not is_anomaly:
            risk_type = "NORMAL"
            severity = "NONE"
            explanation = "User profile within normal parameters"
        else:
            # Map reasons to risk types
            if "resignation_filed" in risk_reasons:
                risk_type = "NOTICE_PERIOD_RISK"
                severity = "HIGH"
                explanation = f"Employee on notice period with access to {len(u['approved_data_assets'].split('|'))} data assets. Elevated exfiltration risk."
            elif "account_compromise_risk" in risk_reasons:
                risk_type = "STALE_ACCOUNT"
                severity = "HIGH"
                explanation = f"{u['failed_logins_30d']} failed login attempts in 30 days suggests potential credential compromise."
            elif "contractor_sensitive_access" in risk_reasons:
                risk_type = "OVER_PRIVILEGED"
                severity = "MEDIUM"
                explanation = f"Contractor with {u['access_tier']} tier has access to high/restricted sensitivity assets."
            elif "low_performance_pip" in risk_reasons:
                risk_type = "NOTICE_PERIOD_RISK"
                severity = "MEDIUM"
                explanation = f"Employee on PIP (rating: {u['last_performance_rating']}) with active data access — disgruntled insider risk."
            else:
                risk_type = random.choice(["STALE_ACCOUNT", "OVER_PRIVILEGED", "EXCESSIVE_SYSTEMS"])
                severity = "MEDIUM"
                reason = risk_reasons[0] if risk_reasons else "flagged_by_security"
                explanation = f"User flagged for: {reason}. Requires periodic access review."

        labels.append({
            "user_id": u["user_id"],
            "is_anomaly": is_anomaly,
            "risk_type": risk_type,
            "severity": severity,
            "explanation": explanation,
        })

    return labels


# ============================================================
# STEP 3: GENERATE ACCESS EVENTS
# ============================================================

def generate_normal_event(event_id, user, ts):
    """Generate a single normal (non-anomalous) access event."""
    dept_cfg = user["_dept_cfg"]
    approved = user["approved_data_assets"].split("|")
    asset = random.choice(approved)
    meta = ASSET_METADATA[asset]

    # Normal rowcount: near user's average
    avg_rc = user["avg_rowcount_per_query"]
    rowcount = max(1, int(gauss_clamp(avg_rc, avg_rc * 0.3, 1, avg_rc * 3)))

    query_type = weighted_choice(QUERY_TYPES, NORMAL_QUERY_WEIGHTS)
    access_method = weighted_choice(ACCESS_METHODS, NORMAL_METHOD_WEIGHTS)
    destination = weighted_choice(NORMAL_DESTINATIONS, NORMAL_DEST_WEIGHTS)
    client_app = weighted_choice(CLIENT_APPS, NORMAL_APP_WEIGHTS)
    auth_method = weighted_choice(AUTH_METHODS, NORMAL_AUTH_WEIGHTS)

    # Bytes proportional to rowcount
    bytes_per_row = random.randint(80, 500)
    bytes_transferred = rowcount * bytes_per_row

    session_duration = round(gauss_clamp(15, 10, 1, 120), 1)

    geo = random.choice(GEO_NORMAL)
    is_vpn = user["vpn_user"] and random.random() < 0.6
    source_ip = random_ip(internal=True)

    time_class = classify_time(ts.hour, ts.weekday())

    return {
        "access_id": f"ACC-{event_id:06d}",
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user["user_id"],
        "username": user["username"],
        "department": user["department"],
        "data_asset": asset,
        "data_sensitivity": meta["sensitivity"],
        "data_category": meta["category"],
        "query_type": query_type,
        "rowcount": rowcount,
        "access_method": access_method,
        "destination": destination,
        "status": "success",
        "source_ip": source_ip,
        "bytes_transferred": bytes_transferred,
        "session_duration_min": session_duration,
        "client_application": client_app,
        "authentication_method": auth_method,
        "is_vpn": is_vpn,
        "geo_location": geo,
        "time_classification": time_class,
        "anomaly_marker": "",
    }


def generate_anomalous_event(event_id, user, anomaly_type, ts=None):
    """Generate a single anomalous access event of the specified type."""

    dept_cfg = user["_dept_cfg"]
    approved = user["approved_data_assets"].split("|")

    # Defaults (will be overridden per anomaly type)
    if ts is None:
        ts = random_timestamp(DATE_START, DATE_END)
    asset = random.choice(approved)
    meta = ASSET_METADATA[asset]
    avg_rc = user["avg_rowcount_per_query"]
    rowcount = max(1, int(gauss_clamp(avg_rc, avg_rc * 0.3, 1, avg_rc * 3)))
    query_type = weighted_choice(QUERY_TYPES, NORMAL_QUERY_WEIGHTS)
    access_method = weighted_choice(ACCESS_METHODS, NORMAL_METHOD_WEIGHTS)
    destination = weighted_choice(NORMAL_DESTINATIONS, NORMAL_DEST_WEIGHTS)
    client_app = weighted_choice(CLIENT_APPS, NORMAL_APP_WEIGHTS)
    auth_method = weighted_choice(AUTH_METHODS, NORMAL_AUTH_WEIGHTS)
    status = "success"
    source_ip = random_ip(internal=True)
    geo = random.choice(GEO_NORMAL)
    is_vpn = user["vpn_user"] and random.random() < 0.6

    explanation = ""

    # ---- ANOMALY-SPECIFIC OVERRIDES ----

    if anomaly_type == "BULK_EXPORT":
        rowcount = int(avg_rc * random.uniform(10, 100))
        query_type = random.choice(["EXPORT", "BULK_DOWNLOAD", "SELECT"])
        access_method = random.choice(["Direct_Download", "SQL_Client", "CLI"])
        # Pick a sensitive asset
        sensitive_assets = [a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        asset = random.choice(sensitive_assets)
        meta = ASSET_METADATA[asset]
        explanation = f"Exported {rowcount} records from {asset} (avg is {avg_rc}). Volume is {rowcount/max(1,avg_rc):.0f}x normal."

    elif anomaly_type == "AFTER_HOURS_RESTRICTED":
        # Force timestamp to after-hours (22:00-05:59)
        hour = random.choice(list(range(22, 24)) + list(range(0, 6)))
        ts = ts.replace(hour=hour, minute=random.randint(0, 59))
        # Access high/restricted data
        sensitive_assets = [a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        asset = random.choice(sensitive_assets)
        meta = ASSET_METADATA[asset]
        explanation = f"Accessed {meta['sensitivity']}-sensitivity {asset} at {ts.strftime('%H:%M')} (outside {user['typical_access_hours']} hours)."

    elif anomaly_type == "CROSS_DEPT_ACCESS":
        # Access asset NOT in approved list
        unapproved = [a for a in ALL_ASSETS if a not in approved]
        # Prefer sensitive unapproved assets
        sensitive_unapproved = [a for a in unapproved if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        if sensitive_unapproved:
            asset = random.choice(sensitive_unapproved)
        else:
            asset = random.choice(unapproved)
        meta = ASSET_METADATA[asset]
        explanation = f"{user['department']} employee accessed {asset} ({meta['sensitivity']} sensitivity) — not in approved assets."

    elif anomaly_type == "STALE_ACCOUNT_ACCESS":
        # This user has been inactive but suddenly makes access
        explanation = f"Account last active {user['last_access_date']}; sudden access to {asset} after extended dormancy."

    elif anomaly_type == "PRIVILEGE_ESCALATION":
        # Junior/standard accessing admin-level resources
        admin_assets = ["Admin_Console", "Infrastructure_DB", "SIEM_Dashboard", "Backup_System",
                        "Board_Materials", "Strategic_Plans", "Merger_Documents", "All_Financial"]
        asset = random.choice(admin_assets)
        meta = ASSET_METADATA[asset]
        explanation = f"{user['access_tier']}-tier user accessed {asset} ({meta['sensitivity']}). Exceeds privilege level."

    elif anomaly_type == "DEVICE_ANOMALY":
        # Wrong equipment type for the action
        if user["equipment"] == "contractor_machine":
            sensitive_assets = [a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
            asset = random.choice(sensitive_assets)
            meta = ASSET_METADATA[asset]
            explanation = f"Contractor machine accessing {meta['sensitivity']}-sensitivity {asset}. Equipment policy violation."
        else:
            # Simulate access from unregistered device
            client_app = random.choice(["curl", "Python_Script", "Custom_ETL"])
            source_ip = random_ip(internal=False)
            explanation = f"Access from unregistered device (IP: {source_ip}) using {client_app}. Not matching registered equipment."

    elif anomaly_type == "EXFILTRATION_RISK":
        destination = random.choice(RISKY_DESTINATIONS)
        query_type = random.choice(["EXPORT", "BULK_DOWNLOAD"])
        sensitive_assets = [a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        asset = random.choice(sensitive_assets)
        meta = ASSET_METADATA[asset]
        rowcount = int(avg_rc * random.uniform(3, 20))
        explanation = f"Exported {rowcount} records from {asset} to {destination}. High exfiltration risk."

    elif anomaly_type == "NIGHT_BULK_CRITICAL":
        # Compound: night + bulk + external destination
        hour = random.choice(list(range(0, 5)))
        ts = ts.replace(hour=hour, minute=random.randint(0, 59))
        rowcount = int(avg_rc * random.uniform(20, 150))
        query_type = random.choice(["EXPORT", "BULK_DOWNLOAD"])
        destination = random.choice(["external_email", "usb_drive", "cloud_storage"])
        access_method = random.choice(["Direct_Download", "CLI"])
        sensitive_assets = [a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        asset = random.choice(sensitive_assets)
        meta = ASSET_METADATA[asset]
        geo = random.choice(GEO_SUSPICIOUS + GEO_NORMAL[:2])
        auth_method = random.choice(["Password_Only", "API_Key"])
        explanation = (f"CRITICAL: {rowcount} records bulk exported from {asset} at {ts.strftime('%H:%M')} "
                      f"to {destination}. Weak auth ({auth_method}), geo: {geo}.")

    elif anomaly_type == "PRE_RESIGNATION_DOWNLOAD":
        # User on notice period doing bulk downloads
        query_type = random.choice(["EXPORT", "BULK_DOWNLOAD", "SELECT"])
        rowcount = int(avg_rc * random.uniform(5, 50))
        destination = random.choice(["cloud_storage", "usb_drive", "personal_device", "external_email"])
        # Access their own department's sensitive data (they know where the good stuff is)
        dept_assets = [a for a in dept_cfg["assets"] if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]]
        if dept_assets:
            asset = random.choice(dept_assets)
        else:
            asset = random.choice([a for a in ALL_ASSETS if ASSET_METADATA[a]["sensitivity"] in ["high", "restricted"]])
        meta = ASSET_METADATA[asset]
        explanation = (f"Employee on notice period downloaded {rowcount} records from {asset} to {destination}. "
                      f"Tenure: {user['tenure_months']}mo, notice_period=True.")

    elif anomaly_type == "FAILED_AUTH_BURST":
        status = "failure"
        auth_method = random.choice(["Password_Only", "MFA", "SSO"])
        # Sometimes from suspicious location
        if random.random() < 0.4:
            geo = random.choice(GEO_SUSPICIOUS)
            source_ip = random_ip(internal=False)
        rowcount = 0
        explanation = f"Failed {auth_method} authentication attempt from {geo} (IP: {source_ip}). Part of burst pattern."

    # Compute derived fields
    bytes_per_row = random.randint(80, 500)
    bytes_transferred = rowcount * bytes_per_row
    session_duration = round(gauss_clamp(15, 10, 1, 120), 1) if status == "success" else round(random.uniform(0.1, 2.0), 1)
    time_class = classify_time(ts.hour, ts.weekday())

    event = {
        "access_id": f"ACC-{event_id:06d}",
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user["user_id"],
        "username": user["username"],
        "department": user["department"],
        "data_asset": asset,
        "data_sensitivity": meta["sensitivity"],
        "data_category": meta["category"],
        "query_type": query_type,
        "rowcount": rowcount,
        "access_method": access_method,
        "destination": destination,
        "status": status,
        "source_ip": source_ip,
        "bytes_transferred": bytes_transferred,
        "session_duration_min": session_duration,
        "client_application": client_app,
        "authentication_method": auth_method,
        "is_vpn": is_vpn,
        "geo_location": geo,
        "time_classification": time_class,
        "anomaly_marker": anomaly_type,
    }

    label = {
        "access_id": event["access_id"],
        "is_anomaly": True,
        "anomaly_type": anomaly_type,
        "severity": EVENT_ANOMALY_TYPES[anomaly_type]["severity"],
        "explanation": explanation,
    }

    return event, label


def generate_access_logs(users):
    """Generate 1,200 access events with ~46% anomaly density."""
    events = []
    event_labels = []
    event_id = 0

    # ---- Generate anomalous events ----
    # Separate users into pools
    high_risk_users = [u for u in users if u["high_risk_flag"]]
    stale_users = [u for u in users if u["last_access_date"] < (DATE_END - timedelta(days=50)).strftime("%Y-%m-%d")]
    notice_users = [u for u in users if u["notice_period"]]
    junior_users = [u for u in users if u["access_tier"] in ["junior", "standard"]]
    contractor_users = [u for u in users if u["access_tier"] == "contractor"]

    for anomaly_type, cfg in EVENT_ANOMALY_TYPES.items():
        count = cfg["count"]
        for _ in range(count):
            ts = random_timestamp(DATE_START, DATE_END)

            # Pick appropriate user for anomaly type
            if anomaly_type == "STALE_ACCOUNT_ACCESS" and stale_users:
                user = random.choice(stale_users)
            elif anomaly_type == "PRE_RESIGNATION_DOWNLOAD" and notice_users:
                user = random.choice(notice_users)
            elif anomaly_type in ["PRIVILEGE_ESCALATION"] and junior_users:
                user = random.choice(junior_users)
            elif anomaly_type == "DEVICE_ANOMALY" and contractor_users:
                user = random.choice(contractor_users + random.sample(users, min(5, len(users))))
            elif anomaly_type in ["NIGHT_BULK_CRITICAL", "EXFILTRATION_RISK"] and high_risk_users:
                user = random.choice(high_risk_users + random.sample(users, min(10, len(users))))
            else:
                user = random.choice(users)

            event, label = generate_anomalous_event(event_id, user, anomaly_type, ts)
            events.append(event)
            event_labels.append(label)
            event_id += 1

    num_anomalous = event_id
    num_normal = NUM_EVENTS - num_anomalous

    # ---- Generate normal events ----
    for _ in range(num_normal):
        user = random.choice(users)
        dept_cfg = user["_dept_cfg"]

        # Parse typical hours for this user
        hour_parts = user["typical_access_hours"].split("-")
        h_lo, h_hi = int(hour_parts[0]), int(hour_parts[1])

        # Timestamp during typical hours (with weekday bias)
        ts = random_timestamp(DATE_START, DATE_END, hour_lo=h_lo, hour_hi=min(h_hi, 23))
        # Slight weekday bias
        for _ in range(5):
            if ts.weekday() < 5:
                break
            ts = random_timestamp(DATE_START, DATE_END, hour_lo=h_lo, hour_hi=min(h_hi, 23))

        event = generate_normal_event(event_id, user, ts)
        events.append(event)

        event_labels.append({
            "access_id": event["access_id"],
            "is_anomaly": False,
            "anomaly_type": "NORMAL",
            "severity": "NONE",
            "explanation": "Access within normal behavioral parameters.",
        })
        event_id += 1

    # ---- Shuffle events chronologically ----
    combined = list(zip(events, event_labels))
    combined.sort(key=lambda x: x[0]["timestamp"])

    # Re-assign sequential access IDs after sorting
    for idx, (ev, lab) in enumerate(combined):
        new_id = f"ACC-{idx:06d}"
        ev["access_id"] = new_id
        lab["access_id"] = new_id

    events = [c[0] for c in combined]
    event_labels = [c[1] for c in combined]

    return events, event_labels


# ============================================================
# STEP 4: WRITE OUTPUT FILES
# ============================================================

def write_csv(filepath, data, fieldnames):
    """Write a list of dicts to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"  [OK] {os.path.basename(filepath):35s} -> {len(data):>5,} rows  [{os.path.getsize(filepath):>10,} bytes]")


def main():
    print("=" * 70)
    print("  Neuro-SOC PS04 Synthetic Data Generator")
    print("=" * 70)
    print()

    # Step 1: Users
    print("[1/4] Generating user profiles...")
    users = generate_user_profiles()

    # Step 2: User labels
    print("[2/4] Generating user profile labels...")
    user_labels = generate_user_labels(users)

    # Step 3: Access logs + labels
    print("[3/4] Generating access logs with embedded anomalies...")
    events, event_labels = generate_access_logs(users)

    # Step 4: Write files
    print("[4/4] Writing output files...")
    print()

    # Clean internal metadata from user dicts before writing
    user_fields = [
        "user_id", "username", "email", "department", "job_title", "access_tier",
        "tenure_months", "approved_data_assets", "avg_queries_per_day",
        "typical_access_hours", "avg_rowcount_per_query", "high_risk_flag",
        "equipment", "clearance_level", "notice_period", "last_performance_rating",
        "security_training_current", "failed_logins_30d", "vpn_user", "last_access_date",
    ]
    users_clean = [{k: u[k] for k in user_fields} for u in users]

    event_fields = [
        "access_id", "timestamp", "user_id", "username", "department",
        "data_asset", "data_sensitivity", "data_category", "query_type",
        "rowcount", "access_method", "destination", "status",
        "source_ip", "bytes_transferred", "session_duration_min",
        "client_application", "authentication_method", "is_vpn",
        "geo_location", "time_classification", "anomaly_marker",
    ]

    user_label_fields = ["user_id", "is_anomaly", "risk_type", "severity", "explanation"]
    event_label_fields = ["access_id", "is_anomaly", "anomaly_type", "severity", "explanation"]

    write_csv(os.path.join(OUTPUT_DIR, "user_profiles.csv"), users_clean, user_fields)
    write_csv(os.path.join(OUTPUT_DIR, "data_access_logs.csv"), events, event_fields)
    write_csv(os.path.join(OUTPUT_DIR, "user_profile_labels.csv"), user_labels, user_label_fields)
    write_csv(os.path.join(OUTPUT_DIR, "data_access_labels.csv"), event_labels, event_label_fields)

    # Print summary statistics
    print()
    print("-" * 70)
    print("  SUMMARY")
    print("-" * 70)

    n_risk_users = sum(1 for l in user_labels if l["is_anomaly"])
    n_anomalous_events = sum(1 for l in event_labels if l["is_anomaly"])
    n_total_events = len(events)

    print(f"  Users:              {len(users_clean)}")
    print(f"  High-risk users:    {n_risk_users} ({n_risk_users/len(users_clean)*100:.1f}%)")
    print()
    print(f"  Total events:       {n_total_events}")
    print(f"  Anomalous events:   {n_anomalous_events} ({n_anomalous_events/n_total_events*100:.1f}%)")
    print(f"  Normal events:      {n_total_events - n_anomalous_events} ({(n_total_events - n_anomalous_events)/n_total_events*100:.1f}%)")
    print()

    # Anomaly type breakdown
    type_counts = defaultdict(int)
    for l in event_labels:
        if l["is_anomaly"]:
            type_counts[l["anomaly_type"]] += 1

    print("  Anomaly breakdown:")
    for atype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        sev = EVENT_ANOMALY_TYPES[atype]["severity"]
        print(f"    {atype:35s} {count:>4} events  [{sev}]")

    # Department distribution
    print()
    print("  Department distribution:")
    dept_counts = defaultdict(int)
    for u in users_clean:
        dept_counts[u["department"]] += 1
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"    {dept:20s} {count:>3} users")

    # Sensitivity distribution in events
    print()
    print("  Data sensitivity in events:")
    sens_counts = defaultdict(int)
    for e in events:
        sens_counts[e["data_sensitivity"]] += 1
    for sens, count in sorted(sens_counts.items(), key=lambda x: -x[1]):
        print(f"    {sens:15s} {count:>5} events")

    # Additional feature coverage
    print()
    print("  Additional features (beyond PS04 README):")
    print(f"    notice_period=True:            {sum(1 for u in users_clean if u['notice_period']):>3} users")
    print(f"    failed_logins_30d >= 5:        {sum(1 for u in users_clean if u['failed_logins_30d'] >= 5):>3} users")
    print(f"    vpn_user=True:                 {sum(1 for u in users_clean if u['vpn_user']):>3} users")
    print(f"    security_training_current=F:   {sum(1 for u in users_clean if not u['security_training_current']):>3} users")
    print(f"    events with external dest:     {sum(1 for e in events if e['destination'] in ['external_email', 'usb_drive', 'cloud_storage', 'personal_device']):>5} events")
    print(f"    events from external IP:       {sum(1 for e in events if not e['source_ip'].startswith('10.')):>5} events")
    print(f"    events with suspicious geo:    {sum(1 for e in events if e['geo_location'] in GEO_SUSPICIOUS):>5} events")

    print()
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print()
    print("=" * 70)
    print("  Generation complete. All 4 files ready for Neuro-SOC pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
