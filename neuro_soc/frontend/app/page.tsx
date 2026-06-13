"use client";

import React, { useState } from "react";
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  Activity,
  Clock,
  User,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  Lock,
  Flag,
  Server,
  ArrowRight,
  Radio,
  FileWarning,
  BarChart3,
  ChevronRight,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════════════════
//  TYPE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

interface EventData {
  user_id: string;
  days_inactive: number;
  tenure_months: number;
  high_risk_flag: number;
  Tenure_Risk_Modifier: number;
  Equipment_Mismatch_Score: number;
  Temporal_Velocity: number;
  rowcount_deviation: number;
  After_Hours_High_Sensitivity: number;
  Failed_Action_Flag: number;
  systems_access_count: number;
  Privilege_Sensitivity_Mismatch: number;
  _timestamp?: string;
  _label?: string;
}

interface ShapFeature {
  feature: string;
  shap_value: number;
  event_value: number;
}

interface LlmNarrative {
  threat_narrative: string;
  evidence_list: string[];
  recommended_action: string;
}

interface AnalysisResult {
  user_id: string;
  is_anomaly: boolean;
  anomaly_score: number;
  prediction: number;
  top_shap_features: ShapFeature[];
  llm_narrative: LlmNarrative | null;
  cache_hit: boolean;
  analyzed_at: string;
}

// ═══════════════════════════════════════════════════════════════════════════
//  MOCK EVENTS
// ═══════════════════════════════════════════════════════════════════════════

const MOCK_EVENTS: EventData[] = [
  {
    user_id: "USR00015",
    days_inactive: 24,
    tenure_months: 6,
    high_risk_flag: 1,
    Tenure_Risk_Modifier: 2.0,
    Equipment_Mismatch_Score: 0,
    Temporal_Velocity: 18,
    rowcount_deviation: 4.8,
    After_Hours_High_Sensitivity: 1,
    Failed_Action_Flag: 1,
    systems_access_count: 7,
    Privilege_Sensitivity_Mismatch: 0.29,
    _timestamp: "2026-06-13 23:41",
    _label: "Bulk export after hours — 7 systems",
  },
  {
    user_id: "USR00057",
    days_inactive: 14,
    tenure_months: 3,
    high_risk_flag: 1,
    Tenure_Risk_Modifier: 3.0,
    Equipment_Mismatch_Score: 1,
    Temporal_Velocity: 25,
    rowcount_deviation: 5.2,
    After_Hours_High_Sensitivity: 1,
    Failed_Action_Flag: 1,
    systems_access_count: 1,
    Privilege_Sensitivity_Mismatch: 3.0,
    _timestamp: "2026-06-13 22:18",
    _label: "New hire burst — device mismatch",
  },
  {
    user_id: "USR00074",
    days_inactive: 0,
    tenure_months: 48,
    high_risk_flag: 0,
    Tenure_Risk_Modifier: 1.0,
    Equipment_Mismatch_Score: 0,
    Temporal_Velocity: 1,
    rowcount_deviation: 1.0,
    After_Hours_High_Sensitivity: 0,
    Failed_Action_Flag: 0,
    systems_access_count: 2,
    Privilege_Sensitivity_Mismatch: 1.0,
    _timestamp: "2026-06-13 14:22",
    _label: "Routine DB read — standard access",
  },
  {
    user_id: "USR00039",
    days_inactive: 25,
    tenure_months: 31,
    high_risk_flag: 0,
    Tenure_Risk_Modifier: 1.0,
    Equipment_Mismatch_Score: 1,
    Temporal_Velocity: 3,
    rowcount_deviation: 1.0,
    After_Hours_High_Sensitivity: 0,
    Failed_Action_Flag: 1,
    systems_access_count: 2,
    Privilege_Sensitivity_Mismatch: 0.5,
    _timestamp: "2026-06-13 11:30",
    _label: "Unusual device — failed login attempt",
  },
  {
    user_id: "USR00022",
    days_inactive: 52,
    tenure_months: 11,
    high_risk_flag: 0,
    Tenure_Risk_Modifier: 1.0,
    Equipment_Mismatch_Score: 0,
    Temporal_Velocity: 1,
    rowcount_deviation: 0.92,
    After_Hours_High_Sensitivity: 0,
    Failed_Action_Flag: 0,
    systems_access_count: 2,
    Privilege_Sensitivity_Mismatch: 1.0,
    _timestamp: "2026-06-13 09:15",
    _label: "Standard CRM access",
  },
  {
    user_id: "USR00081",
    days_inactive: 44,
    tenure_months: 3,
    high_risk_flag: 1,
    Tenure_Risk_Modifier: 3.0,
    Equipment_Mismatch_Score: 0,
    Temporal_Velocity: 12,
    rowcount_deviation: 3.1,
    After_Hours_High_Sensitivity: 1,
    Failed_Action_Flag: 0,
    systems_access_count: 3,
    Privilege_Sensitivity_Mismatch: 1.0,
    _timestamp: "2026-06-13 03:22",
    _label: "Early-AM data pull — short tenure",
  },
];

// ═══════════════════════════════════════════════════════════════════════════
//  API
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = "http://localhost:8000";

async function analyzeEvent(eventData: EventData): Promise<AnalysisResult> {
  const { _timestamp, _label, ...payload } = eventData;
  const res = await fetch(`${API_BASE}/analyze_event`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function normalizeScore(raw: number): number {
  const clamped = Math.max(-0.5, Math.min(0.5, raw));
  return Math.round(((0.5 - clamped) / 1.0) * 100);
}

function getRisk(score: number) {
  if (score >= 80)
    return { label: "CRITICAL", fg: "text-red-400", bg: "bg-red-500/8", border: "border-red-500/20" };
  if (score >= 50)
    return { label: "ELEVATED", fg: "text-amber-400", bg: "bg-amber-500/8", border: "border-amber-500/20" };
  return { label: "NORMAL", fg: "text-emerald-400", bg: "bg-emerald-500/8", border: "border-emerald-500/20" };
}

function actionStyle(a: string) {
  if (a === "Quarantine") return "text-red-400 border-red-500/25 bg-red-500/8";
  if (a === "Investigate") return "text-amber-400 border-amber-500/25 bg-amber-500/8";
  return "text-emerald-400 border-emerald-500/25 bg-emerald-500/8";
}

function fmtFeature(s: string) {
  return s.replace(/_/g, " ").replace(/([A-Z])/g, " $1").trim();
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE
// ═══════════════════════════════════════════════════════════════════════════

export default function Dashboard() {
  const [selected, setSelected] = useState<EventData | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClick = async (ev: EventData) => {
    setSelected(ev);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      setResult(await analyzeEvent(ev));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const handleQuarantine = () => {
    if (!result) return;
    console.log("🔒 QUARANTINE:", result.user_id, result);
    alert(`${result.user_id} quarantined. Access revoked.`);
    setSelected(null);
    setResult(null);
  };

  const handleFP = () => {
    if (!result) return;
    console.log("✅ FALSE POSITIVE:", result.user_id, result);
    alert(`${result.user_id} marked as false positive.`);
    setSelected(null);
    setResult(null);
  };

  const norm = result ? normalizeScore(result.anomaly_score) : 0;
  const risk = getRisk(norm);

  return (
    <div className="flex h-screen bg-[#101010] text-[#e0e0e0]">
      {/* ──────────── SIDEBAR ──────────── */}
      <aside className="flex w-[340px] flex-shrink-0 flex-col border-r border-[#1e1e1e]">
        {/* Brand */}
        <div className="border-b border-[#1e1e1e] px-5 py-5">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-[#808080]" />
            <span className="text-[13px] font-semibold tracking-wider text-[#c0c0c0]">
              NEURO-SOC
            </span>
          </div>
          <div className="mt-3 flex gap-4 text-[10px] text-[#555]">
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Online
            </span>
            <span className="flex items-center gap-1.5">
              <Server className="h-2.5 w-2.5" /> Redis
            </span>
            <span className="flex items-center gap-1.5">
              <Radio className="h-2.5 w-2.5" /> LLM
            </span>
          </div>
        </div>

        {/* List header */}
        <div className="flex items-center justify-between px-5 py-3">
          <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#555]">
            Event Feed
          </span>
          <span className="text-[10px] text-[#444]">{MOCK_EVENTS.length}</span>
        </div>

        {/* Events */}
        <div className="flex-1 overflow-y-auto px-3 pb-3">
          {MOCK_EVENTS.map((ev, i) => {
            const risky =
              ev.Temporal_Velocity > 5 ||
              ev.rowcount_deviation > 2 ||
              ev.After_Hours_High_Sensitivity === 1 ||
              ev.Failed_Action_Flag === 1;
            const active =
              selected?.user_id === ev.user_id &&
              selected?._timestamp === ev._timestamp;

            return (
              <button
                key={`${ev.user_id}-${i}`}
                id={`event-${ev.user_id}-${i}`}
                onClick={() => handleClick(ev)}
                className={`group mb-1 w-full rounded-md border px-3.5 py-3 text-left transition-colors
                  ${active
                    ? "border-[#333] bg-[#1a1a1a]"
                    : "border-transparent hover:bg-[#161616]"
                  }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div
                      className={`flex h-7 w-7 items-center justify-center rounded ${
                        risky ? "bg-red-500/8 text-red-400" : "bg-[#1a1a1a] text-[#555]"
                      }`}
                    >
                      <User className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <p className="text-[12px] font-medium text-[#ccc]">
                        {ev.user_id}
                      </p>
                      <p className="text-[10px] text-[#555] leading-tight">
                        {ev._label}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {risky && (
                      <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
                    )}
                    <ChevronRight className="h-3 w-3 text-[#333] group-hover:text-[#555] transition-colors" />
                  </div>
                </div>

                <div className="mt-2 flex gap-4 text-[9px] text-[#444]">
                  <span className="flex items-center gap-1">
                    <Clock className="h-2.5 w-2.5" />
                    {ev._timestamp?.split(" ")[1]}
                  </span>
                  <span>Vel {ev.Temporal_Velocity}</span>
                  <span>Dev {ev.rowcount_deviation}×</span>
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ──────────── MAIN ──────────── */}
      <main className="flex flex-1 flex-col overflow-y-auto">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-[#1e1e1e] px-8 py-3.5">
          <span className="text-[11px] font-semibold uppercase tracking-[0.15em] text-[#555]">
            Investigation
          </span>
          {result?.cache_hit && (
            <span className="text-[10px] text-[#555]">
              ● Cached response
            </span>
          )}
        </header>

        <div className="flex-1 px-8 py-6">
          {/* Empty */}
          {!selected && !loading && (
            <div className="flex h-full flex-col items-center justify-center">
              <Shield className="h-12 w-12 text-[#222]" />
              <p className="mt-5 text-[13px] text-[#444]">
                Select an event to begin investigation
              </p>
              <p className="mt-1 text-[11px] text-[#333]">
                Events are scored in real time with anomaly detection
              </p>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="flex h-full flex-col items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-[#444]" />
              <p className="mt-5 text-[12px] text-[#555]">
                Analyzing event…
              </p>
            </div>
          )}

          {/* Error */}
          {error && !loading && (
            <div className="flex h-full flex-col items-center justify-center">
              <XCircle className="h-8 w-8 text-red-400/60" />
              <p className="mt-4 text-[12px] text-red-400/80">
                {error}
              </p>
              <button
                onClick={() => selected && handleClick(selected)}
                className="mt-4 rounded border border-[#2a2a2a] bg-[#181818] px-4 py-1.5 text-[11px] text-[#888] hover:bg-[#1e1e1e] transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {/* ──────────── RESULT ──────────── */}
          {result && !loading && !error && (
            <div className="mx-auto max-w-3xl space-y-4">
              {/* Row 1: Score + Action */}
              <div className="grid grid-cols-5 gap-4">
                {/* Score */}
                <div className={`col-span-3 rounded-lg border ${risk.border} ${risk.bg} p-5`}>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                      Risk Score
                    </span>
                    <span className={`rounded border px-2 py-0.5 text-[10px] font-bold ${risk.border} ${risk.fg}`}>
                      {risk.label}
                    </span>
                  </div>
                  <div className="mt-3 flex items-end gap-2">
                    <span className={`text-4xl font-bold tabular-nums ${risk.fg}`}>
                      {norm}
                    </span>
                    <span className="mb-1 text-[12px] text-[#444]">/ 100</span>
                  </div>
                  <div className="mt-3 h-1 w-full rounded-full bg-[#1a1a1a]">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        norm >= 80 ? "bg-red-500" : norm >= 50 ? "bg-amber-500" : "bg-emerald-500"
                      }`}
                      style={{ width: `${norm}%`, opacity: 0.7 }}
                    />
                  </div>
                  <div className="mt-2 flex justify-between text-[9px] text-[#444]">
                    <span>Raw: {result.anomaly_score.toFixed(4)}</span>
                    <span>{result.user_id}</span>
                  </div>
                </div>

                {/* Action */}
                <div className="col-span-2 flex flex-col items-center justify-center rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                    Recommended
                  </span>
                  <span className={`mt-3 rounded-lg border px-4 py-2 text-[14px] font-bold ${actionStyle(result.llm_narrative?.recommended_action || "Monitor")}`}>
                    {result.llm_narrative?.recommended_action || "—"}
                  </span>
                </div>
              </div>

              {/* Row 2: Narrative */}
              {result.llm_narrative?.threat_narrative && (
                <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <div className="flex items-center gap-2">
                    <FileWarning className="h-3.5 w-3.5 text-[#555]" />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                      Threat Summary
                    </span>
                  </div>
                  <p className="mt-3 text-[12px] leading-[1.7] text-[#999]">
                    {result.llm_narrative.threat_narrative}
                  </p>
                </div>
              )}

              {/* Row 3: Evidence + SHAP */}
              <div className="grid grid-cols-2 gap-4">
                {/* Evidence */}
                {result.llm_narrative && result.llm_narrative.evidence_list.length > 0 && (
                  <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-3.5 w-3.5 text-[#555]" />
                      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                        Evidence
                      </span>
                    </div>
                    <div className="mt-3 space-y-2">
                      {result.llm_narrative.evidence_list.map((ev, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2.5 rounded border border-[#1e1e1e] bg-[#111] px-3 py-2.5"
                        >
                          <span className="mt-px flex h-4 w-4 flex-shrink-0 items-center justify-center rounded text-[9px] font-bold text-[#666] bg-[#1a1a1a]">
                            {i + 1}
                          </span>
                          <span className="text-[11px] leading-relaxed text-[#777]">
                            {ev}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* SHAP */}
                <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-3.5 w-3.5 text-[#555]" />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                      Feature Weights
                    </span>
                  </div>
                  <div className="mt-3 space-y-2.5">
                    {result.top_shap_features.map((f, i) => {
                      const abs = Math.abs(f.shap_value);
                      const max = Math.max(
                        ...result.top_shap_features.map((x) => Math.abs(x.shap_value))
                      );
                      const pct = max > 0 ? (abs / max) * 100 : 0;
                      return (
                        <div key={i} className="rounded border border-[#1e1e1e] bg-[#111] px-3 py-2.5">
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] text-[#999]">
                              {fmtFeature(f.feature)}
                            </span>
                            <span className="text-[10px] tabular-nums text-[#555]">
                              {f.event_value}
                            </span>
                          </div>
                          <div className="mt-1.5 h-1 w-full rounded-full bg-[#1a1a1a]">
                            <div
                              className="h-full rounded-full bg-[#555] transition-all duration-500"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="mt-1 block text-[9px] tabular-nums text-[#444]">
                            SHAP {f.shap_value.toFixed(4)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Normal event */}
              {!result.is_anomaly && (
                <div className="flex items-center gap-3 rounded-lg border border-emerald-500/15 bg-emerald-500/5 px-5 py-4">
                  <CheckCircle2 className="h-5 w-5 text-emerald-400/70" />
                  <div>
                    <p className="text-[12px] font-medium text-emerald-400/80">
                      No anomaly detected
                    </p>
                    <p className="text-[10px] text-[#555]">
                      Event is within normal parameters. No action needed.
                    </p>
                  </div>
                </div>
              )}

              {/* Human-in-the-Loop */}
              {result.is_anomaly && (
                <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                    Analyst Decision
                  </span>
                  <p className="mt-1 text-[10px] text-[#444]">
                    Take action on this event. Decision will be logged for audit.
                  </p>
                  <div className="mt-4 flex gap-3">
                    <button
                      id="btn-quarantine"
                      onClick={handleQuarantine}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md bg-red-600 py-3 text-[12px] font-semibold text-white transition-colors hover:bg-red-500 active:bg-red-700"
                    >
                      <Lock className="h-3.5 w-3.5" />
                      Quarantine User & Revoke Access
                    </button>
                    <button
                      id="btn-false-positive"
                      onClick={handleFP}
                      className="flex flex-1 items-center justify-center gap-2 rounded-md border border-[#2a2a2a] bg-[#181818] py-3 text-[12px] text-[#888] transition-colors hover:bg-[#1e1e1e]"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Mark as False Positive
                    </button>
                  </div>
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center justify-between px-1 text-[9px] text-[#333]">
                <span>{result.analyzed_at}</span>
                <span>{result.prediction === -1 ? "Anomaly" : "Normal"}</span>
                <span>{result.cache_hit ? "Cached" : "Fresh"}</span>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
