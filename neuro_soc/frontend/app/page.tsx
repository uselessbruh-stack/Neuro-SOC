"use client";

import React, { useState, useEffect } from "react";
import {
  Shield,
  Activity,
  Clock,
  User,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  Lock,
  Server,
  Radio,
  FileWarning,
  BarChart3,
  ChevronRight,
  Database,
  Eye,
  Target,
  RefreshCw,
  TrendingUp,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════════════════
//  TYPE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

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

interface GroundTruth {
  is_anomaly: boolean;
  anomaly_type: string;
  severity: string;
  explanation: string;
}

interface FlaggedEvent {
  access_id: string;
  user_id: string;
  username: string;
  department: string;
  timestamp: string;
  data_asset: string;
  data_sensitivity: string;
  query_type: string;
  destination: string;
  anomaly_score: number;
  risk_score: number;
  is_anomaly_predicted: boolean;
  prediction: number;
  top_shap_features: ShapFeature[];
  llm_narrative: LlmNarrative | null;
  ground_truth: GroundTruth;
}

interface EvaluationMetrics {
  total_events: number;
  total_predicted_anomalies: number;
  total_ground_truth_anomalies: number;
  precision: number;
  recall: number;
  f1_score: number;
}

interface FlaggedEventsResponse {
  metrics: EvaluationMetrics;
  flagged_events: FlaggedEvent[];
}

// ═══════════════════════════════════════════════════════════════════════════
//  API
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE = "http://localhost:8000";

async function fetchFlaggedEvents(
  limit = 50,
  refresh = false
): Promise<FlaggedEventsResponse> {
  const res = await fetch(
    `${API_BASE}/flagged_events?limit=${limit}&refresh=${refresh}`
  );
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// ═══════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

function getRisk(score: number) {
  if (score >= 80)
    return {
      label: "CRITICAL",
      fg: "text-red-400",
      bg: "bg-red-500/8",
      border: "border-red-500/20",
    };
  if (score >= 60)
    return {
      label: "HIGH",
      fg: "text-amber-400",
      bg: "bg-amber-500/8",
      border: "border-amber-500/20",
    };
  if (score >= 40)
    return {
      label: "ELEVATED",
      fg: "text-yellow-400",
      bg: "bg-yellow-500/8",
      border: "border-yellow-500/20",
    };
  return {
    label: "NORMAL",
    fg: "text-emerald-400",
    bg: "bg-emerald-500/8",
    border: "border-emerald-500/20",
  };
}

function actionStyle(a: string) {
  if (a === "Quarantine")
    return "text-red-400 border-red-500/25 bg-red-500/8";
  if (a === "Investigate")
    return "text-amber-400 border-amber-500/25 bg-amber-500/8";
  return "text-emerald-400 border-emerald-500/25 bg-emerald-500/8";
}

function fmtFeature(s: string) {
  return s.replace(/_/g, " ").replace(/([A-Z])/g, " $1").trim();
}

function severityDot(sev: string) {
  switch (sev) {
    case "CRITICAL":
      return "bg-red-400";
    case "HIGH":
      return "bg-amber-400";
    case "MEDIUM":
      return "bg-yellow-400";
    default:
      return "bg-emerald-400";
  }
}

function severityBadge(sev: string) {
  switch (sev) {
    case "CRITICAL":
      return "text-red-400 bg-red-500/10 border-red-500/20";
    case "HIGH":
      return "text-amber-400 bg-amber-500/10 border-amber-500/20";
    case "MEDIUM":
      return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
    default:
      return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  }
}

function fmtAnomalyType(t: string) {
  return t
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function fmtTimestamp(ts: string) {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return `${d.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
  })} ${d.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

// ═══════════════════════════════════════════════════════════════════════════
//  PAGE
// ═══════════════════════════════════════════════════════════════════════════

export default function Dashboard() {
  const [events, setEvents] = useState<FlaggedEvent[]>([]);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [selected, setSelected] = useState<FlaggedEvent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("ALL");
  const [refreshing, setRefreshing] = useState(false);

  // Fetch flagged events on mount
  useEffect(() => {
    loadEvents(false);
  }, []);

  async function loadEvents(refresh: boolean) {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const data = await fetchFlaggedEvents(50, refresh);
      setMetrics(data.metrics);
      setEvents(data.flagged_events);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load events");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  const filteredEvents =
    filter === "ALL"
      ? events
      : events.filter((e) => e.ground_truth.severity === filter);

  const handleQuarantine = () => {
    if (!selected) return;
    console.log("🔒 QUARANTINE:", selected.user_id, selected);
    alert(`${selected.user_id} quarantined. Access revoked.`);
    setSelected(null);
  };

  const handleFP = () => {
    if (!selected) return;
    console.log("✅ FALSE POSITIVE:", selected.user_id, selected);
    alert(`${selected.user_id} marked as false positive.`);
    setSelected(null);
  };

  const risk = selected ? getRisk(selected.risk_score) : null;

  // Count by severity
  const counts = {
    ALL: events.length,
    CRITICAL: events.filter((e) => e.ground_truth.severity === "CRITICAL")
      .length,
    HIGH: events.filter((e) => e.ground_truth.severity === "HIGH").length,
    MEDIUM: events.filter((e) => e.ground_truth.severity === "MEDIUM").length,
  };

  return (
    <div className="flex h-screen bg-[#101010] text-[#e0e0e0]">
      {/* ──────────── SIDEBAR ──────────── */}
      <aside className="flex w-[380px] flex-shrink-0 flex-col border-r border-[#1e1e1e]">
        {/* Brand */}
        <div className="border-b border-[#1e1e1e] px-5 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="h-5 w-5 text-[#808080]" />
              <span className="text-[13px] font-semibold tracking-wider text-[#c0c0c0]">
                NEURO-SOC
              </span>
            </div>
            <button
              id="btn-refresh"
              onClick={() => loadEvents(true)}
              disabled={refreshing}
              className="rounded p-1.5 text-[#555] transition-colors hover:bg-[#1a1a1a] hover:text-[#888] disabled:opacity-50"
              title="Refresh events"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`}
              />
            </button>
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

        {/* Metrics */}
        {metrics && (
          <div className="border-b border-[#1e1e1e] px-4 py-3">
            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-md border border-[#1e1e1e] bg-[#141414] px-3 py-2 text-center">
                <p className="text-[9px] font-semibold uppercase tracking-wider text-[#555]">
                  Precision
                </p>
                <p className="mt-0.5 text-[16px] font-bold tabular-nums text-emerald-400">
                  {(metrics.precision * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md border border-[#1e1e1e] bg-[#141414] px-3 py-2 text-center">
                <p className="text-[9px] font-semibold uppercase tracking-wider text-[#555]">
                  Recall
                </p>
                <p className="mt-0.5 text-[16px] font-bold tabular-nums text-blue-400">
                  {(metrics.recall * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md border border-[#1e1e1e] bg-[#141414] px-3 py-2 text-center">
                <p className="text-[9px] font-semibold uppercase tracking-wider text-[#555]">
                  F1
                </p>
                <p className="mt-0.5 text-[16px] font-bold tabular-nums text-purple-400">
                  {metrics.f1_score.toFixed(3)}
                </p>
              </div>
            </div>
            <div className="mt-2 flex justify-between text-[9px] text-[#444]">
              <span>{metrics.total_events.toLocaleString()} events</span>
              <span>
                {metrics.total_predicted_anomalies.toLocaleString()} predicted
              </span>
              <span>
                {metrics.total_ground_truth_anomalies.toLocaleString()} actual
              </span>
            </div>
          </div>
        )}

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 border-b border-[#1e1e1e] px-4 py-2">
          {(["ALL", "CRITICAL", "HIGH", "MEDIUM"] as const).map((f) => (
            <button
              key={f}
              id={`filter-${f.toLowerCase()}`}
              onClick={() => setFilter(f)}
              className={`rounded px-2.5 py-1 text-[9px] font-semibold uppercase tracking-wider transition-colors ${
                filter === f
                  ? "bg-[#1e1e1e] text-[#ccc]"
                  : "text-[#555] hover:text-[#888]"
              }`}
            >
              {f}{" "}
              <span className="ml-0.5 text-[#444]">
                {counts[f as keyof typeof counts]}
              </span>
            </button>
          ))}
        </div>

        {/* Event List */}
        <div className="flex-1 overflow-y-auto px-3 pb-3 pt-1">
          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-20">
              <Loader2 className="h-6 w-6 animate-spin text-[#444]" />
              <p className="mt-4 text-[11px] text-[#555]">
                Loading flagged events...
              </p>
              <p className="mt-1 text-[9px] text-[#333]">
                First load may take 30-60 seconds (SHAP computation)
              </p>
            </div>
          )}

          {/* Error */}
          {error && !loading && (
            <div className="flex flex-col items-center justify-center py-20">
              <XCircle className="h-6 w-6 text-red-400/60" />
              <p className="mt-3 text-[11px] text-red-400/80">{error}</p>
              <p className="mt-1 text-[9px] text-[#444]">
                Make sure the backend is running on port 8000
              </p>
              <button
                onClick={() => loadEvents(false)}
                className="mt-4 rounded border border-[#2a2a2a] bg-[#181818] px-4 py-1.5 text-[10px] text-[#888] transition-colors hover:bg-[#1e1e1e]"
              >
                Retry
              </button>
            </div>
          )}

          {/* Events */}
          {!loading &&
            !error &&
            filteredEvents.map((ev) => {
              const active = selected?.access_id === ev.access_id;
              return (
                <button
                  key={ev.access_id}
                  id={`event-${ev.access_id}`}
                  onClick={() => setSelected(ev)}
                  className={`group mb-1 w-full rounded-md border px-3.5 py-2.5 text-left transition-colors ${
                    active
                      ? "border-[#333] bg-[#1a1a1a]"
                      : "border-transparent hover:bg-[#161616]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="relative flex h-7 w-7 items-center justify-center rounded bg-[#1a1a1a]">
                        <User className="h-3.5 w-3.5 text-[#555]" />
                        <span
                          className={`absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full ${severityDot(ev.ground_truth.severity)}`}
                        />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-[11px] font-medium text-[#ccc]">
                            {ev.user_id}
                          </p>
                          <span className="truncate text-[9px] text-[#444]">
                            {ev.username}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[9px] text-[#555] leading-tight truncate">
                          {fmtAnomalyType(ev.ground_truth.anomaly_type)}
                        </p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <span
                        className={`rounded border px-1.5 py-0.5 text-[8px] font-bold ${severityBadge(ev.ground_truth.severity)}`}
                      >
                        {ev.risk_score}
                      </span>
                      <ChevronRight className="h-3 w-3 text-[#333] transition-colors group-hover:text-[#555]" />
                    </div>
                  </div>
                  <div className="mt-1.5 flex gap-3 text-[8px] text-[#444]">
                    <span className="flex items-center gap-1">
                      <Clock className="h-2.5 w-2.5" />
                      {fmtTimestamp(ev.timestamp)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Database className="h-2.5 w-2.5" />
                      {ev.data_asset}
                    </span>
                  </div>
                </button>
              );
            })}

          {/* No results */}
          {!loading && !error && filteredEvents.length === 0 && (
            <div className="flex flex-col items-center py-16">
              <CheckCircle2 className="h-6 w-6 text-[#333]" />
              <p className="mt-3 text-[11px] text-[#444]">
                No events matching filter
              </p>
            </div>
          )}
        </div>
      </aside>

      {/* ──────────── MAIN ──────────── */}
      <main className="flex flex-1 flex-col overflow-y-auto">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-[#1e1e1e] px-8 py-3.5">
          <span className="text-[11px] font-semibold uppercase tracking-[0.15em] text-[#555]">
            Investigation
          </span>
          {selected && (
            <span className="text-[10px] text-[#444]">
              {selected.access_id}
            </span>
          )}
        </header>

        <div className="flex-1 px-8 py-6">
          {/* Empty state */}
          {!selected && (
            <div className="flex h-full flex-col items-center justify-center">
              <Shield className="h-12 w-12 text-[#222]" />
              <p className="mt-5 text-[13px] text-[#444]">
                Select an event to begin investigation
              </p>
              <p className="mt-1 text-[11px] text-[#333]">
                {events.length} flagged events loaded from ML pipeline
              </p>
            </div>
          )}

          {/* ──────────── SELECTED EVENT DETAIL ──────────── */}
          {selected && risk && (
            <div className="mx-auto max-w-3xl space-y-4">
              {/* Row 1: Score + Action */}
              <div className="grid grid-cols-5 gap-4">
                {/* Score */}
                <div
                  className={`col-span-3 rounded-lg border ${risk.border} ${risk.bg} p-5`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                      Risk Score
                    </span>
                    <span
                      className={`rounded border px-2 py-0.5 text-[10px] font-bold ${risk.border} ${risk.fg}`}
                    >
                      {risk.label}
                    </span>
                  </div>
                  <div className="mt-3 flex items-end gap-2">
                    <span
                      className={`text-4xl font-bold tabular-nums ${risk.fg}`}
                    >
                      {selected.risk_score}
                    </span>
                    <span className="mb-1 text-[12px] text-[#444]">/ 100</span>
                  </div>
                  <div className="mt-3 h-1 w-full rounded-full bg-[#1a1a1a]">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        selected.risk_score >= 80
                          ? "bg-red-500"
                          : selected.risk_score >= 60
                            ? "bg-amber-500"
                            : selected.risk_score >= 40
                              ? "bg-yellow-500"
                              : "bg-emerald-500"
                      }`}
                      style={{
                        width: `${selected.risk_score}%`,
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <div className="mt-2 flex justify-between text-[9px] text-[#444]">
                    <span>Raw: {selected.anomaly_score.toFixed(4)}</span>
                    <span>
                      {selected.user_id} · {selected.username}
                    </span>
                  </div>
                </div>

                {/* Action */}
                <div className="col-span-2 flex flex-col items-center justify-center rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                    Recommended
                  </span>
                  <span
                    className={`mt-3 rounded-lg border px-4 py-2 text-[14px] font-bold ${actionStyle(selected.llm_narrative?.recommended_action || "Monitor")}`}
                  >
                    {selected.llm_narrative?.recommended_action || "—"}
                  </span>
                  <span className="mt-2 text-[9px] text-[#444]">
                    {selected.department} · {selected.data_sensitivity}
                  </span>
                </div>
              </div>

              {/* Row 2: Ground Truth Comparison */}
              <div className="rounded-lg border border-purple-500/20 bg-purple-500/5 p-4">
                <div className="flex items-center gap-2">
                  <Target className="h-3.5 w-3.5 text-purple-400/70" />
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-purple-400/70">
                    Ground Truth Label
                  </span>
                  <span
                    className={`ml-auto rounded border px-2 py-0.5 text-[9px] font-bold ${severityBadge(selected.ground_truth.severity)}`}
                  >
                    {selected.ground_truth.severity}
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <span
                    className={`flex items-center gap-1.5 text-[11px] font-medium ${
                      selected.ground_truth.is_anomaly
                        ? "text-red-400/80"
                        : "text-emerald-400/80"
                    }`}
                  >
                    {selected.ground_truth.is_anomaly ? (
                      <ShieldAlert className="h-3.5 w-3.5" />
                    ) : (
                      <ShieldCheck className="h-3.5 w-3.5" />
                    )}
                    {selected.ground_truth.is_anomaly
                      ? "Confirmed Anomaly"
                      : "Normal Event"}
                  </span>
                  <span className="rounded bg-[#1a1a1a] px-2 py-0.5 text-[9px] font-medium text-[#888]">
                    {fmtAnomalyType(selected.ground_truth.anomaly_type)}
                  </span>
                </div>
                <p className="mt-2 text-[10px] leading-relaxed text-[#777]">
                  {selected.ground_truth.explanation}
                </p>
              </div>

              {/* Row 3: Narrative */}
              {selected.llm_narrative?.threat_narrative && (
                <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                  <div className="flex items-center gap-2">
                    <FileWarning className="h-3.5 w-3.5 text-[#555]" />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                      Threat Summary
                    </span>
                  </div>
                  <p className="mt-3 text-[12px] leading-[1.7] text-[#999]">
                    {selected.llm_narrative.threat_narrative}
                  </p>
                </div>
              )}

              {/* Row 4: Evidence + SHAP */}
              <div className="grid grid-cols-2 gap-4">
                {/* Evidence */}
                {selected.llm_narrative &&
                  selected.llm_narrative.evidence_list.length > 0 && (
                    <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-3.5 w-3.5 text-[#555]" />
                        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                          Evidence
                        </span>
                      </div>
                      <div className="mt-3 space-y-2">
                        {selected.llm_narrative.evidence_list.map((ev, i) => (
                          <div
                            key={i}
                            className="flex items-start gap-2.5 rounded border border-[#1e1e1e] bg-[#111] px-3 py-2.5"
                          >
                            <span className="mt-px flex h-4 w-4 flex-shrink-0 items-center justify-center rounded bg-[#1a1a1a] text-[9px] font-bold text-[#666]">
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
                      Feature Weights (SHAP)
                    </span>
                  </div>
                  <div className="mt-3 space-y-2.5">
                    {selected.top_shap_features.map((f, i) => {
                      const abs = Math.abs(f.shap_value);
                      const max = Math.max(
                        ...selected.top_shap_features.map((x) =>
                          Math.abs(x.shap_value)
                        )
                      );
                      const pct = max > 0 ? (abs / max) * 100 : 0;
                      return (
                        <div
                          key={i}
                          className="rounded border border-[#1e1e1e] bg-[#111] px-3 py-2.5"
                        >
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

              {/* Row 5: Event Context */}
              <div className="rounded-lg border border-[#1e1e1e] bg-[#141414] p-5">
                <div className="flex items-center gap-2">
                  <Eye className="h-3.5 w-3.5 text-[#555]" />
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#555]">
                    Event Context
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3">
                  {[
                    ["Data Asset", selected.data_asset],
                    ["Sensitivity", selected.data_sensitivity],
                    ["Department", selected.department],
                    ["Query Type", selected.query_type],
                    ["Destination", selected.destination],
                    ["Timestamp", fmtTimestamp(selected.timestamp)],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded border border-[#1e1e1e] bg-[#111] px-3 py-2"
                    >
                      <p className="text-[8px] font-semibold uppercase tracking-wider text-[#444]">
                        {label}
                      </p>
                      <p className="mt-0.5 text-[11px] text-[#999]">{value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Row 6: Human-in-the-Loop */}
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

              {/* Footer */}
              <div className="flex items-center justify-between px-1 text-[9px] text-[#333]">
                <span>{selected.timestamp}</span>
                <span>Anomaly Score: {selected.anomaly_score.toFixed(4)}</span>
                <span>
                  {selected.ground_truth.is_anomaly
                    ? "✓ True Positive"
                    : "✗ False Positive"}
                </span>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
