# Neuro-SOC — 5-Minute Pitch Script

**Duration:** 5 minutes (timed)  
**Presenter Notes:** Speak at a measured pace. Pause at transition markers [PAUSE]. Click demo elements at [DEMO] markers.

---

## SLIDE 1 — The Problem (60 seconds)

Good morning. I want to start with a number that should concern everyone in this room.

**4,100.** That's the average number of security alerts a banking SOC receives per day. Of those, over 40% are false positives. Analysts spend their entire shift triaging noise instead of catching threats.

[PAUSE]

But here's the real problem. The industry's response has been to throw more AI at it. Plug an LLM directly into your security pipeline and let it decide who's a threat. That sounds great until your LLM hallucinates a threat that doesn't exist and locks out your CFO during a quarterly earnings call. Or worse — it misses a real insider exfiltrating customer PII because the model's confidence threshold was tuned for a different data distribution.

The compliance teams know this. GDPR Article 22 requires meaningful human oversight of automated decisions. NIST SP 800-61 requires documented incident handling with clear decision rationale. You can't satisfy either if your threat detection is a black box.

So the question isn't whether to use AI. The question is: **how do you use AI without losing control?**

---

## SLIDE 2 — Our Solution: The Three-Tiered Architecture (90 seconds)

We built Neuro-SOC. And the core design principle is what we call **Inversion of Responsibility**.

The AI doesn't make decisions. Humans make decisions. The AI's only job is to translate math into English.

Here's how it works. Three layers.

**Layer 1: Machine Learning Baselining.**  
We run an Isolation Forest — a well-understood, auditable anomaly detection algorithm — on 11 engineered behavioral features. Things like: how fast is this user accessing systems compared to their own historical baseline? Are they accessing resources above their clearance level? Are they working at 3 AM when they've never done that before?

The model produces a numeric anomaly score and SHAP feature weights. SHAP tells us exactly which features drove the decision. No black box. Every score is fully decomposable.

[PAUSE]

**Layer 2: Constrained LLM Translation.**  
When the ML engine flags an anomaly, we send the score and SHAP weights — and nothing else — to Meta's Llama-3 via Hugging Face. But here's the critical constraint: the LLM operates at temperature 0.1, near-deterministic. Its system prompt explicitly states: "Do not infer malicious intent. State deviations objectively." It returns a structured JSON with exactly three fields: a threat narrative, an evidence list, and a recommended action.

The LLM never sees raw data. It only sees the math. It cannot hallucinate a threat that the ML engine didn't detect.

**Layer 3: Human-in-the-Loop Dashboard.**  
The SOC analyst sees everything: the risk score, the SHAP features, the plain-English narrative, and two buttons. Red: quarantine the user. Gray: mark as false positive. No automated enforcement. The human always makes the final call.

---

## SLIDE 3 — Live Demo (90 seconds)

Let me show you this working.

[DEMO: Open browser to localhost:3000]

This is our SOC dashboard. On the left, you see the live event feed. Each row is an access event from our banking dataset — real employee access logs, anonymized.

[DEMO: Click on USR00057 — the "New hire burst — device mismatch" event]

I'm clicking on this event. USR00057. Three months tenure, device mismatch detected, 25 events in one hour.

Watch the main panel. The spinner shows our ML engine running the Isolation Forest, computing SHAP values, and sending the results to Llama-3.

[PAUSE — wait for result to load]

There it is. Risk score: 82 out of 100. Critical.

The threat narrative reads — and I want you to notice the language — it says "unusual temporal velocity" and "device mismatch detected." It does NOT say "this user is stealing data." It states deviations objectively. That's the constraint working.

The evidence log shows the top three SHAP features driving this score. Temporal Velocity, Failed Action Flag, Rowcount Deviation. Each with the exact numeric value and SHAP weight.

Recommended action: Investigate.

[DEMO: Click on USR00074 — the normal event]

Now watch what happens with a normal event. USR00074. 48 months tenure, standard access, business hours.

Risk score: 28. Normal. Green. No narrative generated — we don't waste LLM compute on events the ML engine clears.

[DEMO: Go back to the anomalous event, hover over the Quarantine button]

And here are the human-in-the-loop controls. If I were a real SOC analyst, I'd review the evidence and press one of these two buttons. Every decision is logged for audit compliance.

---

## SLIDE 4 — Enterprise Alignment & Compliance (60 seconds)

Let me address the compliance angle directly.

**NIST IR-4 — Incident Response.** Our three-tier architecture maps directly to NIST's detection-analysis-containment-eradication lifecycle. Layer 1 is detection. Layer 2 is analysis. Layer 3 is containment. Each stage has a documented decision rationale — the SHAP values, the LLM narrative, and the analyst's action.

**GDPR Article 32 — Security of Processing.** We implement technical measures proportionate to risk. The ML engine provides the risk quantification. The LLM provides the human-readable explanation. The dashboard provides the proportionate response mechanism.

**GDPR Article 22 — Automated Decision-Making.** This is where most AI security tools fail. They automate enforcement. We don't. The AI informs. The human decides. There is always meaningful human oversight.

**The Semantic Cache.** Powered by Redis Cloud. When the same behavioral pattern is seen twice, we serve the cached result instantly — no redundant LLM calls, no redundant compute cost. This keeps our operational cost predictable and our response time under 200 milliseconds for repeat patterns.

---

## SLIDE 5 — Close (30 seconds)

To summarize.

Neuro-SOC doesn't replace the security analyst. It makes them faster, more accurate, and fully compliant.

The math decides if something is anomalous. The AI translates the math into English. The human decides what to do about it.

We call this the Inversion of Responsibility. And it's how you build AI security tools that enterprises will actually trust.

Thank you. Happy to take questions.

---

## Appendix — Key Metrics for Q&A

| Metric | Value |
|--------|-------|
| Training data | 1,200 enriched events, 11 features |
| Anomaly detection rate | 5% (60/1,200 events) |
| False positive target | < 20% (projected < 15%) |
| LLM temperature | 0.1 (near-deterministic) |
| Cache backend | Redis Cloud, 1-hour TTL |
| LLM model | Meta-Llama-3-8B-Instruct |
| Response time (cache miss) | ~3-5 seconds |
| Response time (cache hit) | < 200 milliseconds |
| Frontend | Next.js 16, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.12 |
| ML Engine | Scikit-learn IsolationForest + SHAP |
