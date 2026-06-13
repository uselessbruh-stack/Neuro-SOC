An excellent pivot to Node.js. While Streamlit is great for rapid prototyping, a Node.js (specifically Next.js or React) frontend is the true enterprise standard for building a scalable, secure, and highly customized Security Operations Center (SOC) dashboard.

A traditional Software Requirements Specification (SRS) is too theoretical for a hackathon. In the banking sector, when we bridge complex Machine Learning pipelines with software execution, we use an **Enterprise Architecture & System Blueprint**. This document format perfectly aligns with what the judges expect: a clear mapping of business problems to technical implementations, pipeline optimizations, and success metrics.

Here is the comprehensive blueprint for your Neuro-SOC application.

---

# Enterprise Architecture & System Blueprint: Neuro-SOC (PS4)

## 1. Executive Summary

This document outlines the architecture for an advanced User and Entity Behavior Analytics (UEBA) platform designed to detect insider threats and unauthorized data exfiltration. The system utilizes a "Human-in-the-Loop" workflow, combining deep learning anomaly detection with deterministically constrained Small Language Models (SLMs) to eliminate "black-box" AI risks while maintaining microsecond inference latency.

## 2. Technology Stack & Infrastructure

To maximize efficiency and maintain Pythonic harmony for ML while delivering a polished enterprise UI, the architecture utilizes a hybrid stack:

| Architecture Layer | Technology | Enterprise Justification |
| --- | --- | --- |
| **Data Ingestion** | Apache Kafka / Asyncio | Handles the massive 1M+ daily enterprise event requirement smoothly.

 |
| **State Management** | Redis | Crucial for holding rolling time-windows for sequential tracking.

 |
| **Backend API** | FastAPI (Python) | Native async support prevents blocking during ML and LLM operations.

 |
| **ML Engine** | PyTorch + SHAP | Handles the complex tensor math; SHAP extracts explainable feature weights.

 |
| **Agentic LLM** | Llama-3-8B (Hugging Face) | Instruction-tuned SLM capable of strict JSON adherence.

 |
| **Frontend Dashboard** | **Node.js (Next.js/React)** | Provides a highly responsive, enterprise-grade, custom UI for SOC analysts.

 |
| **Audit Database** | PostgreSQL | Immutable, ACID-compliant storage for user profiles and SOC decisions.

 |

## 3. Data Engineering & Feature Set

Do not utilize synthetic data generators; rely strictly on the provided `data_access_logs.csv` and `user_profiles.csv`. The raw data will be enriched into mathematical tensors using the following custom features:

* 
**`Tenure_Risk_Modifier`:** Combines `tenure_months` and `high_risk_flag` to weight anomalous actions heavily for new or departing employees.


* 
**`Equipment_Mismatch_Score`:** A boolean trigger for when restricted data is accessed via a contractor machine.


* 
**`Privilege_Creep_Index`:** Flags access to assets outside the user's `approved_data_assets` array.


* 
**`Temporal_Velocity`:** Tracks rolling queries to catch "low and slow" sequence-based exfiltration.



## 4. The Three-Tiered Detection Pipeline

### Tier 1: The ML Baselining Engine (UEBA)

The pipeline branches into specialized detectors to evaluate both point-in-time and sequential anomalies:

* 
**Volumetric/Categorical Detection:** Handled by Deep Autoencoding Gaussian Mixture Models (DAGMM) to understand the complex density of normal behaviors.


* 
**Sequential Detection:** Handled by LSTM-Autoencoders processing a rolling window of recent actions to catch altered frequencies of benign actions.



### Tier 2: The Agentic LLM (Explainability)

To prevent the massive compliance risk of LLM hallucinations , the architecture enforces an **Inversion of Responsibility**.

* **Algorithmic Grounding:** The ML engine does the thinking; SHapley Additive exPlanations (SHAP) extracts the exact mathematical deviations.


* 
**Strict Prompt Structuring:** The Llama-3 model acts solely as a deterministic translator, barred from inferring malicious intent.


* 
**Constrained Output:** The LLM is forced to output a strict JSON schema (`threat_narrative`, `evidence_list`, `recommended_action`) at a temperature of 0.0 to ensure 100% predictable parsing by the Node.js frontend.



### Tier 3: Human-in-the-Loop Node.js Dashboard

The Next.js frontend acts as the SOC analyst's command center. It digests the JSON payload from FastAPI and renders actionable threat cards. Crucially, the system cannot autonomously block users; the analyst must review the LLM's explanation and click "Quarantine" or "False Positive," fulfilling strict auditability requirements.

---

## 5. System Bottlenecks & Engineered Remedies

To prove the system scales to millions of events without fatal latency, the following architectural optimizations are mandatory:

| Pipeline Stage | Identified Bottleneck | Architectural Remedy |
| --- | --- | --- |
| **Deep Learning Inference** | Passing high-dimensional data through DAGMM/LSTM in raw PyTorch carries excessive Python overhead.

 | Convert `.pt` models to **ONNX format** to strip Python overhead. Implement dynamic micro-batching to process multiple logs simultaneously.

 |
| **LLM API Latency** | GenAI calls take 1–5 seconds, causing a fatal throughput flaw for high-volume SOC ingestion.

 | Deploy **Semantic Caching** to hash core feature deviations and instantly return previously generated narratives from Redis.

 |

---

## 6. Output Expectations & Success Criteria

The final Node.js application and ML pipeline must strictly meet the quantitative and qualitative metrics defined in the problem statement parameters:

**Quantitative Targets:**

* **Precision:** > 75% (Minimize alert fatigue for analysts)
* **Recall:** > 70% (Ensure no actual threats are missed)
* **F1 Score:** > 0.72 (Overall detection accuracy)
* **Detection Speed:** < 5 minutes from log ingestion to alert generation.

**Deliverables Required for Demo:**

1. **Risk Dashboard (Node.js):** A highly polished UI showing top alerts, user profiles, and data assets.
2. **Investigation Toolkit:** Contextual XAI information (SHAP features + LLM narrative) for every flagged alert.
3. **Anomaly Detection Output:** 20+ clearly flagged accesses with detailed JSON explanations.
4. **Evaluation Metrics:** Empirical proof of the Precision/Recall scores against the ground truth labels.

## 7. Regulatory Alignment

* 
**NIST IR-4 (Incident Response):** LLM integration drastically reduces the Mean Time to Understand (MTTU).


* 
**GDPR Article 32 & 22:** The system monitors for unauthorized exfiltration while legally bypassing automated decision-making restrictions via the mandatory Node.js Human-in-the-Loop interface.