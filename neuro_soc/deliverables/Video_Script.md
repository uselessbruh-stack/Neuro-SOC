# Neuro-SOC — Video Presentation Script
## 5-Minute Demo with AI Voiceover

---

## RECORDING PLAN

### Tools You Need:
1. **Screen recorder**: OBS Studio (free) or Windows Game Bar (Win+G)
2. **AI Voiceover**: Use one of these (all free or cheap):
   - **ElevenLabs** (elevenlabs.io) — Best quality, 10 min free/month
   - **Murf AI** (murf.ai) — Professional narrator voices, free tier
   - **Natural Reader** (naturalreader.com) — Simple, free online
3. **Video editor** (optional): CapCut (free) or DaVinci Resolve (free)

### Recording Flow:
1. Record your screen walking through each section below (NO audio needed)
2. Copy each voiceover script section below into your AI voice tool
3. Generate the audio clips
4. Sync the audio to the screen recording in your editor
5. Export as MP4

---

## SCENE 1: TITLE CARD (0:00 – 0:15)
**Show**: A title slide or the README header in VS Code

**AI Voiceover Script:**
> "Neuro-SOC — an AI-powered insider threat detection system for banking security operations centres. Built for Problem Statement 04: Data Access Audit and Insider Threat Detection."

---

## SCENE 2: THE PROBLEM (0:15 – 0:50)
**Show**: Scroll through the Problem Statement file, or show a slide with key stats

**AI Voiceover Script:**
> "Enterprise banks process over one million data access events every day. Hidden within this massive volume of legitimate activity are insider threats — employees exfiltrating customer data, accessing resources beyond their clearance, or using compromised credentials at 3 AM.
>
> Traditional rule-based security systems generate an overwhelming 40 to 60 percent false positive rate, causing alert fatigue. Analysts simply stop trusting the system, and real threats go undetected for weeks or months.
>
> Neuro-SOC solves this with a three-tiered detection pipeline that achieves over 80 percent precision while keeping the human analyst in full control."

---

## SCENE 3: ARCHITECTURE OVERVIEW (0:50 – 1:30)
**Show**: Scroll through the Architecture diagram in README or the EASB document

**AI Voiceover Script:**
> "Our architecture follows what we call the Inversion of Responsibility principle. It has three distinct tiers.
>
> Tier One is the Machine Learning Engine. An Isolation Forest trained on 23 engineered features scores every access event. SHAP Tree Explainer then extracts the exact features that contributed to each anomaly — eliminating the black-box problem entirely.
>
> Tier Two is the LLM Translator. Llama-3 receives only the pre-computed SHAP values — never raw data — and translates them into plain-English threat narratives. It is explicitly barred from inferring malicious intent. Temperature is set to 0.1 for near-deterministic output.
>
> Tier Three is the Human-in-the-Loop Dashboard. A Next.js SOC console where the analyst sees the full evidence and makes the final enforcement decision. The system cannot quarantine a user without human confirmation — satisfying GDPR Article 22."

---

## SCENE 4: DATA GENERATION (1:30 – 2:10)
**Show**: Open `generate_ps4_data.py` in VS Code, scroll through the anomaly types and department configurations

**AI Voiceover Script:**
> "We built a comprehensive synthetic data generator that produces 50,000 realistic access events across 500 employees and 10 departments. The generator embeds 10 distinct anomaly types — from bulk data exports and after-hours restricted access, to pre-resignation downloads and credential compromise attempts.
>
> Each anomaly type targets specific user pools. For example, pre-resignation downloads are assigned to employees already flagged as being on their notice period. Night bulk critical events target users with high-risk HR flags.
>
> The dataset includes 50 data assets across sensitivity levels, 8 geographic locations, 6 access tiers, and realistic behavioural baselines per department. The overall anomaly rate is 44 percent, matching real enterprise threat distributions."

---

## SCENE 5: FEATURE ENGINEERING (2:10 – 2:45)
**Show**: Open `enrichment.py` in VS Code, show the FEATURE_COLUMNS list

**AI Voiceover Script:**
> "The enrichment pipeline transforms raw access logs and user profiles into 23 engineered features across four categories.
>
> User profile features capture tenure, HR risk flags, and account staleness. Computed risk modifiers weight anomalies by employee seniority and device type. Per-event features measure data volume deviation, destination risk, and geographic anomalies.
>
> Most importantly, we engineered four compound features that amplify weak individual signals. For example, Volume-Destination Compound multiplies unusual data volume by destination risk — turning two moderate signals into a strong exfiltration indicator."

---

## SCENE 6: MODEL TRAINING & RESULTS (2:45 – 3:20)
**Show**: Run `python retrain.py` in terminal, show the metrics output. Then show the EDA notebook's SHAP visualisations.

**AI Voiceover Script:**
> "Training the Isolation Forest on 50,000 events completes in under 3 seconds. The model is then exported to ONNX format for production-speed inference.
>
> On evaluation against ground truth labels, Neuro-SOC achieves 80.3 percent precision, 78.5 percent recall, and an F1 score of 0.794 — all exceeding the problem statement targets of 75 percent precision, 70 percent recall, and 0.72 F1.
>
> Compared to the naive baseline approach of flagging all night-time access — which achieves only 40 percent precision and 35 percent recall — our system delivers more than double the accuracy while reducing false positives by over 60 percent."

---

## SCENE 7: LIVE DASHBOARD DEMO (3:20 – 4:30)
**Show**: Open the frontend at localhost:3000. Click through 2-3 events showing the investigation flow.

**AI Voiceover Script:**
> "Here is the SOC analyst dashboard. On the left sidebar, we see real-time precision, recall, and F1 metrics computed against ground truth. Below that, flagged events are sorted by severity — Critical, High, and Medium.
>
> Let me click on a Critical event. The main investigation panel shows the risk score — 74 out of 100 — with a recommended action of Investigate.
>
> Below that, the Ground Truth Label panel confirms this is indeed a true positive — a Night Bulk Critical anomaly where an employee exported restricted financial data at midnight.
>
> The Threat Summary reads in plain English: a high-severity event was detected for this employee from the Marketing department. The primary indicator was Rapid Access Burst — an unusually high number of access events in a short time window.
>
> The Evidence panel lists three contributing factors, and the SHAP Feature Weights show the exact mathematical contribution of each feature.
>
> At the bottom, the analyst has two options: Quarantine User and Revoke Access, or Mark as False Positive. Every decision is logged with a full audit trail for compliance."

---

## SCENE 8: FALSE POSITIVE HANDLING (4:30 – 4:50)
**Show**: Scroll through the False_Positive_Analysis.md or show a slide

**AI Voiceover Script:**
> "We've specifically analysed three high-frequency false positive scenarios in banking: month-end bulk exports, temporary role changes, and seasonal audit activity. Through a combination of rolling baseline absorption, SHAP-based feature differentiation, and the human-in-the-loop final filter, our projected blended false positive rate is under 15 percent — compared to the 40 to 55 percent industry average."

---

## SCENE 9: CLOSING (4:50 – 5:00)
**Show**: Return to the dashboard or show a summary slide

**AI Voiceover Script:**
> "Neuro-SOC proves that insider threat detection can be accurate, explainable, and compliant. By inverting the AI responsibility model — where the ML engine detects, the LLM translates, and the human decides — we deliver enterprise-grade security without the black-box risk. Thank you."

---

## POST-PRODUCTION TIPS

1. **Record at 1080p** minimum (1920×1080)
2. **Use dark theme** in VS Code and the terminal — matches the dashboard aesthetic
3. **Close unnecessary tabs** before recording — keep it clean
4. **Add subtle background music**: Search "royalty-free ambient tech music" on YouTube Audio Library
5. **Export as MP4** with H.264 codec, 30fps
6. **Keep it under 5 minutes** — judges appreciate conciseness
