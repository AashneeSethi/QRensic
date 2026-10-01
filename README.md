# QRensic — An Agentic Vision System for Physical QR Fraud Investigation

> **An AI investigator that knows when to ask for a second look.**

[![OpenCV AI Competition 2026](https://img.shields.io/badge/OpenCV%20AI%20Competition-2026%20Target-blue.svg)](https://opencv.org/)
[![Track](https://img.shields.io/badge/Category-Agentic%20Vision-purple.svg)]()
[![OpenCV Version](https://img.shields.io/badge/OpenCV-5.0.0-green.svg)](https://pypi.org/project/opencv-python/)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()

---

## 🎯 Overview

**QRensic** is an agentic computer vision system designed for the **OpenCV AI Competition 2026**, specifically targeting the **Agentic Vision** award. 

Unlike traditional passive computer vision pipelines that make one-shot judgments from a single image, QRensic behaves like a forensic investigator. When inspecting a physical QR code poster, it actively evaluates the sufficiency of its visual evidence and can request targeted second observations (such as an angled perspective or a flash photograph) to verify physical surface continuity.

### The Hero Scenario: Physical QR Tampering
The primary real-world threat addressed by QRensic is **physical QR overlay tampering**:
* A legitimate merchant displays a printed payment or informational poster.
* A scammer prints a replacement QR sticker and physically overlays it on top of the merchant's original QR code.
* The sticker often exhibits subtle physical tells: specular mismatch under directional lighting, misaligned boundaries, edge thickness shadows, or material texture discontinuities.

---

## 🧠 Core Architecture Principles

QRensic enforces a strict separation of concerns to avoid hallucinatory verdicts or black-box errors:

```
┌─────────────────────────────────────────────────────────────┐
│                       QRensic Pipeline                      │
│                                                             │
│   Observe                                                   │
│      │                                                      │
│      ▼                                                      │
│   Decide Next Observation                                   │
│      │                                                      │
│      ▼                                                      │
│   Collect Evidence                                          │
│      │                                                      │
│      ▼                                                      │
│   Re-evaluate                                               │
│      │                                                      │
│      ▼                                                      │
│   Stop or Request Human Review                              │
└─────────────────────────────────────────────────────────────┘
```

Three pillars govern the architecture:
1. **OpenCV 5 answers:** *"What can I observe?"*  
   Computer vision provides deterministic measurements: edge gradients, specular reflection ratios, texture variances, keypoint homography, and QR geometry.
2. **A bounded LLM agent answers:** *"What should I investigate next?"*  
   The agent acts as an orchestrator choosing strategic follow-up actions dynamically rather than adhering to a rigid static pipeline.
3. **Deterministic rules answer:** *"What does the evidence justify?"*  
   Evidence metrics are evaluated against deterministic thresholds and rules. **The LLM must NOT directly decide whether something is fraud.**

---

## 🚫 What QRensic Is NOT

To maintain forensic rigor and avoid scope creep, QRensic is explicitly NOT:
* **NOT a generic chatbot:** It does not hold open-ended conversations.
* **NOT a universal scam detector:** It focuses specifically on visual and physical evidence of tampering.
* **NOT a live surveillance system:** It is designed for interactive, point-of-interest investigation sessions.
* **NOT a payment platform:** It does not process transactions, handle wallets, or transmit funds.
* **NOT a real phishing detector:** It does not crawl malicious external websites or execute browser sandboxing.
* **NOT a generic sticker classifier:** It specifically investigates physical QR overlay discrepancies against host surfaces.

---

## 🛠️ The 8 Forensic Agent Actions

The QRensic agent orchestrates investigation using **EXACTLY 8 bounded actions**, selecting its next step dynamically based on gathered visual evidence:

| # | Action | Purpose & Scope |
|---|---|---|
| 1 | `inspect_scene` | Initial scene capture; checks image quality, blur, framing, and global lighting. |
| 2 | `analyze_qr` | QR detection, QR decoding, payload parsing, QR type classification, and URL/EMV-style structural checks where applicable. |
| 3 | `extract_region` | Crops and isolates the QR bounding polygon and the surrounding host poster ring/margin. |
| 4 | `validate_identity` | Compares the decoded payload context with the visible merchant branding / text in the surrounding scene. |
| 5 | `analyze_qr_surface` | Measures surface physical signals: edge discontinuity, boundary gradients, local texture variance, and color transition deltas. |
| 6 | `request_observation` | Directs the capture device or user to provide a targeted follow-up view. Supports: `clearer` (re-capture for blur/occlusion) and `flash` (directional light for specular inspection). |
| 7 | `analyze_second_view` | Multi-view analysis: ORB feature detection, descriptor matching, RANSAC homography, scene registration, and differential surface comparison (e.g. relative specular response). |
| 8 | `human_review` | Flags ambiguous or high-uncertainty cases to a human specialist with the compiled forensic dossier. |

---

## 🔒 Safety, Ethics & Security

* **Controlled Synthetic Payloads Only:** All experiments and prop generators use strictly harmless, synthetic test strings (e.g. `QRFORENSIC_TEST_PAYEE_NOVA_COFFEE` and `QRFORENSIC_TEST_PAYEE_RAMESH_KUMAR`). No real UPI IDs, payment destinations, bank details, or live URLs are used.
* **Prompt Injection Defense:** All OCR, visual text, and decoded QR payloads are treated strictly as **untrusted data**. Decoded strings are never interpolated directly into system prompts or instructions.
* **Evidence-Based Decisions:** The system reports empirical visual signals (RSR ratios, gradient strength, homography inliers) rather than speculative "fraud probability" scores.

---

## 📁 Repository Structure

```text
QRensic/
├── README.md               # Main project documentation & design principles
├── .gitignore              # Repository exclusion rules
├── requirements.txt        # Core dependencies (OpenCV 5, qrcode, numpy)
├── feasibility/            # Physical feasibility experiment
│   ├── README.md           # Experiment design, 4 conditions, & 16-photo protocol
│   ├── generate_props.py   # Synthetic test QR generator
│   └── feasibility.py      # Feature extraction & measurement script
├── backend/                # Application backend (future milestone)
├── frontend/               # Interactive UI (future milestone)
├── docs/                   # Extended design & architecture specifications
└── tests/                  # Automated verification tests
```

---

## 🚀 Quickstart: Feasibility Experiment

### 1. Prerequisites
* Python 3.11+
* OpenCV 5 (`opencv-python>=5,<6`)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/AashneeSethi/QRensic.git
cd QRensic

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Harmless Test Props
```bash
python feasibility/generate_props.py
```

### 4. Run Feasibility Measurements
```bash
python feasibility/feasibility.py --input-dir feasibility/photos --output-csv feasibility/results/measurements.csv
```

See [feasibility/README.md](feasibility/README.md) for full details on the physical experiment protocol, conditions, and relative specular response calculations.

---

## 🔬 QRensic Vision Engine Foundation

The vision engine provides a deterministic, decoupled computer vision pipeline that transforms raw input images into structured forensic evidence models for downstream bounded agent reasoning:

```text
              QRensic Vision Engine
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Scene scan       QR analysis    Text/identity
 (inspect_scene)   (analyze_qr)   (extract_region &
        │              │          validate_identity)
        └──────────────┼──────────────┘
                       ↓
                Surface evidence
              (analyze_qr_surface)
                       ↓
              Structured Evidence
              (StructuredEvidence)
```

### Module Breakdown (`backend/vision/`):
* `models.py`: Structured dataclasses (`SceneEvidence`, `QREvidence`, `ExtractedRegionEvidence`, `IdentityEvidence`, `SurfaceEvidence`, `StructuredEvidence`) and state enums (`EvidenceState`, `IdentityConsistency`, `PayloadType`).
* `inspect_scene.py`: Scene dimensions, Laplacian blur score, illumination stats, and QR/text candidate regions using OpenCV 5.
* `analyze_qr.py`: OpenCV 5 QR detection, decoding, quad polygon bounding points, and URL/EMV/plain payload classification.
* `extract_region.py`: Surrounding text candidate detection interface. All visual text is tagged with `is_untrusted = True` as a prompt injection defense.
* `validate_identity.py`: Deterministic entity normalization (stripping legal suffixes and test prefixes). Personal payees are held as `AMBIGUOUS` rather than treated as automatic fraud.
* `analyze_surface.py`: Boundary gradient strength (Sobel), edge discontinuity ratios, local texture energy, and color transition deltas across the QR perimeter and host poster ring.
* `engine.py`: Orchestrates the sequential pipeline into a serializable `StructuredEvidence` dossier.
* `cli.py`: Command-line interface to inspect images and export structured JSON evidence.

### Running the Vision Engine CLI
```bash
# Analyze a test QR code with simulated poster brand text
python -m backend.vision.cli feasibility/props/generated/genuine_qr.png --poster-text "NOVA COFFEE Pvt Ltd"

# Or run directly
python backend/vision/cli.py feasibility/props/generated/sticker_qr.png --poster-text "NOVA COFFEE"
```

### Running Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🔄 Second-View Registration

A hallmark of **Agentic Vision** is active investigation: rather than making high-stakes decisions from a single compromised snapshot, an agentic system can request a second observation (e.g. an angled view or flash lighting) and verify visual continuity.

```text
Photo 1 (Initial View)
      │
      ▼
Ambiguous / Insufficient Evidence
      │
      ▼
Request Second Observation
      │
      ▼
Photo 2 (Follow-up View)
      │
      ▼
ORB Feature Matching & RANSAC Homography
      │
      ▼
Same-Scene Verification (Geometric Sanity Checks)
      │
      ▼
Transform QR Polygon & Extract Differential Surface Signals
      │
      ▼
Updated Structured Evidence
```

### Key Technical Components:
* **ORB Feature Matching:** Extracts rotation-invariant local keypoints and matches binary descriptors using Hamming distance filtered through Lowe's ratio test ($0.75$).
* **RANSAC Homography Estimation:** Robustly estimates the planar perspective projection matrix ($H$) while rejecting outlier matches.
* **Rigorous Same-Scene Verification:** A candidate registration is not accepted on raw feature count alone. QRensic enforces strict geometric sanity checks:
  1. *Determinant Sign & Scale:* $\det(H_{2\times2}) > 0$ strictly prevents mirror reflections/inversions; scale limits prevent physically unviable viewing distance discrepancies.
  2. *Projected Boundary Convexity:* Projecting scene corners must form a strictly convex, non-self-intersecting quadrilateral (`cv2.isContourConvex`).
  3. *Area Viability:* Projected scene area must fall within realistic sensor field-of-view limits.
  4. *Inlier Gating:* Rejects registrations with insufficient RANSAC consensus ($\text{inliers} < 15$ or inlier ratio $< 0.20$).
* **QR Region Projection:** Projects the detected QR bounding quad from Photo 1 into Photo 2 coordinates, enabling localized surface comparisons even when perspective causes standard QR detection to miss in the second shot.
* **Differential Evidence Extraction:** Measures changes in QR brightness, surrounding ring brightness, Relative Specular Response (RSR), boundary gradient steps, and texture energy across views.

### Running Second-View Analysis CLI
```bash
# Register two visual observations and compute differential surface evidence
python -m backend.vision.cli_second_view first.jpg second.jpg

# Or invoke directly
python backend/vision/cli_second_view.py first.jpg second.jpg
```

> [!IMPORTANT]
> **Limitations & Physical Feasibility:**
> Passing geometric unit tests confirms that ORB matching, planar homography, and projective sanity checks execute reliably on synthetic transformations. However, **physical sticker detection in real-world environments has not yet been validated**. Empirical validation will be established through the planned 16-photo physical feasibility experiment.

---

## ⚖️ Deterministic Evidence Engine

The **Evidence Engine** (`backend/vision/evidence_rules.py`) serves as the deterministic bridge between raw computer vision metrics and downstream agent decision-making:

```text
┌────────────────────────┐      ┌─────────────────────────────┐      ┌────────────────────────────┐
│      OpenCV 5          │ ───► │ Deterministic Evidence Engine│ ───► │   Bounded LLM Agent        │
│ (Raw Optical Signals)  │      │ (Interprets Measurements)   │      │(Decides Next Action/Tool)  │
└────────────────────────┘      └─────────────────────────────┘      └────────────────────────────┘
                                               │
                                               ▼
                                  Observable Factual Reasons &
                                  Recommended Next Step:
                                  [STOP | REQUEST_SECOND_VIEW | HUMAN_REVIEW]
```

### Architectural Principles:
1. **OpenCV produces measurements:** Boundary gradients, edge steps, color deltas, and relative specular response.
2. **Deterministic rules interpret evidence:** Structured metrics are mapped into discrete states (`CONSISTENT`, `AMBIGUOUS`, `HIGH_RISK`, `INCONCLUSIVE`, `HUMAN_REVIEW`).
3. **The future agent decides actions:** Decides *when* and *which* tool to call (e.g. requesting a flash or oblique re-shot).
4. **The LLM NEVER assigns fraud risk:** Machine learning does not output speculative "fraud probability" scores.

### Conservative Forensic Rules:
* **No Premature HIGH_RISK:** An identity contradiction alone (e.g., mismatching payee name) does **NOT** produce `HIGH_RISK`. It is held as `AMBIGUOUS` or routed to `HUMAN_REVIEW` unless corroborated by an independent, strong physical surface anomaly.
* **Personal Payee Protection:** Sole proprietorships often link personal accounts (e.g. `Ramesh Kumar` on a `Nova Coffee` poster). This is classified as `AMBIGUOUS`, never fraud.
* **Missing Evidence $\neq$ Negative Evidence:** If physical surface signals are unavailable, the system does not assume "no anomaly found"; it reports surface evidence as `UNAVAILABLE` and recommends a second look.
* **Conflicting Signal Handling:** If identity matches but physical surface metrics exhibit anomalous edge steps, the case is immediately routed to `HUMAN_REVIEW`.

> [!WARNING]
> **Provisional Thresholds Notice:**
> All quantitative thresholds configured in `EvidenceThresholdConfig` (such as edge discontinuity ratios and specular RSR deltas) are **provisional development defaults**. Definitive physical thresholds will be calibrated empirically using the planned 16-photo physical feasibility dataset.

---

## 🤖 Local Agent Orchestration Layer

The **Local Agent Orchestration Layer** (`backend/agent/`) coordinates dynamic, multi-turn forensic investigation sessions before connecting to external cloud LLMs (such as AWS Bedrock).

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      Agentic Investigation Loop                        │
│                                                                        │
│               ┌──────────────────────────────────────┐                 │
│               │             OBSERVE                  │                 │
│               │  Inspect optical scene quality & QR  │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                                  ▼                                     │
│               ┌──────────────────────────────────────┐                 │
│               │              DECIDE                  │                 │
│               │ Reasoning model chooses next action  │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                                  ▼                                     │
│               ┌──────────────────────────────────────┐                 │
│               │               ACT                    │                 │
│               │   Execute bounded CV vision tool     │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                                  ▼                                     │
│               ┌──────────────────────────────────────┐                 │
│               │            RE-OBSERVE                │                 │
│               │ Request/register second observation  │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                                  ▼                                     │
│               ┌──────────────────────────────────────┐                 │
│               │             EVALUATE                 │                 │
│               │ Deterministic evidence re-evaluation │                 │
│               └──────────────────┬───────────────────┘                 │
│                                  │                                     │
│                                  ▼                                     │
│               ┌──────────────────────────────────────┐                 │
│               │       STOP / HUMAN REVIEW            │                 │
│               │  Terminal verdict or specialist handoff│               │
│               └──────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────┘
```

### Architectural Principles:
1. **Dynamic Decision-Making, Not Static Pipelines:** The agent does not follow a hardcoded script. At each turn, it inspects the current evidence state and decides which observation to perform next.
2. **Strict 8-Action Whitelist:** The agent may only choose from the 8 allowed forensic actions (`inspect_scene`, `analyze_qr`, `extract_region`, `validate_identity`, `analyze_qr_surface`, `request_observation`, `analyze_second_view`, `human_review`). Any unknown or prohibited action (e.g. `calculate_fraud`, `classify_fraud`, `analyze_url`) is strictly rejected without crashing.
3. **Observable Factual Rationale:** Every agent decision requires an observable, non-empty `reason` justifying the step based strictly on visual facts—never internal or hidden chain-of-thought.
4. **Deterministic Evidence Authority:** The agent NEVER calculates or declares fraud probabilities. The deterministic evidence engine remains the sole authority for final evidence states.
5. **Execution Budget Guardrails:** Investigations are strictly bounded to a maximum of **5 iterations** (`MAX_AGENT_ITERATIONS = 5`). Reaching this limit triggers a safe forced escalation to `HUMAN_REVIEW`.
6. **Graceful Tool Failure Recovery:** If a tool encounters an error or corrupted data, the orchestrator logs the failure in the trace without crashing or fabricating hallucinated values, allowing the agent to observe the error and route to human review.

### Investigation Scenarios Tested with `MockModel`:
* **Scenario A (Consistent Case):** `inspect_scene` → `analyze_qr` → `validate_identity` → `analyze_qr_surface` → Concludes investigation (`human_review` / archive) with `CONSISTENT` evidence state.
* **Scenario B (Ambiguous Case):** Personal-name payee triggers ambiguous identity, requests oblique second observation, and safely escalates to `HUMAN_REVIEW`.
* **Scenario C (Second-Look Hero Scenario):** Initial pass detects QR → requests flash observation → receives second image → registers views via RANSAC homography (`analyze_second_view`) → prepares multi-view differential evidence dossier.
* **Scenario D (Tool Failure Recovery):** Simulates a tool fault; agent observes operational error in state and executes safe recovery fallback.
* **Scenario E (Prohibited Action Injection):** Tests unauthorized action rejection; orchestrator intercepts malformed/prohibited decisions, logs `[REJECTED]` in trace, and enforces security policy.

### Running the Agent CLI
```bash
# Run Scenario A (Consistent poster inspection)
python -m backend.agent.cli feasibility/props/generated/genuine_qr.png --scenario A --poster-text "NOVA COFFEE"

# Run Scenario C (Hero Second-Look with multi-view registration)
python -m backend.agent.cli feasibility/props/generated/genuine_qr.png --second-image feasibility/props/generated/genuine_qr.png --scenario C --poster-text "NOVA COFFEE"

# Run Scenario E (Security guardrail test - prohibited action rejection)
python -m backend.agent.cli feasibility/props/generated/genuine_qr.png --scenario E

# Output complete investigation state as structured JSON
python -m backend.agent.cli feasibility/props/generated/genuine_qr.png --scenario A --json
```




