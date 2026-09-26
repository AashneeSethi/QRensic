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
