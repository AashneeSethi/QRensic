# QRensic Feasibility Experiment: Physical Second-Look Investigation

## 🔬 Purpose

The sole objective of this initial feasibility study is:

> **Determine whether a second-look investigation using OpenCV can obtain useful visual evidence from real photographs of physical QR posters.**

Rather than jumping directly to training deep models or building cloud infrastructure, this experiment investigates the physical optics and computer vision feasibility of detecting physical sticker overlays through targeted, multi-view observations.

---

## 🧪 The Four Physical Conditions

To establish baseline validity, we test four distinct physical configurations:

| # | Condition Name | Description | Rationale |
|---|---|---|---|
| 1 | **Genuine Matte Poster** | Entire poster and QR printed on uniform matte paper. | Represents baseline non-reflective merchant poster without tampering. |
| 2 | **Genuine Glossy Poster** | Entire poster and QR printed on uniform glossy paper. | **Crucial negative control:** tests whether a glossy surface alone causes false positives under flash. |
| 3 | **Glossy QR Sticker Overlay** | Glossy sticker (`sticker_qr.png`) applied over a matte host poster (`genuine_qr.png`). | The classic physical scam: shiny sticker placed over matte paper, causing strong specular mismatch. |
| 4 | **Matte QR Sticker Overlay** | Matte sticker (`sticker_qr.png`) applied over a glossy or matte host poster. | **Edge case test:** tests whether matte stickers can be detected via boundary step/edge discontinuities rather than flash flare alone. |

### Why These Four Conditions?
1. **Disentangling Material vs. Boundary:** A naive vision system might classify any glossy reflection as a "sticker". By including a **genuine glossy poster**, we ensure our metrics measure *differential discontinuity* across the boundary rather than global glossiness.
2. **Defeating Low-Reflectance Stickers:** Matte stickers do not create bright flash highlights. Testing **matte stickers** forces the system to evaluate geometric edge steps, thickness shadows, and boundary gradient transitions.

---

## 📷 Photography & Capture Protocol

Each of the 4 conditions is captured across 2 viewing angles with both ambient light and camera flash, producing **16 photographs total**.

```text
4 Physical Props × 2 Angles × 2 Lighting States (No-Flash / Flash) = 16 Photos
```

### Protocol Guidelines
* **Distance:** Keep camera / smartphone approximately **30–40 cm** from the poster plane.
* **Angle 1 (Straight-ish):** Normal incidence (~0–5 degrees from perpendicular).
  * 1 photo: **No Flash** (ambient room lighting)
  * 1 photo: **Flash** (camera flash enabled; same position)
* **Angle 2 (Oblique):** Angled perspective (~20 degrees tilt from normal).
  * 1 photo: **No Flash** (ambient room lighting)
  * 1 photo: **Flash** (camera flash enabled; same position)
* **Stability:** For each no-flash / flash pair, keep the camera position as stationary as possible so that spatial registration is direct.

### Standardized Filename Convention

Store captured photos in `feasibility/photos/` (excluded from git):

| Prop Condition | Angle 1 (No Flash) | Angle 1 (Flash) | Angle 2 (No Flash) | Angle 2 (Flash) |
|---|---|---|---|---|
| 1. Genuine Matte | `genuine_matte_a1_noflash.jpg` | `genuine_matte_a1_flash.jpg` | `genuine_matte_a2_noflash.jpg` | `genuine_matte_a2_flash.jpg` |
| 2. Genuine Glossy | `genuine_glossy_a1_noflash.jpg` | `genuine_glossy_a1_flash.jpg` | `genuine_glossy_a2_noflash.jpg` | `genuine_glossy_a2_flash.jpg` |
| 3. Glossy Sticker | `glossy_sticker_a1_noflash.jpg` | `glossy_sticker_a1_flash.jpg` | `glossy_sticker_a2_noflash.jpg` | `glossy_sticker_a2_flash.jpg` |
| 4. Matte Sticker | `matte_sticker_a1_noflash.jpg` | `matte_sticker_a1_flash.jpg` | `matte_sticker_a2_noflash.jpg` | `matte_sticker_a2_flash.jpg` |

---

## 📐 Signal Formulations & OpenCV Measurements

### 1. Relative Specular Response (RSR)
Absolute brightness cannot distinguish a sticker from variable room lighting. QRensic evaluates the **differential response** between the inner QR bounding region and an outer surrounding ring on the host poster:

$$\Delta \mu_{\text{QR}} = \mu(\text{QR}_{\text{flash}}) - \mu(\text{QR}_{\text{noflash}})$$

$$\Delta \mu_{\text{Ring}} = \mu(\text{Ring}_{\text{flash}}) - \mu(\text{Ring}_{\text{noflash}})$$

$$\text{RSR}_{\text{ratio}} = \frac{\Delta \mu_{\text{QR}} + \epsilon}{\Delta \mu_{\text{Ring}} + \epsilon}$$

$$\text{RSR}_{\text{diff}} = \Delta \mu_{\text{QR}} - \Delta \mu_{\text{Ring}}$$

* **Uniform Surface (Genuine Matte or Genuine Glossy):** $\text{RSR}_{\text{ratio}} \approx 1.0$ and $\text{RSR}_{\text{diff}} \approx 0$ (both regions brighten equally).
* **Glossy Sticker on Matte Poster:** $\Delta \mu_{\text{QR}} \gg \Delta \mu_{\text{Ring}}$, producing a significant positive $\text{RSR}_{\text{diff}}$ and $\text{RSR}_{\text{ratio}} > 1.5$.

### 2. Physical Boundary Discontinuity
* **Boundary Gradient Strength:** Mean Sobel gradient magnitude along the dilated perimeter separating the QR quad from the host poster.
* **Edge Discontinuity Score:** Ratio of outer perimeter gradient to interior module gradients.
* **Color Discontinuity ($\Delta E$ / Euclidean HSV):** Chromatic distance between the QR quiet zone margin and the host poster ring.

### 3. Multi-View Registration (Second-Look View)
* **Keypoint Detection:** ORB feature extraction on View 1 and View 2.
* **Feature Matching:** Hamming distance with Lowe's ratio test ($0.75$).
* **RANSAC Homography:** Estimating planar projective transformation matrix $H$ and computing inlier count and inlier ratio.

---

## ⚙️ Environment & OpenCV 5 Requirement

* **Strict Dependency:** QRensic specifically targets **OpenCV 5** (`opencv-python>=5,<6`).
* **Environment Status:** Tested and validated on Python 3.11 with `opencv-python 5.0.0.93`.
* **Policy:** Do **NOT** silently downgrade to OpenCV 4. If an environment cannot obtain OpenCV 5 packages, document the blocker explicitly rather than falling back.

---

## 🛡️ Safety & Security

* **Harmless Payloads:** Props generated via `generate_props.py` use ONLY harmless test strings:
  * `QRFORENSIC_TEST_PAYEE_NOVA_COFFEE`
  * `QRFORENSIC_TEST_PAYEE_RAMESH_KUMAR`
* **Zero Real Financial Data:** Never use real bank UPI IDs, personal URLs, or live payment gateways in physical or digital tests.
* **Prompt Injection Boundary:** Any payload or text read by OpenCV/OCR must be treated as untrusted user input, never executed or interpolated as system prompt instructions.
* **No Premature Classification:** The feasibility script extracts deterministic measurements only. It deliberately does not output a speculative "fraud probability".
