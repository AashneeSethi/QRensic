"""
QRensic Vision Engine - Deterministic Evidence Engine
------------------------------------------------------
Interprets structured computer vision measurements and derives an explainable
evidence state and actionable investigation recommendation.

ARCHITECTURAL PRINCIPLE:
- OpenCV 5 produces deterministic measurements (what can be observed).
- Evidence engine interprets measured evidence against deterministic rules.
- The future bounded agent decides which investigation tool/action to invoke next.
- The LLM NEVER directly assigns fraud risk or probability.

PROVISIONAL THRESHOLD NOTICE:
All default thresholds in this module are PROVISIONAL DEVELOPMENT DEFAULTS.
They have NOT yet been calibrated against the physical 16-photo experiment dataset.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.vision.models import (
    EvidenceEvaluation,
    EvidenceState,
    IdentityConsistency,
    RecommendedAction,
    RegistrationStatus,
    StructuredEvidence,
)


class EvidenceQuality(str, Enum):
    """Quality and availability grade of an evidence modality."""
    UNAVAILABLE = "UNAVAILABLE"
    WEAK = "WEAK"
    AMBIGUOUS = "AMBIGUOUS"
    STRONG = "STRONG"


@dataclass
class EvidenceThresholdConfig:
    """
    Configurable decision parameters for the deterministic evidence engine.
    
    IMPORTANT:
    These thresholds are PROVISIONAL DEVELOPMENT DEFAULTS.
    Physical calibration will be performed once the 16-photo physical dataset is gathered.
    """
    # Optical quality
    blur_min_laplacian_var: float = 100.0

    # Surface anomaly thresholds (Development-only provisional values)
    edge_discontinuity_strong_threshold: float = 3.5
    edge_discontinuity_weak_threshold: float = 2.0
    relative_specular_strong_ratio: float = 2.5
    relative_specular_strong_diff: float = 40.0
    color_discontinuity_strong_delta: float = 50.0

    # Multi-view registration
    min_registration_inliers: int = 15
    min_registration_inlier_ratio: float = 0.20

    is_provisional_development_config: bool = True


DEFAULT_THRESHOLD_CONFIG = EvidenceThresholdConfig()


def _assess_surface_anomaly(
    surface: Any,
    second_view: Optional[Any],
    config: EvidenceThresholdConfig,
) -> Tuple[EvidenceQuality, bool, List[str]]:
    """
    Evaluate surface evidence availability and whether an anomalous physical step/flare exists.

    Returns:
      (quality: EvidenceQuality, is_strong_anomaly: bool, anomaly_notes: List[str])
    """
    if not surface or not surface.signals_available or surface.qr_area_px <= 0:
        return EvidenceQuality.UNAVAILABLE, False, ["Physical surface analysis is unavailable."]

    anomaly_notes: List[str] = []
    is_strong = False
    is_weak = False

    # Check boundary gradient & edge discontinuity
    if "edge_discontinuity" in surface.signals_available:
        if surface.edge_discontinuity_ratio >= config.edge_discontinuity_strong_threshold:
            is_strong = True
            anomaly_notes.append(
                f"Pronounced edge discontinuity ratio ({surface.edge_discontinuity_ratio:.2f} >= "
                f"{config.edge_discontinuity_strong_threshold:.1f}) across QR boundary perimeter."
            )
        elif surface.edge_discontinuity_ratio >= config.edge_discontinuity_weak_threshold:
            is_weak = True
            anomaly_notes.append(
                f"Moderate edge discontinuity ({surface.edge_discontinuity_ratio:.2f}) observed."
            )

    # Check Relative Specular Response (from second view comparison or single-image paired flash)
    rsr_ratio = None
    rsr_diff = None

    if second_view and second_view.comparison:
        rsr_ratio = second_view.comparison.relative_specular_ratio
        rsr_diff = second_view.comparison.relative_specular_diff
    elif surface.relative_specular_ratio is not None:
        rsr_ratio = surface.relative_specular_ratio
        rsr_diff = surface.relative_specular_diff

    if rsr_ratio is not None and rsr_diff is not None:
        if rsr_ratio >= config.relative_specular_strong_ratio and rsr_diff >= config.relative_specular_strong_diff:
            is_strong = True
            anomaly_notes.append(
                f"Elevated Relative Specular Response (RSR ratio: {rsr_ratio:.2f}, diff: {rsr_diff:.1f}) "
                f"indicates distinct reflectivity between QR region and surrounding poster."
            )
        elif rsr_ratio > 1.35:
            is_weak = True
            anomaly_notes.append(f"Mild specular variance (RSR ratio: {rsr_ratio:.2f}) observed.")

    if is_strong:
        return EvidenceQuality.STRONG, True, anomaly_notes
    elif is_weak:
        return EvidenceQuality.AMBIGUOUS, False, anomaly_notes
    else:
        return EvidenceQuality.STRONG, False, ["Surface boundary and specular response appear uniform without step anomalies."]


def evaluate_evidence(
    evidence: StructuredEvidence,
    config: Optional[EvidenceThresholdConfig] = None,
) -> EvidenceEvaluation:
    """
    Deterministically interpret structured visual observations into an evidence state,
    observable factual reasons, and a recommended next investigation step.

    CONSERVATIVE RULES ENFORCED:
    1. Identity contradiction ALONE does not produce HIGH_RISK.
    2. Personal-name payee ALONE does not produce HIGH_RISK (held as AMBIGUOUS).
    3. Missing evidence is NOT treated as absence of an anomaly.
    4. Second-view registration failure triggers INCONCLUSIVE / HUMAN_REVIEW.
    5. Conflicting modalities trigger HUMAN_REVIEW.
    6. No machine-learning fraud probabilities are generated.
    """
    cfg = config or DEFAULT_THRESHOLD_CONFIG
    reasons: List[str] = []
    signals_summary: Dict[str, str] = {}

    scene = evidence.scene
    qr = evidence.qr
    identity = evidence.identity
    surface = evidence.surface
    second_view = evidence.second_view

    # -------------------------------------------------------------------------
    # 1. Optical Quality Screening
    # -------------------------------------------------------------------------
    if scene.width == 0:
        return EvidenceEvaluation(
            state=EvidenceState.INCONCLUSIVE,
            recommended_next_step=RecommendedAction.HUMAN_REVIEW,
            reasons=["Input image is empty, missing, or unreadable by OpenCV."],
            signals_summary={"scene": "UNREADABLE"},
        )

    if scene.is_blurry:
        signals_summary["scene_quality"] = f"BLURRY (Laplacian: {scene.blur_laplacian_var:.1f})"
        reasons.append(
            f"Optical blur detected in scene observation (Laplacian variance {scene.blur_laplacian_var:.1f} < "
            f"{cfg.blur_min_laplacian_var:.1f})."
        )
    else:
        signals_summary["scene_quality"] = "ADEQUATE"

    # -------------------------------------------------------------------------
    # 2. QR Detection & Payload Availability
    # -------------------------------------------------------------------------
    if not qr.detected or not qr.decoded:
        reasons.append("QR code was not detected or payload decoding failed.")
        signals_summary["qr"] = "UNDETECTED_OR_UNDECODED"
        return EvidenceEvaluation(
            state=EvidenceState.INCONCLUSIVE,
            recommended_next_step=RecommendedAction.REQUEST_SECOND_VIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    signals_summary["qr"] = f"DECODED ({qr.payload_type.value})"
    reasons.append(f"QR payload decoded successfully as {qr.payload_type.value}.")

    # -------------------------------------------------------------------------
    # 3. Second-View Registration Gating (if second view was submitted)
    # -------------------------------------------------------------------------
    if second_view is not None:
        if second_view.registration_status != RegistrationStatus.PASS:
            signals_summary["second_view"] = f"REGISTRATION_FAILED ({second_view.registration_status.value})"
            reasons.append(
                f"Second visual observation failed geometric same-scene registration: {second_view.reason}"
            )
            reasons.append("Cross-view differential evidence cannot be validated without geometric continuity.")
            return EvidenceEvaluation(
                state=EvidenceState.INCONCLUSIVE,
                recommended_next_step=RecommendedAction.HUMAN_REVIEW,
                reasons=reasons,
                signals_summary=signals_summary,
            )
        else:
            signals_summary["second_view"] = "REGISTERED_PASS"
            reasons.append("Second visual observation confirmed geometric same-scene continuity.")

    # -------------------------------------------------------------------------
    # 4. Surface Evidence Assessment
    # -------------------------------------------------------------------------
    surface_quality, is_strong_surface_anomaly, surface_notes = _assess_surface_anomaly(surface, second_view, cfg)
    signals_summary["surface_quality"] = surface_quality.value
    signals_summary["surface_anomaly"] = "STRONG" if is_strong_surface_anomaly else "NONE_OR_AMBIGUOUS"

    # -------------------------------------------------------------------------
    # 5. Identity Evidence Assessment
    # -------------------------------------------------------------------------
    signals_summary["identity_status"] = identity.status.value
    signals_summary["is_personal_payee"] = str(identity.is_personal_payee)

    # -------------------------------------------------------------------------
    # 6. Deterministic Evidence Synthesis
    # -------------------------------------------------------------------------

    # Condition: Severe optical blur preventing reliable inspection
    if scene.is_blurry:
        reasons.append("Reliable surface and boundary evidence cannot be established under optical blur.")
        return EvidenceEvaluation(
            state=EvidenceState.INCONCLUSIVE,
            recommended_next_step=RecommendedAction.REQUEST_SECOND_VIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 1: Identity is CONTRADICTORY + STRONG independent physical surface anomaly
    if identity.status == IdentityConsistency.CONTRADICTORY and is_strong_surface_anomaly:
        reasons.append("Decoded payee identity directly conflicts with visible merchant poster branding.")
        reasons.extend(surface_notes)
        reasons.append("Multiple independent evidence sources corroborate physical tampering risk.")
        return EvidenceEvaluation(
            state=EvidenceState.HIGH_RISK,
            recommended_next_step=RecommendedAction.HUMAN_REVIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 2: Conflicting evidence sources (e.g. Identity is CONSISTENT but strong physical surface anomaly exists)
    if identity.status == IdentityConsistency.CONSISTENT and is_strong_surface_anomaly:
        reasons.append("Decoded QR identity matches poster branding, yet strong physical boundary/specular anomalies were detected.")
        reasons.extend(surface_notes)
        reasons.append("Discrepancy between consistent branding and physical surface discontinuity requires human specialist review.")
        return EvidenceEvaluation(
            state=EvidenceState.HUMAN_REVIEW,
            recommended_next_step=RecommendedAction.HUMAN_REVIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 3: Identity is CONTRADICTORY ALONE (no strong independent physical anomaly)
    # CRITICAL: Identity mismatch alone MUST NOT produce HIGH_RISK.
    if identity.status == IdentityConsistency.CONTRADICTORY:
        reasons.append(
            f"Decoded payee name ('{identity.normalized_payee_name}') conflicts with poster branding {identity.normalized_poster_names}."
        )
        if surface_quality == EvidenceQuality.UNAVAILABLE:
            reasons.append("Physical surface evidence is unavailable to verify physical sticker presence.")
        else:
            reasons.append("Independent physical surface anomaly was not confirmed across the QR perimeter.")
        reasons.append("Identity discrepancy alone does not justify a HIGH_RISK physical tampering determination.")
        
        # If second view already evaluated, escalate to human review; otherwise request second view
        next_step = RecommendedAction.STOP if second_view else RecommendedAction.REQUEST_SECOND_VIEW
        return EvidenceEvaluation(
            state=EvidenceState.AMBIGUOUS,
            recommended_next_step=RecommendedAction.HUMAN_REVIEW if second_view else RecommendedAction.REQUEST_SECOND_VIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 4: Personal-name payee on commercial poster (AMBIGUOUS by design)
    # CRITICAL: Personal-name payee MUST NOT automatically produce HIGH_RISK.
    if identity.is_personal_payee and identity.status == IdentityConsistency.AMBIGUOUS:
        reasons.append(
            f"Payee appears to be an individual/personal name ('{identity.normalized_payee_name}') "
            f"whereas poster displays business branding {identity.normalized_poster_names}."
        )
        reasons.append("Personal-name payees are common in sole proprietorships and are held as AMBIGUOUS pending corroboration.")
        if is_strong_surface_anomaly:
            reasons.extend(surface_notes)
            reasons.append("Physical anomaly paired with personal payee elevates case for human review.")
            return EvidenceEvaluation(
                state=EvidenceState.HUMAN_REVIEW,
                recommended_next_step=RecommendedAction.HUMAN_REVIEW,
                reasons=reasons,
                signals_summary=signals_summary,
            )
        else:
            reasons.append("Physical surface evidence does not exhibit a confirmed overlay step anomaly.")
            return EvidenceEvaluation(
                state=EvidenceState.AMBIGUOUS,
                recommended_next_step=RecommendedAction.REQUEST_SECOND_VIEW if not second_view else RecommendedAction.HUMAN_REVIEW,
                reasons=reasons,
                signals_summary=signals_summary,
            )

    # Rule 5: Missing surface evidence
    # CRITICAL: Missing evidence must NOT collapse into "no anomaly found".
    if surface_quality == EvidenceQuality.UNAVAILABLE:
        reasons.append("Decoded QR identity is consistent or unverified, but physical surface analysis was unavailable.")
        reasons.append("Physical continuity cannot be confirmed without measurable surface signals.")
        return EvidenceEvaluation(
            state=EvidenceState.AMBIGUOUS,
            recommended_next_step=RecommendedAction.REQUEST_SECOND_VIEW,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 6: Identity is CONSISTENT and surface evidence confirmed uniform (no anomaly)
    if identity.status == IdentityConsistency.CONSISTENT and not is_strong_surface_anomaly:
        reasons.append("Decoded QR identity aligns consistently with visible merchant poster branding.")
        reasons.extend(surface_notes)
        reasons.append("No physical boundary, specular, or geometric anomalies observed.")
        return EvidenceEvaluation(
            state=EvidenceState.CONSISTENT,
            recommended_next_step=RecommendedAction.STOP,
            reasons=reasons,
            signals_summary=signals_summary,
        )

    # Rule 7: General Ambiguous Baseline (e.g. unverified poster text or weak evidence)
    reasons.append("Evidence signals are currently inconclusive or incomplete.")
    if surface_notes:
        reasons.extend(surface_notes)
    return EvidenceEvaluation(
        state=EvidenceState.AMBIGUOUS,
        recommended_next_step=RecommendedAction.REQUEST_SECOND_VIEW if not second_view else RecommendedAction.HUMAN_REVIEW,
        reasons=reasons,
        signals_summary=signals_summary,
    )
