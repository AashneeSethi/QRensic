"""
QRensic Vision Engine
---------------------
Deterministic computer vision layer for physical QR fraud investigation.
Targeted for OpenCV AI Competition 2026.
"""

from backend.vision.analyze_qr import analyze_qr
from backend.vision.analyze_second_view import analyze_second_view
from backend.vision.analyze_surface import analyze_qr_surface
from backend.vision.engine import QRensicVisionEngine
from backend.vision.evidence_rules import (
    DEFAULT_THRESHOLD_CONFIG,
    EvidenceQuality,
    EvidenceThresholdConfig,
    evaluate_evidence,
)
from backend.vision.extract_region import extract_region
from backend.vision.inspect_scene import inspect_scene
from backend.vision.models import (
    EvidenceEvaluation,
    EvidenceState,
    ExtractedRegionEvidence,
    IdentityConsistency,
    IdentityEvidence,
    PayloadType,
    QREvidence,
    RecommendedAction,
    RegistrationStatus,
    SceneEvidence,
    SecondViewComparison,
    SecondViewEvidence,
    StructuredEvidence,
    SurfaceEvidence,
    URLInfo,
)
from backend.vision.validate_identity import normalize_entity_name, validate_identity

__all__ = [
    "QRensicVisionEngine",
    "inspect_scene",
    "analyze_qr",
    "extract_region",
    "validate_identity",
    "analyze_qr_surface",
    "analyze_second_view",
    "evaluate_evidence",
    "normalize_entity_name",
    "EvidenceThresholdConfig",
    "DEFAULT_THRESHOLD_CONFIG",
    "EvidenceQuality",
    "StructuredEvidence",
    "EvidenceEvaluation",
    "SceneEvidence",
    "QREvidence",
    "ExtractedRegionEvidence",
    "IdentityEvidence",
    "SurfaceEvidence",
    "SecondViewEvidence",
    "SecondViewComparison",
    "RegistrationStatus",
    "RecommendedAction",
    "URLInfo",
    "EvidenceState",
    "IdentityConsistency",
    "PayloadType",
]
