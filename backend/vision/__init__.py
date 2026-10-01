"""
QRensic Vision Engine
---------------------
Deterministic computer vision layer for physical QR fraud investigation.
Targeted for OpenCV AI Competition 2026.
"""

from backend.vision.analyze_qr import analyze_qr
from backend.vision.analyze_surface import analyze_qr_surface
from backend.vision.engine import QRensicVisionEngine
from backend.vision.extract_region import extract_region
from backend.vision.inspect_scene import inspect_scene
from backend.vision.models import (
    EvidenceState,
    ExtractedRegionEvidence,
    IdentityConsistency,
    IdentityEvidence,
    PayloadType,
    QREvidence,
    SceneEvidence,
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
    "normalize_entity_name",
    "StructuredEvidence",
    "SceneEvidence",
    "QREvidence",
    "ExtractedRegionEvidence",
    "IdentityEvidence",
    "SurfaceEvidence",
    "URLInfo",
    "EvidenceState",
    "IdentityConsistency",
    "PayloadType",
]
