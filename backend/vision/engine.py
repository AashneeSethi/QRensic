"""
QRensic Vision Engine - Orchestration Engine
---------------------------------------------
Coordinates the full deterministic vision pipeline:
  IMAGE -> inspect_scene -> analyze_qr -> extract_region -> validate_identity -> analyze_qr_surface -> StructuredEvidence

CRITICAL CONSTRAINTS:
- No LLM in the loop for fraud decisions.
- All extracted visual text is flagged UNTRUSTED.
- State evaluation uses preliminary interface states without speculative thresholds.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import numpy as np

from backend.vision.analyze_qr import analyze_qr
from backend.vision.analyze_second_view import analyze_second_view
from backend.vision.analyze_surface import analyze_qr_surface
from backend.vision.evidence_rules import (
    DEFAULT_THRESHOLD_CONFIG,
    EvidenceThresholdConfig,
    evaluate_evidence,
)
from backend.vision.extract_region import extract_region
from backend.vision.inspect_scene import inspect_scene
from backend.vision.models import (
    EvidenceEvaluation,
    EvidenceState,
    IdentityConsistency,
    RecommendedAction,
    RegistrationStatus,
    SecondViewEvidence,
    StructuredEvidence,
)
from backend.vision.validate_identity import validate_identity


class QRensicVisionEngine:
    """
    Core deterministic computer vision engine for physical QR inspection.
    """

    def __init__(self) -> None:
        pass

    def process_image(
        self,
        image_input: Union[np.ndarray, str, Path],
        poster_text: Optional[Union[str, List[str]]] = None,
        second_image_input: Optional[Union[np.ndarray, str, Path]] = None,
    ) -> StructuredEvidence:
        """
        Execute the full vision pipeline on an input image.

        Args:
          image_input: Primary image or path to image.
          poster_text: Optional visible merchant brand text (simulating OCR).
          second_image_input: Optional second observation (e.g., flash photo).

        Returns:
          StructuredEvidence dataclass serializable to JSON.
        """
        warnings: List[str] = []

        # 1. Inspect Scene
        scene_ev = inspect_scene(image_input)
        if scene_ev.width == 0:
            warnings.append("Invalid or unreadable input image.")
            return StructuredEvidence(
                scene=scene_ev,
                preliminary_state=EvidenceState.INCONCLUSIVE,
                warnings=warnings,
            )

        if scene_ev.is_blurry:
            warnings.append(f"Image blur detected (Laplacian variance: {scene_ev.blur_laplacian_var:.1f}).")
        if scene_ev.is_low_light:
            warnings.append(f"Low ambient lighting detected (mean brightness: {scene_ev.mean_brightness:.1f}).")
        if scene_ev.is_overexposed:
            warnings.append(f"Overexposure detected (mean brightness: {scene_ev.mean_brightness:.1f}).")

        # 2. Analyze QR
        qr_ev = analyze_qr(image_input)
        if not qr_ev.detected:
            warnings.append("No QR code detected in the image.")
        elif not qr_ev.decoded:
            warnings.append("QR code detected but payload decoding failed.")

        # 3. Extract Region (Surrounding text interface)
        region_ev = extract_region(
            image_input,
            qr_bounding_box=qr_ev.bounding_box,
            mock_ocr_text=[poster_text] if isinstance(poster_text, str) else poster_text,
        )

        # 4. Validate Identity
        # Use provided poster text or extracted text
        effective_poster_text = poster_text or region_ev.extracted_text
        identity_ev = validate_identity(effective_poster_text, qr_ev.payload)

        # 5. Analyze QR Surface
        if qr_ev.detected and qr_ev.bounding_box:
            surface_ev = analyze_qr_surface(image_input, qr_ev.bounding_box, second_image_input)
        else:
            surface_ev = analyze_qr_surface(image_input, [])
            warnings.append("Surface analysis skipped due to missing QR bounding box.")

        # 6. Analyze Second View (if second observation provided)
        second_view_ev: Optional[SecondViewEvidence] = None
        if second_image_input is not None:
            second_view_ev = analyze_second_view(
                first_image=image_input,
                second_image=second_image_input,
                qr_bbox=qr_ev.bounding_box,
            )
            if second_view_ev.registration_status == RegistrationStatus.PASS:
                warnings.append("Second visual observation successfully registered with initial scene.")
            else:
                warnings.append(f"Second observation registration failed: {second_view_ev.reason}")

        # 7. Compile Initial Structured Evidence Dossier
        structured = StructuredEvidence(
            scene=scene_ev,
            qr=qr_ev,
            region=region_ev,
            identity=identity_ev,
            surface=surface_ev,
            second_view=second_view_ev,
            warnings=warnings,
        )

        # 8. Deterministic Evidence Evaluation
        evaluation = self.evaluate_evidence(structured)
        structured.evaluation = evaluation
        structured.preliminary_state = evaluation.state

        return structured

    def evaluate_evidence(
        self,
        evidence: StructuredEvidence,
        config: Optional[EvidenceThresholdConfig] = None,
    ) -> EvidenceEvaluation:
        """
        Deterministically interpret structured visual observations into an evidence state,
        observable factual reasons, and a recommended next investigation step.
        """
        return evaluate_evidence(evidence, config)

    def analyze_second_view(
        self,
        first_image: Union[np.ndarray, str, Path],
        second_image: Union[np.ndarray, str, Path],
        qr_bbox: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> SecondViewEvidence:
        """
        Standalone second-view registration and differential evidence extraction.

        Can be called independently by future agents or test harnesses.
        """
        return analyze_second_view(
            first_image=first_image,
            second_image=second_image,
            qr_bbox=qr_bbox,
            **kwargs,
        )

