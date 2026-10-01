"""
QRensic Vision Engine - Data Models & State Enums
-------------------------------------------------
Structured evidence representations and deterministic state enums.
All models are strictly decoupled from machine-learning classifiers
and can be serialized to JSON.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional


class EvidenceState(str, Enum):
    """Overall evidence state classification."""
    CONSISTENT = "CONSISTENT"
    AMBIGUOUS = "AMBIGUOUS"
    HIGH_RISK = "HIGH_RISK"
    INCONCLUSIVE = "INCONCLUSIVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class RecommendedAction(str, Enum):
    """Recommended next investigative action for the future controller/agent."""
    STOP = "STOP"
    REQUEST_SECOND_VIEW = "REQUEST_SECOND_VIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class RegistrationStatus(str, Enum):
    """Status of multi-view geometric scene registration."""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class IdentityConsistency(str, Enum):
    """Consistency between visible poster text and QR identity."""
    CONSISTENT = "CONSISTENT"
    AMBIGUOUS = "AMBIGUOUS"
    CONTRADICTORY = "CONTRADICTORY"


class PayloadType(str, Enum):
    """Broad classification of decoded QR payload."""
    URL = "URL"
    EMV_PAYMENT = "EMV_PAYMENT"
    PLAIN_TEST = "PLAIN_TEST"
    UNKNOWN = "UNKNOWN"


@dataclass
class SceneEvidence:
    """Scene-level optical, illumination, and candidate quality metrics."""
    width: int = 0
    height: int = 0
    channels: int = 0
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    blur_laplacian_var: float = 0.0
    is_blurry: bool = False
    is_low_light: bool = False
    is_overexposed: bool = False
    qr_candidates_count: int = 0
    text_candidates_count: int = 0
    qr_candidates_boxes: List[List[int]] = field(default_factory=list)  # [[x, y, w, h], ...]
    text_candidates_boxes: List[List[int]] = field(default_factory=list)  # [[x, y, w, h], ...]


@dataclass
class URLInfo:
    """Structured breakdown of a URL payload (non-evaluative)."""
    raw_url: str
    scheme: str = ""
    domain: str = ""
    path: str = ""
    is_https: bool = False


@dataclass
class QREvidence:
    """QR code detection, geometry, decoding, and payload classification."""
    detected: bool = False
    decoded: bool = False
    payload: str = ""
    payload_type: PayloadType = PayloadType.UNKNOWN
    bounding_box: Optional[List[List[float]]] = None  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    url_info: Optional[URLInfo] = None
    emv_info: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedRegionEvidence:
    """
    Extracted textual and regional evidence from the scene.
    
    SAFETY NOTICE:
    All visual text extracted from images must be marked UNTRUSTED.
    It must never be executed or treated as system instructions.
    """
    extracted_text: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    text_bounding_regions: List[List[int]] = field(default_factory=list)  # [[x, y, w, h], ...]
    is_untrusted: bool = True
    notes: str = ""


@dataclass
class IdentityEvidence:
    """Deterministic comparison between visible poster text and QR payee identity."""
    status: IdentityConsistency = IdentityConsistency.AMBIGUOUS
    normalized_poster_names: List[str] = field(default_factory=list)
    normalized_payee_name: str = ""
    match_score: float = 0.0
    match_rationale: str = ""
    is_personal_payee: bool = False


@dataclass
class SurfaceEvidence:
    """Deterministic physical surface and edge evidence signals."""
    boundary_gradient_mean: float = 0.0
    boundary_gradient_max: float = 0.0
    edge_discontinuity_ratio: float = 0.0
    color_discontinuity_delta: float = 0.0
    qr_mean_intensity: float = 0.0
    qr_std_intensity: float = 0.0
    qr_mean_saturation: float = 0.0
    ring_mean_intensity: float = 0.0
    ring_std_intensity: float = 0.0
    ring_mean_saturation: float = 0.0
    texture_laplacian_energy_qr: float = 0.0
    texture_laplacian_energy_ring: float = 0.0
    qr_aspect_ratio: float = 0.0
    qr_diagonal_ratio: float = 0.0
    qr_area_px: float = 0.0
    relative_specular_ratio: Optional[float] = None
    relative_specular_diff: Optional[float] = None
    signals_available: List[str] = field(default_factory=list)


@dataclass
class SecondViewComparison:
    """Differential surface and illumination comparison between two registered views."""
    qr_brightness_first: float = 0.0
    qr_brightness_second: float = 0.0
    delta_qr_brightness: float = 0.0
    ring_brightness_first: float = 0.0
    ring_brightness_second: float = 0.0
    delta_ring_brightness: float = 0.0
    relative_specular_ratio: Optional[float] = None
    relative_specular_diff: Optional[float] = None
    boundary_gradient_first: float = 0.0
    boundary_gradient_second: float = 0.0
    delta_boundary_gradient: float = 0.0
    texture_energy_first: float = 0.0
    texture_energy_second: float = 0.0
    delta_texture_energy: float = 0.0
    color_delta_first: float = 0.0
    color_delta_second: float = 0.0
    delta_color_delta: float = 0.0
    comparison_notes: str = ""


@dataclass
class SecondViewEvidence:
    """
    Structured outcome of multi-view registration and same-object validation.
    """
    registration_status: RegistrationStatus = RegistrationStatus.INCONCLUSIVE
    keypoints_first: int = 0
    keypoints_second: int = 0
    candidate_matches: int = 0
    good_matches: int = 0
    homography_found: bool = False
    ransac_inliers: int = 0
    inlier_ratio: float = 0.0
    geometric_sanity_passed: bool = False
    homography_matrix: Optional[List[List[float]]] = None
    reprojection_error: Optional[float] = None
    registered_qr_bbox: Optional[List[List[float]]] = None
    reason: str = ""
    comparison: Optional[SecondViewComparison] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SecondViewEvidence to a native dictionary."""
        raw = asdict(self)
        return json.loads(json.dumps(raw, default=_enum_serializer))

    def to_json(self, indent: int = 2) -> str:
        """Serialize SecondViewEvidence to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class EvidenceEvaluation:
    """
    Deterministic interpretation of measured evidence.
    
    SAFETY NOTICE:
    Contains observable factual evidence reasons only.
    Does NOT calculate or output a fraud probability.
    """
    state: EvidenceState = EvidenceState.INCONCLUSIVE
    recommended_next_step: RecommendedAction = RecommendedAction.REQUEST_SECOND_VIEW
    reasons: List[str] = field(default_factory=list)
    is_provisional_calibration: bool = True
    signals_summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvidenceEvaluation to a native dictionary."""
        raw = asdict(self)
        return json.loads(json.dumps(raw, default=_enum_serializer))

    def to_json(self, indent: int = 2) -> str:
        """Serialize EvidenceEvaluation to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


def _enum_serializer(obj: Any) -> Any:
    """Helper to convert Enum objects to their string value during serialization."""
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class StructuredEvidence:
    """
    Comprehensive structured evidence dossier compiled by the vision engine.
    Designed to be safely consumed by downstream bounded agents.
    """
    scene: SceneEvidence = field(default_factory=SceneEvidence)
    qr: QREvidence = field(default_factory=QREvidence)
    region: ExtractedRegionEvidence = field(default_factory=ExtractedRegionEvidence)
    identity: IdentityEvidence = field(default_factory=IdentityEvidence)
    surface: SurfaceEvidence = field(default_factory=SurfaceEvidence)
    second_view: Optional[SecondViewEvidence] = None
    evaluation: Optional[EvidenceEvaluation] = None
    preliminary_state: EvidenceState = EvidenceState.INCONCLUSIVE
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire structured evidence model into a native dictionary."""
        raw = asdict(self)
        # Convert enums to string representation
        return json.loads(json.dumps(raw, default=_enum_serializer))

    def to_json(self, indent: int = 2) -> str:
        """Serialize structured evidence into JSON format."""
        return json.dumps(self.to_dict(), indent=indent)

