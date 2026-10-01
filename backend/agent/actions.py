"""
QRensic Agent - Allowed Forensic Actions
-----------------------------------------
Defines the EXACT 8 allowed investigative actions the bounded agent may select.

PROHIBITED ACTIONS:
Actions like 'calculate_fraud', 'classify_fraud', 'analyze_url', or arbitrary code execution
are strictly prohibited by design. The agent investigates; deterministic evidence evaluates.
"""

from enum import Enum
from typing import Set


class AgentActionType(str, Enum):
    """The exact 8 bounded forensic investigation actions."""
    INSPECT_SCENE = "inspect_scene"
    ANALYZE_QR = "analyze_qr"
    EXTRACT_REGION = "extract_region"
    VALIDATE_IDENTITY = "validate_identity"
    ANALYZE_QR_SURFACE = "analyze_qr_surface"
    REQUEST_OBSERVATION = "request_observation"
    ANALYZE_SECOND_VIEW = "analyze_second_view"
    HUMAN_REVIEW = "human_review"


class ObservationRequestType(str, Enum):
    """Standardized observation modalities requested by the agent."""
    CLEARER = "clearer"      # Re-shot due to optical blur or occlusion
    FLASH = "flash"          # Flash illumination for specular discontinuity
    OBLIQUE = "oblique"      # Angled observation for physical edge step relief


ALLOWED_ACTIONS: Set[AgentActionType] = set(AgentActionType)
