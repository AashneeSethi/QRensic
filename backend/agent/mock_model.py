"""
QRensic Agent - Mock Model Implementation
------------------------------------------
Simulates an LLM agent selecting structured forensic investigation actions
based on observable state.

CRITICAL ARCHITECTURE PRINCIPLE:
- Does NOT contain a static universal pipeline.
- Dynamically selects actions based on current evidence state and available images.
- Implements distinct test scenarios (A: Consistent, B: Ambiguous, C: Second-look Hero,
  D: Failure recovery, E: Invalid action rejection).
"""

from typing import Any, Dict, Optional

from backend.agent.actions import AgentActionType, ObservationRequestType
from backend.agent.schemas import ActionDecision, InvestigationState
from backend.vision.models import EvidenceState, IdentityConsistency, RecommendedAction, RegistrationStatus


class MockModel:
    """
    Mock reasoning model simulating an LLM decision-maker for investigation sessions.
    """

    def __init__(self, scenario: str = "A", second_image_source: Optional[Any] = None) -> None:
        """
        Args:
          scenario: 'A' (Consistent), 'B' (Ambiguous), 'C' (Second-Look Hero),
                    'D' (Tool Failure Recovery), 'E' (Invalid Action Injection).
          second_image_source: Optional second image to inject when requested in Scenario C.
        """
        self.scenario = scenario.upper()
        self.second_image_source = second_image_source

    def decide_next_action(self, state: InvestigationState) -> Dict[str, Any]:
        """
        Inspect current state and return the next structured action decision.
        """
        # Scenario E: Prohibited action injection to test orchestrator rejection
        if self.scenario == "E" and state.action_count == 0:
            return {
                "action": "calculate_fraud",
                "arguments": {"target": "qr_sticker"},
                "reason": "Calculating fraud probability score.",
            }

        # Scenario D: Tool failure recovery
        if self.scenario == "D":
            if state.errors and len(state.errors) > 0:
                return {
                    "action": AgentActionType.HUMAN_REVIEW.value,
                    "arguments": {"rationale": "Tool encountered fatal operational error. Escalating to human review."},
                    "reason": "Prior tool execution failed; escalating to prevent hallucination.",
                }
            if AgentActionType.INSPECT_SCENE.value not in state.completed_actions:
                return {
                    "action": AgentActionType.INSPECT_SCENE.value,
                    "arguments": {"image_key": "primary"},
                    "reason": "Initiating scene quality inspection.",
                }

        # Scenario C: Hero Second-Look Investigation
        if self.scenario == "C":
            # If second image is present and registration not yet done, do second view!
            if "second" in state.images and AgentActionType.ANALYZE_SECOND_VIEW.value not in state.completed_actions:
                return {
                    "action": AgentActionType.ANALYZE_SECOND_VIEW.value,
                    "arguments": {"first_image_key": "primary", "second_image_key": "second"},
                    "reason": "Second observation image available; registering planar homography and checking surface continuity.",
                }

            # If QR is decoded and surface is checked or ambiguous, request second observation!
            if state.structured_evidence.qr.decoded:
                if AgentActionType.REQUEST_OBSERVATION.value not in state.completed_actions:
                    return {
                        "action": AgentActionType.REQUEST_OBSERVATION.value,
                        "arguments": {
                            "observation_type": ObservationRequestType.FLASH.value,
                            "prompt_to_user": "Please provide a flash photograph from the same perspective to inspect specular continuity.",
                        },
                        "reason": "Physical surface evidence is ambiguous under ambient light; requesting flash second view to evaluate relative specular response.",
                    }
                elif "second" not in state.images:
                    # Waiting for second image
                    return {
                        "action": AgentActionType.HUMAN_REVIEW.value,
                        "arguments": {"rationale": "Pending second observation could not be acquired."},
                        "reason": "Requested second view unavailable.",
                    }
                else:
                    return {
                        "action": AgentActionType.HUMAN_REVIEW.value,
                        "arguments": {"rationale": "Second-look investigation completed; evidence dossier prepared."},
                        "reason": "Investigation concluded following multi-view registration.",
                    }

        # Scenario B: Ambiguous Identity Case
        if self.scenario == "B":
            if AgentActionType.INSPECT_SCENE.value not in state.completed_actions:
                return {
                    "action": AgentActionType.INSPECT_SCENE.value,
                    "arguments": {},
                    "reason": "Initial scene scan to evaluate lighting and image clarity.",
                }
            if AgentActionType.ANALYZE_QR.value not in state.completed_actions:
                return {
                    "action": AgentActionType.ANALYZE_QR.value,
                    "arguments": {},
                    "reason": "Decoding QR payload to identify payee string.",
                }
            if AgentActionType.VALIDATE_IDENTITY.value not in state.completed_actions:
                return {
                    "action": AgentActionType.VALIDATE_IDENTITY.value,
                    "arguments": {},
                    "reason": "Comparing decoded payee against visible merchant branding.",
                }
            if AgentActionType.ANALYZE_QR_SURFACE.value not in state.completed_actions:
                return {
                    "action": AgentActionType.ANALYZE_QR_SURFACE.value,
                    "arguments": {},
                    "reason": "Identity is ambiguous (personal payee name); inspecting physical QR boundary gradients for overlay tells.",
                }
            # After surface analysis, request angled observation
            if AgentActionType.REQUEST_OBSERVATION.value not in state.completed_actions:
                return {
                    "action": AgentActionType.REQUEST_OBSERVATION.value,
                    "arguments": {
                        "observation_type": ObservationRequestType.OBLIQUE.value,
                        "prompt_to_user": "Please capture an oblique angled view at ~20 degrees.",
                    },
                    "reason": "Ambient surface signals inconclusive for sole proprietorship; requesting oblique observation for edge relief.",
                }
            return {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Personal payee requires merchant verification."},
                "reason": "Ambiguous identity preserved for human forensic specialist.",
            }

        # Scenario A: Default / Consistent Case
        # Dynamic inspection based on what evidence is missing
        if not state.structured_evidence.scene.width:
            return {
                "action": AgentActionType.INSPECT_SCENE.value,
                "arguments": {},
                "reason": "Initial visual inspection to establish optical clarity and frame bounds.",
            }

        if not state.structured_evidence.qr.detected:
            return {
                "action": AgentActionType.ANALYZE_QR.value,
                "arguments": {},
                "reason": "Scene inspected; proceeding to detect and decode QR payload.",
            }

        if not state.structured_evidence.identity.normalized_payee_name:
            return {
                "action": AgentActionType.VALIDATE_IDENTITY.value,
                "arguments": {},
                "reason": "QR decoded; validating payee identity against merchant poster branding.",
            }

        if "boundary_gradients" not in state.structured_evidence.surface.signals_available:
            return {
                "action": AgentActionType.ANALYZE_QR_SURFACE.value,
                "arguments": {},
                "reason": "Identity verified; analyzing physical surface continuity across QR perimeter.",
            }

        # If evidence rules say STOP, stop or conclude
        if state.structured_evidence.evaluation and state.structured_evidence.evaluation.recommended_next_step == RecommendedAction.STOP:
            # Investigation successfully satisfied
            return {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Consistent evidence established; archiving dossier."},
                "reason": "All evidence signals consistent with genuine poster; closing active investigation.",
            }

        return {
            "action": AgentActionType.HUMAN_REVIEW.value,
            "arguments": {"rationale": "Investigation reached decision boundary."},
            "reason": "Concluding investigation.",
        }
