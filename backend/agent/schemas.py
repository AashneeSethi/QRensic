"""
QRensic Agent - Schemas & Investigation State
----------------------------------------------
Strict typed data structures for agent decisions, trace events, and investigation states.
Rejects malformed outputs, unknown actions, and invalid parameters.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType, ObservationRequestType
from backend.vision.models import (
    EvidenceState,
    RecommendedAction,
    StructuredEvidence,
    _enum_serializer,
)


@dataclass
class ActionDecision:
    """
    A single structured decision made by the agent.
    
    SAFETY PRINCIPLE:
    'reason' must be a concise, observable factual rationale,
    NOT internal or hidden chain-of-thought.
    """
    action: AgentActionType
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return json.loads(json.dumps(raw, default=_enum_serializer))


@dataclass
class TraceEvent:
    """A single observable step in the forensic investigation trace."""
    step: int
    action: str
    arguments: Dict[str, Any]
    reason: str
    status: str  # "SUCCESS" | "FAILED" | "REJECTED"
    evidence_change: str
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return json.loads(json.dumps(raw, default=_enum_serializer))


@dataclass
class InvestigationState:
    """
    Complete observable state of a physical QR fraud investigation session.
    """
    investigation_id: str
    images: Dict[str, Any] = field(default_factory=dict)  # {"primary": ..., "second": ...}
    structured_evidence: StructuredEvidence = field(default_factory=StructuredEvidence)
    completed_actions: List[str] = field(default_factory=list)
    action_count: int = 0
    pending_observation_request: Optional[Dict[str, Any]] = None
    final_state: Optional[EvidenceState] = None
    final_recommendation: Optional[RecommendedAction] = None
    trace: List[TraceEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    is_terminated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary (omitting raw image arrays for clarity)."""
        d = {
            "investigation_id": self.investigation_id,
            "images_available": list(self.images.keys()),
            "structured_evidence": self.structured_evidence.to_dict(),
            "completed_actions": self.completed_actions,
            "action_count": self.action_count,
            "pending_observation_request": self.pending_observation_request,
            "final_state": self.final_state.value if self.final_state else None,
            "final_recommendation": self.final_recommendation.value if self.final_recommendation else None,
            "trace": [t.to_dict() for t in self.trace],
            "errors": self.errors,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_terminated": self.is_terminated,
        }
        return json.loads(json.dumps(d, default=_enum_serializer))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SchemaValidationError(Exception):
    """Raised when an agent action or decision violates schema constraints."""
    pass


def validate_action_decision(decision_dict: Any) -> ActionDecision:
    """
    Validate that an agent decision conforms to the strict schema.
    
    Rejects:
    - Unknown action names
    - Malformed structures
    - Missing observable reasons
    - Invalid argument structures
    """
    if isinstance(decision_dict, ActionDecision):
        return decision_dict

    if not isinstance(decision_dict, dict):
        raise SchemaValidationError(f"Agent output must be a JSON dictionary, got {type(decision_dict).__name__}.")

    action_raw = decision_dict.get("action")
    if not action_raw:
        raise SchemaValidationError("Agent decision missing required 'action' field.")

    # Check action whitelist
    try:
        action_type = AgentActionType(action_raw)
    except ValueError:
        raise SchemaValidationError(
            f"Prohibited or unknown action '{action_raw}'. "
            f"Allowed actions: {[a.value for a in ALLOWED_ACTIONS]}"
        )

    # Validate reason
    reason = decision_dict.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise SchemaValidationError(f"Action '{action_raw}' requires an observable non-empty 'reason' string.")

    # Validate arguments
    arguments = decision_dict.get("arguments", {})
    if not isinstance(arguments, dict):
        raise SchemaValidationError(f"Arguments for action '{action_raw}' must be a dictionary.")

    # Specific action argument validations
    if action_type == AgentActionType.REQUEST_OBSERVATION:
        obs_type = arguments.get("observation_type")
        if obs_type:
            valid_obs = {o.value for o in ObservationRequestType}
            if obs_type not in valid_obs:
                raise SchemaValidationError(
                    f"Invalid observation_type '{obs_type}'. Allowed types: {list(valid_obs)}"
                )

    return ActionDecision(
        action=action_type,
        arguments=arguments,
        reason=reason.strip(),
    )
