"""
QRensic Agent - Orchestrator
----------------------------
Manages the forensic investigation lifecycle:
  1. Initializes InvestigationState.
  2. Bounded iteration loop (max 5 actions).
  3. Queries reasoning model (e.g. MockModel) for the next action.
  4. Validates decision via SecurityPolicy and action schemas.
  5. Executes deterministic OpenCV vision tools.
  6. Re-evaluates evidence deterministically via evaluate_evidence().
  7. Records structured TraceEvent with observable factual rationale.
  8. Terminates on human_review, STOP recommendation, or iteration limit.

SAFETY & GOVERNANCE PRINCIPLES:
- The agent decides WHICH tool to invoke next; deterministic rules decide WHAT the evidence means.
- The agent NEVER calculates or declares fraud probabilities.
- Any unauthorized or malformed action is rejected without crashing.
- Tool failures are recorded in the trace; no evidence is ever fabricated.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import numpy as np

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType, ObservationRequestType
from backend.agent.mock_model import MockModel
from backend.agent.policy import MAX_AGENT_ITERATIONS, SecurityPolicy
from backend.agent.schemas import (
    ActionDecision,
    InvestigationState,
    SchemaValidationError,
    TraceEvent,
    validate_action_decision,
)
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
    RecommendedAction,
    StructuredEvidence,
)
from backend.vision.validate_identity import validate_identity


class AgentOrchestrator:
    """
    Forensic investigation agent orchestrator coordinating dynamic multi-turn
    investigations bounded by strict safety policies and deterministic evidence evaluation.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        threshold_config: Optional[EvidenceThresholdConfig] = None,
        default_poster_text: Optional[Union[str, List[str]]] = None,
        second_image_source: Optional[Any] = None,
    ) -> None:
        self.model = model or MockModel()
        self.threshold_config = threshold_config or DEFAULT_THRESHOLD_CONFIG
        self.default_poster_text = default_poster_text
        self.second_image_source = second_image_source
        self._staged_second_image: Optional[Any] = None

        # Tool registry mapping all 8 permitted actions to concrete implementations
        self.tool_registry: Dict[AgentActionType, Callable[..., str]] = {
            AgentActionType.INSPECT_SCENE: self._tool_inspect_scene,
            AgentActionType.ANALYZE_QR: self._tool_analyze_qr,
            AgentActionType.EXTRACT_REGION: self._tool_extract_region,
            AgentActionType.VALIDATE_IDENTITY: self._tool_validate_identity,
            AgentActionType.ANALYZE_QR_SURFACE: self._tool_analyze_qr_surface,
            AgentActionType.REQUEST_OBSERVATION: self._tool_request_observation,
            AgentActionType.ANALYZE_SECOND_VIEW: self._tool_analyze_second_view,
            AgentActionType.HUMAN_REVIEW: self._tool_human_review,
        }

    def run_investigation(
        self,
        primary_image: Union[np.ndarray, str, Path],
        second_image: Optional[Union[np.ndarray, str, Path]] = None,
        poster_text: Optional[Union[str, List[str]]] = None,
        investigation_id: Optional[str] = None,
    ) -> InvestigationState:
        """
        Execute an end-to-end bounded forensic investigation loop.

        Args:
          primary_image: Initial observation image (file path or numpy array).
          second_image: Optional second observation (staged until requested).
          poster_text: Optional merchant brand text visible on the poster.
          investigation_id: Optional custom investigation identifier.

        Returns:
          InvestigationState containing structured evidence, trace, and final state.
        """
        inv_id = investigation_id or f"inv_{uuid.uuid4().hex[:8]}"
        state = InvestigationState(investigation_id=inv_id)
        state.images["primary"] = primary_image

        effective_poster_text = poster_text or self.default_poster_text
        if second_image is not None:
            self._staged_second_image = second_image
        elif getattr(self.model, "second_image_source", None) is not None:
            self._staged_second_image = getattr(self.model, "second_image_source", None)
        else:
            self._staged_second_image = self.second_image_source

        # Multi-turn bounded investigation loop
        while not state.is_terminated and state.action_count < MAX_AGENT_ITERATIONS:
            self._step(state, effective_poster_text)

        # Guardrail: check if loop ended without termination because iteration limit was reached
        if not state.is_terminated:
            state.is_terminated = True
            state.final_state = EvidenceState.HUMAN_REVIEW
            state.final_recommendation = RecommendedAction.HUMAN_REVIEW
            state.trace.append(
                TraceEvent(
                    step=state.action_count + 1,
                    action="budget_limit",
                    arguments={},
                    reason=f"Maximum iteration limit ({MAX_AGENT_ITERATIONS}) reached. Escalating to human specialist.",
                    status="REJECTED",
                    evidence_change="Iteration budget exhausted; forced escalation to human review.",
                )
            )

        state.end_time = datetime.now(timezone.utc).isoformat()
        return state

    def _step(
        self,
        state: InvestigationState,
        effective_poster_text: Optional[Union[str, List[str]]],
    ) -> None:
        """Execute a single investigation step."""
        # 1. Verify iteration budget
        can_exec, budget_reason = SecurityPolicy.validate_execution_budget(state)
        if not can_exec:
            state.is_terminated = True
            state.final_state = EvidenceState.HUMAN_REVIEW
            state.final_recommendation = RecommendedAction.HUMAN_REVIEW
            state.trace.append(
                TraceEvent(
                    step=state.action_count + 1,
                    action="budget_limit",
                    arguments={},
                    reason=budget_reason,
                    status="REJECTED",
                    evidence_change="Iteration budget exhausted; forced escalation to human review.",
                )
            )
            return

        # 2. Query reasoning model for next structured action
        try:
            raw_decision = self.model.decide_next_action(state)
        except Exception as e:
            state.action_count += 1
            error_msg = f"Model execution error: {str(e)}"
            state.errors.append(error_msg)
            state.trace.append(
                TraceEvent(
                    step=state.action_count,
                    action="model_query",
                    arguments={},
                    reason="Model raised unhandled exception.",
                    status="FAILED",
                    evidence_change="None (model query failed)",
                    error_message=error_msg,
                )
            )
            return

        # 3. Validate action schema
        try:
            decision = validate_action_decision(raw_decision)
        except SchemaValidationError as e:
            state.action_count += 1
            error_msg = f"Schema validation error: {str(e)}"
            state.errors.append(error_msg)
            action_name = (
                raw_decision.get("action")
                if isinstance(raw_decision, dict)
                else "malformed"
            )
            state.trace.append(
                TraceEvent(
                    step=state.action_count,
                    action=str(action_name),
                    arguments=raw_decision.get("arguments", {})
                    if isinstance(raw_decision, dict)
                    else {},
                    reason=str(raw_decision.get("reason", ""))
                    if isinstance(raw_decision, dict)
                    else "",
                    status="REJECTED",
                    evidence_change="None (action rejected)",
                    error_message=error_msg,
                )
            )
            return

        # 4. Enforce security authorization policy
        is_auth, auth_reason = SecurityPolicy.validate_action_authorization(decision)
        if not is_auth:
            state.action_count += 1
            state.errors.append(auth_reason)
            state.trace.append(
                TraceEvent(
                    step=state.action_count,
                    action=decision.action.value,
                    arguments=decision.arguments,
                    reason=decision.reason,
                    status="REJECTED",
                    evidence_change="None (unauthorized action)",
                    error_message=auth_reason,
                )
            )
            return

        # 5. Check termination condition (human_review)
        is_term, term_reason = SecurityPolicy.check_termination_condition(decision, state)
        if is_term:
            handler = self.tool_registry.get(decision.action)
            evidence_change = (
                handler(state, decision.arguments, effective_poster_text)
                if handler
                else "Escalated to human review."
            )
            state.action_count += 1
            state.completed_actions.append(decision.action.value)
            state.is_terminated = True
            state.final_recommendation = RecommendedAction.HUMAN_REVIEW

            # Retain deterministic evidence evaluation if present
            if state.structured_evidence.evaluation:
                state.final_state = state.structured_evidence.evaluation.state
            else:
                state.final_state = EvidenceState.INCONCLUSIVE

            state.trace.append(
                TraceEvent(
                    step=state.action_count,
                    action=decision.action.value,
                    arguments=decision.arguments,
                    reason=decision.reason,
                    status="SUCCESS",
                    evidence_change=evidence_change,
                )
            )
            return

        # 6. Execute deterministic tool handler
        handler = self.tool_registry.get(decision.action)
        if not handler:
            state.action_count += 1
            error_msg = f"No handler registered for action '{decision.action.value}'."
            state.errors.append(error_msg)
            state.trace.append(
                TraceEvent(
                    step=state.action_count,
                    action=decision.action.value,
                    arguments=decision.arguments,
                    reason=decision.reason,
                    status="FAILED",
                    evidence_change="None (missing tool handler)",
                    error_message=error_msg,
                )
            )
            return

        status = "SUCCESS"
        error_msg = None
        evidence_change = ""

        try:
            evidence_change = handler(state, decision.arguments, effective_poster_text)
        except Exception as e:
            status = "FAILED"
            error_msg = f"Tool '{decision.action.value}' execution failed: {str(e)}"
            state.errors.append(error_msg)
            evidence_change = "None (tool execution failed)"

        state.action_count += 1
        state.completed_actions.append(decision.action.value)

        # 7. Re-evaluate evidence deterministically
        eval_result = evaluate_evidence(state.structured_evidence, self.threshold_config)
        state.structured_evidence.evaluation = eval_result
        state.structured_evidence.preliminary_state = eval_result.state
        state.final_state = eval_result.state
        state.final_recommendation = eval_result.recommended_next_step

        state.trace.append(
            TraceEvent(
                step=state.action_count,
                action=decision.action.value,
                arguments=decision.arguments,
                reason=decision.reason,
                status=status,
                evidence_change=evidence_change,
                error_message=error_msg,
            )
        )

    # -------------------------------------------------------------------------
    # Tool Handlers
    # -------------------------------------------------------------------------

    def _tool_inspect_scene(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        image_key = args.get("image_key", "primary")
        img = state.images.get(image_key)
        if img is None:
            raise ValueError(f"Image key '{image_key}' not found in state.")

        scene_ev = inspect_scene(img)
        if scene_ev.width == 0:
            raise ValueError(f"Image for key '{image_key}' is empty or invalid.")

        state.structured_evidence.scene = scene_ev
        return (
            f"Scene inspected: {scene_ev.width}x{scene_ev.height}, "
            f"blur_var={scene_ev.blur_laplacian_var:.1f}, is_blurry={scene_ev.is_blurry}"
        )

    def _tool_analyze_qr(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        image_key = args.get("image_key", "primary")
        img = state.images.get(image_key)
        if img is None:
            raise ValueError(f"Image key '{image_key}' not found in state.")

        qr_ev = analyze_qr(img)
        state.structured_evidence.qr = qr_ev
        return (
            f"QR detected={qr_ev.detected}, decoded={qr_ev.decoded}, "
            f"payload_type={qr_ev.payload_type.value}"
        )

    def _tool_extract_region(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        image_key = args.get("image_key", "primary")
        img = state.images.get(image_key)
        if img is None:
            raise ValueError(f"Image key '{image_key}' not found in state.")

        ocr_mock = args.get("mock_ocr_text", poster_text)
        if isinstance(ocr_mock, str):
            ocr_mock = [ocr_mock]

        reg_ev = extract_region(
            img,
            qr_bounding_box=state.structured_evidence.qr.bounding_box,
            mock_ocr_text=ocr_mock,
        )
        state.structured_evidence.region = reg_ev
        return (
            f"Extracted {len(reg_ev.text_bounding_regions)} regions, "
            f"text={reg_ev.extracted_text} (UNTRUSTED)"
        )

    def _tool_validate_identity(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        p_text = (
            args.get("poster_text")
            or poster_text
            or state.structured_evidence.region.extracted_text
        )
        payee = args.get("payee_identity") or state.structured_evidence.qr.payload
        id_ev = validate_identity(p_text, payee)
        state.structured_evidence.identity = id_ev
        return (
            f"Identity status={id_ev.status.value}, "
            f"score={id_ev.match_score:.2f}, personal={id_ev.is_personal_payee}"
        )

    def _tool_analyze_qr_surface(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        image_key = args.get("image_key", "primary")
        img = state.images.get(image_key)
        if img is None:
            raise ValueError(f"Image key '{image_key}' not found in state.")

        bbox = state.structured_evidence.qr.bounding_box or []
        second_img = state.images.get("second")
        surf_ev = analyze_qr_surface(img, bbox, second_image_input=second_img)
        state.structured_evidence.surface = surf_ev
        return (
            f"Surface signals: {surf_ev.signals_available}, "
            f"edge_ratio={surf_ev.edge_discontinuity_ratio:.2f}"
        )

    def _tool_request_observation(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        obs_type = args.get("observation_type", ObservationRequestType.FLASH.value)
        prompt = args.get("prompt_to_user", "Additional observation required.")
        state.pending_observation_request = {
            "observation_type": obs_type,
            "prompt_to_user": prompt,
        }

        # Inject staged second image if available and not yet present in state
        if "second" not in state.images:
            staged = None
            if self._staged_second_image is not None:
                staged = self._staged_second_image
            elif getattr(self.model, "second_image_source", None) is not None:
                staged = getattr(self.model, "second_image_source", None)
            else:
                staged = self.second_image_source

            if staged is not None:
                state.images["second"] = staged

        return f"Requested second observation (type: {obs_type})"

    def _tool_analyze_second_view(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        k1 = args.get("first_image_key", "primary")
        k2 = args.get("second_image_key", "second")
        img1 = state.images.get(k1)
        img2 = state.images.get(k2)

        if img1 is None:
            raise ValueError(f"First image '{k1}' not found in state.")
        if img2 is None:
            raise ValueError(f"Second observation image '{k2}' not found in state.")

        bbox = state.structured_evidence.qr.bounding_box
        sec_ev = analyze_second_view(img1, img2, qr_bbox=bbox)
        state.structured_evidence.second_view = sec_ev
        return (
            f"Second-view registration: {sec_ev.registration_status.value}, "
            f"inliers={sec_ev.ransac_inliers}, ratio={sec_ev.inlier_ratio:.2f}"
        )

    def _tool_human_review(
        self,
        state: InvestigationState,
        args: Dict[str, Any],
        poster_text: Optional[Union[str, List[str]]],
    ) -> str:
        rationale = args.get("rationale", "Specialist review requested.")
        state.is_terminated = True
        state.final_recommendation = RecommendedAction.HUMAN_REVIEW
        return f"Escalated to human review: {rationale}"
