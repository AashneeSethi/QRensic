"""
QRensic Agent - Orchestrator Test Suite
----------------------------------------
Comprehensive test suite validating bounded agent execution, security policies,
error handling, trace generation, and deterministic evidence authority.

TEST MATRIX:
 1. Valid action execution
 2. Unknown action rejection
 3. Malformed action rejection
 4. Maximum iteration limit (5 iterations -> HUMAN_REVIEW)
 5. human_review stops loop
 6. Tool failure recovery
 7. Second-look scenario (request observation -> second view -> registration)
 8. Deterministic evidence remains authoritative
 9. No fabricated evidence after tool failure
10. Trace is generated with timestamps, actions, evidence changes
11. Only 8 allowed actions can execute
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

import cv2
import numpy as np

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType, ObservationRequestType
from backend.agent.mock_model import MockModel
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.policy import MAX_AGENT_ITERATIONS, SecurityPolicy
from backend.agent.schemas import (
    ActionDecision,
    InvestigationState,
    SchemaValidationError,
    validate_action_decision,
)
from backend.vision.models import (
    EvidenceState,
    IdentityConsistency,
    RecommendedAction,
    RegistrationStatus,
)


class CustomScriptedModel:
    """Helper mock model that yields a predefined sequence of action dictionaries."""

    def __init__(self, script: List[Dict[str, Any]]) -> None:
        self.script = list(script)
        self.step_idx = 0

    def decide_next_action(self, state: InvestigationState) -> Dict[str, Any]:
        if self.step_idx < len(self.script):
            action = self.script[self.step_idx]
            self.step_idx += 1
            return action
        return {
            "action": AgentActionType.HUMAN_REVIEW.value,
            "arguments": {"rationale": "End of script reached."},
            "reason": "Concluding scripted test execution.",
        }


class TestAgentOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.props_dir = cls.repo_root / "feasibility" / "props" / "generated"
        cls.genuine_qr_path = cls.props_dir / "genuine_qr.png"
        cls.sticker_qr_path = cls.props_dir / "sticker_qr.png"

        if not cls.genuine_qr_path.is_file():
            from feasibility.generate_props import main as gen_props
            gen_props()

        cls.base_img = cv2.imread(str(cls.genuine_qr_path))
        assert cls.base_img is not None, "Failed to load genuine_qr.png for agent tests"

    # -------------------------------------------------------------------------
    # Test 1: Valid action execution
    # -------------------------------------------------------------------------
    def test_01_valid_action_execution(self):
        """Test 1: Valid actions execute properly, update evidence and trace, and increment count."""
        model = CustomScriptedModel([
            {
                "action": AgentActionType.INSPECT_SCENE.value,
                "arguments": {"image_key": "primary"},
                "reason": "Evaluating optical quality and resolution.",
            },
            {
                "action": AgentActionType.ANALYZE_QR.value,
                "arguments": {"image_key": "primary"},
                "reason": "Detecting and decoding QR code.",
            },
            {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Finished preliminary check."},
                "reason": "Concluding test.",
            },
        ])

        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Verify completed actions
        self.assertIn(AgentActionType.INSPECT_SCENE.value, state.completed_actions)
        self.assertIn(AgentActionType.ANALYZE_QR.value, state.completed_actions)
        self.assertIn(AgentActionType.HUMAN_REVIEW.value, state.completed_actions)
        self.assertEqual(state.action_count, 3)

        # Verify structured evidence was populated
        self.assertGreater(state.structured_evidence.scene.width, 0)
        self.assertTrue(state.structured_evidence.qr.detected)
        self.assertTrue(state.structured_evidence.qr.decoded)

        # Verify trace entries
        self.assertEqual(len(state.trace), 3)
        self.assertEqual(state.trace[0].status, "SUCCESS")
        self.assertEqual(state.trace[0].action, AgentActionType.INSPECT_SCENE.value)
        self.assertIn("Scene inspected", state.trace[0].evidence_change)
        self.assertEqual(state.trace[1].status, "SUCCESS")
        self.assertEqual(state.trace[1].action, AgentActionType.ANALYZE_QR.value)
        self.assertIn("QR detected=True", state.trace[1].evidence_change)

    # -------------------------------------------------------------------------
    # Test 2: Unknown action rejection
    # -------------------------------------------------------------------------
    def test_02_unknown_action_rejection(self):
        """Test 2: Prohibited/unknown action names are rejected, logged in trace, and do not crash."""
        model = CustomScriptedModel([
            {
                "action": "calculate_fraud",
                "arguments": {"target": "qr_sticker"},
                "reason": "Attempting unauthorized fraud calculation.",
            },
            {
                "action": "classify_fraud",
                "arguments": {},
                "reason": "Attempting prohibited classification.",
            },
            {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Safe exit."},
                "reason": "Conclude safely after rejections.",
            },
        ])

        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Verify rejections were caught without crashing
        self.assertEqual(len(state.errors), 2)
        self.assertIn("calculate_fraud", state.errors[0])
        self.assertIn("classify_fraud", state.errors[1])

        # Verify trace recorded REJECTED status
        self.assertEqual(state.trace[0].status, "REJECTED")
        self.assertEqual(state.trace[0].action, "calculate_fraud")
        self.assertEqual(state.trace[1].status, "REJECTED")
        self.assertEqual(state.trace[1].action, "classify_fraud")

        # Prohibited actions must NOT appear in completed_actions
        self.assertNotIn("calculate_fraud", state.completed_actions)
        self.assertNotIn("classify_fraud", state.completed_actions)
        self.assertIn(AgentActionType.HUMAN_REVIEW.value, state.completed_actions)

    # -------------------------------------------------------------------------
    # Test 3: Malformed action rejection
    # -------------------------------------------------------------------------
    def test_03_malformed_action_rejection(self):
        """Test 3: Malformed decisions (missing fields, invalid types) are rejected with SchemaValidationError."""
        # 1. Non-dict input
        with self.assertRaises(SchemaValidationError):
            validate_action_decision("not_a_dict")

        # 2. Missing action field
        with self.assertRaises(SchemaValidationError):
            validate_action_decision({"reason": "Valid reason without action."})

        # 3. Missing reason
        with self.assertRaises(SchemaValidationError):
            validate_action_decision({"action": "inspect_scene", "reason": ""})

        # 4. Whitespace-only reason
        with self.assertRaises(SchemaValidationError):
            validate_action_decision({"action": "inspect_scene", "reason": "   \n\t "})

        # 5. Non-dict arguments
        with self.assertRaises(SchemaValidationError):
            validate_action_decision({
                "action": "inspect_scene",
                "arguments": "invalid_args_format",
                "reason": "Valid reason.",
            })

        # 6. Invalid observation type
        with self.assertRaises(SchemaValidationError):
            validate_action_decision({
                "action": "request_observation",
                "arguments": {"observation_type": "invalid_infrared_type"},
                "reason": "Requesting unsupported observation modality.",
            })

        # Test orchestrator handling of malformed input during run
        model = CustomScriptedModel([
            {"action": "inspect_scene"},  # Missing reason
            {"action": AgentActionType.HUMAN_REVIEW.value, "reason": "Recover from error."},
        ])
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        self.assertEqual(state.trace[0].status, "REJECTED")
        self.assertIn("reason", state.errors[0])

    # -------------------------------------------------------------------------
    # Test 4: Maximum iteration limit (5 iterations -> HUMAN_REVIEW)
    # -------------------------------------------------------------------------
    def test_04_maximum_iteration_limit(self):
        """Test 4: When agent hits 5 iterations without concluding, loop halts and forces HUMAN_REVIEW."""
        class InfiniteLoopModel:
            def decide_next_action(self, state: InvestigationState) -> Dict[str, Any]:
                return {
                    "action": AgentActionType.INSPECT_SCENE.value,
                    "arguments": {"image_key": "primary"},
                    "reason": "Looping action without terminating.",
                }

        orchestrator = AgentOrchestrator(model=InfiniteLoopModel())
        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Strictly max 5 action iterations
        self.assertEqual(state.action_count, MAX_AGENT_ITERATIONS)
        self.assertTrue(state.is_terminated)
        self.assertEqual(state.final_state, EvidenceState.HUMAN_REVIEW)
        self.assertEqual(state.final_recommendation, RecommendedAction.HUMAN_REVIEW)

        # Budget exhaustion event recorded in trace
        budget_events = [t for t in state.trace if t.action == "budget_limit"]
        self.assertGreaterEqual(len(budget_events), 1)
        self.assertIn("Maximum iteration limit", budget_events[0].reason)

    # -------------------------------------------------------------------------
    # Test 5: human_review stops loop
    # -------------------------------------------------------------------------
    def test_05_human_review_stops_loop(self):
        """Test 5: Selecting human_review immediately halts the loop and preserves evidence."""
        model = CustomScriptedModel([
            {
                "action": AgentActionType.INSPECT_SCENE.value,
                "arguments": {"image_key": "primary"},
                "reason": "Inspect first.",
            },
            {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Early specialist review needed."},
                "reason": "Escalating early.",
            },
            {
                "action": AgentActionType.ANALYZE_QR.value,
                "arguments": {},
                "reason": "This action should NEVER be reached.",
            },
        ])

        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        self.assertEqual(state.action_count, 2)
        self.assertTrue(state.is_terminated)
        self.assertEqual(state.completed_actions, [
            AgentActionType.INSPECT_SCENE.value,
            AgentActionType.HUMAN_REVIEW.value,
        ])
        self.assertNotIn(AgentActionType.ANALYZE_QR.value, state.completed_actions)
        self.assertEqual(state.final_recommendation, RecommendedAction.HUMAN_REVIEW)
        # Scene evidence was collected and preserved
        self.assertGreater(state.structured_evidence.scene.width, 0)

    # -------------------------------------------------------------------------
    # Test 6: Tool failure recovery
    # -------------------------------------------------------------------------
    def test_06_tool_failure_recovery(self):
        """Test 6: When a tool fails, orchestrator catches error, logs it, and agent recovers safely."""
        def failing_tool(state, args, poster_text):
            raise RuntimeError("Hardware camera interface timeout or file corruption")

        orchestrator = AgentOrchestrator(model=MockModel(scenario="D"))
        # Inject the failing tool into the registry
        orchestrator.tool_registry[AgentActionType.INSPECT_SCENE] = failing_tool

        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Tool failure handled gracefully
        self.assertGreaterEqual(len(state.errors), 1)
        self.assertIn("Hardware camera interface timeout", state.errors[0])

        # Trace recorded FAILED status
        failed_trace = state.trace[0]
        self.assertEqual(failed_trace.status, "FAILED")
        self.assertIn("failed", failed_trace.error_message)

        # Scenario D model detected the error and recovered via human_review
        self.assertTrue(state.is_terminated)
        self.assertIn(AgentActionType.HUMAN_REVIEW.value, state.completed_actions)
        self.assertEqual(state.trace[1].action, AgentActionType.HUMAN_REVIEW.value)
        self.assertEqual(state.trace[1].status, "SUCCESS")

    # -------------------------------------------------------------------------
    # Test 7: Second-look scenario (request observation -> second view -> registration)
    # -------------------------------------------------------------------------
    def test_07_second_look_scenario(self):
        """Test 7: Scenario C executes initial pass, requests observation, injects view, and registers."""
        # Synthesize a slightly transformed second observation
        h, w = self.base_img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
        second_img = cv2.warpAffine(self.base_img, M, (w, h))

        mock_model = MockModel(scenario="C", second_image_source=second_img)
        orchestrator = AgentOrchestrator(model=mock_model, default_poster_text="NOVA COFFEE")

        state = orchestrator.run_investigation(
            primary_image=self.base_img,
            second_image=second_img,
            poster_text="NOVA COFFEE",
        )

        # Verify expected flow of actions executed
        self.assertIn(AgentActionType.INSPECT_SCENE.value, state.completed_actions)
        self.assertIn(AgentActionType.ANALYZE_QR.value, state.completed_actions)
        self.assertIn(AgentActionType.REQUEST_OBSERVATION.value, state.completed_actions)
        self.assertIn(AgentActionType.ANALYZE_SECOND_VIEW.value, state.completed_actions)
        self.assertIn(AgentActionType.HUMAN_REVIEW.value, state.completed_actions)

        # Verify second view registration passed
        sec_ev = state.structured_evidence.second_view
        self.assertIsNotNone(sec_ev)
        self.assertEqual(sec_ev.registration_status, RegistrationStatus.PASS)
        self.assertTrue(sec_ev.homography_found)
        self.assertGreater(sec_ev.ransac_inliers, 15)
        self.assertIsNotNone(sec_ev.comparison)

        # Pending observation request recorded
        self.assertIsNotNone(state.pending_observation_request)
        self.assertEqual(
            state.pending_observation_request.get("observation_type"),
            ObservationRequestType.FLASH.value,
        )

    # -------------------------------------------------------------------------
    # Test 8: Deterministic evidence remains authoritative
    # -------------------------------------------------------------------------
    def test_08_deterministic_evidence_authoritative(self):
        """Test 8: Agent reason/claims cannot dictate fraud state; deterministic rules remain authoritative."""
        # Script a model claiming 100% fraud in its reason on a perfectly consistent prop
        model = CustomScriptedModel([
            {
                "action": AgentActionType.INSPECT_SCENE.value,
                "arguments": {},
                "reason": "This poster looks completely fake and fraudulent!",
            },
            {
                "action": AgentActionType.ANALYZE_QR.value,
                "arguments": {},
                "reason": "QR definitely generated by scammers.",
            },
            {
                "action": AgentActionType.VALIDATE_IDENTITY.value,
                "arguments": {"poster_text": "NOVA COFFEE"},
                "reason": "Payee name mismatch fraud detected.",
            },
            {
                "action": AgentActionType.ANALYZE_QR_SURFACE.value,
                "arguments": {},
                "reason": "Physical sticker confirmed with 99.9% probability.",
            },
            {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Confirm my fraud detection."},
                "reason": "Closing with fraud finding.",
            },
        ])

        orchestrator = AgentOrchestrator(model=model, default_poster_text="NOVA COFFEE")
        state = orchestrator.run_investigation(primary_image=self.base_img, poster_text="NOVA COFFEE")

        # Despite the model's subjective claims of fraud, the genuine prop has consistent identity and no physical anomaly
        # Therefore, the authoritative deterministic evidence state must be CONSISTENT!
        self.assertEqual(state.final_state, EvidenceState.CONSISTENT)
        self.assertEqual(state.structured_evidence.evaluation.state, EvidenceState.CONSISTENT)
        self.assertNotEqual(state.final_state, EvidenceState.HIGH_RISK)

    # -------------------------------------------------------------------------
    # Test 9: No fabricated evidence after tool failure
    # -------------------------------------------------------------------------
    def test_09_no_fabricated_evidence_after_tool_failure(self):
        """Test 9: Failed tool executions do not produce fabricated or hallucinated evidence values."""
        def broken_qr_analyzer(state, args, poster_text):
            raise ValueError("Corrupt QR stream")

        model = CustomScriptedModel([
            {
                "action": AgentActionType.ANALYZE_QR.value,
                "arguments": {},
                "reason": "Attempting QR analysis on corrupted source.",
            },
            {
                "action": AgentActionType.HUMAN_REVIEW.value,
                "arguments": {"rationale": "Tool failed."},
                "reason": "Escalate safely.",
            },
        ])

        orchestrator = AgentOrchestrator(model=model)
        orchestrator.tool_registry[AgentActionType.ANALYZE_QR] = broken_qr_analyzer

        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Verify no fabricated values in QR evidence
        qr_ev = state.structured_evidence.qr
        self.assertFalse(qr_ev.detected)
        self.assertFalse(qr_ev.decoded)
        self.assertEqual(qr_ev.payload, "")
        self.assertIsNone(qr_ev.bounding_box)

    # -------------------------------------------------------------------------
    # Test 10: Trace is generated with timestamps, actions, evidence changes
    # -------------------------------------------------------------------------
    def test_10_trace_generation_and_serialization(self):
        """Test 10: Trace events contain timestamps, actions, reasons, statuses, and serialize to JSON."""
        orchestrator = AgentOrchestrator(
            model=MockModel(scenario="A"),
            default_poster_text="NOVA COFFEE",
        )
        state = orchestrator.run_investigation(primary_image=self.base_img, poster_text="NOVA COFFEE")

        self.assertGreaterEqual(len(state.trace), 4)

        for idx, event in enumerate(state.trace, start=1):
            self.assertEqual(event.step, idx)
            self.assertIn(event.action, [a.value for a in ALLOWED_ACTIONS] + ["budget_limit"])
            self.assertIsInstance(event.arguments, dict)
            self.assertTrue(len(event.reason.strip()) > 0)
            self.assertIn(event.status, ["SUCCESS", "FAILED", "REJECTED"])
            self.assertTrue(len(event.evidence_change.strip()) > 0)
            self.assertTrue(len(event.timestamp.strip()) > 0)
            self.assertIn("T", event.timestamp)  # ISO 8601 format

            # Dict serialization check
            d = event.to_dict()
            self.assertEqual(d["step"], idx)
            self.assertIn("status", d)

        # Entire state serialization check
        state_dict = state.to_dict()
        self.assertEqual(state_dict["investigation_id"], state.investigation_id)
        self.assertIsInstance(state_dict["trace"], list)
        self.assertEqual(len(state_dict["trace"]), len(state.trace))

        json_str = state.to_json()
        self.assertIn(state.investigation_id, json_str)
        self.assertIn("trace", json_str)

    # -------------------------------------------------------------------------
    # Test 11: Only 8 allowed actions can execute
    # -------------------------------------------------------------------------
    def test_11_only_eight_allowed_actions_can_execute(self):
        """Test 11: Whitelist enforces EXACTLY the 8 approved forensic actions."""
        # Verify the whitelist contains exactly 8 actions
        self.assertEqual(len(ALLOWED_ACTIONS), 8)
        expected_actions = {
            "inspect_scene",
            "analyze_qr",
            "extract_region",
            "validate_identity",
            "analyze_qr_surface",
            "request_observation",
            "analyze_second_view",
            "human_review",
        }
        self.assertEqual({a.value for a in ALLOWED_ACTIONS}, expected_actions)

        # Verify all 8 are accepted by validate_action_decision
        for act in ALLOWED_ACTIONS:
            decision = validate_action_decision({
                "action": act.value,
                "arguments": {},
                "reason": f"Testing action {act.value}",
            })
            self.assertEqual(decision.action, act)
            is_auth, _ = SecurityPolicy.validate_action_authorization(decision)
            self.assertTrue(is_auth)

        # Prohibited actions that MUST be rejected
        prohibited_actions = [
            "calculate_fraud",
            "classify_fraud",
            "retry_decode",
            "analyze_url",
            "register_image",
            "predict_risk",
            "execute_code",
            "scan_database",
        ]

        for bad_action in prohibited_actions:
            with self.assertRaises(SchemaValidationError):
                validate_action_decision({
                    "action": bad_action,
                    "arguments": {},
                    "reason": "Attempting prohibited action.",
                })


if __name__ == "__main__":
    unittest.main()
