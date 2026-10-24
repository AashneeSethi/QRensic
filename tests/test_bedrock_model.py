"""
QRensic Agent - Amazon Bedrock Model Adapter Test Suite
-------------------------------------------------------
Comprehensive mocked unit tests validating the Bedrock Runtime Converse API adapter:
 1. Valid Bedrock tool-use response
 2. Multiple tool-use cycles
 3. Correct tool-result handoff
 4. Invalid JSON arguments
 5. Unknown or prohibited action
 6. Bedrock timeout and throttling
 7. Malformed or unexpected model response
 8. Iteration-limit enforcement
 9. Deterministic evidence rules remaining authoritative
10. Mock model continuing to work without AWS credentials

CRITICAL RULE:
No real AWS API calls are made during unit tests. All Bedrock interactions are mocked.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import MagicMock

from botocore.exceptions import ClientError, ReadTimeoutError
import cv2

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType, ObservationRequestType
from backend.agent.bedrock_model import (
    BEDROCK_FORENSIC_TOOLS,
    BedrockModel,
    BedrockModelError,
)
from backend.agent.mock_model import MockModel
from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.policy import MAX_AGENT_ITERATIONS
from backend.agent.schemas import (
    ActionDecision,
    InvestigationState,
    SchemaValidationError,
    validate_action_decision,
)
from backend.vision.models import (
    EvidenceState,
    RecommendedAction,
    RegistrationStatus,
)


def make_converse_response(
    tool_use_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_input: Optional[Dict[str, Any]] = None,
    text: Optional[str] = None,
    stop_reason: str = "tool_use",
) -> Dict[str, Any]:
    """Helper to build a realistic Amazon Bedrock Converse API response dictionary."""
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"text": text})
    if tool_use_id and tool_name:
        content.append({
            "toolUse": {
                "toolUseId": tool_use_id,
                "name": tool_name,
                "input": tool_input if tool_input is not None else {},
            }
        })
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content,
            }
        },
        "stopReason": stop_reason,
    }


class TestBedrockModelAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.props_dir = cls.repo_root / "feasibility" / "props" / "generated"
        cls.genuine_qr_path = cls.props_dir / "genuine_qr.png"

        if not cls.genuine_qr_path.is_file():
            from feasibility.generate_props import main as gen_props
            gen_props()

        cls.base_img = cv2.imread(str(cls.genuine_qr_path))
        assert cls.base_img is not None, "Failed to load genuine_qr.png for Bedrock tests"

    # -------------------------------------------------------------------------
    # Test 1: Valid Bedrock tool-use response
    # -------------------------------------------------------------------------
    def test_01_valid_bedrock_tool_use_response(self):
        """Test 1: Valid toolUse from Converse API parses correctly into an ActionDecision."""
        mock_client = MagicMock()
        mock_client.converse.return_value = make_converse_response(
            tool_use_id="tooluse_001",
            tool_name="inspect_scene",
            tool_input={"image_key": "primary", "reason": "Evaluating lighting and blur."},
        )

        model = BedrockModel(client=mock_client)
        state = InvestigationState(investigation_id="test_inv_01")
        decision_dict = model.decide_next_action(state)

        self.assertEqual(decision_dict["action"], "inspect_scene")
        self.assertEqual(decision_dict["arguments"]["image_key"], "primary")
        self.assertEqual(decision_dict["reason"], "Evaluating lighting and blur.")

        # Verify ActionDecision validation succeeds
        decision = validate_action_decision(decision_dict)
        self.assertIsInstance(decision, ActionDecision)
        self.assertEqual(decision.action, AgentActionType.INSPECT_SCENE)
        self.assertEqual(model.last_tool_use_id, "tooluse_001")

        # Verify tool definitions sent to Converse match the 8 permitted tools
        call_kwargs = mock_client.converse.call_args[1]
        tools_sent = call_kwargs["toolConfig"]["tools"]
        self.assertEqual(len(tools_sent), 8)
        tool_names = [t["toolSpec"]["name"] for t in tools_sent]
        self.assertEqual(set(tool_names), {a.value for a in ALLOWED_ACTIONS})

    # -------------------------------------------------------------------------
    # Test 2: Multiple tool-use cycles
    # -------------------------------------------------------------------------
    def test_02_multiple_tool_use_cycles(self):
        """Test 2: Multi-turn tool execution loop runs smoothly with Bedrock adapter."""
        mock_client = MagicMock()
        # Sequence of assistant tool responses
        mock_client.converse.side_effect = [
            make_converse_response(
                tool_use_id="tool_1",
                tool_name="inspect_scene",
                tool_input={"image_key": "primary", "reason": "Scene optical inspection."},
            ),
            make_converse_response(
                tool_use_id="tool_2",
                tool_name="analyze_qr",
                tool_input={"image_key": "primary", "reason": "Decoding QR code."},
            ),
            make_converse_response(
                tool_use_id="tool_3",
                tool_name="human_review",
                tool_input={"rationale": "Investigation complete.", "reason": "Closing case."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        self.assertEqual(mock_client.converse.call_count, 3)
        self.assertEqual(state.action_count, 3)
        self.assertTrue(state.is_terminated)
        self.assertEqual(
            state.completed_actions,
            ["inspect_scene", "analyze_qr", "human_review"],
        )
        self.assertEqual(len(state.trace), 3)
        for t in state.trace:
            self.assertEqual(t.status, "SUCCESS")

    # -------------------------------------------------------------------------
    # Test 3: Correct tool-result handoff
    # -------------------------------------------------------------------------
    def test_03_correct_tool_result_handoff(self):
        """Test 3: Orchestrator hands tool execution results back to Bedrock via toolResult message."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            make_converse_response(
                tool_use_id="tool_step_1",
                tool_name="inspect_scene",
                tool_input={"image_key": "primary", "reason": "First step."},
            ),
            make_converse_response(
                tool_use_id="tool_step_2",
                tool_name="human_review",
                tool_input={"rationale": "Done.", "reason": "Finished."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        orchestrator.run_investigation(primary_image=self.base_img)

        # Inspect the conversation history recorded on model.messages
        # Total messages: 5 (initial user, assistant tool 1, user toolResult 1, assistant tool 2, user toolResult 2)
        self.assertEqual(len(model.messages), 5)

        # Message 0: User initial instruction
        self.assertEqual(model.messages[0]["role"], "user")

        # Message 1: Assistant toolUse request for step 1
        self.assertEqual(model.messages[1]["role"], "assistant")
        self.assertEqual(
            model.messages[1]["content"][0]["toolUse"]["toolUseId"],
            "tool_step_1",
        )

        # Message 2: User toolResult payload handed back for step 1
        tool_result_msg = model.messages[2]
        self.assertEqual(tool_result_msg["role"], "user")
        tool_result_block = tool_result_msg["content"][0]["toolResult"]
        self.assertEqual(tool_result_block["toolUseId"], "tool_step_1")
        self.assertEqual(tool_result_block["status"], "success")

        # Verify json content inside toolResult
        json_content = tool_result_block["content"][0]["json"]
        self.assertEqual(json_content["action"], "inspect_scene")
        self.assertEqual(json_content["status"], "SUCCESS")
        self.assertIn("Scene inspected", json_content["evidence_change"])

    # -------------------------------------------------------------------------
    # Test 4: Invalid JSON arguments
    # -------------------------------------------------------------------------
    def test_04_invalid_json_arguments(self):
        """Test 4: Malformed or invalid arguments from Bedrock are rejected safely by the orchestrator."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            # Call 1: invalid observation_type not in enum
            make_converse_response(
                tool_use_id="tool_bad_args",
                tool_name="request_observation",
                tool_input={
                    "observation_type": "invalid_infrared_modality",
                    "prompt_to_user": "Please use infrared.",
                    "reason": "Testing invalid argument enum.",
                },
            ),
            # Call 2: safe recovery
            make_converse_response(
                tool_use_id="tool_recover",
                tool_name="human_review",
                tool_input={"rationale": "Recover from bad args.", "reason": "Safe exit."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Step 1 must be recorded as REJECTED without crashing
        self.assertEqual(state.trace[0].status, "REJECTED")
        self.assertEqual(state.trace[0].action, "request_observation")
        self.assertIn("Invalid observation_type", state.errors[0])

        # Step 2 recovered cleanly
        self.assertEqual(state.trace[1].status, "SUCCESS")
        self.assertEqual(state.trace[1].action, "human_review")
        self.assertTrue(state.is_terminated)

    # -------------------------------------------------------------------------
    # Test 5: Unknown or prohibited action
    # -------------------------------------------------------------------------
    def test_05_unknown_or_prohibited_action(self):
        """Test 5: Prohibited action (e.g. calculate_fraud) requested by Bedrock is strictly rejected."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            make_converse_response(
                tool_use_id="tool_prohibited",
                tool_name="calculate_fraud",
                tool_input={"target": "qr_code", "reason": "Attempting fraud score calculation."},
            ),
            make_converse_response(
                tool_use_id="tool_safe_conclude",
                tool_name="human_review",
                tool_input={"rationale": "Closing.", "reason": "Conclude safely."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        self.assertEqual(state.trace[0].status, "REJECTED")
        self.assertEqual(state.trace[0].action, "calculate_fraud")
        self.assertIn("calculate_fraud", state.errors[0])
        self.assertNotIn("calculate_fraud", state.completed_actions)

    # -------------------------------------------------------------------------
    # Test 6: Bedrock timeout and throttling
    # -------------------------------------------------------------------------
    def test_06_bedrock_timeout_and_throttling(self):
        """Test 6: Transient errors (throttling, read timeouts) are caught gracefully without fabricating evidence."""
        # 6a: ThrottlingException
        mock_client_throttled = MagicMock()
        mock_client_throttled.converse.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate limit exceeded."}},
            "Converse",
        )

        model_throttled = BedrockModel(client=mock_client_throttled)
        orchestrator = AgentOrchestrator(model=model_throttled)
        state_throttled = orchestrator.run_investigation(primary_image=self.base_img)

        # Budget exhaustion will escalate after 5 failed attempts
        self.assertTrue(state_throttled.is_terminated)
        self.assertEqual(state_throttled.action_count, MAX_AGENT_ITERATIONS)
        self.assertIn("ThrottlingException", state_throttled.errors[0])
        self.assertEqual(state_throttled.trace[0].status, "FAILED")
        self.assertEqual(state_throttled.final_state, EvidenceState.HUMAN_REVIEW)

        # 6b: ReadTimeoutError
        mock_client_timeout = MagicMock()
        mock_client_timeout.converse.side_effect = ReadTimeoutError(
            endpoint_url="https://bedrock.us-east-1.amazonaws.com",
            error="Socket timed out",
        )

        model_timeout = BedrockModel(client=mock_client_timeout)
        orchestrator_to = AgentOrchestrator(model=model_timeout)
        state_to = orchestrator_to.run_investigation(primary_image=self.base_img)

        self.assertTrue(state_to.is_terminated)
        self.assertIn("network/transport error", state_to.errors[0])
        self.assertEqual(state_to.trace[0].status, "FAILED")

    # -------------------------------------------------------------------------
    # Test 7: Malformed or unexpected model response
    # -------------------------------------------------------------------------
    def test_07_malformed_or_unexpected_model_response(self):
        """Test 7: Model responding with plain conversational text without toolUse is rejected safely."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            # Plain text only (no toolUse block)
            make_converse_response(
                text="The QR poster looks genuine to me. I see no issues.",
                stop_reason="end_turn",
            ),
            # Safe follow-up
            make_converse_response(
                tool_use_id="tool_recover_text",
                tool_name="human_review",
                tool_input={"rationale": "Specialist check.", "reason": "Recovery."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        # Step 1 rejected because no permitted tool was invoked
        self.assertEqual(state.trace[0].status, "REJECTED")
        self.assertIn("unauthorized", state.trace[0].action)

        # Step 2 completed cleanly
        self.assertEqual(state.trace[1].status, "SUCCESS")
        self.assertEqual(state.trace[1].action, "human_review")
        self.assertTrue(state.is_terminated)

    # -------------------------------------------------------------------------
    # Test 8: Iteration-limit enforcement
    # -------------------------------------------------------------------------
    def test_08_iteration_limit_enforcement(self):
        """Test 8: If Bedrock keeps requesting tools indefinitely, loop strictly halts at 5 iterations."""
        mock_client = MagicMock()
        # Always return inspect_scene
        mock_client.converse.return_value = make_converse_response(
            tool_use_id="tool_loop",
            tool_name="inspect_scene",
            tool_input={"image_key": "primary", "reason": "Continuous loop check."},
        )

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model)
        state = orchestrator.run_investigation(primary_image=self.base_img)

        self.assertEqual(state.action_count, MAX_AGENT_ITERATIONS)
        self.assertTrue(state.is_terminated)
        self.assertEqual(state.final_state, EvidenceState.HUMAN_REVIEW)
        self.assertEqual(state.final_recommendation, RecommendedAction.HUMAN_REVIEW)

        budget_events = [t for t in state.trace if t.action == "budget_limit"]
        self.assertGreaterEqual(len(budget_events), 1)

    # -------------------------------------------------------------------------
    # Test 9: Deterministic evidence rules remaining authoritative
    # -------------------------------------------------------------------------
    def test_09_deterministic_evidence_rules_authoritative(self):
        """Test 9: Model reason claiming fraud cannot override deterministic evidence rules."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            make_converse_response(
                tool_use_id="t1",
                tool_name="inspect_scene",
                tool_input={"reason": "Scam detected!"},
            ),
            make_converse_response(
                tool_use_id="t2",
                tool_name="analyze_qr",
                tool_input={"reason": "Fake QR confirmed!"},
            ),
            make_converse_response(
                tool_use_id="t3",
                tool_name="validate_identity",
                tool_input={"poster_text": "NOVA COFFEE", "reason": "Total fraud!"},
            ),
            make_converse_response(
                tool_use_id="t4",
                tool_name="analyze_qr_surface",
                tool_input={"reason": "Sticker 100% verified!"},
            ),
            make_converse_response(
                tool_use_id="t5",
                tool_name="human_review",
                tool_input={"rationale": "I declared fraud.", "reason": "Closing."},
            ),
        ]

        model = BedrockModel(client=mock_client)
        orchestrator = AgentOrchestrator(model=model, default_poster_text="NOVA COFFEE")
        state = orchestrator.run_investigation(primary_image=self.base_img, poster_text="NOVA COFFEE")

        # Despite the model's claims of fraud, the genuine prop has consistent identity and no physical anomaly
        # Therefore, deterministic evidence engine evaluates CONSISTENT, not HIGH_RISK!
        self.assertEqual(state.final_state, EvidenceState.CONSISTENT)
        self.assertNotEqual(state.final_state, EvidenceState.HIGH_RISK)

    # -------------------------------------------------------------------------
    # Test 10: Mock model continuing to work without AWS credentials
    # -------------------------------------------------------------------------
    def test_10_mock_model_works_without_aws_credentials(self):
        """Test 10: MockModel operates 100% offline without any AWS credentials or boto3 configuration."""
        # Ensure MockModel works across scenarios A, B, C, D, E without AWS credentials
        for sc in ["A", "B", "C", "D", "E"]:
            mock = MockModel(scenario=sc)
            orch = AgentOrchestrator(model=mock, default_poster_text="NOVA COFFEE")
            st = orch.run_investigation(primary_image=self.base_img, poster_text="NOVA COFFEE")
            self.assertTrue(st.is_terminated)
            self.assertGreaterEqual(st.action_count, 1)
            self.assertLessEqual(st.action_count, MAX_AGENT_ITERATIONS)


if __name__ == "__main__":
    unittest.main()
