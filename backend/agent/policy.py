"""
QRensic Agent - Policy & Guardrails
------------------------------------
Enforces execution bounds, maximum iteration limits, and security constraints
on agent action execution.

SAFETY GUARANTEES:
1. Maximum 5 iterations per investigation session.
2. Strict action whitelist (only the 8 approved actions).
3. The agent/model cannot declare fraud or assign fraud probability scores.
4. human_review terminates the loop safely with evidence preserved.
5. No arbitrary tool invocation or code execution.
"""

from typing import Tuple

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType
from backend.agent.schemas import ActionDecision, InvestigationState
from backend.vision.models import EvidenceState


MAX_AGENT_ITERATIONS = 5


class SecurityPolicy:
    """Security guardrails governing bounded agent execution."""

    @staticmethod
    def validate_execution_budget(state: InvestigationState) -> Tuple[bool, str]:
        """
        Ensure the investigation has not exceeded the strict iteration budget.
        """
        if state.action_count >= MAX_AGENT_ITERATIONS:
            return False, f"Maximum iteration limit ({MAX_AGENT_ITERATIONS}) reached. Escalating to human specialist."
        return True, ""

    @staticmethod
    def validate_action_authorization(decision: ActionDecision) -> Tuple[bool, str]:
        """
        Ensure the requested action is strictly within the allowed 8 forensic tools.
        """
        if decision.action not in ALLOWED_ACTIONS:
            return False, f"Action '{decision.action}' is unauthorized and prohibited."
        return True, ""

    @staticmethod
    def check_termination_condition(
        decision: ActionDecision, state: InvestigationState
    ) -> Tuple[bool, str]:
        """
        Check if the action triggers an immediate forensic termination.
        """
        if decision.action == AgentActionType.HUMAN_REVIEW:
            return True, "Human specialist review explicitly requested by investigator."
        return False, ""
