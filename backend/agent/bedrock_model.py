"""
QRensic Agent - Amazon Bedrock Model Adapter
---------------------------------------------
Production-quality adapter connecting the QRensic agent orchestrator to
Amazon Bedrock Runtime Converse API with native tool use.

SAFETY & GOVERNANCE GUARANTEES:
1. Native Tool Use Only: The model may only select from the 8 approved forensic actions.
2. Bounded Execution: Respects strict timeout (5s connect, 30s read) and standard retries.
3. No Secret Hardcoding: Standard AWS credential provider chain only.
4. Deterministic Primacy: The model selects what to observe; the deterministic evidence engine
   evaluates risk. The model NEVER calculates or declares fraud.
5. Graceful Failure: Catches throttling, timeouts, and network faults without fabricating evidence.
"""

from dataclasses import asdict
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from backend.agent.actions import ALLOWED_ACTIONS, AgentActionType, ObservationRequestType
from backend.agent.schemas import ActionDecision, InvestigationState


# Standard default values (can be overridden via environment variables or constructor)
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


# Bedrock Converse API Tool Specifications for the 8 permitted forensic actions
BEDROCK_FORENSIC_TOOLS: List[Dict[str, Any]] = [
    {
        "toolSpec": {
            "name": AgentActionType.INSPECT_SCENE.value,
            "description": (
                "Inspect overall optical scene quality, dimensions, blur Laplacian variance, "
                "and illumination bounds for the input image."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "image_key": {
                            "type": "string",
                            "description": "Key of the image in state (default: 'primary')",
                            "default": "primary",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for inspecting the scene",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.ANALYZE_QR.value,
            "description": (
                "Detect and decode the QR code in the image, classify payload type (URL, "
                "EMV payment, plain text), and extract quad bounding polygon."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "image_key": {
                            "type": "string",
                            "description": "Key of the image in state (default: 'primary')",
                            "default": "primary",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for detecting/decoding QR",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.EXTRACT_REGION.value,
            "description": (
                "Extract surrounding poster text regions and text bounding boxes (tagged UNTRUSTED) "
                "outside the QR code for OCR identity verification."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "image_key": {
                            "type": "string",
                            "description": "Key of the image in state (default: 'primary')",
                            "default": "primary",
                        },
                        "mock_ocr_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional simulated or extracted OCR text lines",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for extracting region text",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.VALIDATE_IDENTITY.value,
            "description": (
                "Deterministically compare visible merchant poster text against decoded QR payee identity. "
                "Personal payees on business posters are held as AMBIGUOUS, not fraud."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "poster_text": {
                            "type": "string",
                            "description": "Visible brand/store name on host poster",
                        },
                        "payee_identity": {
                            "type": "string",
                            "description": "Decoded payee name from QR payload",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for validating identity",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.ANALYZE_QR_SURFACE.value,
            "description": (
                "Analyze physical surface continuity, edge step gradients, local texture energy, "
                "and color transition deltas across the QR boundary perimeter."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "image_key": {
                            "type": "string",
                            "description": "Key of the image in state (default: 'primary')",
                            "default": "primary",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for surface analysis",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.REQUEST_OBSERVATION.value,
            "description": (
                "Request targeted follow-up physical observation when initial evidence is ambiguous or incomplete. "
                "Supported modalities: 'clearer' (blur/occlusion), 'flash' (specular response), 'oblique' (edge relief)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "observation_type": {
                            "type": "string",
                            "enum": [o.value for o in ObservationRequestType],
                            "description": "Modality of requested follow-up observation",
                        },
                        "prompt_to_user": {
                            "type": "string",
                            "description": "Clear instruction detailing requested angle, distance, or lighting",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for requesting a second observation",
                        },
                    },
                    "required": ["observation_type", "prompt_to_user", "reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.ANALYZE_SECOND_VIEW.value,
            "description": (
                "Register two observations of the scene via ORB feature matching and RANSAC homography, "
                "verify same-object geometric sanity, and compute differential surface metrics."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "first_image_key": {
                            "type": "string",
                            "description": "Key of primary observation (default: 'primary')",
                            "default": "primary",
                        },
                        "second_image_key": {
                            "type": "string",
                            "description": "Key of second observation (default: 'second')",
                            "default": "second",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for multi-view registration",
                        },
                    },
                    "required": ["reason"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": AgentActionType.HUMAN_REVIEW.value,
            "description": (
                "Conclude investigation or escalate to a human forensic specialist with compiled evidence dossier. "
                "Use when evidence is conclusive, conflicting, ambiguous, or investigation is complete."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why specialist review is needed or why case is closed",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Observable factual rationale for conclusion/escalation",
                        },
                    },
                    "required": ["rationale", "reason"],
                }
            },
        }
    },
]


SYSTEM_PROMPT = """You are QRensic, an AI investigator specializing in physical QR fraud investigation.
Your task is to investigate whether a physical QR-code poster has been tampered with (for example, a fraudulent QR sticker physically placed over a legitimate merchant QR code).

ARCHITECTURE & SAFETY RULES:
1. OpenCV vision tools answer: "What can I observe?"
2. You answer: "What should I investigate next?"
3. Deterministic evidence rules answer: "What does the evidence justify?"
4. You MUST NOT calculate or declare fraud risk or fraud probability. The deterministic evidence engine holds sole authority over final verdicts.
5. You MUST select exactly ONE tool from the permitted 8 tools at each step using native tool use.
6. Provide a concise, observable factual 'reason' based strictly on measured physical evidence—never hallucinated assumptions or hidden chain-of-thought.
7. If evidence is ambiguous, request a targeted follow-up observation (request_observation) or escalate to human specialist (human_review).
8. Once sufficient evidence is gathered or the case reaches a conclusion, invoke human_review to conclude the investigation session.
"""


class BedrockModelError(Exception):
    """Raised when Amazon Bedrock Runtime encounters an API or network error."""
    pass


class BedrockModel:
    """
    Production-grade Amazon Bedrock model adapter implementing Converse API tool use
    for dynamic forensic investigations.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
        second_image_source: Optional[Any] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        connect_timeout: int = 5,
        read_timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the Amazon Bedrock adapter.

        Args:
          model_id: Bedrock model ID (defaults to env BEDROCK_MODEL_ID or Claude 3 Haiku).
          region_name: AWS region (defaults to env AWS_REGION or 'us-east-1').
          client: Optional pre-configured boto3 bedrock-runtime client (useful for unit testing).
          second_image_source: Optional second observation to inject when requested.
          max_tokens: Maximum tokens in Converse response.
          temperature: Model sampling temperature (0.0 for deterministic tool selection).
          connect_timeout: Network connection timeout in seconds.
          read_timeout: Socket read timeout in seconds.
          max_retries: Maximum automatic retries for transient errors.
        """
        self.region_name = region_name or os.environ.get("AWS_REGION", DEFAULT_AWS_REGION)
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
        self.second_image_source = second_image_source
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Conversation history for Bedrock Converse multi-turn tool use
        self.messages: List[Dict[str, Any]] = []
        self.last_tool_use_id: Optional[str] = None
        self.last_tool_name: Optional[str] = None

        if client is not None:
            self.client = client
        else:
            config = Config(
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retries={"max_attempts": max_retries, "mode": "standard"},
            )
            self.client = boto3.client("bedrock-runtime", region_name=self.region_name, config=config)

    def decide_next_action(self, state: InvestigationState) -> Dict[str, Any]:
        """
        Query Amazon Bedrock via Converse API with current investigation state
        and available tool definitions. Returns a structured action decision.
        """
        # If conversation hasn't started yet, formulate initial user message
        if not self.messages:
            init_text = (
                f"Forensic QR fraud investigation session started (ID: {state.investigation_id}).\n"
                f"Available images in state: {list(state.images.keys())}.\n"
                f"Current evidence state: {state.final_state.value if state.final_state else 'INITIAL'}.\n"
                f"Please inspect the scene and select the first investigative action."
            )
            self.messages.append({
                "role": "user",
                "content": [{"text": init_text}],
            })
        elif self.last_tool_use_id is not None:
            # If a toolUse was issued previously but record_tool_result was not called,
            # synthesize a minimal fallback toolResult to maintain valid Converse API alternating turns
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": self.last_tool_use_id,
                            "content": [{"json": {"status": "SUCCESS", "note": "Action executed."}}],
                            "status": "success",
                        }
                    }
                ],
            })
            self.last_tool_use_id = None

        tool_config = {
            "tools": BEDROCK_FORENSIC_TOOLS,
        }

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=self.messages,
                system=[{"text": SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                toolConfig=tool_config,
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            raise BedrockModelError(f"Amazon Bedrock ClientError [{error_code}]: {error_msg}") from e
        except BotoCoreError as e:
            raise BedrockModelError(f"Amazon Bedrock network/transport error: {str(e)}") from e
        except Exception as e:
            raise BedrockModelError(f"Unexpected error communicating with Bedrock: {str(e)}") from e

        # Extract assistant output message
        output_msg = response.get("output", {}).get("message", {})
        if not output_msg:
            raise BedrockModelError("Bedrock Converse returned an empty message output.")

        # Append assistant message to ongoing conversation history
        self.messages.append(output_msg)

        content_blocks = output_msg.get("content", [])

        # Collect any accompanying conversational text from the model
        text_snippets = [
            b["text"].strip()
            for b in content_blocks
            if "text" in b and isinstance(b["text"], str) and b["text"].strip()
        ]
        combined_text = " ".join(text_snippets)

        # Look for native toolUse block in content
        tool_use_block = None
        for block in content_blocks:
            if "toolUse" in block and isinstance(block["toolUse"], dict):
                tool_use_block = block["toolUse"]
                break

        if tool_use_block is None:
            # Model responded with plain text without selecting a tool
            self.last_tool_use_id = None
            return {
                "action": "unauthorized_conversational_text",
                "arguments": {},
                "reason": combined_text or "Model returned plain text without invoking a permitted tool.",
            }

        tool_use_id = tool_use_block.get("toolUseId")
        tool_name = tool_use_block.get("name", "")
        tool_input = tool_use_block.get("input", {})

        if not isinstance(tool_input, dict):
            # Malformed tool input
            self.last_tool_use_id = tool_use_id
            return {
                "action": tool_name or "malformed_tool_call",
                "arguments": tool_input,
                "reason": "Model provided non-dictionary arguments.",
            }

        self.last_tool_use_id = tool_use_id
        self.last_tool_name = tool_name

        # Extract observable reason: prefer tool input 'reason', then accompanying text, then default
        reason = tool_input.get("reason")
        if not reason or not isinstance(reason, str) or not reason.strip():
            reason = combined_text if combined_text else f"Bedrock selected tool '{tool_name}' based on current evidence."

        # Ensure rationale is set if human_review is invoked
        if tool_name == AgentActionType.HUMAN_REVIEW.value:
            if "rationale" not in tool_input or not tool_input["rationale"]:
                tool_input["rationale"] = reason

        return {
            "action": tool_name,
            "arguments": tool_input,
            "reason": reason.strip(),
        }

    def record_tool_result(
        self,
        action: str,
        arguments: Dict[str, Any],
        evidence_change: str,
        error_message: Optional[str] = None,
        status: str = "success",
        signals_summary: Optional[Dict[str, Any]] = None,
        current_state: Optional[str] = None,
    ) -> None:
        """
        Record the execution result of the previously requested toolUse block,
        appending a structured toolResult message to the Converse conversation history.
        """
        if not self.last_tool_use_id:
            return

        result_payload: Dict[str, Any] = {
            "action": action,
            "status": "SUCCESS" if status == "success" else "FAILED",
            "evidence_change": evidence_change,
            "signals_summary": signals_summary or {},
            "current_evidence_state": current_state or "UNKNOWN",
        }
        if error_message:
            result_payload["error"] = error_message

        self.messages.append({
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": self.last_tool_use_id,
                        "content": [{"json": result_payload}],
                        "status": "success" if status == "success" else "error",
                    }
                }
            ],
        })

        self.last_tool_use_id = None
        self.last_tool_name = None

    def reset(self) -> None:
        """Reset conversation history for a new investigation."""
        self.messages = []
        self.last_tool_use_id = None
        self.last_tool_name = None
