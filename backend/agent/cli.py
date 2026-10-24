"""
QRensic Agent - Command Line Interface
---------------------------------------
Executes local agent-orchestrated forensic investigations using the MockModel.

Usage:
  python -m backend.agent.cli <primary_image> [--second-image <path>] [--scenario A|B|C|D|E] [--poster-text <text>] [--json]
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

from backend.agent.bedrock_model import BedrockModel, BedrockModelError
from backend.agent.mock_model import MockModel
from backend.agent.orchestrator import AgentOrchestrator


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="QRensic - Local Agent-Orchestrated Forensic Investigation CLI"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the primary observation image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mock", "bedrock"],
        default="mock",
        help="Reasoning model backend: 'mock' (default offline simulation) or 'bedrock' (AWS Bedrock Runtime Converse API).",
    )
    parser.add_argument(
        "--bedrock-model-id",
        type=str,
        default=None,
        help="Amazon Bedrock model ID (defaults to env BEDROCK_MODEL_ID or anthropic.claude-3-haiku-20240307-v1:0).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region for Bedrock (defaults to env AWS_REGION or us-east-1).",
    )
    parser.add_argument(
        "--second-image",
        type=str,
        default=None,
        help="Path to optional second observation image (e.g. angled or flash photo).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        default="A",
        help="MockModel scenario: A (Consistent), B (Ambiguous), C (Second-Look Hero), D (Failure Recovery), E (Invalid Action Rejection). Default: A.",
    )
    parser.add_argument(
        "--poster-text",
        type=str,
        default=None,
        help="Visible merchant branding text on the poster (simulated OCR input).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON serialized state instead of human-readable trace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save complete investigation state JSON to specified file path.",
    )

    args = parser.parse_args(argv)

    primary_path = Path(args.image_path)
    if not primary_path.is_file():
        print(f"[ERROR] Primary image not found: {primary_path}", file=sys.stderr)
        return 1

    second_path = Path(args.second_image) if args.second_image else None
    if second_path and not second_path.is_file():
        print(f"[ERROR] Second image not found: {second_path}", file=sys.stderr)
        return 1

    # Instantiate reasoning model backend
    second_source = str(second_path) if second_path else None
    if args.model == "bedrock":
        try:
            model = BedrockModel(
                model_id=args.bedrock_model_id,
                region_name=args.aws_region,
                second_image_source=second_source,
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize Amazon Bedrock adapter: {e}", file=sys.stderr)
            return 1
    else:
        model = MockModel(
            scenario=args.scenario,
            second_image_source=second_source,
        )

    orchestrator = AgentOrchestrator(
        model=model,
        default_poster_text=args.poster_text,
    )

    state = orchestrator.run_investigation(
        primary_image=str(primary_path),
        second_image=str(second_path) if second_path else None,
        poster_text=args.poster_text,
    )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(state.to_json(indent=2), encoding="utf-8")
        print(f"[OK] Investigation state saved to {out_p}")

    if args.json:
        print(state.to_json(indent=2))
        return 0

    # Human-readable formatted forensic report
    print("\n========================================================")
    print("  Q R E N S I C   F O R E N S I C   I N V E S T I G A T I O N")
    print("========================================================")
    print(f"Investigation ID : {state.investigation_id}")
    print(f"Primary Image    : {primary_path.name}")
    if second_path:
        print(f"Second Image     : {second_path.name}")
    if args.model == "bedrock":
        print(f"Model Backend    : Amazon Bedrock ({model.model_id} in {model.region_name})")
    else:
        print(f"Model Backend    : MockModel (Scenario {args.scenario})")
    print(f"Action Steps     : {state.action_count} / 5")
    print(f"Terminated       : {state.is_terminated}")

    print("\n--------------------------------------------------------")
    print("  DETERMINISTIC EVIDENCE CONCLUSION")
    print("--------------------------------------------------------")
    final_state_str = state.final_state.value if state.final_state else "UNKNOWN"
    rec_str = state.final_recommendation.value if state.final_recommendation else "UNKNOWN"
    print(f"Final Evidence State  : [{final_state_str}]")
    print(f"Recommended Next Step : [{rec_str}]")

    eval_data = state.structured_evidence.evaluation
    if eval_data and eval_data.reasons:
        print("\nObservable Forensic Rationale:")
        for r in eval_data.reasons:
            print(f"  * {r}")

    if eval_data and eval_data.signals_summary:
        print("\nMeasured Evidence Signals:")
        for k, v in eval_data.signals_summary.items():
            print(f"  * {k:20s}: {v}")

    print("\n--------------------------------------------------------")
    print("  INVESTIGATION ACTION TRACE")
    print("--------------------------------------------------------")
    for event in state.trace:
        status_tag = f"[{event.status}]"
        print(f"Step {event.step} {status_tag:10s} Action: {event.action}")
        print(f"  Reason : {event.reason}")
        print(f"  Effect : {event.evidence_change}")
        if event.error_message:
            print(f"  Error  : {event.error_message}")
        print()

    if state.pending_observation_request:
        print("--------------------------------------------------------")
        print("  PENDING OBSERVATION REQUEST")
        print("--------------------------------------------------------")
        req = state.pending_observation_request
        print(f"Type   : {req.get('observation_type')}")
        print(f"Prompt : {req.get('prompt_to_user')}")
        print()

    print("========================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
