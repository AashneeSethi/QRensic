#!/usr/bin/env python3
"""
QRensic Vision Engine - CLI Entry Point
---------------------------------------
Command-line interface to inspect an image through the vision pipeline
and output serialized structured evidence JSON.

Usage:
  python -m backend.vision.cli <image_path> [--poster-text "..."] [--second-image "..."]
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vision.engine import QRensicVisionEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QRensic Vision Engine: Extract deterministic forensic evidence from a QR image"
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the primary photograph or synthetic QR image to analyze",
    )
    parser.add_argument(
        "--poster-text",
        type=str,
        default=None,
        help="Optional visible merchant brand name observed on the poster (simulating OCR)",
    )
    parser.add_argument(
        "--second-image",
        type=Path,
        default=None,
        help="Optional second observation image (e.g., flash photograph) for multi-view comparison",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for formatted output (default: 2)",
    )

    args = parser.parse_args()

    if not args.image_path.is_file():
        print(f"[Error] Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    engine = QRensicVisionEngine()
    evidence = engine.process_image(
        image_input=args.image_path,
        poster_text=args.poster_text,
        second_image_input=args.second_image,
    )

    # Print serialized structured evidence JSON to stdout
    print(evidence.to_json(indent=args.indent))


if __name__ == "__main__":
    main()
