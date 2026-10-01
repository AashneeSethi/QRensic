#!/usr/bin/env python3
"""
QRensic Vision Engine - Second-View Registration CLI
-----------------------------------------------------
Command-line interface to register two visual observations of a QR scene
and extract multi-view surface comparison evidence.

Usage:
  python -m backend.vision.cli_second_view first.jpg second.jpg
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vision.analyze_qr import analyze_qr
from backend.vision.analyze_second_view import analyze_second_view


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QRensic: Register two visual observations and compute differential evidence"
    )
    parser.add_argument(
        "first_image",
        type=Path,
        help="Path to the initial scene photograph (e.g. ambient or straight-on view)",
    )
    parser.add_argument(
        "second_image",
        type=Path,
        help="Path to the follow-up observation photograph (e.g. angled or flash view)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for formatted output (default: 2)",
    )

    args = parser.parse_args()

    if not args.first_image.is_file():
        print(f"[Error] First image file not found: {args.first_image}", file=sys.stderr)
        sys.exit(1)

    if not args.second_image.is_file():
        print(f"[Error] Second image file not found: {args.second_image}", file=sys.stderr)
        sys.exit(1)

    # Detect QR code in first image to provide bounding box if available
    qr_ev = analyze_qr(args.first_image)
    qr_bbox = qr_ev.bounding_box if qr_ev.detected else None

    second_view_ev = analyze_second_view(
        first_image=args.first_image,
        second_image=args.second_image,
        qr_bbox=qr_bbox,
    )

    print(second_view_ev.to_json(indent=args.indent))


if __name__ == "__main__":
    main()
