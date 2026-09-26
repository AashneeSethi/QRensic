#!/usr/bin/env python3
"""
QRensic Feasibility Study - Synthetic QR Prop Generator
-------------------------------------------------------
Generates harmless synthetic QR code images for physical poster and sticker
feasibility experiments.

SAFETY NOTICE:
Only controlled, non-financial test strings are used. Real UPI strings,
bank details, live payment destinations, and phishing URLs are strictly avoided.
"""

from pathlib import Path
import qrcode
from qrcode.image.pil import PilImage

# Controlled test payloads
GENUINE_PAYLOAD = "QRFORENSIC_TEST_PAYEE_NOVA_COFFEE"
STICKER_PAYLOAD = "QRFORENSIC_TEST_PAYEE_RAMESH_KUMAR"


def generate_qr(payload: str, output_path: Path, box_size: int = 10, border: int = 4) -> None:
    """Generate and save a single QR code image from a synthetic payload."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white", image_factory=PilImage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"[+] Generated: {output_path} (payload: {payload})")


def main() -> None:
    # Anchor paths relative to this script
    script_dir = Path(__file__).resolve().parent
    generated_dir = script_dir / "props" / "generated"

    genuine_qr_path = generated_dir / "genuine_qr.png"
    sticker_qr_path = generated_dir / "sticker_qr.png"

    print("=== QRensic Synthetic Prop Generator ===")
    print("Generating controlled, harmless test QR codes...")

    generate_qr(GENUINE_PAYLOAD, genuine_qr_path)
    generate_qr(STICKER_PAYLOAD, sticker_qr_path)

    print("\n[OK] Prop generation complete.")
    print("Print these QRs onto matte/glossy paper and sticker paper for physical testing.")


if __name__ == "__main__":
    main()
