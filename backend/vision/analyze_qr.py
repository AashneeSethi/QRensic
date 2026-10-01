"""
QRensic Vision Engine - QR Detection & Payload Analysis Module
---------------------------------------------------------------
Detects QR codes, decodes payloads, extracts quadrilateral bounding points,
and classifies payload categories (URL, EMV payment, synthetic test payload).
SAFETY NOTICE: Does NOT judge maliciousness or execute live payment APIs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from backend.vision.models import PayloadType, QREvidence, URLInfo


SYNTHETIC_TEST_PREFIXES = (
    "QRFORENSIC_TEST_",
    "TEST_PAYEE_",
)


def _classify_payload(payload: str) -> Tuple[PayloadType, Optional[URLInfo], Optional[Dict[str, Any]]]:
    """
    Classify a decoded QR payload and extract basic structural metadata.
    Avoids making subjective or heuristic security/maliciousness judgements.
    """
    if not payload:
        return PayloadType.UNKNOWN, None, None

    # Check for synthetic test payloads
    if any(payload.startswith(prefix) for prefix in SYNTHETIC_TEST_PREFIXES) or "TEST_PAYEE" in payload:
        return PayloadType.PLAIN_TEST, None, {"synthetic_identifier": payload}

    # Check for UPI / EMV payment formats
    if payload.startswith("upi://pay?") or payload.startswith("000201"):
        emv_info: Dict[str, Any] = {"format": "UPI" if payload.startswith("upi://") else "EMVCo"}
        if payload.startswith("upi://pay?"):
            parsed_url = urlparse(payload)
            params = parse_qs(parsed_url.query)
            # Flatten query params
            emv_info["params"] = {k: v[0] if v else "" for k, v in params.items()}
            emv_info["payee_address"] = emv_info["params"].get("pa", "")
            emv_info["payee_name"] = emv_info["params"].get("pn", "")
        return PayloadType.EMV_PAYMENT, None, emv_info

    # Check for standard URLs
    parsed = urlparse(payload)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        url_info = URLInfo(
            raw_url=payload,
            scheme=parsed.scheme.lower(),
            domain=parsed.netloc.lower(),
            path=parsed.path,
            is_https=(parsed.scheme.lower() == "https"),
        )
        return PayloadType.URL, url_info, None

    # Plain text or unrecognized payload
    return PayloadType.UNKNOWN, None, None


def analyze_qr(image_input: Union[np.ndarray, str, Path]) -> QREvidence:
    """
    Detect and decode QR code in the provided image using OpenCV 5.
    
    Returns:
      QREvidence containing detection flag, payload, corner quad points, and category.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            return QREvidence()
        img = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return QREvidence()

    if img is None or img.size == 0 or img.ndim < 2:
        return QREvidence()

    detector = cv2.QRCodeDetector()

    # Attempt detection and decoding
    decoded_text, points, _ = detector.detectAndDecode(img)

    # Fallback to grayscale if color attempt missed
    if (points is None or len(points) == 0) and img.ndim > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded_text, points, _ = detector.detectAndDecode(gray)

    # In case detection succeeded but decode failed, detect points only
    if (points is None or len(points) == 0):
        retval, points = detector.detect(img)
        if not retval or points is None:
            return QREvidence(detected=False, decoded=False)

    pts = points.reshape(-1, 2)
    if len(pts) < 4:
        return QREvidence(detected=False, decoded=False)

    bounding_box = [[float(pt[0]), float(pt[1])] for pt in pts[:4]]
    detected = True
    decoded = bool(decoded_text and len(decoded_text.strip()) > 0)
    payload = decoded_text if decoded else ""

    payload_type, url_info, emv_info = _classify_payload(payload)

    return QREvidence(
        detected=detected,
        decoded=decoded,
        payload=payload,
        payload_type=payload_type,
        bounding_box=bounding_box,
        url_info=url_info,
        emv_info=emv_info,
    )
