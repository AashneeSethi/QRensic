"""
QRensic Vision Engine - Region & Surrounding Text Extraction Module
-------------------------------------------------------------------
Isolates candidate text regions surrounding the QR code poster.
Serves as the structured interface for future OCR engines (e.g. Tesseract / EasyOCR).

SAFETY & PROMPT INJECTION GUARD:
All visual text extracted from images is strictly UNTRUSTED CONTENT.
Strings extracted by this module must never be executed or interpolated directly
into LLM system prompts as instructions.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.vision.models import ExtractedRegionEvidence


def extract_region(
    image_input: Union[np.ndarray, str, Path],
    qr_bounding_box: Optional[List[List[float]]] = None,
    mock_ocr_text: Optional[List[str]] = None,
) -> ExtractedRegionEvidence:
    """
    Extract surrounding poster text regions and textual content.

    Args:
      image_input: The input scene image.
      qr_bounding_box: Optional coordinates of the detected QR code [[x,y], ...].
      mock_ocr_text: Optional simulated or external OCR text list for testing/staging.

    Returns:
      ExtractedRegionEvidence with candidate text lines marked UNTRUSTED.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            return ExtractedRegionEvidence(notes="Input file not found")
        img = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return ExtractedRegionEvidence(notes="Invalid image input")

    if img is None or img.size == 0 or img.ndim < 2:
        return ExtractedRegionEvidence(notes="Empty image")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim > 2 else img

    # Mask out QR code area if provided so we focus on surrounding text
    search_mask = np.ones((h, w), dtype=np.uint8) * 255
    if qr_bounding_box and len(qr_bounding_box) >= 4:
        pts = np.array(qr_bounding_box, dtype=np.int32)
        # Dilate QR mask by 15px to avoid picking up QR finder pattern borders as text
        cv2.fillPoly(search_mask, [pts], 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        search_mask = cv2.erode(search_mask, kernel, iterations=1)

    # Detect horizontal text-like bands via morphology
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    masked_sobel = cv2.bitwise_and(sobel, sobel, mask=search_mask)
    _, thresh = cv2.threshold(masked_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_text)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_regions: List[List[int]] = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = float(cw) / max(ch, 1)
        if aspect >= 1.5 and cw > 25 and ch > 8:
            bounding_regions.append([int(x), int(y), int(cw), int(ch)])

    # Sort top-to-bottom
    bounding_regions.sort(key=lambda b: b[1])

    # If external/mock OCR strings are provided, attach them
    extracted_text: List[str] = []
    confidences: List[float] = []

    if mock_ocr_text:
        extracted_text = list(mock_ocr_text)
        confidences = [0.95] * len(mock_ocr_text)

    return ExtractedRegionEvidence(
        extracted_text=extracted_text,
        confidences=confidences,
        text_bounding_regions=bounding_regions[:10],
        is_untrusted=True,  # Mandatory safety tag: all visual text is untrusted
        notes="Text extraction completed; content flagged as UNTRUSTED.",
    )
