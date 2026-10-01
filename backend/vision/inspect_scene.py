"""
QRensic Vision Engine - Scene Inspection Module
------------------------------------------------
Analyzes overall scene properties, optical clarity, illumination levels,
and identifies regions containing QR and text candidates.
Does NOT make fraud or authenticity decisions.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.vision.models import SceneEvidence


# Heuristic thresholds for initial quality screening
BLUR_THRESHOLD_LAPLACIAN = 100.0
LOW_LIGHT_THRESHOLD = 50.0
OVEREXPOSURE_THRESHOLD = 215.0


def _load_image(image_input: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
    """Load an image from array or filesystem path safely."""
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            return None
        return cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0 or image_input.ndim < 2:
            return None
        return image_input
    return None


def _find_qr_candidates(gray: np.ndarray) -> List[List[int]]:
    """
    Identify potential QR candidate bounding boxes using morphological
    filtering and contour aspect ratio analysis.
    """
    candidates: List[List[int]] = []
    h, w = gray.shape[:2]
    min_dim = min(h, w)

    # Adaptive threshold to highlight module patterns
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )

    # Look for square-ish components
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (min_dim * 0.05) ** 2 or area > (min_dim * 0.95) ** 2:
            continue
        aspect = float(min(cw, ch)) / max(cw, ch)
        if aspect > 0.75:  # Nearly square
            candidates.append([int(x), int(y), int(cw), int(ch)])

    # Deduplicate overlapping candidates using non-maximum style filter
    if not candidates:
        return []

    # Sort by area descending and pick top distinct boxes
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    deduped: List[List[int]] = []
    for b in candidates:
        bx, by, bw, bh = b
        overlap = False
        for db in deduped:
            dx, dy, dw, dh = db
            # Check center distance
            cx_dist = abs((bx + bw / 2) - (dx + dw / 2))
            cy_dist = abs((by + bh / 2) - (dy + dh / 2))
            if cx_dist < (bw + dw) / 3 and cy_dist < (bh + dh) / 3:
                overlap = True
                break
        if not overlap:
            deduped.append(b)
        if len(deduped) >= 5:
            break

    return deduped


def _find_text_candidates(gray: np.ndarray) -> List[List[int]]:
    """
    Identify candidate text regions using horizontal gradient density.
    """
    h, w = gray.shape[:2]
    text_boxes: List[List[int]] = []

    # Sobel horizontal gradient
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Horizontal morphological kernel to connect character fragments into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = float(cw) / max(ch, 1)
        # Text lines are typically wider than they are tall
        if aspect >= 1.5 and cw > 20 and ch > 8 and cw < w * 0.95:
            text_boxes.append([int(x), int(y), int(cw), int(ch)])

    # Limit to top 10 most prominent candidate bands
    text_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return text_boxes[:10]


def inspect_scene(image_input: Union[np.ndarray, str, Path]) -> SceneEvidence:
    """
    Execute initial scene-level inspection on the input image.

    Returns:
      SceneEvidence with dimensions, blur metric, brightness, and candidate locations.
    """
    img = _load_image(image_input)
    if img is None:
        return SceneEvidence()

    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim > 2 else 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c > 1 else img

    # Blur estimation via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < BLUR_THRESHOLD_LAPLACIAN

    # Brightness metrics
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    is_low_light = mean_val < LOW_LIGHT_THRESHOLD
    is_overexposed = mean_val > OVEREXPOSURE_THRESHOLD

    # Region candidate discovery
    qr_boxes = _find_qr_candidates(gray)
    text_boxes = _find_text_candidates(gray)

    return SceneEvidence(
        width=w,
        height=h,
        channels=c,
        mean_brightness=mean_val,
        std_brightness=std_val,
        blur_laplacian_var=blur_score,
        is_blurry=is_blurry,
        is_low_light=is_low_light,
        is_overexposed=is_overexposed,
        qr_candidates_count=len(qr_boxes),
        text_candidates_count=len(text_boxes),
        qr_candidates_boxes=qr_boxes,
        text_candidates_boxes=text_boxes,
    )
