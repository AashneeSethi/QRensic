"""
QRensic Vision Engine - Physical Surface & Edge Analysis Module
---------------------------------------------------------------
Extracts deterministic optical, gradient, texture, and specular discontinuity
signals across the QR perimeter and host poster background.

FORENSIC INTEGRITY RULE:
Do NOT output a "sticker probability" or claim fraud.
Only output empirical physical signals and signal availability.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.vision.models import SurfaceEvidence


def _extract_masks(
    h: int, w: int, pts: np.ndarray, dilation_px: int = 6, expansion_scale: float = 1.40
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate QR mask, dilated QR mask, surrounding poster ring mask, and boundary band mask.
    """
    # 1. Base QR polygon mask
    qr_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(qr_mask, [pts.astype(np.int32)], 255)

    # 2. Dilated QR mask
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    dilated_qr_mask = cv2.dilate(qr_mask, kernel_dilate, iterations=1)

    # 3. Surrounding poster ring (expanded outward from centroid)
    center = np.mean(pts, axis=0)
    expanded_pts = center + expansion_scale * (pts - center)
    expanded_pts = np.clip(expanded_pts, [0, 0], [w - 1, h - 1]).astype(np.int32)

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(outer_mask, [expanded_pts], 255)
    ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(dilated_qr_mask))

    # 4. Narrow boundary band straddling the physical perimeter
    kernel_band = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_boundary = cv2.dilate(qr_mask, kernel_band, iterations=1)
    eroded_boundary = cv2.erode(qr_mask, kernel_band, iterations=1)
    boundary_band_mask = cv2.bitwise_and(dilated_boundary, cv2.bitwise_not(eroded_boundary))

    return qr_mask, ring_mask, boundary_band_mask, eroded_boundary


def analyze_qr_surface(
    image_input: Union[np.ndarray, str, Path],
    qr_bounding_box: List[List[float]],
    second_image_input: Optional[Union[np.ndarray, str, Path]] = None,
) -> SurfaceEvidence:
    """
    Analyze optical surface metrics across the QR bounding box and surrounding poster ring.

    Args:
      image_input: Primary image (e.g. ambient lighting).
      qr_bounding_box: 4 corner points [[x, y], ...].
      second_image_input: Optional second image (e.g. camera flash observation).

    Returns:
      SurfaceEvidence containing raw/normalized optical evidence signals.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            return SurfaceEvidence()
        img = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return SurfaceEvidence()

    if img is None or img.size == 0 or img.ndim < 2 or not qr_bounding_box or len(qr_bounding_box) < 4:
        return SurfaceEvidence()

    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim > 2 else 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c > 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if c > 1 else None

    pts = np.array(qr_bounding_box[:4], dtype=np.float32)

    # Calculate geometry
    area = float(cv2.contourArea(pts))
    rect = cv2.minAreaRect(pts)
    rect_w, rect_h = rect[1]
    aspect_ratio = float(min(rect_w, rect_h) / max(rect_w, rect_h)) if (rect_w > 0 and rect_h > 0) else 0.0

    d1 = float(np.linalg.norm(pts[0] - pts[2]))
    d2 = float(np.linalg.norm(pts[1] - pts[3]))
    diagonal_ratio = float(d1 / d2) if d2 > 0 else 0.0

    # Masks
    qr_mask, ring_mask, boundary_band_mask, eroded_boundary = _extract_masks(h, w, pts)

    signals_available: List[str] = ["geometry"]

    # QR Region & Ring Statistics
    qr_pixels = gray[qr_mask > 0]
    qr_mean_intensity = float(np.mean(qr_pixels)) if len(qr_pixels) > 0 else 0.0
    qr_std_intensity = float(np.std(qr_pixels)) if len(qr_pixels) > 0 else 0.0

    qr_mean_sat = 0.0
    if hsv is not None and len(qr_pixels) > 0:
        qr_mean_sat = float(np.mean(hsv[:, :, 1][qr_mask > 0]))

    ring_pixels = gray[ring_mask > 0]
    ring_mean_intensity = float(np.mean(ring_pixels)) if len(ring_pixels) > 0 else 0.0
    ring_std_intensity = float(np.std(ring_pixels)) if len(ring_pixels) > 0 else 0.0

    ring_mean_sat = 0.0
    if hsv is not None and len(ring_pixels) > 0:
        ring_mean_sat = float(np.mean(hsv[:, :, 1][ring_mask > 0]))

    if len(qr_pixels) > 0 and len(ring_pixels) > 0:
        signals_available.append("region_statistics")

    # Local Texture (Laplacian high-frequency energy)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_energy_qr = float(laplacian[qr_mask > 0].var()) if len(qr_pixels) > 0 else 0.0
    texture_energy_ring = float(laplacian[ring_mask > 0].var()) if len(ring_pixels) > 0 else 0.0
    if len(qr_pixels) > 0 and len(ring_pixels) > 0:
        signals_available.append("texture_energy")

    # Sobel Boundary Gradient Analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    boundary_grads = grad_mag[boundary_band_mask > 0]
    boundary_mean = float(np.mean(boundary_grads)) if len(boundary_grads) > 0 else 0.0
    boundary_max = float(np.max(boundary_grads)) if len(boundary_grads) > 0 else 0.0

    interior_grads = grad_mag[eroded_boundary > 0]
    interior_mean = float(np.mean(interior_grads)) if len(interior_grads) > 0 else 0.0
    edge_discontinuity = float(boundary_mean / (interior_mean + 1e-5))

    if len(boundary_grads) > 0:
        signals_available.append("boundary_gradients")
        signals_available.append("edge_discontinuity")

    # Color Discontinuity Delta across boundary
    color_delta = 0.0
    if c > 1 and len(boundary_grads) > 0 and len(ring_pixels) > 0:
        bgr_boundary = np.mean(img[boundary_band_mask > 0], axis=0)
        bgr_ring = np.mean(img[ring_mask > 0], axis=0)
        color_delta = float(np.linalg.norm(bgr_boundary - bgr_ring))
        signals_available.append("color_discontinuity")

    # Optional Second View / Flash Specular Analysis
    rsr_ratio: Optional[float] = None
    rsr_diff: Optional[float] = None

    if second_image_input is not None:
        if isinstance(second_image_input, (str, Path)):
            sec_p = Path(second_image_input)
            img2 = cv2.imread(str(sec_p)) if sec_p.is_file() else None
        elif isinstance(second_image_input, np.ndarray):
            img2 = second_image_input
        else:
            img2 = None

        if img2 is not None and img2.shape[:2] == (h, w):
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim > 2 else img2
            qr_mean_2 = float(np.mean(gray2[qr_mask > 0])) if len(qr_pixels) > 0 else 0.0
            ring_mean_2 = float(np.mean(gray2[ring_mask > 0])) if len(ring_pixels) > 0 else 0.0

            delta_qr = qr_mean_2 - qr_mean_intensity
            delta_ring = ring_mean_2 - ring_mean_intensity
            epsilon = 1e-4

            rsr_ratio = float((delta_qr + epsilon) / (delta_ring + epsilon))
            rsr_diff = float(delta_qr - delta_ring)
            signals_available.append("relative_specular_response")

    return SurfaceEvidence(
        boundary_gradient_mean=boundary_mean,
        boundary_gradient_max=boundary_max,
        edge_discontinuity_ratio=edge_discontinuity,
        color_discontinuity_delta=color_delta,
        qr_mean_intensity=qr_mean_intensity,
        qr_std_intensity=qr_std_intensity,
        qr_mean_saturation=qr_mean_sat,
        ring_mean_intensity=ring_mean_intensity,
        ring_std_intensity=ring_std_intensity,
        ring_mean_saturation=ring_mean_sat,
        texture_laplacian_energy_qr=texture_energy_qr,
        texture_laplacian_energy_ring=texture_energy_ring,
        qr_aspect_ratio=aspect_ratio,
        qr_diagonal_ratio=diagonal_ratio,
        qr_area_px=area,
        relative_specular_ratio=rsr_ratio,
        relative_specular_diff=rsr_diff,
        signals_available=signals_available,
    )
