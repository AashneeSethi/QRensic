"""
QRensic Vision Engine - Second-View Multi-View Registration & Comparison
-------------------------------------------------------------------------
Registers two separate visual observations of a physical QR scene using ORB
features and RANSAC homography. Performs strict geometric sanity checks to verify
same-scene/same-object continuity before projecting regions and comparing surface evidence.

CRITICAL FORENSIC PRINCIPLE:
Do NOT output a "fraud probability" or claim that visual discrepancies prove a sticker.
If registration fails or is geometrically ambiguous, report INCONCLUSIVE/FAIL rather
than inventing evidence.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.vision.analyze_surface import analyze_qr_surface
from backend.vision.models import (
    RegistrationStatus,
    SecondViewComparison,
    SecondViewEvidence,
)


def _load_image(image_input: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
    """Safely load and validate an image array or file path."""
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


def _check_geometric_sanity(
    H: np.ndarray,
    w1: int,
    h1: int,
    w2: int,
    h2: int,
) -> Tuple[bool, str]:
    """
    Perform rigorous geometric checks on the estimated homography matrix.
    Rejects physically impossible transformations (e.g. reflections, degenerate areas).
    """
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False, "Homography matrix contains invalid or non-finite values."

    # 1. Determinant of affine component (must be strictly positive to avoid mirror inversion)
    det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
    if det <= 0.0:
        return False, f"Homography determinant ({det:.3f}) is non-positive; indicates physical reflection/inversion."

    # 2. Plausible scale range between observations
    if det < 0.04 or det > 25.0:
        return False, f"Homography scale factor ({det:.3f}) falls outside plausible viewing distances."

    # 3. Project corners of Image 1 into Image 2 coordinate frame
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners1, H).reshape(-1, 2)
    except cv2.error as e:
        return False, f"Perspective transformation of boundary corners failed: {e}"

    # 4. Check polygon convexity (must not fold over or self-intersect)
    if not cv2.isContourConvex(projected.astype(np.int32)):
        return False, "Projected scene boundary forms a non-convex or self-intersecting polygon."

    # 5. Check projected area
    proj_area = float(cv2.contourArea(projected.astype(np.float32)))
    img2_area = float(w2 * h2)
    if proj_area < img2_area * 0.04 or proj_area > img2_area * 10.0:
        return False, f"Projected scene boundary area ({proj_area:.0f} px) is physically unviable."

    return True, "Geometric sanity checks passed."


def analyze_second_view(
    first_image: Union[np.ndarray, str, Path],
    second_image: Union[np.ndarray, str, Path],
    qr_bbox: Optional[List[List[float]]] = None,
    min_good_matches: int = 10,
    min_inliers: int = 15,
    min_inlier_ratio: float = 0.20,
    max_reprojection_error: float = 4.0,
    lowe_ratio: float = 0.75,
    nfeatures: int = 2000,
) -> SecondViewEvidence:
    """
    Register and compare a second visual observation against the initial scene view.

    Args:
      first_image: Initial observation image (ambient or straight-on).
      second_image: Follow-up observation image (angled or flash).
      qr_bbox: Optional QR bounding polygon in first image [[x1,y1], ...].
      min_good_matches: Minimum Lowe-ratio matches required to attempt RANSAC.
      min_inliers: Minimum inliers required for valid scene registration.
      min_inlier_ratio: Minimum ratio of inliers to good matches.
      max_reprojection_error: RANSAC reprojection error threshold in pixels.
      lowe_ratio: Lowe's ratio test threshold.
      nfeatures: Maximum ORB features to extract per image.

    Returns:
      SecondViewEvidence containing registration metrics, transformed QR quad,
      and differential surface comparison.
    """
    img1 = _load_image(first_image)
    img2 = _load_image(second_image)

    if img1 is None or img2 is None:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.INCONCLUSIVE,
            reason="One or both input images could not be read or are empty.",
        )

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim > 2 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim > 2 else img2

    # 1. Feature detection & description using ORB
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    k1_count = len(kp1) if kp1 else 0
    k2_count = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or k1_count < 4 or k2_count < 4:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.INCONCLUSIVE,
            keypoints_first=k1_count,
            keypoints_second=k2_count,
            reason="Insufficient feature keypoints found in one or both views to register scene.",
        )

    # 2. Feature matching with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    candidate_count = len(raw_matches)

    # 3. Lowe's ratio test filtering
    good_matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < lowe_ratio * m[1].distance:
            good_matches.append(m[0])

    good_count = len(good_matches)

    if good_count < min_good_matches or good_count < 4:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.FAIL,
            keypoints_first=k1_count,
            keypoints_second=k2_count,
            candidate_matches=candidate_count,
            good_matches=good_count,
            reason=f"Insufficient distinct matching features ({good_count} < {min_good_matches}) between views.",
        )

    # 4. RANSAC Homography Estimation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, max_reprojection_error)

    if H is None or mask is None:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.FAIL,
            keypoints_first=k1_count,
            keypoints_second=k2_count,
            candidate_matches=candidate_count,
            good_matches=good_count,
            homography_found=False,
            reason="RANSAC failed to find a valid planar homography between views.",
        )

    inliers_count = int(np.sum(mask))
    inlier_ratio = float(inliers_count / good_count)

    # 5. Inlier threshold checks
    if inliers_count < min_inliers or inlier_ratio < min_inlier_ratio:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.FAIL,
            keypoints_first=k1_count,
            keypoints_second=k2_count,
            candidate_matches=candidate_count,
            good_matches=good_count,
            homography_found=True,
            ransac_inliers=inliers_count,
            inlier_ratio=inlier_ratio,
            reason=(
                f"RANSAC inlier agreement insufficient (inliers={inliers_count} < {min_inliers}, "
                f"ratio={inlier_ratio:.2f} < {min_inlier_ratio:.2f})."
            ),
        )

    # 6. Geometric sanity checks
    sanity_passed, sanity_reason = _check_geometric_sanity(H, w1, h1, w2, h2)
    if not sanity_passed:
        return SecondViewEvidence(
            registration_status=RegistrationStatus.FAIL,
            keypoints_first=k1_count,
            keypoints_second=k2_count,
            candidate_matches=candidate_count,
            good_matches=good_count,
            homography_found=True,
            ransac_inliers=inliers_count,
            inlier_ratio=inlier_ratio,
            geometric_sanity_passed=False,
            reason=f"Geometric verification rejected homography: {sanity_reason}",
        )

    # Reprojection error over inliers
    inlier_indices = np.where(mask.ravel() == 1)[0]
    src_inliers = src_pts[inlier_indices]
    dst_inliers = dst_pts[inlier_indices]
    projected_inliers = cv2.perspectiveTransform(src_inliers, H)
    reproj_err = float(np.mean(np.linalg.norm(dst_inliers - projected_inliers, axis=2)))

    h_matrix_list = [[float(val) for val in row] for row in H]

    # 7. QR Region Registration (if provided)
    registered_qr: Optional[List[List[float]]] = None
    if qr_bbox and len(qr_bbox) >= 4:
        qr_in_pts = np.float32(qr_bbox[:4]).reshape(-1, 1, 2)
        try:
            trans_qr = cv2.perspectiveTransform(qr_in_pts, H).reshape(-1, 2)
            if cv2.isContourConvex(trans_qr.astype(np.int32)):
                area_trans = float(cv2.contourArea(trans_qr.astype(np.float32)))
                if area_trans > 50.0:
                    registered_qr = [[float(pt[0]), float(pt[1])] for pt in trans_qr]
        except cv2.error:
            registered_qr = None

    # 8. Differential Surface Evidence Comparison
    comparison: Optional[SecondViewComparison] = None
    if registered_qr is not None and qr_bbox is not None:
        surf1 = analyze_qr_surface(img1, qr_bbox)
        surf2 = analyze_qr_surface(img2, registered_qr)

        delta_qr = surf2.qr_mean_intensity - surf1.qr_mean_intensity
        delta_ring = surf2.ring_mean_intensity - surf1.ring_mean_intensity
        epsilon = 1e-4

        rsr_ratio = float((delta_qr + epsilon) / (delta_ring + epsilon))
        rsr_diff = float(delta_qr - delta_ring)

        delta_boundary = float(surf2.boundary_gradient_mean - surf1.boundary_gradient_mean)
        delta_texture = float(surf2.texture_laplacian_energy_qr - surf1.texture_laplacian_energy_qr)
        delta_color = float(surf2.color_discontinuity_delta - surf1.color_discontinuity_delta)

        comparison = SecondViewComparison(
            qr_brightness_first=surf1.qr_mean_intensity,
            qr_brightness_second=surf2.qr_mean_intensity,
            delta_qr_brightness=float(delta_qr),
            ring_brightness_first=surf1.ring_mean_intensity,
            ring_brightness_second=surf2.ring_mean_intensity,
            delta_ring_brightness=float(delta_ring),
            relative_specular_ratio=rsr_ratio,
            relative_specular_diff=rsr_diff,
            boundary_gradient_first=surf1.boundary_gradient_mean,
            boundary_gradient_second=surf2.boundary_gradient_mean,
            delta_boundary_gradient=delta_boundary,
            texture_energy_first=surf1.texture_laplacian_energy_qr,
            texture_energy_second=surf2.texture_laplacian_energy_qr,
            delta_texture_energy=delta_texture,
            color_delta_first=surf1.color_discontinuity_delta,
            color_delta_second=surf2.color_discontinuity_delta,
            delta_color_delta=delta_color,
            comparison_notes=(
                f"Multi-view surface comparison extracted across registered views. "
                f"Relative Specular Response ratio: {rsr_ratio:.3f}, diff: {rsr_diff:.2f}. "
                f"Surface signal changes observed; evidence remains empirical."
            ),
        )

    return SecondViewEvidence(
        registration_status=RegistrationStatus.PASS,
        keypoints_first=k1_count,
        keypoints_second=k2_count,
        candidate_matches=candidate_count,
        good_matches=good_count,
        homography_found=True,
        ransac_inliers=inliers_count,
        inlier_ratio=inlier_ratio,
        geometric_sanity_passed=True,
        homography_matrix=h_matrix_list,
        reprojection_error=reproj_err,
        registered_qr_bbox=registered_qr,
        reason="Successful planar registration with geometric sanity validation.",
        comparison=comparison,
    )
