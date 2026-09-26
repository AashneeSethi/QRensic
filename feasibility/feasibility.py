#!/usr/bin/env python3
"""
QRensic Feasibility Study - Optical & Vision Evidence Measurement
-----------------------------------------------------------------
First technical feasibility script for QRensic.
Evaluates single-image metrics, flash/no-flash differential specular response (RSR),
surface boundary discontinuities, and second-view ORB/RANSAC homography registration.

IMPORTANT FORENSIC PRINCIPLE:
This script computes deterministic computer vision evidence measurements.
It does NOT output a machine learning classification or "fraud probability".
"""

import argparse
import csv
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Filename Parsing Helpers
# -----------------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"^(?P<condition>[a-zA-Z0-9_-]+)_(?P<angle>a[12])_(?P<flash>flash|noflash)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """Extract condition, angle, and flash state from standardized filename."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.groupdict()
    return {"condition": "unknown", "angle": "unknown", "flash": "unknown", "ext": ""}


# -----------------------------------------------------------------------------
# Single Image Analysis
# -----------------------------------------------------------------------------

def analyze_single_image(image_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive visual and geometric measurements from a single photograph.
    
    Measurements:
    - Dimensions, blur estimate (Laplacian variance), brightness stats.
    - QR detection, decoding, quad corners, aspect ratio, perspective skew.
    - QR region vs. host poster surrounding ring statistics.
    - Physical edge & boundary gradient discontinuity.
    - Color discontinuity (Euclidean delta in BGR and HSV).
    """
    row: Dict[str, Any] = {
        "filename": image_path.name,
        "filepath": str(image_path),
        "condition": "unknown",
        "angle": "unknown",
        "lighting": "unknown",
        "image_width": 0,
        "image_height": 0,
        "channels": 0,
        "blur_laplacian_var": 0.0,
        "mean_brightness": 0.0,
        "std_brightness": 0.0,
        "qr_detected": False,
        "qr_decoded": False,
        "qr_payload": "",
        "qr_area_px": 0.0,
        "qr_aspect_ratio": 0.0,
        "qr_skew_angle": 0.0,
        "qr_diagonal_ratio": 0.0,
        "qr_mean_intensity": 0.0,
        "qr_std_intensity": 0.0,
        "qr_mean_saturation": 0.0,
        "qr_std_saturation": 0.0,
        "ring_mean_intensity": 0.0,
        "ring_std_intensity": 0.0,
        "ring_mean_saturation": 0.0,
        "ring_std_saturation": 0.0,
        "boundary_gradient_mean": 0.0,
        "boundary_gradient_max": 0.0,
        "edge_discontinuity_ratio": 0.0,
        "color_discontinuity_delta": 0.0,
        "texture_laplacian_energy_qr": 0.0,
        "texture_laplacian_energy_ring": 0.0,
        "error_message": "",
    }

    meta = parse_filename_metadata(image_path.name)
    row["condition"] = meta["condition"]
    row["angle"] = meta["angle"]
    row["lighting"] = meta["flash"]

    if not image_path.is_file():
        row["error_message"] = "File not found"
        return row

    img = cv2.imread(str(image_path))
    if img is None:
        row["error_message"] = "Failed to decode image file via OpenCV"
        return row

    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim > 2 else 1
    row["image_width"] = w
    row["image_height"] = h
    row["channels"] = c

    # Convert color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c > 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if c > 1 else None

    # Quality & illumination
    row["blur_laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    row["mean_brightness"] = float(np.mean(gray))
    row["std_brightness"] = float(np.std(gray))

    # QR Detection & Decoding using OpenCV QRCodeDetector
    detector = cv2.QRCodeDetector()
    decoded_text, points, _ = detector.detectAndDecode(img)

    # In case color detection missed, attempt on grayscale
    if (points is None or len(points) == 0) and c > 1:
        decoded_text, points, _ = detector.detectAndDecode(gray)

    if points is not None and len(points) > 0:
        pts = points.reshape(-1, 2)
        if len(pts) >= 4:
            row["qr_detected"] = True
            if decoded_text:
                row["qr_decoded"] = True
                row["qr_payload"] = decoded_text

            # Geometry & perspective
            area = float(cv2.contourArea(pts.astype(np.float32)))
            row["qr_area_px"] = area

            # Min area bounding rectangle
            rect = cv2.minAreaRect(pts.astype(np.float32))
            rect_w, rect_h = rect[1]
            if rect_w > 0 and rect_h > 0:
                row["qr_aspect_ratio"] = float(min(rect_w, rect_h) / max(rect_w, rect_h))
            row["qr_skew_angle"] = float(rect[2])

            # Quadrilateral diagonal perspective check
            d1 = float(np.linalg.norm(pts[0] - pts[2]))
            d2 = float(np.linalg.norm(pts[1] - pts[3]))
            if d2 > 0:
                row["qr_diagonal_ratio"] = float(d1 / d2)

            # Region Segmentation Masks
            # 1. QR Region Mask
            qr_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(qr_mask, [pts.astype(np.int32)], 255)

            # 2. Surrounding Poster Ring Mask (dilated convex hull around QR)
            center = np.mean(pts, axis=0)
            expanded_pts = center + 1.40 * (pts - center)
            expanded_pts = np.clip(expanded_pts, [0, 0], [w - 1, h - 1]).astype(np.int32)

            outer_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(outer_mask, [expanded_pts], 255)

            # Dilate QR mask by 6 px to exclude direct boundary transition from poster ring
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            dilated_qr_mask = cv2.dilate(qr_mask, kernel_dilate, iterations=1)
            ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(dilated_qr_mask))

            # QR Region intensity and saturation statistics
            qr_pixels = gray[qr_mask > 0]
            if len(qr_pixels) > 0:
                row["qr_mean_intensity"] = float(np.mean(qr_pixels))
                row["qr_std_intensity"] = float(np.std(qr_pixels))
                row["texture_laplacian_energy_qr"] = float(cv2.Laplacian(gray, cv2.CV_64F)[qr_mask > 0].var())
            if hsv is not None:
                row["qr_mean_saturation"] = float(np.mean(hsv[:, :, 1][qr_mask > 0]))
                row["qr_std_saturation"] = float(np.std(hsv[:, :, 1][qr_mask > 0]))

            # Surrounding Poster Ring statistics
            ring_pixels = gray[ring_mask > 0]
            if len(ring_pixels) > 0:
                row["ring_mean_intensity"] = float(np.mean(ring_pixels))
                row["ring_std_intensity"] = float(np.std(ring_pixels))
                row["texture_laplacian_energy_ring"] = float(cv2.Laplacian(gray, cv2.CV_64F)[ring_mask > 0].var())
            if hsv is not None and len(ring_pixels) > 0:
                row["ring_mean_saturation"] = float(np.mean(hsv[:, :, 1][ring_mask > 0]))
                row["ring_std_saturation"] = float(np.std(hsv[:, :, 1][ring_mask > 0]))

            # Physical Boundary & Edge Discontinuity Signals
            # Compute Sobel gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)

            # Boundary band: narrow 8px band straddling the perimeter
            kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated_boundary = cv2.dilate(qr_mask, kernel_boundary, iterations=1)
            eroded_boundary = cv2.erode(qr_mask, kernel_boundary, iterations=1)
            boundary_band_mask = cv2.bitwise_and(dilated_boundary, cv2.bitwise_not(eroded_boundary))

            boundary_grads = grad_mag[boundary_band_mask > 0]
            if len(boundary_grads) > 0:
                row["boundary_gradient_mean"] = float(np.mean(boundary_grads))
                row["boundary_gradient_max"] = float(np.max(boundary_grads))

            interior_grads = grad_mag[eroded_boundary > 0]
            if len(interior_grads) > 0 and len(boundary_grads) > 0:
                interior_mean = float(np.mean(interior_grads))
                row["edge_discontinuity_ratio"] = float(row["boundary_gradient_mean"] / (interior_mean + 1e-5))

            # Color Discontinuity Delta (Euclidean distance between outer QR edge and poster ring)
            if img.ndim > 2 and len(ring_pixels) > 0:
                qr_edge_bgr = np.mean(img[boundary_band_mask > 0], axis=0)
                ring_bgr = np.mean(img[ring_mask > 0], axis=0)
                row["color_discontinuity_delta"] = float(np.linalg.norm(qr_edge_bgr - ring_bgr))

    return row


# -----------------------------------------------------------------------------
# Flash / No-Flash Paired Analysis (Relative Specular Response)
# -----------------------------------------------------------------------------

def analyze_flash_pair(noflash_row: Dict[str, Any], flash_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate differential specular response between No-Flash and Flash images.
    
    Formula:
      Delta_QR   = Mean_Intensity(Flash_QR)   - Mean_Intensity(NoFlash_QR)
      Delta_Ring = Mean_Intensity(Flash_Ring) - Mean_Intensity(NoFlash_Ring)
      RSR_Ratio  = (Delta_QR + epsilon) / (Delta_Ring + epsilon)
      RSR_Diff   = Delta_QR - Delta_Ring
    """
    pair_result: Dict[str, Any] = {
        "condition": noflash_row["condition"],
        "angle": noflash_row["angle"],
        "noflash_file": noflash_row["filename"],
        "flash_file": flash_row["filename"],
        "both_qr_detected": noflash_row["qr_detected"] and flash_row["qr_detected"],
        "delta_qr_intensity": 0.0,
        "delta_ring_intensity": 0.0,
        "relative_specular_ratio": 0.0,
        "relative_specular_diff": 0.0,
        "delta_boundary_gradient": 0.0,
        "delta_color_discontinuity": 0.0,
        "notes": "",
    }

    if not pair_result["both_qr_detected"]:
        pair_result["notes"] = "QR missing or undetected in one or both views"
        return pair_result

    delta_qr = flash_row["qr_mean_intensity"] - noflash_row["qr_mean_intensity"]
    delta_ring = flash_row["ring_mean_intensity"] - noflash_row["ring_mean_intensity"]

    pair_result["delta_qr_intensity"] = float(delta_qr)
    pair_result["delta_ring_intensity"] = float(delta_ring)

    epsilon = 1e-4
    pair_result["relative_specular_ratio"] = float((delta_qr + epsilon) / (delta_ring + epsilon))
    pair_result["relative_specular_diff"] = float(delta_qr - delta_ring)

    pair_result["delta_boundary_gradient"] = float(
        flash_row["boundary_gradient_mean"] - noflash_row["boundary_gradient_mean"]
    )
    pair_result["delta_color_discontinuity"] = float(
        flash_row["color_discontinuity_delta"] - noflash_row["color_discontinuity_delta"]
    )

    return pair_result


# -----------------------------------------------------------------------------
# Multi-View Registration (ORB + RANSAC Homography)
# -----------------------------------------------------------------------------

def analyze_second_view(view1_path: Path, view2_path: Path) -> Dict[str, Any]:
    """
    Perform second-view feature registration using ORB and RANSAC Homography.
    """
    reg_result: Dict[str, Any] = {
        "view1": view1_path.name,
        "view2": view2_path.name,
        "keypoints_view1": 0,
        "keypoints_view2": 0,
        "good_matches": 0,
        "homography_inliers": 0,
        "inlier_ratio": 0.0,
        "registration_success": False,
        "homography_det": 0.0,
        "error_message": "",
    }

    img1 = cv2.imread(str(view1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(view2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        reg_result["error_message"] = "Failed to load one or both images for registration"
        return reg_result

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    reg_result["keypoints_view1"] = len(kp1)
    reg_result["keypoints_view2"] = len(kp2)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        reg_result["error_message"] = "Insufficient keypoints for matching"
        return reg_result

    # Match descriptors using BFMatcher with Hamming norm
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    reg_result["good_matches"] = len(good_matches)

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            num_inliers = int(np.sum(mask))
            inlier_ratio = float(num_inliers / len(good_matches))
            reg_result["homography_inliers"] = num_inliers
            reg_result["inlier_ratio"] = inlier_ratio
            reg_result["registration_success"] = bool(num_inliers >= 15 and inlier_ratio >= 0.25)
            reg_result["homography_det"] = float(np.linalg.det(H[:2, :2]))

    return reg_result


# -----------------------------------------------------------------------------
# Batch Processing & Orchestration
# -----------------------------------------------------------------------------

def write_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write list of dictionaries to a CSV file."""
    if not data:
        print(f"[!] Warning: No data to write to {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(data[0].keys())

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"[OK] Wrote {len(data)} rows to {output_path}")


def run_experiment(input_dir: Path, output_dir: Path) -> None:
    """Execute complete feasibility analysis on an experiment directory."""
    print("=" * 60)
    print("QRensic Physical Feasibility Measurement Engine")
    print("=" * 60)
    print(f"Scanning directory: {input_dir}")
    print(f"Target output:      {output_dir}\n")

    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    if not image_files:
        print(f"[!] No valid image files (.jpg, .jpeg, .png) found in {input_dir}.")
        print("    Capture test photos according to the protocol in feasibility/README.md.")
        return

    print(f"Found {len(image_files)} photos to analyze.")

    # 1. Single Image Analysis
    single_results: List[Dict[str, Any]] = []
    print("\n[Phase 1] Extracting single-image optical & geometric signals...")
    for idx, img_path in enumerate(image_files, 1):
        res = analyze_single_image(img_path)
        single_results.append(res)
        detected_str = "QR [OK]" if res["qr_detected"] else "QR [x]"
        print(f"  [{idx:02d}/{len(image_files):02d}] {img_path.name:<32} {detected_str} (blur: {res['blur_laplacian_var']:6.1f}, R_mean: {res['ring_mean_intensity']:5.1f})")

    single_csv = output_dir / "measurements_single.csv"
    write_csv(single_results, single_csv)

    # 2. Flash / No-Flash Paired Analysis (Relative Specular Response)
    print("\n[Phase 2] Computing Relative Specular Response (RSR) on flash/no-flash pairs...")
    by_condition_angle: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for row in single_results:
        cond, angle, flash = row["condition"], row["angle"], row["lighting"]
        if cond != "unknown" and angle != "unknown":
            key = (cond, angle)
            if key not in by_condition_angle:
                by_condition_angle[key] = {}
            by_condition_angle[key][flash] = row

    flash_pair_results: List[Dict[str, Any]] = []
    for (cond, angle), pair_map in by_condition_angle.items():
        if "noflash" in pair_map and "flash" in pair_map:
            pair_res = analyze_flash_pair(pair_map["noflash"], pair_map["flash"])
            flash_pair_results.append(pair_res)
            print(f"  Pair: {cond}_{angle} -> RSR Ratio: {pair_res['relative_specular_ratio']:.3f}, RSR Diff: {pair_res['relative_specular_diff']:.2f}")

    flash_pair_csv = output_dir / "measurements_flash_pairs.csv"
    if flash_pair_results:
        write_csv(flash_pair_results, flash_pair_csv)

    # 3. Second-View Multi-View Registration (ORB + RANSAC)
    print("\n[Phase 3] Computing multi-view registration across Angle 1 & Angle 2...")
    multiview_results: List[Dict[str, Any]] = []
    # Match a1 with a2 for the same condition and lighting
    by_cond_light: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for f in image_files:
        meta = parse_filename_metadata(f.name)
        if meta["condition"] != "unknown" and meta["angle"] != "unknown":
            key = (meta["condition"], meta["flash"])
            if key not in by_cond_light:
                by_cond_light[key] = {}
            by_cond_light[key][meta["angle"]] = f

    for (cond, light), angles in by_cond_light.items():
        if "a1" in angles and "a2" in angles:
            mv_res = analyze_second_view(angles["a1"], angles["a2"])
            mv_res["condition"] = cond
            mv_res["lighting"] = light
            multiview_results.append(mv_res)
            success_str = "SUCCESS" if mv_res["registration_success"] else "FAILED"
            print(f"  Multi-view: {cond} ({light}) a1->a2: {success_str} (inliers: {mv_res['homography_inliers']}, ratio: {mv_res['inlier_ratio']:.2f})")

    multiview_csv = output_dir / "measurements_multiview.csv"
    if multiview_results:
        write_csv(multiview_results, multiview_csv)

    print("\n[OK] Feasibility measurements completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QRensic Feasibility Measurement Script: Optical & Vision Signal Extraction"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "photos",
        help="Directory containing physical test photographs (default: feasibility/photos)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to save output CSV measurements (default: feasibility/results)",
    )
    args = parser.parse_args()
    run_experiment(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
