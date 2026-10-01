"""
QRensic Vision Engine - Second-View Registration Test Suite
------------------------------------------------------------
Unit tests verifying ORB feature matching, RANSAC homography estimation,
geometric sanity checks, same-scene validation, and transformed QR projection.

IMPORTANT DISTINCTION:
These unit tests validate geometric registration algorithms and failure-handling
under synthetic transformations. They do NOT validate physical sticker detection
on real multi-view poster photos, which requires physical testing.
"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from backend.vision.analyze_qr import analyze_qr
from backend.vision.analyze_second_view import analyze_second_view
from backend.vision.engine import QRensicVisionEngine
from backend.vision.models import RegistrationStatus


class TestSecondViewRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.props_dir = cls.repo_root / "feasibility" / "props" / "generated"
        cls.genuine_qr_path = cls.props_dir / "genuine_qr.png"
        cls.sticker_qr_path = cls.props_dir / "sticker_qr.png"

        # Ensure synthetic props exist
        if not cls.genuine_qr_path.is_file():
            from feasibility.generate_props import main as gen_props
            gen_props()

        # Load base test image
        cls.base_img = cv2.imread(str(cls.genuine_qr_path))
        assert cls.base_img is not None, "Failed to load genuine_qr.png for test"

    def test_01_identical_image_self_registration(self):
        """Test 1: Identical image registered against itself must pass with high inliers."""
        result = analyze_second_view(self.base_img, self.base_img)

        self.assertEqual(result.registration_status, RegistrationStatus.PASS)
        self.assertTrue(result.homography_found)
        self.assertTrue(result.geometric_sanity_passed)
        self.assertGreater(result.ransac_inliers, 30)
        self.assertGreaterEqual(result.inlier_ratio, 0.70)
        self.assertIsNotNone(result.homography_matrix)

    def test_02_synthetic_transformed_image_registration(self):
        """
        Test 2: Synthetic affine/perspective transformation of the image
        (rotated ~12 degrees, scaled 0.95x, translated) must successfully register.
        """
        h, w = self.base_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        # Affine rotation matrix
        M_rot = cv2.getRotationMatrix2D(center, angle=12.0, scale=0.95)
        transformed_img = cv2.warpAffine(self.base_img, M_rot, (w, h), borderValue=(255, 255, 255))

        result = analyze_second_view(self.base_img, transformed_img)

        self.assertEqual(result.registration_status, RegistrationStatus.PASS)
        self.assertTrue(result.homography_found)
        self.assertTrue(result.geometric_sanity_passed)
        self.assertGreaterEqual(result.ransac_inliers, 15)
        self.assertGreaterEqual(result.inlier_ratio, 0.20)

    def test_03_unrelated_images_registration_rejected(self):
        """
        Test 3: Two unrelated visual scenes (QR code vs random noise or blank canvas)
        must fail or report inconclusive registration, never falsely claim same-scene agreement.
        """
        h, w = self.base_img.shape[:2]
        # Generate random uniform noise image
        noise_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        result_noise = analyze_second_view(self.base_img, noise_img)
        self.assertIn(
            result_noise.registration_status,
            [RegistrationStatus.FAIL, RegistrationStatus.INCONCLUSIVE],
            "Unrelated noise image must NOT pass registration",
        )
        self.assertFalse(result_noise.geometric_sanity_passed)

        # Blank white image
        blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        result_blank = analyze_second_view(self.base_img, blank_img)
        self.assertIn(
            result_blank.registration_status,
            [RegistrationStatus.FAIL, RegistrationStatus.INCONCLUSIVE],
            "Blank featureless image must NOT pass registration",
        )

    def test_04_invalid_or_missing_inputs_graceful(self):
        """Test 4: Invalid file paths or empty image inputs must fail gracefully without exceptions."""
        result_missing = analyze_second_view(
            Path("nonexistent_view1.jpg"),
            Path("nonexistent_view2.jpg"),
        )
        self.assertEqual(result_missing.registration_status, RegistrationStatus.INCONCLUSIVE)
        self.assertIn("could not be read", result_missing.reason)

        # Empty numpy array
        result_empty = analyze_second_view(self.base_img, np.array([]))
        self.assertEqual(result_empty.registration_status, RegistrationStatus.INCONCLUSIVE)

    def test_05_qr_region_registration_projection(self):
        """
        Test 5: Providing a QR bounding box correctly transforms the quad polygon
        through the homography and returns a geometrically valid registered QR bbox.
        """
        qr_ev = analyze_qr(self.base_img)
        self.assertTrue(qr_ev.detected)
        self.assertIsNotNone(qr_ev.bounding_box)

        # Rotate base image by 8 degrees
        h, w = self.base_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, angle=8.0, scale=1.0)
        rotated_img = cv2.warpAffine(self.base_img, M_rot, (w, h), borderValue=(255, 255, 255))

        result = analyze_second_view(
            first_image=self.base_img,
            second_image=rotated_img,
            qr_bbox=qr_ev.bounding_box,
        )

        self.assertEqual(result.registration_status, RegistrationStatus.PASS)
        self.assertIsNotNone(result.registered_qr_bbox)
        self.assertEqual(len(result.registered_qr_bbox), 4)

        # Transformed polygon must have reasonable area and convexity
        trans_pts = np.array(result.registered_qr_bbox, dtype=np.float32)
        self.assertTrue(cv2.isContourConvex(trans_pts.astype(np.int32)))
        self.assertGreater(cv2.contourArea(trans_pts), 1000.0)

        # Differential comparison should be populated
        self.assertIsNotNone(result.comparison)
        self.assertIsNotNone(result.comparison.relative_specular_ratio)

    def test_06_engine_integration_standalone_and_combined(self):
        """Test 6: QRensicVisionEngine executes second-view analysis both standalone and within process_image."""
        engine = QRensicVisionEngine()

        # 1. Standalone invocation
        ev_standalone = engine.analyze_second_view(self.base_img, self.base_img)
        self.assertEqual(ev_standalone.registration_status, RegistrationStatus.PASS)

        # 2. Combined invocation inside process_image
        ev_combined = engine.process_image(
            image_input=self.genuine_qr_path,
            second_image_input=self.genuine_qr_path,
        )
        self.assertIsNotNone(ev_combined.second_view)
        self.assertEqual(ev_combined.second_view.registration_status, RegistrationStatus.PASS)
        self.assertTrue(any("Second visual observation successfully registered" in w for w in ev_combined.warnings))


if __name__ == "__main__":
    unittest.main()
