"""
QRensic Vision Engine - Deterministic Evidence Engine Test Suite
----------------------------------------------------------------
Unit tests verifying conservative deterministic evidence rules, state transitions,
reasons synthesis, and action recommendations.

TEST CRITERIA:
1. consistent identity + no anomaly -> CONSISTENT
2. personal payee + business poster -> AMBIGUOUS
3. identity contradiction alone -> NOT HIGH_RISK
4. strong identity contradiction + strong independent surface anomaly -> HIGH_RISK
5. failed second-view registration -> INCONCLUSIVE or HUMAN_REVIEW
6. missing surface evidence -> does NOT count as "no anomaly" (cannot be CONSISTENT)
7. conflicting evidence -> HUMAN_REVIEW
8. missing QR -> INCONCLUSIVE
9. every result contains observable reasons
10. recommended next step is strictly one of: STOP, REQUEST_SECOND_VIEW, HUMAN_REVIEW
"""

from pathlib import Path
import unittest

from backend.vision.evidence_rules import (
    DEFAULT_THRESHOLD_CONFIG,
    EvidenceQuality,
    EvidenceThresholdConfig,
    evaluate_evidence,
)
from backend.vision.models import (
    EvidenceEvaluation,
    EvidenceState,
    ExtractedRegionEvidence,
    IdentityConsistency,
    IdentityEvidence,
    PayloadType,
    QREvidence,
    RecommendedAction,
    RegistrationStatus,
    SceneEvidence,
    SecondViewComparison,
    SecondViewEvidence,
    StructuredEvidence,
    SurfaceEvidence,
)


class TestDeterministicEvidenceEngine(unittest.TestCase):
    def setUp(self):
        self.config = DEFAULT_THRESHOLD_CONFIG

        # Base valid scene evidence
        self.valid_scene = SceneEvidence(
            width=800,
            height=600,
            channels=3,
            mean_brightness=150.0,
            blur_laplacian_var=350.0,
            is_blurry=False,
        )

        # Base valid QR evidence
        self.valid_qr = QREvidence(
            detected=True,
            decoded=True,
            payload="QRFORENSIC_TEST_PAYEE_NOVA_COFFEE",
            payload_type=PayloadType.PLAIN_TEST,
            bounding_box=[[50.0, 50.0], [250.0, 50.0], [250.0, 250.0], [50.0, 250.0]],
        )

        # Base uniform surface evidence (no anomaly)
        self.uniform_surface = SurfaceEvidence(
            boundary_gradient_mean=80.0,
            boundary_gradient_max=120.0,
            edge_discontinuity_ratio=1.1,  # Well below 3.5 strong threshold
            color_discontinuity_delta=10.0,
            qr_mean_intensity=140.0,
            ring_mean_intensity=145.0,
            texture_laplacian_energy_qr=200.0,
            texture_laplacian_energy_ring=190.0,
            qr_aspect_ratio=1.0,
            qr_diagonal_ratio=1.0,
            qr_area_px=40000.0,
            signals_available=[
                "geometry",
                "region_statistics",
                "boundary_gradients",
                "edge_discontinuity",
                "color_discontinuity",
            ],
        )

        # Strong anomalous surface evidence
        self.anomalous_surface = SurfaceEvidence(
            boundary_gradient_mean=450.0,
            boundary_gradient_max=950.0,
            edge_discontinuity_ratio=4.8,  # > 3.5 strong threshold
            color_discontinuity_delta=85.0,
            qr_mean_intensity=210.0,
            ring_mean_intensity=110.0,
            relative_specular_ratio=3.2,  # > 2.5
            relative_specular_diff=75.0,  # > 40.0
            qr_aspect_ratio=1.0,
            qr_diagonal_ratio=1.0,
            qr_area_px=40000.0,
            signals_available=[
                "geometry",
                "region_statistics",
                "boundary_gradients",
                "edge_discontinuity",
                "color_discontinuity",
                "relative_specular_response",
            ],
        )

    def test_01_consistent_identity_no_anomaly(self):
        """Test 1: consistent identity + no anomaly -> CONSISTENT with STOP."""
        identity = IdentityEvidence(
            status=IdentityConsistency.CONSISTENT,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="nova coffee",
            match_score=1.0,
            is_personal_payee=False,
        )

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity,
            surface=self.uniform_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertEqual(evaluation.state, EvidenceState.CONSISTENT)
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.STOP)
        self.assertGreater(len(evaluation.reasons), 0)
        self.assertTrue(any("aligns consistently" in r for r in evaluation.reasons))

    def test_02_personal_payee_business_poster_is_ambiguous(self):
        """Test 2: personal payee + business poster -> AMBIGUOUS (not fraud)."""
        identity = IdentityEvidence(
            status=IdentityConsistency.AMBIGUOUS,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="ramesh kumar",
            match_score=0.4,
            is_personal_payee=True,
        )

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity,
            surface=self.uniform_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertEqual(evaluation.state, EvidenceState.AMBIGUOUS)
        self.assertNotEqual(evaluation.state, EvidenceState.HIGH_RISK)
        self.assertIn(evaluation.recommended_next_step, [RecommendedAction.REQUEST_SECOND_VIEW, RecommendedAction.HUMAN_REVIEW])
        self.assertTrue(any("individual/personal name" in r for r in evaluation.reasons))

    def test_03_identity_contradiction_alone_not_high_risk(self):
        """Test 3: identity contradiction alone MUST NOT produce HIGH_RISK."""
        identity_contradiction = IdentityEvidence(
            status=IdentityConsistency.CONTRADICTORY,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="quick mart",
            match_score=0.1,
            is_personal_payee=False,
        )

        # Pair contradictory identity with UNIFORM surface (no anomaly)
        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity_contradiction,
            surface=self.uniform_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        # Critical: Contradiction alone must NOT produce HIGH_RISK
        self.assertNotEqual(evaluation.state, EvidenceState.HIGH_RISK)
        self.assertEqual(evaluation.state, EvidenceState.AMBIGUOUS)
        self.assertTrue(any("does not justify a HIGH_RISK" in r for r in evaluation.reasons))

    def test_04_contradiction_plus_strong_surface_anomaly_is_high_risk(self):
        """Test 4: strong identity contradiction + strong independent surface anomaly -> HIGH_RISK."""
        identity_contradiction = IdentityEvidence(
            status=IdentityConsistency.CONTRADICTORY,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="quick mart",
            match_score=0.1,
            is_personal_payee=False,
        )

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity_contradiction,
            surface=self.anomalous_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertEqual(evaluation.state, EvidenceState.HIGH_RISK)
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.HUMAN_REVIEW)
        self.assertTrue(any("Multiple independent evidence sources" in r for r in evaluation.reasons))

    def test_05_failed_second_view_registration_is_inconclusive_or_review(self):
        """Test 5: failed second-view registration -> INCONCLUSIVE or HUMAN_REVIEW."""
        identity = IdentityEvidence(
            status=IdentityConsistency.CONSISTENT,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="nova coffee",
            match_score=1.0,
        )

        failed_second_view = SecondViewEvidence(
            registration_status=RegistrationStatus.FAIL,
            reason="RANSAC inliers insufficient (inliers=8 < 15).",
        )

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity,
            surface=self.uniform_surface,
            second_view=failed_second_view,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertIn(evaluation.state, [EvidenceState.INCONCLUSIVE, EvidenceState.HUMAN_REVIEW])
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.HUMAN_REVIEW)
        self.assertTrue(any("failed geometric same-scene registration" in r for r in evaluation.reasons))

    def test_06_missing_surface_evidence_does_not_count_as_no_anomaly(self):
        """Test 6: missing surface evidence does not count as 'no anomaly' (cannot be CONSISTENT)."""
        identity = IdentityEvidence(
            status=IdentityConsistency.CONSISTENT,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="nova coffee",
            match_score=1.0,
        )

        # Missing surface evidence (empty/unavailable)
        empty_surface = SurfaceEvidence(signals_available=[])

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity,
            surface=empty_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        # MUST NOT be CONSISTENT when surface evidence was never measured!
        self.assertNotEqual(evaluation.state, EvidenceState.CONSISTENT)
        self.assertEqual(evaluation.state, EvidenceState.AMBIGUOUS)
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.REQUEST_SECOND_VIEW)
        self.assertTrue(any("physical surface analysis was unavailable" in r for r in evaluation.reasons))

    def test_07_conflicting_evidence_sources_is_human_review(self):
        """Test 7: conflicting evidence (e.g. matching brand + severe physical surface step) -> HUMAN_REVIEW."""
        # Payee matches brand perfectly ("Nova Coffee")
        identity = IdentityEvidence(
            status=IdentityConsistency.CONSISTENT,
            normalized_poster_names=["nova coffee"],
            normalized_payee_name="nova coffee",
            match_score=1.0,
        )

        # But surface exhibits strong edge anomaly (e.g. physical overlay cut detected)
        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=self.valid_qr,
            identity=identity,
            surface=self.anomalous_surface,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertEqual(evaluation.state, EvidenceState.HUMAN_REVIEW)
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.HUMAN_REVIEW)
        self.assertTrue(any("Discrepancy between consistent branding and physical surface discontinuity" in r for r in evaluation.reasons))

    def test_08_missing_qr_is_inconclusive(self):
        """Test 8: missing or unreadable QR code -> INCONCLUSIVE."""
        undetected_qr = QREvidence(detected=False, decoded=False)

        evidence = StructuredEvidence(
            scene=self.valid_scene,
            qr=undetected_qr,
        )

        evaluation = evaluate_evidence(evidence, self.config)

        self.assertEqual(evaluation.state, EvidenceState.INCONCLUSIVE)
        self.assertEqual(evaluation.recommended_next_step, RecommendedAction.REQUEST_SECOND_VIEW)
        self.assertTrue(any("QR code was not detected" in r for r in evaluation.reasons))

    def test_09_every_result_contains_observable_reasons(self):
        """Test 9: every evaluation output must contain factual, observable reasons."""
        for state_scenario in [
            (self.uniform_surface, IdentityConsistency.CONSISTENT),
            (self.anomalous_surface, IdentityConsistency.CONTRADICTORY),
            (self.uniform_surface, IdentityConsistency.AMBIGUOUS),
        ]:
            surf, id_status = state_scenario
            evidence = StructuredEvidence(
                scene=self.valid_scene,
                qr=self.valid_qr,
                identity=IdentityEvidence(status=id_status, normalized_payee_name="test"),
                surface=surf,
            )
            ev = evaluate_evidence(evidence, self.config)
            self.assertGreater(len(ev.reasons), 0, "Evaluation must always contain non-empty reasons")
            self.assertTrue(all(isinstance(r, str) and len(r) > 5 for r in ev.reasons))

    def test_10_recommended_step_is_strictly_valid_enum(self):
        """Test 10: recommended next step must be strictly one of STOP, REQUEST_SECOND_VIEW, HUMAN_REVIEW."""
        valid_actions = {
            RecommendedAction.STOP,
            RecommendedAction.REQUEST_SECOND_VIEW,
            RecommendedAction.HUMAN_REVIEW,
        }

        # Test across various combinations
        test_evidences = [
            StructuredEvidence(scene=self.valid_scene, qr=self.valid_qr, identity=IdentityEvidence(status=IdentityConsistency.CONSISTENT), surface=self.uniform_surface),
            StructuredEvidence(scene=self.valid_scene, qr=self.valid_qr, identity=IdentityEvidence(status=IdentityConsistency.AMBIGUOUS), surface=self.uniform_surface),
            StructuredEvidence(scene=self.valid_scene, qr=self.valid_qr, identity=IdentityEvidence(status=IdentityConsistency.CONTRADICTORY), surface=self.anomalous_surface),
            StructuredEvidence(scene=self.valid_scene, qr=QREvidence(detected=False)),
        ]

        for ev in test_evidences:
            res = evaluate_evidence(ev, self.config)
            self.assertIn(res.recommended_next_step, valid_actions)
            self.assertIsInstance(res.recommended_next_step.value, str)


if __name__ == "__main__":
    unittest.main()
