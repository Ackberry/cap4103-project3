# Check split separation, perturbation units, and claimed-user scoring.
import unittest
import numpy as np
from biometrics import face_features, split_users, paired_probes, distance_scores, score_groups
from evaluation import evaluate


class PipelineTests(unittest.TestCase):
    def test_disjoint_splits(self):
        labels = np.repeat(np.arange(4), [5, 6, 10, 25])
        parts = split_users(labels, np.random.default_rng(42))
        self.assertEqual(len(np.unique(np.concatenate(parts))), len(labels))
        for part in parts:
            self.assertEqual(set(labels[part]), set(labels))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertEqual(len(np.intersect1d(parts[a], parts[b])), 0)

    def test_error_units_and_no_mutation(self):
        face = np.array([[[0., 0.], [10., 0.], [5., 4.], [2., 8.], [8., 8.]]])
        original = face.copy()
        clean = face_features(face).reshape(1, 5, 2)
        changed = face_features(face, 'combined', .075).reshape(1, 5, 2)
        np.testing.assert_allclose(changed[0, 3] - clean[0, 3], [-.075, -.075])
        np.testing.assert_allclose(changed[0, 4] - clean[0, 4], [.075, 0])
        np.testing.assert_array_equal(face, original)
        np.testing.assert_allclose(face_features(face * 3 + 7), face_features(face))

    def test_claimed_identity_distance(self):
        enrollment = np.array([[0.], [2.], [10.], [12.]])
        users = np.array([0, 1])
        scores = distance_scores(enrollment, np.array([0, 0, 1, 1]), np.array([[1.], [11.]]), users, 2)
        genuine, impostor = score_groups(scores, users, users)
        np.testing.assert_allclose(genuine, [-1, -1])
        np.testing.assert_allclose(impostor, [-10, -10])
        self.assertEqual(evaluate(genuine, impostor)['eer'], 0)
        self.assertEqual(evaluate([-1, -1], [-1, -1])['eer'], .5)
        with self.assertRaises(ValueError):
            distance_scores(enrollment, np.array([0, 0, 1, 1]), np.array([[1.]]), users, 3)

    def test_pairing(self):
        fl, vl = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1, 1, 1])
        f, v = paired_probes(np.arange(4), np.arange(6), fl, vl, np.array([0, 1]), np.random.default_rng(42))
        np.testing.assert_array_equal(fl[f], vl[v])
        self.assertEqual(len(set(v)), len(v))


if __name__ == '__main__':
    unittest.main()
