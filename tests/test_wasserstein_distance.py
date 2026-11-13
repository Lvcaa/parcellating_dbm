"""Smoke tests for reusable parcel-similarity distance computations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "graph_building"))

from wasserstein_distance import WassersteinDistanceComputer


class WassersteinDistanceComputerSmokeTest(unittest.TestCase):
    def test_wasserstein_matrix_and_weighted_degree(self) -> None:
        parcels = [
            np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
        ]
        computer = WassersteinDistanceComputer(quantile_count=4)
        features = computer.prepare_features(parcels)

        result = computer.compute(
            features,
            block_size=2,
            compute_weighted_degree=True,
            return_similarity_matrix=True,
        )

        distances = np.array(
            [
                [wasserstein_distance(left, right) for right in parcels]
                for left in parcels
            ]
        )
        expected_similarity = np.exp(-distances)
        np.testing.assert_allclose(result.similarity_matrix, expected_similarity)
        np.testing.assert_allclose(
            result.weighted_degree,
            (expected_similarity.sum(axis=1) - 1.0) / (len(parcels) - 1),
        )
        np.testing.assert_allclose(
            computer.similarity_block(features, 1, 3), expected_similarity[1:3]
        )

        degree_only = computer.compute(
            features,
            block_size=2,
            compute_weighted_degree=True,
            return_similarity_matrix=False,
        )
        self.assertIsNone(degree_only.similarity_matrix)
        np.testing.assert_allclose(degree_only.weighted_degree, result.weighted_degree)


if __name__ == "__main__":
    unittest.main()
