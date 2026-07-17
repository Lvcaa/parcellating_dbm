"""Low-cost synthetic checks for pipeline arithmetic and atomic outputs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "graph_building"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "parcellation"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "parcellation" / "separate_cases"))

from export_masked_jacobian_vectors import export_vectors, write_direct_summaries
from pipeline_integrity import atomic_save_npy
from second_roi_test import connected_subcomponents, grow_component, parcel_targets
from wasserstein_distance_graph2 import (
    SIM_FORMULA_EXP,
    SIM_FORMULA_INV1PW,
    _compute_block,
)


class GraphArithmeticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = np.array(
            [
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0, 3.0],
                [-2.0, -1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.distances = np.array(
            [
                [wasserstein_distance(left, right) for right in self.samples]
                for left in self.samples
            ]
        )

    def test_exact_w1_and_both_similarity_formulas(self) -> None:
        _, _, exp_similarity = _compute_block(
            0, len(self.samples), self.samples, SIM_FORMULA_EXP
        )
        _, _, inv_similarity = _compute_block(
            0, len(self.samples), self.samples, SIM_FORMULA_INV1PW
        )
        np.testing.assert_allclose(exp_similarity, np.exp(-self.distances), rtol=1e-6)
        np.testing.assert_allclose(
            inv_similarity, 1.0 / (1.0 + self.distances), rtol=1e-6
        )

    def test_weighted_degree_excludes_self_and_uses_n_minus_one(self) -> None:
        _, _, similarity = _compute_block(
            0, len(self.samples), self.samples, SIM_FORMULA_EXP
        )
        observed = (similarity.sum(axis=1, dtype=np.float64) - 1.0) / (
            len(self.samples) - 1
        )
        expected = np.array(
            [
                np.delete(similarity[row], row).mean(dtype=np.float64)
                for row in range(len(self.samples))
            ]
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


class ConnectedParcellationTest(unittest.TestCase):
    def test_targets_and_six_connectivity(self) -> None:
        coordinates = np.argwhere(np.ones((4, 4, 3), dtype=bool))
        targets = parcel_targets(len(coordinates), 4)
        owners = grow_component(coordinates, len(targets))
        observed = np.bincount(owners, minlength=len(targets))
        np.testing.assert_array_equal(observed, targets)
        for parcel_id in range(len(targets)):
            self.assertEqual(connected_subcomponents(coordinates[owners == parcel_id]), 1)


class ExtractionAndAtomicIoTest(unittest.TestCase):
    def test_atomic_npy_and_masked_direct_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            npy_path = root / "atomic.npy"
            expected_array = np.array([1.0, 2.0], dtype=np.float32)
            atomic_save_npy(npy_path, expected_array)
            np.testing.assert_array_equal(np.load(npy_path), expected_array)

            affine = np.eye(4)
            jacobian_values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            jacobian_path = root / "log_jacobian.nii.gz"
            nib.save(nib.Nifti1Image(jacobian_values, affine), jacobian_path)

            mask_paths = []
            for parcel_number, coordinates in enumerate(
                ([(0, 0, 0), (0, 0, 1)], [(1, 1, 0), (1, 1, 1)]), start=1
            ):
                mask = np.zeros((2, 2, 2), dtype=np.uint8)
                for coordinate in coordinates:
                    mask[coordinate] = 1
                path = root / f"roi_{parcel_number:04d}.nii.gz"
                nib.save(nib.Nifti1Image(mask, affine), path)
                mask_paths.append(path)

            subject_dir = root / "subject"
            label_dir = subject_dir / "label_42"
            export_vectors(mask_paths, jacobian_path, label_dir, num_workers=1, force=True)
            parcel_ids = ["label_42/roi_0001", "label_42/roi_0002"]
            write_direct_summaries(subject_dir, parcel_ids)

            np.testing.assert_array_equal(
                np.load(label_dir / "roi_0001.npy"), jacobian_values[0, 0, :]
            )
            np.testing.assert_array_equal(
                np.load(label_dir / "roi_0002.npy"), jacobian_values[1, 1, :]
            )
            np.testing.assert_allclose(
                np.fromfile(subject_dir / "direct_mean.dat", dtype=np.float64),
                [0.5, 6.5],
            )
            np.testing.assert_allclose(
                np.fromfile(subject_dir / "direct_median.dat", dtype=np.float64),
                [0.5, 6.5],
            )


if __name__ == "__main__":
    unittest.main()
