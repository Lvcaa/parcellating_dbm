"""Exercise the committed connected-atlas workflow without ANTs or cohort data."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConnectedWorkflowTest(unittest.TestCase):
    def run_script(self, script: str, *arguments: object, succeeds: bool = True):
        command = [sys.executable, str(PROJECT_ROOT / script)]
        command.extend(str(argument) for argument in arguments)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_connected_atlas_to_validated_wasserstein_cohort(self):
        # Retain all small synthetic fixtures for inspection.
        root = Path(tempfile.mkdtemp(prefix="dbm-connected-workflow-"))
        affine = np.eye(4)
        segmentation = root / "segmentation.nii.gz"
        nib.save(nib.Nifti1Image(np.full((4, 4, 3), 3, dtype=np.int16), affine), segmentation)
        atlas = root / "atlas"
        self.run_script(
            "scripts/parcellation/separate_cases/second_parcellation_method.py",
            "--segmentation", segmentation, "--roi-label", 3,
            "--target-parcels", 3, "--workers", 1, "--output-root", atlas,
        )
        atlas_manifest = atlas / "atlas_manifest.json"
        atlas_record = json.loads(atlas_manifest.read_text())
        self.assertEqual(atlas_record["parcel_count"], 3)
        self.assertNotIn("atlas_id", atlas_record)

        subjects = ["sub-0001", "sub-0002"]
        subject_list = root / "subjects.txt"
        subject_list.write_text("\n".join(subjects) + "\n")
        database = root / "cohort.csv"
        with database.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["OASISID", "HStatus"])
            writer.writerow(["OAS30001", "Healthy"])
            writer.writerow(["OAS30002", "Unhealthy"])
        run = root / "run"
        freeze_arguments = (
            "--manifest", run / "run_manifest.json", "--run-id", "synthetic",
            "--atlas-manifest", atlas_manifest, "--subject-list", subject_list,
            "--file", database, "--config", "method=was", "--config", "sim_formula=both",
        )
        freeze_script = "scripts/validation/freeze_pipeline.py"
        self.run_script(freeze_script, "create", *freeze_arguments)
        self.run_script(freeze_script, "check", *freeze_arguments)

        for index, subject in enumerate(subjects):
            jacobian = root / f"{subject}_to_template_logJacobian.nii.gz"
            values = np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 48
            values = values * (index + 1)
            nib.save(nib.Nifti1Image(values, affine), jacobian)
            self.run_script(
                "scripts/parcellation/export_masked_jacobian_vectors.py",
                "--jacobian", jacobian, "--rois-dir", atlas,
                "--output-dir", run / "jacobian_parcel_vectors", "--num-workers", 1,
            )
            for formula, name in [(1, "expW"), (2, "inv1pW")]:
                output = run / f"wasserstein_graphs_{name}" / subject
                self.run_script(
                    "scripts/graph_building/wasserstein_distance_graph2.py",
                    "--input-folder", run / "jacobian_parcel_vectors" / subject,
                    "--output-folder", output, "--method", "was",
                    "--sim-formula", formula, "--num-workers", 1, "--block-size", 2,
                )
                degree = np.fromfile(output / "weighted_degree.dat", dtype=np.float64)
                self.assertEqual(degree.shape, (3,))
                self.assertTrue(np.isfinite(degree).all())
                self.assertFalse((output / "adjacency_matrix.dat").exists())

        self.run_script(
            "scripts/validation/validate_cohort_outputs.py",
            "--run-root", run, "--subjects", subject_list, "--database", database,
            "--manifest", run / "run_manifest.json", "--methods", "was",
            "--expected-healthy", 1, "--expected-unhealthy", 1,
        )
        report = json.loads((run / "cohort_validation.json").read_text())
        self.assertEqual(report["validation_status"], "passed")
        self.assertEqual(report["subject_count"], 2)
        self.assertFalse(report["definitive"])

        # A changed connected manifest must invalidate the frozen run.
        atlas_record["parcel_count"] += 1
        atlas_manifest.write_text(json.dumps(atlas_record))
        self.run_script(freeze_script, "check", *freeze_arguments, succeeds=False)

    def test_legacy_manifest_identity_ignores_timestamp_only_changes(self):
        root = Path(tempfile.mkdtemp(prefix="dbm-legacy-freeze-"))
        atlas = root / "atlas.json"
        atlas.write_text(json.dumps({"atlas_id": "legacy-id", "run_date": "first"}))
        subjects = root / "subjects.txt"
        subjects.write_text("sub-0001\n")
        arguments = (
            "--manifest", root / "run.json", "--run-id", "legacy",
            "--atlas-manifest", atlas, "--subject-list", subjects,
        )
        script = "scripts/validation/freeze_pipeline.py"
        self.run_script(script, "create", *arguments)
        atlas.write_text(json.dumps({"atlas_id": "legacy-id", "run_date": "later"}))
        self.run_script(script, "check", *arguments)
        atlas.write_text(json.dumps({
            "schema_version": 2, "validation_status": "failed",
            "labels_processed_sequentially": True,
        }))
        self.run_script(script, "create", *arguments, succeeds=False)

    def test_documented_cli_entry_points_start_without_source_data(self):
        for script in [
            "scripts/parcellation/separate_cases/run_allowed_parcellations.py",
            "scripts/parcellation/separate_cases/run_high_ram_labels.py",
            "scripts/registration/compute_log_jacobian.py",
            "scripts/validation/validate_batch_inputs.py",
            "scripts/analysis/parcel_statistics.py",
        ]:
            with self.subTest(script=script):
                self.run_script(script, "--help")


if __name__ == "__main__":
    unittest.main()
