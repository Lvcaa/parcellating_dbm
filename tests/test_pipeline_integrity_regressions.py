"""Synthetic regression checks for pipeline input identities."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/parcellation'),
                str(ROOT / 'scripts/graph_building'), str(ROOT / 'scripts/validation')]
from pipeline_integrity import sha256_file, sha256_lines, write_completion
from export_masked_jacobian_vectors import load_atlas_contract
from wasserstein_distance_graph2 import graph_contract, graph_is_complete, read_input_contract
import validate_cohort_outputs


class IntegrityRegressionTest(unittest.TestCase):
    def setUp(self):
        # Keep fixtures available for inspection, as required by repository conventions.
        self.root = Path(tempfile.mkdtemp(prefix='dbm-integrity-'))

    def atlas(self):
        rows = []
        for label in (1, 2):
            directory = self.root / str(label)
            directory.mkdir()
            mask = directory / 'roi_0001.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), mask)
            label_manifest = directory / 'label_manifest.json'
            label_manifest.write_text(json.dumps({'label': label, 'parcel_count': 1,
                'validation_status': 'passed', 'parcels': [{'file': mask.name,
                'voxel_count': 8, 'sha256': sha256_file(mask)}]}))
            rows.append({'label': label, 'parcel_count': 1, 'manifest_sha256': sha256_file(label_manifest)})
        (self.root / 'atlas_manifest.json').write_text(json.dumps({'schema_version': 2,
            'validation_status': 'passed', 'labels_processed_sequentially': True, 'labels': rows}))
        return argparse.Namespace(roi_manifest=None, rois_dir=self.root, allow_unvalidated_rois=False,
                                  label=None, n_parcels=None, sample_mode='first', seed=0)

    def test_modified_mask_rejected(self):
        args = self.atlas()
        ids = ['label_1/roi_0001', 'label_2/roi_0001']
        load_atlas_contract(args, ids)
        mask = self.root / '1/roi_0001.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4)), mask)
        with self.assertRaisesRegex(ValueError, 'Mask hash mismatch'):
            load_atlas_contract(args, ids)

    def test_missing_label_requires_explicit_subset(self):
        args = self.atlas()
        ids = ['label_1/roi_0001']
        with self.assertRaisesRegex(ValueError, 'complete validated atlas'):
            load_atlas_contract(args, ids)
        args.label = 1
        atlas, _ = load_atlas_contract(args, ids)
        self.assertEqual(atlas['selection']['label'], 1)

    def test_changed_vectors_invalidate_graph(self):
        subject = self.root / 'subject'
        subject.mkdir()
        vectors = subject / 'parcel_vectors.dat'
        offsets = subject / 'parcel_offsets.dat'
        np.array([1, 2], dtype=np.float32).tofile(vectors)
        np.array([0, 1, 2], dtype=np.int64).tofile(offsets)
        ids = ['a', 'b']
        completion = {'stage': 'parcel_vectors', 'parcel_count': 2,
                      'parcel_order_sha256': sha256_lines(ids), 'atlas_id': 'atlas',
                      'vector_data_sha256': sha256_file(vectors), 'vector_offsets_sha256': sha256_file(offsets)}
        write_completion(subject, completion)
        contract = graph_contract('subject', 2, sha256_lines(ids), read_input_contract(subject, ids), 'was', 1)
        graph = self.root / 'graph'
        graph.mkdir()
        np.array([0.5, 0.5], dtype=np.float64).tofile(graph / 'weighted_degree.dat')
        write_completion(graph, contract)
        self.assertTrue(graph_is_complete(graph, contract))
        np.array([1, 9], dtype=np.float32).tofile(vectors)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_input_contract(subject, ids)
        completion['vector_data_sha256'] = sha256_file(vectors)
        write_completion(subject, completion)
        new = graph_contract('subject', 2, sha256_lines(ids), read_input_contract(subject, ids), 'was', 1)
        self.assertFalse(graph_is_complete(graph, new))

    def test_swapped_labels_rejected_at_final_gate(self):
        subjects = self.root / 'subjects.txt'
        database = self.root / 'database.csv'
        subjects.write_text('sub-1\nsub-2\n')
        database.write_text('OASISID,HStatus\nOAS31,Healthy\nOAS32,Unhealthy\n')
        manifest = self.root / 'run_manifest.json'
        manifest.write_text(json.dumps({'files': {str(subjects): sha256_file(subjects),
            str(database): sha256_file(database)}, 'configuration': {'method': 'was', 'sim_formula': 'both'}}))
        database.write_text('OASISID,HStatus\nOAS31,Unhealthy\nOAS32,Healthy\n')
        args = argparse.Namespace(run_root=self.root, subjects=subjects, database=database,
                                  expected_healthy=1, expected_unhealthy=1, report=None,
                                  manifest=manifest, methods='was')
        with patch.object(validate_cohort_outputs, 'parse_args', return_value=args):
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                validate_cohort_outputs.main()
        self.assertFalse((self.root / 'cohort_validation.json').exists())


if __name__ == '__main__':
    unittest.main()
