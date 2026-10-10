"""Export NBS matrices from test-split warps using the 1,000-parcel atlas.

Inputs are already validated. Keep parcels with at least 15 voxels and reuse
wasserstein_distance_graph2's quantile representation and was similarities.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import savemat

from wasserstein_distance_graph2 import (
    PROJECT_ROOT,
    SIM_FORMULA_LABELS,
    _compute_block,
    _to_quantile_grid,
)

DEFAULT_ROIS_DIR = PROJECT_ROOT / 'outputs/atlases/atlas-1000-nocsf-from-voxels/rois'


def atlas_parcels(rois_dir):
    """Read voxel counts in numeric anatomical-label and filename order."""
    parcels = []
    for folder in sorted(rois_dir.glob('[0-9]*'), key=lambda path: int(path.name)):
        manifest = json.loads((folder / 'label_manifest.json').read_text())
        for parcel in sorted(manifest['parcels'], key=lambda parcel: parcel['file']):
            parcels.append((folder / parcel['file'], parcel['voxel_count']))
    return parcels


def subject_similarity(log_jacobian_path, parcel_indices, sim_formula):
    values = nib.load(log_jacobian_path).get_fdata(dtype=np.float32).ravel()
    quantiles = np.stack([_to_quantile_grid(values[indices]) for indices in parcel_indices])
    return _compute_block(0, len(parcel_indices), quantiles, sim_formula)[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, default=PROJECT_ROOT / 'data/splits.json')
    parser.add_argument('--warps-root', type=Path, default=PROJECT_ROOT / 'data/warps')
    parser.add_argument('--rois-dir', type=Path, default=DEFAULT_ROIS_DIR)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'outputs/nbs/similarities_was_expW.mat')
    parser.add_argument('--method', choices=['was'], default='was')
    parser.add_argument('--sim-formula', type=int, choices=[1, 2], default=1)
    parser.add_argument('--dry-run', action='store_true', help='Report cohort, parcel exclusion and RAM; do not compute.')
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    subjects, groups, warps = [], [], []


    for group, split in enumerate(['test_healthy', 'test_unhealthy']):
        for subject_id in splits[split]:
            subject = f'sub-{subject_id}'
            warp = args.warps_root / subject / 'Reg_/_SyN1Warp.nii.gz'
            
            # Some test subjects have no local warp; use the available cohort.
            if warp.exists():
                subjects.append(subject)
                groups.append(group)
                warps.append(warp)

    parcels = atlas_parcels(args.rois_dir)
    retained = [path for path, size in parcels if size >= 15]
    excluded = [path for path, size in parcels if size < 15]
    parcel_count = len(retained)
    stack_gib = parcel_count ** 2 * len(subjects) * 8 / 2 ** 30
    print(f'Subjects: {groups.count(0)} healthy + {groups.count(1)} unhealthy', flush=True)
    print(f'Parcels: {parcel_count} retained, {len(excluded)} excluded (<15 voxels)', flush=True)
    print(f'Float64 stack: {stack_gib:.2f} GiB; allow ~{2 * stack_gib + 1:.1f} GiB including MAT export and image buffers.', flush=True)
    print('Parallelism: one subject at a time, one ANTs thread.', flush=True)
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parcel_indices = [np.flatnonzero(np.asanyarray(nib.load(path).dataobj)) for path in retained]
    matrices = np.empty((parcel_count, parcel_count, len(subjects)), dtype=np.float64, order='F')
    environment = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1')
    for index, (subject, warp) in enumerate(zip(subjects, warps)):
        log_path = args.output.parent / 'log_jacobians' / f'{subject}_logJacobian.nii.gz'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists():
            subprocess.run(['CreateJacobianDeterminantImage', '3', str(warp), str(log_path), '1', '0'],
                           env=environment, check=True)
        matrices[:, :, index] = subject_similarity(log_path, parcel_indices, args.sim_formula)
        print(f'{index + 1}/{len(subjects)}: {subject}', flush=True)

    savemat(args.output, {'similarity_matrices': matrices})
    args.output.with_suffix('.json').write_text(json.dumps({
        'subjects_in_matrix_order': subjects,
        'unhealthy_in_matrix_order': groups,
        'retained_parcels_in_matrix_order': [str(path.relative_to(args.rois_dir)) for path in retained],
        'excluded_parcels': [str(path.relative_to(args.rois_dir)) for path in excluded],
        'atlas': str(args.rois_dir),
        'method': args.method,
        'similarity_formula': SIM_FORMULA_LABELS[args.sim_formula],
        'quantile_count': 15,
        'jacobian_geometric': False,
    }, indent=2) + '\n')
    args.output.with_suffix('.subjects.txt').write_text('\n'.join(subjects) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
