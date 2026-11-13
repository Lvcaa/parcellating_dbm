"""Compare ANTs non-geometric and geometric relative log-Jacobians for one warp."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executable = shutil.which("CreateJacobianDeterminantImage")
    if executable is None:
        raise FileNotFoundError("CreateJacobianDeterminantImage is not available in PATH")
    if not args.warp.is_file():
        raise FileNotFoundError(args.warp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "non_geometric": args.output_dir / "relative_logjac_non_geometric.nii.gz",
        "geometric": args.output_dir / "relative_logjac_geometric.nii.gz",
    }
    for geometric, path in ((False, paths["non_geometric"]), (True, paths["geometric"])):
        subprocess.run([
            executable,
            str(args.dimension),
            str(args.warp),
            str(path),
            "1",
            "1" if geometric else "0",
        ], check=True)

    non_geometric = np.asarray(nib.load(str(paths["non_geometric"])).dataobj, dtype=np.float64)
    geometric = np.asarray(nib.load(str(paths["geometric"])).dataobj, dtype=np.float64)
    if non_geometric.shape != geometric.shape:
        raise ValueError("Jacobian mode outputs have different shapes")
    difference = geometric - non_geometric
    report = {
        "warp": str(args.warp.resolve()),
        "shape": list(non_geometric.shape),
        "non_geometric": {
            "min": float(non_geometric.min()),
            "mean": float(non_geometric.mean()),
            "max": float(non_geometric.max()),
        },
        "geometric": {
            "min": float(geometric.min()),
            "mean": float(geometric.mean()),
            "max": float(geometric.max()),
        },
        "difference": {
            "mean": float(difference.mean()),
            "mean_abs": float(np.abs(difference).mean()),
            "max_abs": float(np.abs(difference).max()),
            "correlation": float(np.corrcoef(non_geometric.ravel(), geometric.ravel())[0, 1]),
        },
    }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
