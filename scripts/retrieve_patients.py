"""
Copy the first N healthy and N unhealthy subjects from the OASIS-3 demographics
CSV into data/healthy/ and data/unhealthy/ under the repo root.

Healthy subjects: looked up in Healthy_MRI/ by sub-XXXX, glob *T1w-enhanced.nii.gz
  (these have been fully preprocessed and enhanced).
Unhealthy subjects: sourced from the raw T1w Scan Path in the CSV
  (these were explicitly excluded from the preprocessing pipeline).
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

CSV_PATH = Path(
    "/home/lucagalli/Projects/neuroglocal/OASIS3_Pipeline_Luca"
    "/01_patient_demographics/data/database_finale_labels_corrette.csv"
)

HEALTHY_MRI_ROOT = Path(
    "/home/lucagalli/Projects/OASIS3_Pipeline_Luca"
    "/data/healthy_mri/Healthy_MRI"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def oasis_id_to_sub(oasis_id: str) -> str:
    """OAS30001 → sub-0001"""
    number = oasis_id.removeprefix("OAS3")
    return f"sub-{number}"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_healthy(df: pd.DataFrame, n: int, dest_dir: Path) -> None:
    subset = df[df["HStatus"] == "Healthy"].head(n)
    reset_dir(dest_dir)

    for _, row in subset.iterrows():
        sub_id = oasis_id_to_sub(row["OASISID"])
        matches = list((HEALTHY_MRI_ROOT / sub_id).glob("**/*[Tt]1w-enhanced.nii.gz"))

        if not matches:
            print(f"  [WARN] no T1w-enhanced image found for {sub_id}, skipping")
            continue

        src = matches[0]
        shutil.copy2(src, dest_dir / src.name)
        print(f"  copied {src.name}")


def copy_unhealthy(df: pd.DataFrame, n: int, dest_dir: Path) -> None:
    # Unhealthy subjects were never preprocessed; use the raw T1w path from the CSV.
    subset = df[df["HStatus"] == "Unhealthy"].head(n)
    reset_dir(dest_dir)

    for _, row in subset.iterrows():
        src = Path(row["T1w Scan Path"])

        if not src.exists():
            print(f"  [WARN] raw T1w not found for {row['OASISID']}, skipping: {src}")
            continue

        shutil.copy2(src, dest_dir / src.name)
        print(f"  copied {src.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve the first N healthy and N unhealthy OASIS-3 subjects."
    )
    parser.add_argument(
        "-n", "--n-subjects",
        type=int,
        default=20,
        metavar="N",
        help="Number of healthy AND unhealthy subjects to retrieve (default: 20).",
    )
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)

    print(f"Copying {args.n_subjects} healthy subjects ...")
    copy_healthy(df, args.n_subjects, REPO_ROOT / "data" / "healthy")

    print(f"Copying {args.n_subjects} unhealthy subjects ...")
    copy_unhealthy(df, args.n_subjects, REPO_ROOT / "data" / "unhealthy")

    print("Done.")


if __name__ == "__main__":
    main()
