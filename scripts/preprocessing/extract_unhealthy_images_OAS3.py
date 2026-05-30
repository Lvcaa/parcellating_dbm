"""
Identify the most atrophied OASIS-3 sessions from the configured FreeSurfer CSV.

The script computes an ICV-normalized composite score, keeps the worst session
per subject, prints a summary, and writes a session ID list next to this file.
The input CSV path is configured in ``CSV_PATH``.

Usage:
    python scripts/preprocessing/extract_unhealthy_images_OAS3.py [options]

Parameters:
    --top-n INT         Maximum subjects to keep; 0 removes the cap (default: 30).
    --min-score FLOAT   Minimum composite score for inclusion (default: 2.0).
    --output NAME       Output filename written next to this script
                        (default: unhealthy_sessions_OAS3.txt).

Examples:
    python scripts/preprocessing/extract_unhealthy_images_OAS3.py
    python scripts/preprocessing/extract_unhealthy_images_OAS3.py --top-n 0 --min-score 1.5
    python scripts/preprocessing/extract_unhealthy_images_OAS3.py --top-n 50 --output unhealthy_sessions_top50.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = (
    "/home/lucagalli/Projects/Parkinson_Population_Model/"
    "OASIS3_data_files/scans/FS-Freesurfer_output/resources/csv/files/"
    "OASIS3_Freesurfer_output.csv"
)

OUTPUT_DIR = Path(__file__).parent


def compute_atrophy_scores(df: pd.DataFrame) -> pd.DataFrame:
    icv = df["IntraCranialVol"]

    df = df.copy()
    df["hippo_norm"]      = df["TOTAL_HIPPOCAMPUS_VOLUME"] / icv
    df["latvent_norm"]    = (df["Left-Lateral-Ventricle_volume"] + df["Right-Lateral-Ventricle_volume"]) / icv
    df["wmh_norm"]        = df["WM-hypointensities_volume"] / icv
    df["entorhinal_norm"] = (df["lh_entorhinal_volume"] + df["rh_entorhinal_volume"]) / icv
    df["cortex_norm"]     = df["CortexVol"] / icv

    df["z_hippo"]      = -stats.zscore(df["hippo_norm"],      nan_policy="omit")
    df["z_latvent"]    =  stats.zscore(df["latvent_norm"],    nan_policy="omit")
    df["z_wmh"]        =  stats.zscore(df["wmh_norm"],        nan_policy="omit")
    df["z_entorhinal"] = -stats.zscore(df["entorhinal_norm"], nan_policy="omit")
    df["z_cortex"]     = -stats.zscore(df["cortex_norm"],     nan_policy="omit")

    df["atrophy_score"] = df[["z_hippo", "z_latvent", "z_wmh", "z_entorhinal", "z_cortex"]].mean(axis=1)
    return df


def main(top_n: int, min_score: float, output_name: str) -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} scans from {df['Subject'].nunique()} subjects.")

    df_qc = df[df["FS QC Status"].str.startswith("Pass")].copy()
    print(f"QC-passed scans: {len(df_qc)}")

    df_qc = compute_atrophy_scores(df_qc)

    # One row per subject: keep the session with the highest atrophy score
    worst_per_subject = (
        df_qc.sort_values("atrophy_score", ascending=False)
        .drop_duplicates("Subject")
    )

    selected = worst_per_subject[worst_per_subject["atrophy_score"] >= min_score]
    if top_n > 0:
        selected = selected.head(top_n)

    selected = selected.sort_values("atrophy_score", ascending=False)

    print(f"\nSelected {len(selected)} subjects (score >= {min_score}, top_n={top_n or 'all'}):")
    display_cols = [
        "Subject", "MR_session", "atrophy_score",
        "hippo_norm", "latvent_norm", "wmh_norm", "entorhinal_norm", "cortex_norm",
    ]
    print(selected[display_cols].to_string(index=False))

    # Write session list
    out_path = OUTPUT_DIR / output_name
    sessions = sorted(selected["MR_session"].tolist())
    out_path.write_text("\n".join(sessions) + "\n")
    print(f"\nSession list written to: {out_path}")

    # Summary statistics
    print(f"\nAtrophy score distribution (all QC-passed scans):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:3d}: {df_qc['atrophy_score'].quantile(p / 100):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract most-atrophied OASIS-3 sessions.")
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Keep at most this many subjects (0 = no cap, use --min-score only). Default: 30.",
    )
    parser.add_argument(
        "--min-score", type=float, default=2.0,
        help="Minimum composite atrophy score to include a subject. Default: 2.0 (≈ top 1%%).",
    )
    parser.add_argument(
        "--output", type=str, default="unhealthy_sessions_OAS3.txt",
        help="Output filename (written next to this script). Default: unhealthy_sessions_OAS3.txt.",
    )
    args = parser.parse_args()
    main(top_n=args.top_n, min_score=args.min_score, output_name=args.output)
