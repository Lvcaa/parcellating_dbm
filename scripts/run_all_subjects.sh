#!/bin/bash
set -euo pipefail

echo "run_all_subjects.sh is retired because the validated pipeline requires a frozen subject manifest." >&2
echo "Use: bash scripts/run_batch_pipeline.sh SUBJECTS.txt --run-id ID --rois-dir CORRECTED_ROIS [options]" >&2
exit 2
