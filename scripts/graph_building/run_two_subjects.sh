#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_SCRIPT="$SCRIPT_DIR/wasserstein_distance_graph2.py"

python "$GRAPH_SCRIPT" --subject-id sub-0006 --num-workers 8
python "$GRAPH_SCRIPT" --subject-id sub-0091 --num-workers 8
