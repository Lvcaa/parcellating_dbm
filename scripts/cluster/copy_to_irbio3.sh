#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CLUSTER_HOST="${CLUSTER_HOST:-irbio-3}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/luca.galli-1/neuro3ducate/parcellating_dbm}"

usage() {
    cat <<'EOF'
Usage:
  scripts/cluster/copy_to_irbio3.sh [extra-file-or-alias]

Always copied:
  - requirements.txt
  - containers/Dockerfile

Optional extra file:
  - a repo-relative path, for example:
      scripts/benchmarks/fake_matrix_stress_test.py
  - or one of these aliases:
      fake:matrixstress         -> scripts/benchmarks/fake_matrix_stress_test.py
      fake:matrixstress:sparse  -> scripts/benchmarks/fake_matrix_stress_test_sparse.py
      phase1:feasibility        -> scripts/benchmarks/phase1_feasibility.py
EOF
}

resolve_extra_file() {
    case "$1" in
        fake:matrixstress)
            echo "scripts/benchmarks/fake_matrix_stress_test.py"
            ;;
        fake:matrixstress:sparse)
            echo "scripts/benchmarks/fake_matrix_stress_test_sparse.py"
            ;;
        phase1:feasibility)
            echo "scripts/benchmarks/phase1_feasibility.py"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 1
fi

files=(
    "requirements.txt"
    "containers/Dockerfile"
)

if [[ $# -eq 1 ]]; then
    files+=("$(resolve_extra_file "$1")")
fi

for rel_path in "${files[@]}"; do
    if [[ "$rel_path" = /* ]]; then
        abs_path="$rel_path"
        remote_rel="${rel_path#$PROJECT_ROOT/}"
    else
        abs_path="$PROJECT_ROOT/$rel_path"
        remote_rel="$rel_path"
    fi

    if [[ ! -f "$abs_path" ]]; then
        echo "Error: file not found: $rel_path" >&2
        exit 1
    fi

    remote_path="$REMOTE_PROJECT_DIR/$remote_rel"
    remote_dir="$(dirname "$remote_path")"

    echo "Copying $rel_path -> $CLUSTER_HOST:$remote_path"
    ssh "$CLUSTER_HOST" "mkdir -p '$remote_dir'"
    scp "$abs_path" "$CLUSTER_HOST:$remote_path"
done

echo "Copy complete."
