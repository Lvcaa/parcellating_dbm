# Cluster File Copy Workflow

Use the single helper below to copy files to `irbio-3` with `scp`.

## Defaults

- Host: `irbio-3`
- Remote base dir: `/home/luca.galli-1/neuro3ducate/parcellating_dbm`

The script always copies:

- `requirements.txt`
- `containers/Dockerfile`

You can also pass one extra file or alias to copy.

## Usage

Copy only the always-required files:

```bash
scripts/cluster/copy_to_irbio3.sh
```

Copy those files plus `fake_matrix_stress_test.py`:

```bash
scripts/cluster/copy_to_irbio3.sh fake:matrixstress
```

Copy those files plus `fake_matrix_stress_test_sparse.py`:

```bash
scripts/cluster/copy_to_irbio3.sh fake:matrixstress:sparse
```

Copy those files plus `phase1_feasibility.py`:

```bash
scripts/cluster/copy_to_irbio3.sh phase1:feasibility
```

You can also pass a repo-relative path directly:

```bash
scripts/cluster/copy_to_irbio3.sh scripts/benchmarks/fake_matrix_stress_test_sparse.py
```

## Notes

The helper preserves the relative path on the remote side, so:

- `requirements.txt` goes to `/home/luca.galli-1/neuro3ducate/parcellating_dbm/requirements.txt`
- `containers/Dockerfile` goes to `/home/luca.galli-1/neuro3ducate/parcellating_dbm/containers/Dockerfile`
- `scripts/benchmarks/fake_matrix_stress_test.py` goes to `/home/luca.galli-1/neuro3ducate/parcellating_dbm/scripts/benchmarks/fake_matrix_stress_test.py`

Override the host or destination directory if needed:

```bash
CLUSTER_HOST=irbio-3 REMOTE_PROJECT_DIR=/some/other/path scripts/cluster/copy_to_irbio3.sh fake:matrixstress
```
