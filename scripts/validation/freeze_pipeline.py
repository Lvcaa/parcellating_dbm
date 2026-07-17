"""Create or verify a frozen code/atlas/subject-list manifest for a batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import atomic_write_json, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "check"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--atlas-manifest", type=Path, required=True)
    parser.add_argument("--subject-list", type=Path, required=True)
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    return parser.parse_args()


def current_payload(args: argparse.Namespace) -> dict:
    files = [args.atlas_manifest, args.subject_list, *args.file]
    config = {}
    for item in args.config:
        key, separator, value = item.partition("=")
        if not separator or not key or key in config:
            raise ValueError(f"Invalid or duplicate --config entry: {item!r}")
        config[key] = value
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "configuration": config,
        "files": {
            str(path.resolve()): sha256_file(path)
            for path in files
        },
    }


def main() -> None:
    args = parse_args()
    payload = current_payload(args)
    if args.mode == "create":
        if args.manifest.exists():
            raise FileExistsError(f"Run manifest already exists: {args.manifest}")
        atomic_write_json(args.manifest, payload)
        print(f"Frozen pipeline manifest: {args.manifest}")
        return
    expected = read_json(args.manifest)
    if payload != expected:
        changed = sorted(
            path for path, digest in payload["files"].items()
            if expected.get("files", {}).get(path) != digest
        )
        raise RuntimeError(f"Frozen pipeline inputs changed: {changed}")


if __name__ == "__main__":
    main()
