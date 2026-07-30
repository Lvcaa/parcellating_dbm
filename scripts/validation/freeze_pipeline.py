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


def atlas_manifest_identity(path: Path) -> str:
    """
    Content-stable identity for the atlas manifest: its `atlas_id` field,
    not a raw byte hash. check_roi_counts.py stamps a fresh run_date/run_time
    into atlas_manifest.json on every invocation, so hashing the whole file
    would report a "changed" atlas on every resume even when the underlying
    parcellation (what atlas_id actually encodes) hasn't changed at all.
    """
    manifest = read_json(path)
    atlas_id = manifest.get("atlas_id")
    if not atlas_id:
        raise ValueError(f"Atlas manifest missing atlas_id: {path}")
    return atlas_id


def current_payload(args: argparse.Namespace) -> dict:
    files = [args.subject_list, *args.file]
    config = {}
    for item in args.config:
        key, separator, value = item.partition("=")
        if not separator or not key or key in config:
            raise ValueError(f"Invalid or duplicate --config entry: {item!r}")
        config[key] = value
    return {
        "schema_version": 2,
        "run_id": args.run_id,
        "configuration": config,
        "atlas_manifest": {
            str(args.atlas_manifest.resolve()): atlas_manifest_identity(args.atlas_manifest),
        },
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
            path for path, digest in {**payload["files"], **payload["atlas_manifest"]}.items()
            if {**expected.get("files", {}), **expected.get("atlas_manifest", {})}.get(path) != digest
        )
        if payload["configuration"] != expected.get("configuration"):
            raise RuntimeError(
                f"Frozen pipeline configuration changed: {expected.get('configuration')} -> {payload['configuration']}"
            )
        raise RuntimeError(f"Frozen pipeline inputs changed: {changed}")


if __name__ == "__main__":
    main()
