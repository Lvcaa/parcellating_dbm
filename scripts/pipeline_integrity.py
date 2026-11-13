"""Shared integrity and atomic-I/O helpers for the DBM pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COMPLETION_FILENAME = "complete.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_signature(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    signature: dict[str, Any] = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        signature["sha256"] = sha256_file(resolved)
    return signature


def signatures_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("size", "mtime_ns", "sha256")
    return all(left.get(key) == right.get(key) for key in keys if key in left or key in right)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(destination.suffixes)
    prefix = destination.name[: -len(suffixes)] if suffixes else destination.name
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{prefix}.", suffix=f".tmp{suffixes}", dir=destination.parent
    )
    os.close(fd)
    return Path(raw_path)


def atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(destination: Path, text: str) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npy(destination: Path, array: np.ndarray) -> None:
    temporary = _temporary_path(destination)
    try:
        np.save(temporary, array)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_raw(destination: Path, array: np.ndarray, dtype: str) -> None:
    temporary = _temporary_path(destination)
    try:
        memmap = np.memmap(temporary, dtype=dtype, mode="w+", shape=array.shape)
        memmap[:] = array
        memmap.flush()
        del memmap
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_completion(directory: Path) -> dict[str, Any] | None:
    path = directory / COMPLETION_FILENAME
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def write_completion(directory: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(directory / COMPLETION_FILENAME, payload)


def validate_raw_vector(path: Path, *, length: int, dtype: str = "float64") -> np.memmap:
    expected_size = length * np.dtype(dtype).itemsize
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"Invalid {dtype} vector size for {path}; expected {expected_size} bytes")
    vector = np.memmap(path, dtype=dtype, mode="r", shape=(length,))
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite values in {path}")
    return vector
