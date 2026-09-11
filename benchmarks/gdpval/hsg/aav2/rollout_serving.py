#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the serving image, model code, and parser mounts on the rollout node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
from pathlib import Path, PurePosixPath


if __package__:
    from .rollout_runtime import assert_node_local
else:
    from rollout_runtime import assert_node_local


SCHEMA = "gdpval.rollout-serving.v1"
LOCAL_ROOT = Path("/raid/scratch")
CONTAINER_ENV = ("TMPDIR", "RAY_TMPDIR", "HF_HOME", "HF_DATASETS_CACHE", "PYTHONPYCACHEPREFIX", "XDG_CACHE_HOME")


def _signature(path: Path) -> dict:
    info = path.stat()
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "mode": info.st_mode,
    }


def _source(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or any(character in str(path) for character in "\n\r"):
        raise ValueError(f"unsafe serving source: {path}")
    return path.resolve(strict=True)


def _parser_source(value: str) -> Path | None:
    if not value:
        return None
    fields = value.split(":")
    if len(fields) != 3 or fields[1:] != ["/parsers", "ro"] or "," in value:
        raise ValueError("unsupported parser mount: expected one directory:/parsers:ro or no mount")
    return _source(Path(fields[0]))


def _copy_file(source: Path, destination: Path, signature: dict) -> str:
    digest = hashlib.sha256()
    copied_bytes = 0
    with source.open("rb") as input_file, destination.open("xb") as output:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
            copied_bytes += output.write(chunk)
    if _signature(source) != signature:
        raise ValueError(f"serving source changed while copying: {source}")
    if copied_bytes != signature["size"] or destination.stat().st_size != copied_bytes:
        raise ValueError(f"serving copy differs from source: {destination}")
    destination.chmod(stat.S_IMODE(signature["mode"]) & 0o555 | 0o400)
    return digest.hexdigest()


def _environment_text(environment: dict[str, str]) -> str:
    return "".join(f"export {name}={shlex.quote(value)}\n" for name, value in sorted(environment.items()))


def stage(
    root: Path, image: Path, model: Path, extra_mounts: str = "", *, runtime_root: Path, local_root: Path = LOCAL_ROOT
) -> dict:
    assert_node_local(root.parent, local_root=local_root)
    runtime_root = assert_node_local(runtime_root, local_root=local_root)
    if any(character in str(root) for character in ":,\n\r"):
        raise ValueError(f"unsupported serving mount path: {root}")
    if not root.resolve().is_relative_to(runtime_root) or root.resolve() == runtime_root:
        raise ValueError("serving root must be below the node-local runtime root")
    container_environment = {}
    cache_paths = {}
    runtime_mounts = [str(runtime_root)]
    for name in CONTAINER_ENV:
        if not os.environ.get(name):
            raise ValueError(f"serving container environment is unset: {name}")
        path = assert_node_local(Path(os.environ[name]), local_root=local_root)
        if not path.is_dir():
            raise ValueError(f"serving cache must be a directory: {name}={path}")
        if any(character in str(path) for character in ":,\n\r"):
            raise ValueError(f"unsupported serving cache mount path: {path}")
        cache_paths[name] = path
        container_environment[name] = os.environ[name]
    for name, path in cache_paths.items():
        if path.is_relative_to(runtime_root):
            continue
        if name == "RAY_TMPDIR":
            runtime_mounts.append(str(path))
        elif name == "TMPDIR" and path != cache_paths["RAY_TMPDIR"] and path.is_relative_to(cache_paths["RAY_TMPDIR"]):
            # Ray's self-mount also exposes vLLM's short Unix socket directory.
            continue
        else:
            raise ValueError(f"serving cache is outside the mounted runtime: {name}={path}")
    if root.exists() or root.is_symlink():
        raise ValueError(f"refusing an existing serving stage: {root}")
    image, model = _source(image), _source(model)
    if any(character in str(model) for character in ":,\n\r"):
        raise ValueError(f"unsupported shared weight mount path: {model}")
    if not image.is_file() or not model.is_dir():
        raise ValueError("serving image must be a file and model must be a directory")
    parser_source = _parser_source(extra_mounts)
    for source in (model, parser_source):
        if source is not None and root.resolve().is_relative_to(source):
            raise ValueError(f"serving destination is inside its source: {source}")
    root.mkdir(mode=0o700)
    root = assert_node_local(root, local_root=local_root)
    records = []
    directories = []

    def copy(source: Path, destination: Path, *, weight: bool = False) -> None:
        source = _source(source)
        signature = _signature(source)
        if not stat.S_ISREG(signature["mode"]):
            raise ValueError(f"serving input is not a regular file: {source}")
        record = {"source": str(source), "path": str(destination.relative_to(root)), "signature": signature}
        if weight:
            destination.symlink_to(source)
            record["shared_weight"] = True
        else:
            record["sha256"] = _copy_file(source, destination, signature)
        records.append(record)

    def copy_tree(source: Path, destination: Path, *, model_tree: bool = False) -> None:
        source = _source(source)
        if not source.is_dir():
            raise ValueError(f"parser mount must be a directory: {source}")
        signature = _signature(source)
        directories.append({"source": str(source), "signature": signature})
        destination.mkdir(mode=0o700)
        for item in sorted(source.iterdir()):
            if item.is_symlink():
                raise ValueError(f"symlink in serving input tree: {item}")
            target = destination / item.name
            if item.is_dir():
                copy_tree(item, target, model_tree=model_tree)
            else:
                copy(item, target, weight=model_tree and item.suffix == ".safetensors")
        if _signature(source) != signature:
            raise ValueError(f"serving source changed while copying: {source}")

    copy(image, root / "vllm.sqsh")
    copy_tree(model, root / "model", model_tree=True)
    if not (root / "model/config.json").is_file() or not any(record.get("shared_weight") for record in records):
        raise ValueError("serving requires a model config and Safetensors weights")
    index = root / "model/model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError("model weight index must contain a nonempty weight map")
        for filename in weights.values():
            if not isinstance(filename, str):
                raise ValueError("model weight index must contain relative Safetensors paths")
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
                raise ValueError(f"unsafe indexed weight path: {filename}")
            if not (root / "model" / path).is_file():
                raise ValueError(f"indexed model weight is missing: {filename}")
    rewritten_mounts = []
    if parser_source is not None:
        destination = root / "mount_0"
        copy_tree(parser_source, destination)
        rewritten_mounts.append(f"{destination}:/parsers:ro")
    environment = {
        "CONTAINER_IMAGE": str(root / "vllm.sqsh"),
        "MODEL_PATH": str(root / "model"),
        "EXTRA_MOUNTS": ",".join(
            [*rewritten_mounts, *(f"{path}:{path}" for path in runtime_mounts), f"{model}:{model}:ro"]
        ),
        # Pyxis otherwise gives environment values baked into the image priority.
        "PYXIS_CONTAINER_ENV": ",".join(CONTAINER_ENV),
    }
    manifest = {
        "schema": SCHEMA,
        "environment": environment,
        "container_environment": container_environment,
        "runtime_mounts": runtime_mounts,
        "files": records,
        "directories": directories,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "manifest.json").chmod(0o400)
    (root / "environment.sh").write_text(_environment_text(environment), encoding="utf-8")
    (root / "environment.sh").chmod(0o400)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--extra-mounts", default="")
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        stage(args.root, args.image, args.model, args.extra_mounts, runtime_root=args.runtime_root)
        receipt_digest = hashlib.sha256((args.root / "manifest.json").read_bytes()).hexdigest()
        print(f"MARS_ROLLOUT_SERVING_PASS root={args.root} manifest_sha256={receipt_digest}")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"MARS_ROLLOUT_SERVING_FAIL: {error}") from error


if __name__ == "__main__":
    main()
