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


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or any(character in str(path) for character in "\n\r"):
        raise ValueError(f"unsafe serving source: {path}")
    return path.resolve(strict=True)


def _mounts(value: str) -> list[tuple[Path, str, str]]:
    mounts = []
    targets: list[PurePosixPath] = []
    for item in value.split(",") if value else []:
        fields = item.split(":")
        if len(fields) not in (2, 3) or (len(fields) == 3 and fields[2] != "ro"):
            raise ValueError(f"unsupported parser mount (expected source:target[:ro]): {item}")
        source, target = fields[:2]
        destination = PurePosixPath(target)
        if (
            not destination.is_absolute()
            or ".." in destination.parts
            or destination == PurePosixPath("/")
            or any(character in target for character in "\n\r")
        ):
            raise ValueError(f"unsafe parser mount target: {target}")
        for other in [PurePosixPath("/model"), PurePosixPath("/lustre"), *targets]:
            if destination.is_relative_to(other) or other.is_relative_to(destination):
                raise ValueError(f"overlapping parser mount target: {target}")
        targets.append(destination)
        mounts.append((_source(Path(source)), target, ":ro" if len(fields) == 3 else ""))
    return mounts


def _copy_file(source: Path, destination: Path, signature: dict) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as input_file, destination.open("xb") as output:
        while chunk := input_file.read(8 * 1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if _signature(source) != signature:
        raise ValueError(f"serving source changed while copying: {source}")
    if destination.stat().st_size != signature["size"] or _digest(destination) != digest.hexdigest():
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
    if not image.is_file() or not model.is_dir():
        raise ValueError("serving image must be a file and model must be a directory")
    mounts = _mounts(extra_mounts)
    for _, target, _ in mounts:
        if any(
            PurePosixPath(target).is_relative_to(path) or PurePosixPath(path).is_relative_to(target)
            for path in runtime_mounts
        ):
            raise ValueError(f"parser mount overlaps the node-local runtime: {target}")
    for source in [model, *(source for source, _, _ in mounts)]:
        if root.resolve().is_relative_to(source):
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
        directories.append({"source": str(source), "signature": _signature(source)})
        destination.mkdir(mode=0o700)
        for item in sorted(source.iterdir()):
            if item.is_symlink():
                raise ValueError(f"symlink in serving input tree: {item}")
            target = destination / item.name
            if item.is_dir():
                copy_tree(item, target, model_tree=model_tree)
            else:
                copy(item, target, weight=model_tree and item.suffix == ".safetensors")

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
    for index, (source, target, options) in enumerate(mounts):
        destination = root / f"mount_{index}"
        copy_tree(source, destination)
        rewritten_mounts.append(f"{destination}:{target}{options}")
    environment = {
        "CONTAINER_IMAGE": str(root / "vllm.sqsh"),
        "MODEL_PATH": str(root / "model"),
        "EXTRA_MOUNTS": ",".join([*rewritten_mounts, *(f"{path}:{path}" for path in runtime_mounts)]),
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
    verify(root, local_root=local_root)
    return manifest


def verify(root: Path, *, local_root: Path = LOCAL_ROOT) -> dict:
    root = assert_node_local(root, local_root=local_root)
    manifest_path = assert_node_local(root / "manifest.json", local_root=local_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported serving manifest")
    for path in manifest["runtime_mounts"]:
        assert_node_local(Path(path), local_root=local_root)
    for name, value in manifest["container_environment"].items():
        if os.environ.get(name) != value:
            raise ValueError(f"serving container environment changed: {name}")
        assert_node_local(Path(value), local_root=local_root)
    expected_files = {"manifest.json", "environment.sh"}
    for record in [*manifest["directories"], *manifest["files"]]:
        source = _source(Path(record["source"]))
        if _signature(source) != record["signature"]:
            raise ValueError(f"serving source changed after staging: {source}")
        if "path" not in record:
            continue
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe serving manifest path: {relative}")
        expected_files.add(str(relative))
        target = root / relative
        assert_node_local(target.parent, local_root=local_root)
        if record.get("shared_weight"):
            if relative.parts[0] != "model" or target.suffix != ".safetensors":
                raise ValueError(f"unsupported shared serving asset: {target}")
            if not target.is_symlink() or target.readlink() != source:
                raise ValueError(f"shared weight link changed: {target}")
        elif target.is_symlink() or not target.is_file() or _digest(target) != record["sha256"]:
            raise ValueError(f"local serving asset changed: {target}")
        else:
            assert_node_local(target, local_root=local_root)
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_dir():
            if str(path.relative_to(root)) not in expected_files:
                raise ValueError(f"unexpected serving asset: {path}")
        else:
            assert_node_local(path, local_root=local_root)
    environment_file = root / "environment.sh"
    if (
        not environment_file.is_file()
        or environment_file.is_symlink()
        or environment_file.read_text(encoding="utf-8") != _environment_text(manifest["environment"])
    ):
        raise ValueError("serving environment differs from its manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--extra-mounts", default="")
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "stage":
            if args.image is None or args.model is None or args.runtime_root is None:
                parser.error("stage requires --image, --model, and --runtime-root")
            stage(args.root, args.image, args.model, args.extra_mounts, runtime_root=args.runtime_root)
        else:
            verify(args.root)
        print(f"MARS_ROLLOUT_SERVING_PASS root={args.root} manifest_sha256={_digest(args.root / 'manifest.json')}")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"MARS_ROLLOUT_SERVING_FAIL: {error}") from error


if __name__ == "__main__":
    main()
