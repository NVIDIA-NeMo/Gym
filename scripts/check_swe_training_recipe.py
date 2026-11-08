# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check prepared training assets and the built container before an RL job starts."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash a large model/container file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_files(prepared: Path, roots: dict[str, Path], manifest: dict) -> None:
    """Check exact prepared bytes and production sources, independent of Git metadata in images."""
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(prepared / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    for name, expected in manifest["runtime_files_sha256"].items():
        label, relative = name.split("/", 1)
        if file_sha256(roots[label] / relative) != expected:
            raise ValueError(f"Runtime source does not match the prepared revision: {name}")
    model = Path(manifest["model"]["path"])
    if model.name != manifest["model"]["revision"] or not (model / "config.json").is_file():
        raise ValueError("The pinned model snapshot must exist before submission")
    indexes = list(model.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("Expected one model weight index in the pinned snapshot")
    weights = set(json.loads(indexes[0].read_text())["weight_map"].values()) if indexes else {"model.safetensors"}
    if not weights or any(not (model / name).is_file() for name in weights):
        raise ValueError("Pinned model snapshot is missing weight shards")
    for name, expected in manifest["model"]["files"].items():
        path = model / name
        if expected["algorithm"] == "sha256":
            actual = file_sha256(path)
        elif expected["algorithm"] == "git_blob_sha1":
            data = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        else:
            raise ValueError(f"Unsupported model content hash: {expected['algorithm']}")
        if actual != expected["digest"]:
            raise ValueError(f"Model bytes differ from the pinned Hub revision: {name}")
    parser = manifest["parser"]
    revision = subprocess.check_output(["git", "-C", parser["path"], "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", parser["path"], "diff", "--quiet", "HEAD"], check=True)
    if revision != parser["commit"]:
        raise ValueError("Verifier parser source changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--container", type=Path)
    parser.add_argument("--container-provenance", type=Path)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--shared-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "provenance.json").read_text())
    roots = {"rl": args.rl_root.resolve(), "gym": args.gym_root.resolve()}
    if args.shared_root:
        for path in (manifest["model"]["path"], manifest["parser"]["path"]):
            if not Path(path).resolve().is_relative_to(args.shared_root.resolve()):
                raise ValueError(f"Runtime asset lies outside the shared mount: {path}")
    check_files(args.prepared, roots, manifest)
    if args.runtime:
        import nemo_rl
        import torch

        import nemo_gym

        for label, module in (("gym", nemo_gym), ("rl", nemo_rl)):
            if not Path(module.__file__).resolve().is_relative_to(roots[label]):
                raise ValueError(f"{label} imports outside the checked image source")
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if platform.machine() != "aarch64" or len(names) != 4 or any("B300" not in name for name in names):
            raise ValueError(f"This recipe requires one ARM64 node with four GB300 GPUs; observed {names}")
        receipt = {
            "source_files_checked": len(manifest["runtime_files_sha256"]),
            "gpu_names": names,
            "torch": torch.__version__,
            "model_content_checked_against_hub": manifest["model"]["revision"],
        }
        (args.prepared / "runtime-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        if args.container is None or args.container_provenance is None:
            parser.error("Host checks require --container and --container-provenance")
        image = json.loads(args.container_provenance.read_text())
        for key in ("gym_commit", "rl_commit", "gym_lock_sha256", "rl_lock_sha256"):
            if image[key] != manifest[key]:
                raise ValueError(f"Container build provenance differs from the prepared recipe: {key}")
        if image["platform"] != "linux/arm64" or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image["image"]):
            raise ValueError("Container provenance requires an immutable Linux ARM64 image")
        if file_sha256(args.container) != image["squashfs_sha256"]:
            raise ValueError("Container SquashFS bytes differ from the build provenance")
    print("Prepared inputs, source and runtime prerequisites checked")


if __name__ == "__main__":
    main()
