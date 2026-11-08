# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Check prepared training assets and the built container before an RL job starts."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


sys.dont_write_bytecode = True

from omegaconf import DictConfig, OmegaConf


if __package__:
    from .swe_source_mount import check_image_binding, check_runtime_imports, check_snapshot
else:
    from swe_source_mount import check_image_binding, check_runtime_imports, check_snapshot


@dataclass(frozen=True)
class HardwareProfile:
    """One supported whole-node allocation and its training/generation split."""

    machine: str
    gpu_names: tuple[str, ...]
    image_platform: str
    gpus_per_node: int
    generation_gpus: int


DEFAULT_HARDWARE_PROFILE = "arm64-b300"
HARDWARE_PROFILES = {
    DEFAULT_HARDWARE_PROFILE: HardwareProfile("aarch64", ("B300", "GB300"), "linux/arm64", 4, 2),
    "x86_64-h100": HardwareProfile("x86_64", ("H100",), "linux/amd64", 8, 4),
}


def file_sha256(path: Path) -> str:
    """Hash a large model/container file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_image_identity(image: dict[str, object], *, expected_platform: str) -> None:
    """Keep an exported SquashFS identity separate from its immutable OCI base."""
    image_format = image.get("image_format", "oci")
    if image_format == "oci":
        reference = image.get("image")
        if "base_image" in image:
            raise ValueError("OCI image provenance must not also describe a derived SquashFS")
    elif image_format == "derived-squashfs":
        reference = image.get("base_image")
        if "image" in image:
            raise ValueError("A derived SquashFS has no derived OCI image identity")
        for key in ("base_squashfs_sha256", "source_archive_sha256", "derivation_sha256", "environment_audit_sha256"):
            value = image.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Derived SquashFS provenance requires {key}")
    else:
        raise ValueError(f"Unsupported container provenance format: {image_format}")
    if (
        image.get("platform") != expected_platform
        or not isinstance(reference, str)
        or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", reference)
    ):
        raise ValueError(f"Container provenance requires an immutable {expected_platform} image or base image")


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


def check_megatron_activation(config: DictConfig, *, model_path: Path) -> None:
    """Reject Nemotron-H's unsupported bias/activation fusion before submitting workers.

    Bridge selects squared ReLU for this architecture. RL then applies the recipe's
    fusion flag, so the generic RL default can invalidate an otherwise valid provider.
    This checks that specific conflict without importing GPU libraries or loading weights.
    """
    model = json.loads((model_path / "config.json").read_text())
    if (
        model.get("model_type") == "nemotron_h"
        and config.policy.megatron_cfg.enabled
        and config.policy.megatron_cfg.bias_activation_fusion
    ):
        raise ValueError(
            "Nemotron-H uses squared ReLU; set policy.megatron_cfg.bias_activation_fusion=false. "
            "The fused bias/activation kernel does not support this activation."
        )


def check_rl_config(prepared: Path) -> None:
    """Validate the complete controller schema in the selected RL driver environment.

    This does not start Ray, create a logger, load model weights, or allocate workers.
    Run it inside the prepared runtime before submitting training; the Gym-only host
    environment need not contain RL's dependencies.
    """
    from nemo_rl.algorithms.single_controller_utils import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(prepared / "config.yaml"), resolve=True, throw_on_missing=True)
    MasterConfig.model_validate(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--container", type=Path)
    parser.add_argument("--container-provenance", type=Path)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--source-mount", action="store_true")
    parser.add_argument(
        "--rl-config-only",
        action="store_true",
        help="Validate the full controller schema in the RL driver environment without starting training",
    )
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument(
        "--print-gpus-per-node", action="store_true", help="Print the checked allocation size for Slurm"
    )
    args = parser.parse_args()
    if args.rl_config_only:
        check_rl_config(args.prepared)
        print("RL_CONFIG_SCHEMA_PASSED")
        return
    manifest = json.loads((args.prepared / "provenance.json").read_text())
    if not args.runtime and bool(manifest.get("source_mount")) != args.source_mount:
        raise ValueError("Launch mode must match preparation: use --source-mount only for a mounted preparation")
    # Older preparations used the four-B300 recipe without an explicit profile.
    profile_name = manifest.get("hardware_profile", DEFAULT_HARDWARE_PROFILE)
    if profile_name not in HARDWARE_PROFILES:
        raise ValueError(f"Unsupported hardware profile: {profile_name}")
    profile = HARDWARE_PROFILES[profile_name]
    roots = {"rl": args.rl_root.resolve(), "gym": args.gym_root.resolve()}
    if args.shared_root:
        for path in (manifest["model"]["path"], manifest["parser"]["path"]):
            if not Path(path).resolve().is_relative_to(args.shared_root.resolve()):
                raise ValueError(f"Runtime asset lies outside the shared mount: {path}")
    check_files(args.prepared, roots, manifest)
    if manifest.get("source_mount"):
        check_snapshot(roots["rl"], json.loads((args.prepared / "source-mount.json").read_text()))
    config = OmegaConf.load(args.prepared / "config.yaml")
    check_megatron_activation(config, model_path=Path(manifest["model"]["path"]))
    generation = config.policy.generation.colocated
    if (
        config.cluster.num_nodes != 1
        or config.cluster.gpus_per_node != profile.gpus_per_node
        or generation.enabled
        or generation.resources.num_nodes != 1
        or generation.resources.gpus_per_node != profile.generation_gpus
    ):
        raise ValueError(f"Prepared GPU allocation differs from hardware profile {profile_name}")
    if args.runtime:
        import nemo_rl
        import torch

        import nemo_gym

        for label, module in (("gym", nemo_gym), ("rl", nemo_rl)):
            if not Path(module.__file__).resolve().is_relative_to(roots[label]):
                raise ValueError(f"{label} imports outside the checked image source")
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        machine = platform.machine()
        gpu_pattern = r"\b(?:" + "|".join(map(re.escape, profile.gpu_names)) + r")\b"
        if (
            machine != profile.machine
            or len(names) != profile.gpus_per_node
            or any(not re.search(gpu_pattern, name) for name in names)
        ):
            raise ValueError(
                f"Hardware profile {profile_name} requires one {profile.machine} node with "
                f"{profile.gpus_per_node} {'/'.join(profile.gpu_names)} GPUs; observed {machine}, {names}"
            )
        receipt = {
            "hardware_profile": profile_name,
            "machine": machine,
            "source_files_checked": len(manifest["runtime_files_sha256"]),
            "gpu_names": names,
            "torch": torch.__version__,
            "model_content_checked_against_hub": manifest["model"]["revision"],
        }
        if manifest.get("source_mount"):
            receipt["source_mount"] = {
                "base_image": manifest["source_mount"]["base_image"],
                "executed_repositories": json.loads((args.prepared / "source-mount.json").read_text())["repositories"],
                "source_manifest_sha256": file_sha256(args.prepared / "source-mount.json"),
                "prepared_manifest_sha256": file_sha256(args.prepared / "provenance.json"),
                "imports": check_runtime_imports(config),
                "qualification": "Existing interpreter imports only; actual Ray/Gym startup and training remain separate evidence",
            }
        (args.prepared / "runtime-check.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        if args.container is None or args.container_provenance is None:
            parser.error("Host checks require --container and --container-provenance")
        image = json.loads(args.container_provenance.read_text())
        check_image_binding(manifest, image)
        check_image_identity(image, expected_platform=profile.image_platform)
        if file_sha256(args.container) != image["squashfs_sha256"]:
            raise ValueError("Container SquashFS bytes differ from the build provenance")
    print(
        profile.gpus_per_node
        if args.print_gpus_per_node
        else "Prepared inputs, source and runtime prerequisites checked"
    )


if __name__ == "__main__":
    main()
