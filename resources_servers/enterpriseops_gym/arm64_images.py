# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild native ARM64 EnterpriseOps service images from public service SIFs."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from email.parser import Parser
from pathlib import Path

from resources_servers.enterpriseops_gym.runtime import SERVICES, EnterpriseOpsService


def requirements_from_metadata(rootfs: Path) -> list[str]:
    """Recover the exact Python package versions without executing a foreign image."""
    packages: dict[str, str] = {}
    for metadata_path in rootfs.glob("**/site-packages/*.dist-info/METADATA"):
        metadata = Parser().parsestr(metadata_path.read_text(errors="replace"))
        name = metadata.get("Name")
        version = metadata.get("Version")
        if not name or not version:
            continue
        existing = packages.setdefault(name, version)
        if existing != version:
            raise RuntimeError(f"conflicting versions for {name}: {existing} and {version}")
    if not packages:
        raise RuntimeError(f"no Python package metadata found in {rootfs}")
    return [f"{name}=={packages[name]}" for name in sorted(packages, key=str.lower)]


def render_definition(rootfs: Path, requirements_path: Path) -> str:
    """Return an ARM64 Apptainer definition that preserves the upstream application payload."""
    return f"""Bootstrap: docker
From: python:3.11.14-slim

%files
    {rootfs / "app"} /app
    {requirements_path} /opt/requirements.txt

%post
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates
    rm -rf /var/lib/apt/lists/*
    python -m pip install --no-cache-dir -r /opt/requirements.txt

%environment
    export PYTHONPATH=/app

%runscript
    exec python -m uvicorn main:app --host 127.0.0.1 --port "${{PORT:-8005}}"
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(output_sif: Path, definition_path: Path, *, use_sudo: bool) -> list[str]:
    command = ["apptainer", "build", "--arch", "arm64", "--force", str(output_sif), str(definition_path)]
    if use_sudo:
        return ["sudo", *command]
    return ["apptainer", "build", "--fakeroot", "--arch", "arm64", "--force", str(output_sif), str(definition_path)]


def source_sif_path(service: EnterpriseOpsService, source_cache_dir: Path) -> Path:
    digest = service.image.rsplit("@", 1)[1].removeprefix("sha256:")
    return source_cache_dir / f"{service.domain}-{digest}.sif"


def pull_source_sif(service: EnterpriseOpsService, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["apptainer", "pull", "--arch", "amd64", str(target), f"docker://{service.image}"],
        check=True,
    )


def rebuild(source_sif: Path, output_sif: Path, *, use_sudo: bool = False) -> None:
    """Extract an image payload, install ARM64 wheels, and build a native SIF."""
    source_sif = source_sif.resolve()
    output_sif = output_sif.resolve()
    if not source_sif.is_file():
        raise FileNotFoundError(source_sif)

    with tempfile.TemporaryDirectory(prefix="enterpriseops-arm64-") as temporary_dir:
        temporary = Path(temporary_dir)
        filesystem = temporary / "rootfs.squashfs"
        rootfs = temporary / "rootfs"
        with filesystem.open("wb") as filesystem_handle:
            subprocess.run(
                ["apptainer", "sif", "dump", "4", str(source_sif)],
                check=True,
                stdout=filesystem_handle,
            )
        subprocess.run(["unsquashfs", "-no-progress", "-d", str(rootfs), str(filesystem)], check=True)
        if not (rootfs / "app").is_dir():
            raise RuntimeError(f"EnterpriseOps application payload missing from {source_sif}")

        requirements_path = temporary / "requirements.txt"
        requirements = requirements_from_metadata(rootfs)
        requirements_path.write_text("\n".join(requirements) + "\n")
        definition_path = temporary / "enterpriseops-arm64.def"
        definition_path.write_text(render_definition(rootfs, requirements_path))

        output_sif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_command(output_sif, definition_path, use_sudo=use_sudo), check=True)

    output_sif.with_suffix(output_sif.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "source_sif": str(source_sif),
                "source_sif_sha256": _sha256(source_sif),
                "rebuilt_sif": str(output_sif),
                "rebuilt_sif_sha256": _sha256(output_sif),
                "architecture": "arm64",
            },
            indent=2,
        )
        + "\n"
    )


def rebuild_all(output_dir: Path, source_cache_dir: Path, *, use_sudo: bool = False) -> list[Path]:
    """Build each missing native service SIF from its digest-pinned upstream image."""
    output_dir = output_dir.resolve()
    source_cache_dir = source_cache_dir.resolve()
    outputs = []
    for service in SERVICES.values():
        output_sif = output_dir / f"{service.domain}-arm64.sif"
        outputs.append(output_sif)
        if output_sif.is_file():
            continue
        source_sif = source_sif_path(service, source_cache_dir)
        if not source_sif.is_file():
            pull_source_sif(service, source_sif)
        rebuild(source_sif, output_sif, use_sudo=use_sudo)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="build all seven EnterpriseOps ARM64 SIFs")
    mode.add_argument("--source-sif", type=Path, help="one upstream AMD64 source SIF")
    parser.add_argument("--output-sif", type=Path, help="output path when building one SIF")
    parser.add_argument("--output-dir", type=Path, help="directory for --all native SIF outputs")
    parser.add_argument("--source-cache-dir", type=Path, help="cache for --all upstream AMD64 source SIFs")
    parser.add_argument("--sudo", action="store_true", help="build with sudo when rootless fakeroot is unavailable")
    arguments = parser.parse_args()
    if arguments.all:
        if arguments.output_sif is not None:
            parser.error("--output-sif cannot be used with --all")
        if arguments.output_dir is None:
            parser.error("--output-dir is required with --all")
        source_cache_dir = arguments.source_cache_dir or arguments.output_dir.parent / "source-amd64"
        rebuild_all(arguments.output_dir, source_cache_dir, use_sudo=arguments.sudo)
        return
    if arguments.output_sif is None:
        parser.error("--output-sif is required with --source-sif")
    if arguments.output_dir is not None or arguments.source_cache_dir is not None:
        parser.error("--output-dir and --source-cache-dir can only be used with --all")
    rebuild(arguments.source_sif, arguments.output_sif, use_sudo=arguments.sudo)


if __name__ == "__main__":
    main()
