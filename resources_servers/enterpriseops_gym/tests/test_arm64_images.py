# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from resources_servers.enterpriseops_gym.arm64_images import (
    build_command,
    rebuild_all,
    render_definition,
    requirements_from_metadata,
)
from resources_servers.enterpriseops_gym.runtime import SERVICES


def _metadata(root: Path, directory: str, name: str, version: str) -> None:
    metadata = root / "usr/local/lib/python3.11/site-packages" / directory / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def test_requirements_from_metadata_preserves_exact_versions_and_sorts(tmp_path: Path) -> None:
    _metadata(tmp_path, "uvicorn-0.40.0.dist-info", "uvicorn", "0.40.0")
    _metadata(tmp_path, "fastapi-0.128.0.dist-info", "fastapi", "0.128.0")

    assert requirements_from_metadata(tmp_path) == ["fastapi==0.128.0", "uvicorn==0.40.0"]


def test_render_definition_uses_native_python_and_a_port_override(tmp_path: Path) -> None:
    definition = render_definition(tmp_path / "rootfs", tmp_path / "requirements.txt")

    assert "From: python:3.11.14-slim" in definition
    assert f"{tmp_path / 'rootfs' / 'app'} /app" in definition
    assert "python -m pip install --no-cache-dir -r /opt/requirements.txt" in definition
    assert "curl -LsSf" not in definition
    assert 'exec python -m uvicorn main:app --host 127.0.0.1 --port "${PORT:-8005}"' in definition


def test_build_command_uses_privileged_mode_without_fakeroot(tmp_path: Path) -> None:
    command = build_command(tmp_path / "service.sif", tmp_path / "service.def", use_sudo=True)

    assert command[:3] == ["sudo", "apptainer", "build"]
    assert "--fakeroot" not in command
    assert command[-2:] == [str(tmp_path / "service.sif"), str(tmp_path / "service.def")]


def test_rebuild_all_pulls_missing_sources_and_writes_all_native_sifs(tmp_path: Path, monkeypatch) -> None:
    source_cache = tmp_path / "source"
    output_dir = tmp_path / "native"
    pulled = []
    rebuilt = []

    def pull_source(service, target: Path) -> None:
        pulled.append((service.domain, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source")

    def rebuild_source(source: Path, output: Path, *, use_sudo: bool = False) -> None:
        rebuilt.append((source, output, use_sudo))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"native")

    monkeypatch.setattr("resources_servers.enterpriseops_gym.arm64_images.pull_source_sif", pull_source)
    monkeypatch.setattr("resources_servers.enterpriseops_gym.arm64_images.rebuild", rebuild_source)

    assert rebuild_all(output_dir, source_cache) == [output_dir / f"{domain}-arm64.sif" for domain in SERVICES]
    assert [domain for domain, _ in pulled] == list(SERVICES)
    assert [output.name for _, output, _ in rebuilt] == [f"{domain}-arm64.sif" for domain in SERVICES]
