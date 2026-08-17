# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from pathlib import Path

import pytest

from responses_api_agents.osworld_agent import runtime_dependencies


def test_managed_agent_venv_matches_gym_layout(tmp_path: Path) -> None:
    gym_root = tmp_path / "Gym"

    assert runtime_dependencies.managed_agent_venv_path(gym_root) == (
        gym_root / "responses_api_agents/osworld_agent/.venv"
    )
    assert runtime_dependencies.managed_agent_venv_path(gym_root, tmp_path / "server-venvs") == (
        tmp_path / "server-venvs/responses_api_agents/osworld_agent/.venv"
    )


def test_managed_agent_venv_reads_relative_env_root(tmp_path: Path) -> None:
    gym_root = tmp_path / "Gym"
    env_file = gym_root / "benchmarks/osworld/env.yaml"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("uv_venv_dir: server-venvs\n", encoding="utf-8")

    assert runtime_dependencies.managed_agent_venv_from_env(gym_root, env_file) == (
        env_file.parent / "server-venvs/responses_api_agents/osworld_agent/.venv"
    )


def test_runtime_dependency_validation_accepts_compatible_local_wheel_versions(monkeypatch) -> None:
    versions = {
        "numpy": "1.26.4",
        "cryptography": "46.0.7",
        "opencv-python-headless": "4.8.1.78",
        "torchvision": "0.26.0+cu130",
    }
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", imported.append)

    assert runtime_dependencies.validate_optional_runtime_dependencies() == ()
    assert imported == ["numpy", "cryptography", "cv2", "torchvision"]


def test_runtime_dependency_validation_reports_missing_mismatched_and_broken_imports(monkeypatch) -> None:
    versions = {
        "numpy": "1.26.4",
        "opencv-python-headless": "4.10.0.84",
        "torchvision": "0.26.0",
    }

    def installed_version(distribution: str) -> str:
        if distribution == "cryptography":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    def import_module(import_name: str) -> None:
        if import_name == "torchvision":
            raise RuntimeError("operator ABI mismatch")

    monkeypatch.setattr(runtime_dependencies.importlib.metadata, "version", installed_version)
    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", import_module)

    problems = runtime_dependencies.validate_optional_runtime_dependencies()

    assert any("cryptography~=46.0: package is not installed" in problem for problem in problems)
    assert any("opencv-python-headless~=4.8.1.78" in problem and "does not satisfy" in problem for problem in problems)
    assert any("torchvision==0.26.0" in problem and "operator ABI mismatch" in problem for problem in problems)


def test_runtime_dependency_validation_rejects_numpy_2(monkeypatch) -> None:
    dependencies = (
        runtime_dependencies.RuntimeDependency("numpy", "numpy", "<2"),
        runtime_dependencies.RuntimeDependency("opencv-python-headless", "cv2", "~=4.8.1.78"),
    )
    versions = {"numpy": "2.5.2", "opencv-python-headless": "4.8.1.78"}
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", imported.append)

    assert runtime_dependencies.validate_optional_runtime_dependencies(dependencies) == (
        "numpy<2: installed version '2.5.2' does not satisfy the requirement",
    )
    assert imported == []


def test_runtime_dependency_startup_error_has_copyable_scoped_installer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runtime_dependencies,
        "validate_optional_runtime_dependencies",
        lambda: ("torchvision==0.26.0: package is not installed",),
    )
    installer = tmp_path / "Gym checkout/osworld_agent/install_optional_runtime_deps.sh"
    agent_venv = tmp_path / "managed venv"

    with pytest.raises(RuntimeError) as exc_info:
        runtime_dependencies.require_optional_runtime_dependencies(
            venv_path=agent_venv,
            installer=installer,
        )

    message = str(exc_info.value)
    assert "this agent venv" in message
    assert "torchvision==0.26.0" in message
    assert f"bash '{installer.resolve()}' '{agent_venv.resolve()}'" in message
