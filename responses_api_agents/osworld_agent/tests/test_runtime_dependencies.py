# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import os
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.osworld_agent import runtime_dependencies


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


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
        "numpy": "2.4.6",
        "cryptography": "46.0.7",
        "opencv-python-headless": "4.10.0.84",
        "torchvision": "0.26.0+cu130",
    }
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", imported.append)

    assert runtime_dependencies.validate_optional_runtime_dependencies() == ()
    assert imported == ["numpy", "cryptography", "cv2", "torchvision"]


def test_runtime_dependency_validation_reports_missing_mismatched_and_broken_imports(monkeypatch) -> None:
    versions = {
        "numpy": "2.4.6",
        "opencv-python-headless": "4.8.1.78",
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
    assert any(
        "opencv-python-headless~=4.10.0.84" in problem and "does not satisfy" in problem for problem in problems
    )
    assert any("torchvision==0.26.0" in problem and "operator ABI mismatch" in problem for problem in problems)


@pytest.mark.parametrize("numpy_version", ["1.26.4", "2.5.2"])
def test_runtime_dependency_validation_rejects_unsupported_numpy(monkeypatch, numpy_version: str) -> None:
    dependencies = (
        runtime_dependencies.RuntimeDependency("numpy", "numpy", ">=2.1,<2.5"),
        runtime_dependencies.RuntimeDependency("opencv-python-headless", "cv2", "~=4.10.0.84"),
    )
    versions = {"numpy": numpy_version, "opencv-python-headless": "4.10.0.84"}
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", imported.append)

    assert runtime_dependencies.validate_optional_runtime_dependencies(dependencies) == (
        f"numpy>=2.1,<2.5: installed version {numpy_version!r} does not satisfy the requirement",
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


def test_optional_runtime_installer_matches_agent_torch_backend(tmp_path: Path) -> None:
    agent_dir = Path(runtime_dependencies.__file__).resolve().parent
    installer = agent_dir / "install_optional_runtime_deps.sh"
    backend = (agent_dir / "uv-torch-backend.txt").read_text(encoding="utf-8").strip()
    venv = tmp_path / "managed venv"
    fake_bin = tmp_path / "fake-bin"
    ready = tmp_path / "runtime-ready"
    uv_argv = tmp_path / "uv-argv"
    (venv / "bin").mkdir(parents=True)
    fake_bin.mkdir()
    _write_executable(
        venv / "bin/python",
        '#!/usr/bin/env bash\n[[ -f "${FAKE_RUNTIME_READY}" ]]\n',
    )
    _write_executable(
        fake_bin / "uv",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${FAKE_UV_ARGV}"\ntouch "${FAKE_RUNTIME_READY}"\n',
    )
    env = os.environ | {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_RUNTIME_READY": str(ready),
        "FAKE_UV_ARGV": str(uv_argv),
    }

    result = subprocess.run(
        ["bash", str(installer), str(venv)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    argv = uv_argv.read_text(encoding="utf-8").splitlines()
    assert argv[:5] == ["pip", "install", "--no-config", "--torch-backend", backend]
    assert argv[5:7] == ["--python", str(venv / "bin/python")]
    assert "numpy>=2.1,<2.5" in argv
    assert "opencv-python-headless~=4.10.0.84" in argv
    assert "torchvision==0.26.0" in argv


def test_managed_requirements_cover_the_e2b_provider_imports() -> None:
    """The managed agent venv is built from requirements.txt, not the sandbox extra.

    Both misses cost a full 361-task attempt: `e2b` itself, then
    `httpx_aiohttp`, which the provider imports at configure time rather than
    lazily, so nothing surfaces until a rollout actually starts.
    """
    import ast
    import sys
    from pathlib import Path

    agent_dir = Path(__file__).resolve().parents[1]
    repo_root = agent_dir.parents[1]
    third_party: set[str] = set()
    for name in ("_sdk.py", "provider.py"):
        tree = ast.parse((repo_root / "nemo_gym/sandbox/providers/e2b" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                third_party.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                third_party.add(node.module.split(".")[0])
    third_party -= set(sys.stdlib_module_names) | {"nemo_gym"}

    requirements = (agent_dir / "requirements.txt").read_text()
    # httpx arrives with the e2b SDK; the rest must be pinned here by name.
    for module in sorted(third_party - {"httpx"}):
        assert module.replace("_", "-") in requirements, (
            f"{module} is imported by the e2b provider but absent from requirements.txt"
        )
