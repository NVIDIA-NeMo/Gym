# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from nemo_gym.prefetch_gym_venvs import discover_servers, get_head_server_deps, prefetch_venv


@pytest.fixture
def fake_gym_root(tmp_path: Path) -> Path:
    """Create a minimal fake Gym directory structure for testing discovery."""
    # responses_api_models/vllm_model (pyproject.toml)
    (tmp_path / "responses_api_models" / "vllm_model").mkdir(parents=True)
    (tmp_path / "responses_api_models" / "vllm_model" / "pyproject.toml").write_text("[project]\nname='vllm-model'\n")

    # responses_api_models/openai_model (requirements.txt)
    (tmp_path / "responses_api_models" / "openai_model").mkdir(parents=True)
    (tmp_path / "responses_api_models" / "openai_model" / "requirements.txt").write_text("openai\n")

    # resources_servers/math_with_judge (requirements.txt)
    (tmp_path / "resources_servers" / "math_with_judge").mkdir(parents=True)
    (tmp_path / "resources_servers" / "math_with_judge" / "requirements.txt").write_text("math-verify\n")

    # resources_servers/no_deps_server (no pyproject.toml or requirements.txt — should be skipped)
    (tmp_path / "resources_servers" / "no_deps_server").mkdir(parents=True)

    # responses_api_agents/simple_agent (requirements.txt)
    (tmp_path / "responses_api_agents" / "simple_agent").mkdir(parents=True)
    (tmp_path / "responses_api_agents" / "simple_agent" / "requirements.txt").write_text("nemo-gym\n")

    # A regular file that should be ignored (not a directory)
    (tmp_path / "responses_api_models" / "README.md").write_text("# Models\n")

    return tmp_path


class TestDiscoverServers:
    def test_discovers_all_servers(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers()

        labels = [label for label, _ in servers]
        assert len(servers) == 4
        assert "responses_api_models/vllm_model" in labels
        assert "responses_api_models/openai_model" in labels
        assert "resources_servers/math_with_judge" in labels
        assert "responses_api_agents/simple_agent" in labels

    def test_skips_dirs_without_deps(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers()

        labels = [label for label, _ in servers]
        assert "resources_servers/no_deps_server" not in labels

    def test_skips_non_directories(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers()

        # README.md under responses_api_models should not appear
        labels = [label for label, _ in servers]
        assert not any("README" in label for label in labels)

    def test_filter_single(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers(filters=["vllm_model"])

        assert len(servers) == 1
        assert servers[0][0] == "responses_api_models/vllm_model"

    def test_filter_multiple(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers(filters=["vllm_model", "simple_agent"])

        labels = [label for label, _ in servers]
        assert len(servers) == 2
        assert "responses_api_models/vllm_model" in labels
        assert "responses_api_agents/simple_agent" in labels

    def test_filter_no_match(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers(filters=["nonexistent"])

        assert len(servers) == 0

    def test_filter_is_exact_match(self, fake_gym_root: Path) -> None:
        """Filters match the exact server directory name, not substrings."""
        # Add a local_vllm_model to ensure "vllm_model" doesn't match it
        (fake_gym_root / "responses_api_models" / "local_vllm_model").mkdir(parents=True)
        (fake_gym_root / "responses_api_models" / "local_vllm_model" / "requirements.txt").write_text("vllm\n")

        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers(filters=["vllm_model"])

        labels = [label for label, _ in servers]
        assert len(servers) == 1
        assert "responses_api_models/vllm_model" in labels
        assert "responses_api_models/local_vllm_model" not in labels

    def test_filter_is_case_sensitive(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers(filters=["VLLM_MODEL"])

        assert len(servers) == 0

    def test_missing_server_type_dir(self, fake_gym_root: Path) -> None:
        """If a server type directory doesn't exist, it's silently skipped."""
        import shutil

        shutil.rmtree(fake_gym_root / "responses_api_agents")
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers()

        labels = [label for label, _ in servers]
        assert len(servers) == 3
        assert not any("responses_api_agents" in label for label in labels)

    def test_results_are_sorted(self, fake_gym_root: Path) -> None:
        with patch("nemo_gym.prefetch_gym_venvs.GYM_ROOT", fake_gym_root):
            servers = discover_servers()

        # Within each server type, names should be sorted
        model_servers = [label for label, _ in servers if label.startswith("responses_api_models/")]
        assert model_servers == sorted(model_servers)


class TestPrefetchVenv:
    def test_success_with_pyproject(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "vllm_model"
        server_dir.mkdir()
        (server_dir / "pyproject.toml").write_text("[project]\nname='test'\n")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            result = prefetch_venv(
                "responses_api_models/vllm_model",
                server_dir,
                str(tmp_path / "venvs"),
                "3.12",
                ["ray[default]==2.46.0"],
            )

        assert result is True
        assert mock_run.call_count == 2

        # First call: uv venv
        venv_call = mock_run.call_args_list[0]
        assert "uv" in venv_call[0][0]
        assert "venv" in venv_call[0][0]
        assert str(tmp_path / "venvs" / "vllm_model") in venv_call[0][0]

        # Second call: uv pip install -e .
        pip_call = mock_run.call_args_list[1]
        assert "-e" in pip_call[0][0]
        assert "ray[default]==2.46.0" in pip_call[0][0]

    def test_success_with_requirements(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "math_with_judge"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("math-verify\n")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            result = prefetch_venv(
                "resources_servers/math_with_judge",
                server_dir,
                str(tmp_path / "venvs"),
                "3.12",
                [],
            )

        assert result is True
        pip_call = mock_run.call_args_list[1]
        assert "-r" in pip_call[0][0]

    def test_venv_creation_failure(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "broken_server"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("some-pkg\n")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args=[], returncode=1, stdout="", stderr="error: python 99.99 not found"
            )

            result = prefetch_venv(
                "resources_servers/broken_server",
                server_dir,
                str(tmp_path / "venvs"),
                "99.99",
                [],
            )

        assert result is False
        # Should stop after venv creation failure, not attempt pip install
        assert mock_run.call_count == 1

    def test_pip_install_failure(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "bad_deps"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("nonexistent-pkg-xyz\n")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),  # venv succeeds
                CompletedProcess(args=[], returncode=1, stdout="", stderr="No matching distribution"),  # pip fails
            ]

            result = prefetch_venv(
                "resources_servers/bad_deps",
                server_dir,
                str(tmp_path / "venvs"),
                "3.12",
                [],
            )

        assert result is False
        assert mock_run.call_count == 2

    def test_force_rebuild_removes_existing_venv(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "stale_server"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("some-pkg\n")

        # Create a pre-existing venv directory to simulate a stale venv
        venvs_dir = tmp_path / "venvs"
        existing_venv = venvs_dir / "stale_server"
        existing_venv.mkdir(parents=True)
        (existing_venv / "marker.txt").write_text("i should be deleted")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            result = prefetch_venv(
                "resources_servers/stale_server",
                server_dir,
                str(venvs_dir),
                "3.12",
                [],
                force_rebuild=True,
            )

        assert result is True
        # The old venv directory should have been removed (shutil.rmtree)
        # then recreated by uv venv, so the marker file should be gone
        assert not (existing_venv / "marker.txt").exists()

    def test_force_rebuild_false_keeps_existing_venv(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "good_server"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("some-pkg\n")

        # Create a pre-existing venv directory
        venvs_dir = tmp_path / "venvs"
        existing_venv = venvs_dir / "good_server"
        existing_venv.mkdir(parents=True)
        (existing_venv / "marker.txt").write_text("i should survive")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            result = prefetch_venv(
                "resources_servers/good_server",
                server_dir,
                str(venvs_dir),
                "3.12",
                [],
                force_rebuild=False,
            )

        assert result is True
        # The marker file should still be there
        assert (existing_venv / "marker.txt").exists()

    def test_empty_head_server_deps(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "minimal"
        server_dir.mkdir()
        (server_dir / "requirements.txt").write_text("fastapi\n")

        with patch("nemo_gym.prefetch_gym_venvs.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")

            result = prefetch_venv(
                "resources_servers/minimal",
                server_dir,
                str(tmp_path / "venvs"),
                "3.12",
                [],  # no head_server_deps
            )

        assert result is True
        pip_call = mock_run.call_args_list[1]
        # Should not contain any extra deps beyond -r requirements.txt
        pip_args = pip_call[0][0]
        assert pip_args[-1] == str(server_dir / "requirements.txt")


class TestGetHeadServerDeps:
    def test_with_ray_and_openai(self) -> None:
        with (
            patch.dict("sys.modules", {"ray": type("ray", (), {"__version__": "2.46.0"})()}),
            patch.dict("sys.modules", {"openai": type("openai", (), {"__version__": "2.6.1"})()}),
        ):
            # Need to reimport to pick up the mocked modules
            import importlib

            import nemo_gym.prefetch_gym_venvs as mod

            importlib.reload(mod)
            deps = mod.get_head_server_deps()

        assert "ray[default]==2.46.0" in deps
        assert "openai==2.6.1" in deps

    def test_without_ray_or_openai(self) -> None:
        """When ray/openai aren't importable, returns empty list."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("ray", "openai"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            deps = get_head_server_deps()

        assert deps == []
