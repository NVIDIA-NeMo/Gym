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
import importlib.metadata
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from pytest import MonkeyPatch, raises

import nemo_gym.cli_setup_command
from nemo_gym.cli_setup_command import (
    _get_nemo_gym_install_flags,
    _get_nemo_gym_version_spec,
    run_command,
    setup_env_command,
)
from nemo_gym.global_config import UV_VENV_DIR_KEY_NAME
from tests.unit_tests.test_global_config import TestGlobalConfig as _TestGlobalConfig


class TestCLISetupCommandSetupEnvCommand:
    def _setup_server_dir(self, tmp_path: Path) -> Path:
        server_dir = tmp_path / "first_level" / "second_level"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "pyproject.toml").write_text("")

        return server_dir.absolute()

    def _debug_global_config_dict(self, tmp_path: Path) -> dict:
        return _TestGlobalConfig._default_global_config_dict_values.fget(None) | {UV_VENV_DIR_KEY_NAME: str(tmp_path)}

    def test_sanity(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_skips_install_when_venv_present(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").write_text("")
        (server_dir / ".venv/bin/activate").write_text("")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"skip_venv_if_present": True},
            prefix="my server name",
        )

        expected_command = f"cd {server_dir} && source {server_dir}/.venv/bin/activate"
        assert expected_command == actual_command

    def test_skips_install_still_installs_when_venv_missing(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        # No {server_dir}/.venv.
        # (server_dir / ".venv/bin").mkdir(parents=True)
        # (server_dir / ".venv/bin/python").write_text("")
        # (server_dir / ".venv/bin/activate").write_text("")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"skip_venv_if_present": True},
            prefix="my server name",
        )

        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_head_server_deps(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"head_server_deps": ["dep 1", "dep 2"]},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt dep 1 dep 2 > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_python_version(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"python_version": "my python version"},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python my python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_uv_pip_set_python(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_pip_set_python": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install --python {server_dir}/.venv/bin/python -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_pip_install_verbose(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"pip_install_verbose": True},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install -v -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")

        with raises(RuntimeError, match="Found both pyproject.toml and requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_missing_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "requirements.txt").unlink()

        with raises(RuntimeError, match="Missing pyproject.toml or requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )

    def test_pyproject(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")
        (server_dir / "requirements.txt").unlink()

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path),
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install '-e .' ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_uv_venv_dir_with_install(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        uv_venv_dir = tmp_path / "uv_venv_dir"

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path) | {"uv_venv_dir": str(uv_venv_dir)},
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {uv_venv_dir}/first_level/second_level/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {uv_venv_dir}/first_level/second_level/.venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && (echo 'nemo-gym=={version}' && grep -v -F '../..' requirements.txt) | uv pip install -r /dev/stdin ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    @pytest.mark.parametrize("version", ["0.3.0", "0.3.0rc0", "1.0.0", "2.1.3rc1"])
    def test_installs_from_pypi_when_not_editable_pyproject(
        self, tmp_path: Path, version: str, monkeypatch: MonkeyPatch
    ) -> None:
        server_dir = (tmp_path / "first_level" / "second_level").absolute()
        server_dir.mkdir(parents=True)
        (server_dir / "pyproject.toml").write_text("")
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        with patch("importlib.metadata.version", return_value=version):
            actual_command = setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict(tmp_path),
                prefix="my server name",
            )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version {server_dir}/.venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source {server_dir}/.venv/bin/activate && uv pip install nemo-gym=={version} && uv pip install --no-sources '-e .' ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_uv_venv_dir_and_skip_install_when_venv_present(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        uv_venv_dir = tmp_path / "uv_venv_dir"

        (uv_venv_dir / "first_level/second_level/.venv/bin").mkdir(parents=True)
        (uv_venv_dir / "first_level/second_level/.venv/bin/python").write_text("")
        (uv_venv_dir / "first_level/second_level/.venv/bin/activate").write_text("")

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict(tmp_path)
            | {"skip_venv_if_present": True, "uv_venv_dir": str(uv_venv_dir)},
            prefix="my server name",
        )

        expected_command = f"cd {server_dir} && source {uv_venv_dir}/first_level/second_level/.venv/bin/activate"
        assert expected_command == actual_command


class TestCLISetupCommandRunCommand:
    def _setup(self, monkeypatch: MonkeyPatch) -> tuple[MagicMock, MagicMock]:
        Popen_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.cli_setup_command, "Popen", Popen_mock)

        get_global_config_dict_mock = MagicMock(return_value={"uv_cache_dir": "default uv cache dir"})
        monkeypatch.setattr(nemo_gym.cli_setup_command, "get_global_config_dict", get_global_config_dict_mock)

        monkeypatch.setattr(nemo_gym.cli_setup_command, "environ", dict())

        monkeypatch.setattr(nemo_gym.cli_setup_command, "stdout", "stdout")
        monkeypatch.setattr(nemo_gym.cli_setup_command, "stderr", "stderr")

        return Popen_mock, get_global_config_dict_mock

    def test_sanity(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_pythonpath(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)
        monkeypatch.setattr(nemo_gym.cli_setup_command, "environ", {"PYTHONPATH": "existing pythonpath"})

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path:existing pythonpath", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_custom_uv_cache_dir(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {"uv_cache_dir": "my uv cache dir"}

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "my command",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "my uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args


class TestGetNemoGymInstallFlags:
    """Test _get_nemo_gym_install_flags helper function."""

    def test_no_env_vars_returns_empty(self, monkeypatch: MonkeyPatch) -> None:
        """When no env vars are set, should return empty string."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_prerelease_flag(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=true, should add --pre and --index-strategy."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--pre --index-strategy unsafe-best-match 'fastapi<1.0' "

    def test_prerelease_false(self, monkeypatch: MonkeyPatch) -> None:
        """When NEMO_GYM_ALLOW_PRERELEASE=false, should not add flags."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "false")
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == ""

    def test_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--index-url https://test.pypi.org/simple/ "

    def test_extra_index_url(self, monkeypatch: MonkeyPatch) -> None:
        """Should include UV_EXTRA_INDEX_URL if set."""
        monkeypatch.delenv("NEMO_GYM_ALLOW_PRERELEASE", raising=False)
        monkeypatch.delenv("UV_INDEX_URL", raising=False)
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.delenv("UV_INDEX_STRATEGY", raising=False)

        flags = _get_nemo_gym_install_flags()
        assert flags == "--extra-index-url https://pypi.org/simple/ "

    def test_explicit_index_strategy(self, monkeypatch: MonkeyPatch) -> None:
        """Explicit UV_INDEX_STRATEGY should override auto-set from prerelease."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "first-match")

        flags = _get_nemo_gym_install_flags()
        # Should have --pre but use explicit strategy, not auto-set unsafe-best-match
        assert flags == "--pre 'fastapi<1.0' --index-strategy first-match "

    def test_all_flags_combined(self, monkeypatch: MonkeyPatch) -> None:
        """Test all flags together."""
        monkeypatch.setenv("NEMO_GYM_ALLOW_PRERELEASE", "true")
        monkeypatch.setenv("UV_INDEX_URL", "https://test.pypi.org/simple/")
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://pypi.org/simple/")
        monkeypatch.setenv("UV_INDEX_STRATEGY", "unsafe-best-match")

        flags = _get_nemo_gym_install_flags()
        assert (
            flags
            == "--pre 'fastapi<1.0' --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match "
        )


class TestGetNemoGymVersionSpec:
    """Test _get_nemo_gym_version_spec helper function."""

    def test_editable_install_returns_empty(self) -> None:
        """For editable installs, should return empty string (no version pinning)."""
        version_spec = _get_nemo_gym_version_spec(is_editable_install=True)
        assert version_spec == ""

    def test_non_editable_detects_version(self) -> None:
        """For non-editable installs, should detect and pin to parent version."""
        with patch("importlib.metadata.version", return_value="0.2.1rc0"):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == "==0.2.1rc0"

    def test_non_editable_stable_version(self) -> None:
        """Should work with stable versions too."""
        with patch("importlib.metadata.version", return_value="0.2.0"):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == "==0.2.0"

    def test_package_not_found_returns_empty(self) -> None:
        """If nemo-gym is not installed, should return empty string gracefully."""
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            version_spec = _get_nemo_gym_version_spec(is_editable_install=False)
            assert version_spec == ""


class TestCLISetupCommandRunCommandTeeLog(TestCLISetupCommandRunCommand):
    def test_tee_logs_with_server_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
            server_name="my_resources/my_server",
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my_resources_my_server.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args

    def test_tee_logs_falls_back_to_dir_name(self, monkeypatch: MonkeyPatch) -> None:
        Popen_mock, get_global_config_dict_mock = self._setup(monkeypatch)

        get_global_config_dict_mock.return_value = {
            "uv_cache_dir": "default uv cache dir",
            "nemo_gym_log_dir": "/tmp/gym_logs",
        }

        run_command(
            command="my command",
            working_dir_path=Path("/my path"),
        )

        expected_args = call(
            "set -o pipefail; (my command) 2>&1 | tee -a /tmp/gym_logs/my path.log",
            executable="/bin/bash",
            shell=True,
            env={"PYTHONPATH": "/my path", "UV_CACHE_DIR": "default uv cache dir"},
            stdout="stdout",
            stderr="stderr",
        )
        actual_args = Popen_mock.call_args
        assert expected_args == actual_args
