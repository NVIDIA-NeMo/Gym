from pathlib import Path

from pytest import raises

from nemo_gym.cli_setup_command import setup_env_command
from tests.unit_tests.test_global_config import TestServerUtils


class TestCLISetupCommand:
    def _setup_server_dir(self, tmp_path: Path) -> Path:
        server_dir = tmp_path / "server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").write_text("")
        (server_dir / ".venv/bin/activate").write_text("")

        return server_dir

    @property
    def _debug_global_config_dict(self) -> dict:
        return TestServerUtils._default_global_config_dict_values.fget(None)

    def test_setup_env_command_sanity(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict,
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv && source .venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version"
        assert expected_command == actual_command

    def test_setup_env_command_skips_install_when_venv_present(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict | {"skip_venv_if_present": True},
        )

        expected_command = f"cd {server_dir} && source .venv/bin/activate"
        assert expected_command == actual_command

    def test_setup_env_command_with_prefix_print(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict,
            prefix="my server name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2) && source .venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my server name) /') 2> >(sed 's/^/(my server name) /' >&2)"
        assert expected_command == actual_command

    def test_setup_env_command_head_server_deps(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict | {"head_server_deps": ["dep 1", "dep 2"]},
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv && source .venv/bin/activate && uv pip install -r requirements.txt dep 1 dep 2"
        assert expected_command == actual_command

    def test_setup_env_command_python_version(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict | {"python_version": "my python version"},
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python my python version .venv && source .venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version"
        assert expected_command == actual_command

    def test_setup_env_command_uv_pip_set_python(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict | {"uv_pip_set_python": True},
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv && source .venv/bin/activate && uv pip install --python .venv/bin/python -r requirements.txt ray[default]==test ray version openai==test openai version"
        assert expected_command == actual_command

    def test_setup_env_command_pip_install_verbose(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict | {"pip_install_verbose": True},
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv && source .venv/bin/activate && uv pip install -v -r requirements.txt ray[default]==test ray version openai==test openai version"
        assert expected_command == actual_command

    def test_setup_env_command_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")

        with raises(RuntimeError, match="Found both pyproject.toml and requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict,
            )

    def test_setup_env_command_missing_pyproject_requirements_raises_error(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "requirements.txt").unlink()

        with raises(RuntimeError, match="Missing pyproject.toml or requirements.txt"):
            setup_env_command(
                dir_path=server_dir,
                global_config_dict=self._debug_global_config_dict,
            )

    def test_setup_env_command_pyproject(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        (server_dir / "pyproject.toml").write_text("")
        (server_dir / "requirements.txt").unlink()

        actual_command = setup_env_command(
            dir_path=server_dir,
            global_config_dict=self._debug_global_config_dict,
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv && source .venv/bin/activate && uv pip install '-e .' ray[default]==test ray version openai==test openai version"
        assert expected_command == actual_command
