from pathlib import Path

from omegaconf import OmegaConf

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
            prefix="my_server_name",
        )
        expected_command = f"cd {server_dir} && uv venv --seed --allow-existing --python test python version .venv > >(sed 's/^/(my_server_name) /') 2> >(sed 's/^/(my_server_name) /' >&2) && source .venv/bin/activate && uv pip install -r requirements.txt ray[default]==test ray version openai==test openai version > >(sed 's/^/(my_server_name) /') 2> >(sed 's/^/(my_server_name) /' >&2)"
        assert expected_command == actual_command

    def test_setup_env_command_skips_install_when_venv_present(self, tmp_path: Path) -> None:
        server_dir = self._setup_server_dir(tmp_path)

        global_config_dict = OmegaConf.create(self._debug_global_config_dict | {"skip_venv_if_present": True})

        command = setup_env_command(server_dir, global_config_dict)

        assert command == f"cd {server_dir} && source .venv/bin/activate"
