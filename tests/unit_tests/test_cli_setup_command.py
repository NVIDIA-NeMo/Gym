from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.cli_setup_command import setup_env_command
from nemo_gym.global_config import (
    HEAD_SERVER_DEPS_KEY_NAME,
    PIP_INSTALL_VERBOSE_KEY_NAME,
    PYTHON_VERSION_KEY_NAME,
    SKIP_VENV_IF_PRESENT_KEY_NAME,
    UV_PIP_SET_PYTHON_KEY_NAME,
)


class TestCLISetupCommand:
    def test_setup_env_command_skips_install_when_venv_present(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "server"
        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").write_text("")
        (server_dir / ".venv/bin/activate").write_text("")

        global_config_dict = OmegaConf.create(
            {
                HEAD_SERVER_DEPS_KEY_NAME: ["dep_a", "dep_b"],
                PYTHON_VERSION_KEY_NAME: "3.11.2",
                PIP_INSTALL_VERBOSE_KEY_NAME: False,
                UV_PIP_SET_PYTHON_KEY_NAME: False,
                SKIP_VENV_IF_PRESENT_KEY_NAME: True,
            }
        )

        command = setup_env_command(server_dir, global_config_dict)

        assert command == f"cd {server_dir} && source .venv/bin/activate"

    def test_setup_env_command_installs_when_skip_disabled(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (server_dir / ".venv/bin").mkdir(parents=True)
        (server_dir / ".venv/bin/python").write_text("")
        (server_dir / ".venv/bin/activate").write_text("")

        global_config_dict = OmegaConf.create(
            {
                HEAD_SERVER_DEPS_KEY_NAME: ["dep_a", "dep_b"],
                PYTHON_VERSION_KEY_NAME: "3.11.2",
                PIP_INSTALL_VERBOSE_KEY_NAME: False,
                UV_PIP_SET_PYTHON_KEY_NAME: False,
                SKIP_VENV_IF_PRESENT_KEY_NAME: False,
            }
        )

        command = setup_env_command(server_dir, global_config_dict)

        assert command.startswith(f"cd {server_dir} && ")
        assert "uv venv --seed --allow-existing --python 3.11.2 .venv" in command
        assert "source .venv/bin/activate" in command
        assert "uv pip install -r requirements.txt dep_a dep_b" in command
