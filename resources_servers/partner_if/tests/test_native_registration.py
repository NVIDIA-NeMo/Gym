# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Imported native adapters must satisfy the destination's process lifecycle contract."""

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.server_utils import entrypoint_ray_enabled


@pytest.mark.parametrize(
    "entrypoint,uses_ray",
    [
        ("resources_servers/partner_if/app.py", False),
        ("resources_servers/partner_if/mock_tools_app.py", False),
        ("resources_servers/partner_if/swe_app.py", False),
        ("responses_api_agents/partner_if_harbor_agent/app.py", True),
        ("responses_api_models/openhands_compat_model/app.py", False),
    ],
)
def test_adapters_declare_their_actual_process_dependencies(entrypoint, uses_ray):
    assert entrypoint_ray_enabled(PARENT_DIR / entrypoint) is uses_ray


def test_harbor_installer_explicitly_restores_its_opt_in_packaging_dependency():
    from omegaconf import OmegaConf

    from nemo_gym.cli.setup_command import setup_env_command

    root = (PARENT_DIR / "responses_api_agents/partner_if_harbor_agent").resolve()
    config = OmegaConf.create(
        {
            "head_server_deps": [],
            "uv_venv_dir": str(PARENT_DIR),
            "python_version": "3.13.14",
            "skip_venv_if_present": False,
        }
    )
    command = setup_env_command(root, config, "partner_if_markdown")
    assert "--override overrides.txt" in command
    assert "pathspec==1.1.1" in (root / "overrides.txt").read_text().splitlines()


@pytest.mark.parametrize("component", ["resources_servers/partner_if", "responses_api_agents/partner_if_harbor_agent"])
def test_sandbox_owning_adapters_install_the_sdk_in_their_own_environment(component):
    from nemo_gym.cli.setup_command import _local_nemo_gym_extras

    extras = _local_nemo_gym_extras(PARENT_DIR / component / "requirements.txt").strip("[]").split(",")
    assert "sandbox" in extras
