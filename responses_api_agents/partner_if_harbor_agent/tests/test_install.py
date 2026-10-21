# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nemo_gym import PARENT_DIR


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
