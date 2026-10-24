# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters that own sandboxes install the sandbox SDK in their own environments."""

from nemo_gym import PARENT_DIR


def test_the_sandbox_owning_server_installs_the_sdk_in_its_own_environment():
    from nemo_gym.cli.setup_command import _local_nemo_gym_extras

    extras = (
        _local_nemo_gym_extras(PARENT_DIR / "resources_servers/partner_if/requirements.txt").strip("[]").split(",")
    )
    assert "sandbox" in extras
