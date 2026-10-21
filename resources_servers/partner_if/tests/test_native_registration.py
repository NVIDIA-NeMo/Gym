# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters that own sandboxes install the sandbox SDK in their own environments."""

import pytest

from nemo_gym import PARENT_DIR


@pytest.mark.parametrize("component", ["resources_servers/partner_if", "responses_api_agents/partner_if_harbor_agent"])
def test_sandbox_owning_adapters_install_the_sdk_in_their_own_environment(component):
    from nemo_gym.cli.setup_command import _local_nemo_gym_extras

    extras = _local_nemo_gym_extras(PARENT_DIR / component / "requirements.txt").strip("[]").split(",")
    assert "sandbox" in extras
