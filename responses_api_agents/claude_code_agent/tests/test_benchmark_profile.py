# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This adapter's benchmark profile, seeded through HLE's Environment Server."""

import pytest

from nemo_gym.config import get_first_server_config_dict
from tests.session_harness_profiles import REQUIRED_SETTING_VALUES, check_hle_seed, compose_profile_on_hle


@pytest.mark.parametrize("transports", [None, ["direct_http"]], ids=["as_shipped", "direct_http_granted"])
async def test_hle_seed_reaches_the_agents_own_sandbox_unless_a_direct_http_grant_is_required(
    transports: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await check_hle_seed("claude_code_agent", transports, monkeypatch)


def test_profile_restricts_the_sandbox_to_the_model_server(monkeypatch: pytest.MonkeyPatch) -> None:
    # HLE grades what the model knows, so the profile keeps the harness from looking answers up.
    settings = get_first_server_config_dict(
        compose_profile_on_hle("claude_code_agent", monkeypatch), "hle_equivalence_llm_judge_claude_code_agent"
    )
    assert settings.network_access == "model_only"
    # The Claude Code CLI is uploaded from the Agent Server, so it needs no sandbox network.
    assert all(settings.get(key) is None for key in REQUIRED_SETTING_VALUES)
