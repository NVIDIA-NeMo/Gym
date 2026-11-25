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
    await check_hle_seed("opencode_agent", transports, monkeypatch)


def test_profile_restricts_the_sandbox_to_the_model_server(monkeypatch: pytest.MonkeyPatch) -> None:
    # HLE grades what the model knows, so the profile keeps the harness from looking answers up.
    settings = get_first_server_config_dict(
        compose_profile_on_hle("opencode_agent", monkeypatch), "hle_equivalence_llm_judge_opencode_agent"
    )
    assert settings.network_access == "model_only"
    # OpenCode cannot download its binary through that policy, so the profile requires one in the image.
    assert settings.sandbox_executable_path == REQUIRED_SETTING_VALUES["sandbox_executable_path"]
    assert settings.sandbox_executable_sha256 == REQUIRED_SETTING_VALUES["sandbox_executable_sha256"]


def test_profile_takes_a_runtime_uploaded_from_the_host_in_place_of_a_binary_in_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composed = "hle_equivalence_llm_judge_opencode_agent"
    settings = f"++{composed}.responses_api_agents.opencode_agent"
    # The two required fields are set to null, as the README says: the upload replaces the binary in the image.
    config = compose_profile_on_hle(
        "opencode_agent",
        monkeypatch,
        required=False,
        cli_args=(
            f"{settings}.sandbox_runtime_upload=true",
            f"{settings}.sandbox_executable_path=null",
            f"{settings}.sandbox_executable_sha256=null",
        ),
    )
    agent = get_first_server_config_dict(config, composed)
    assert agent.sandbox_runtime_upload is True
    assert agent.sandbox_executable_path is None and agent.sandbox_executable_sha256 is None
