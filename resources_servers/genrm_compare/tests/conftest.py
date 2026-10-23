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

from unittest.mock import MagicMock

import pytest

import resources_servers.genrm_compare.app as genrm
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


@pytest.fixture
def config():
    return genrm.GenRMCompareConfig(
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        name="genrm_compare",
        genrm_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        num_rollouts_per_prompt=2,
        cohort_collection_timeout_s=1.0,
        cohort_evaluation_timeout_s=1.0,
        judge_request_timeout_s=0.2,
        genrm_parse_retry_sleep_s=0,
    )


@pytest.fixture
async def server(config):
    server = genrm.GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
    yield server
    await server.aclose()


@pytest.fixture(autouse=True)
async def close_owned_cohorts(monkeypatch):
    """Legacy tests construct servers directly; close every server that owned tasks."""
    owners = {}
    own_task = genrm.GenRMCompareResourcesServer._own_task

    def track(self, *args, **kwargs):
        owners[id(self)] = self
        return own_task(self, *args, **kwargs)

    monkeypatch.setattr(genrm.GenRMCompareResourcesServer, "_own_task", track)
    yield
    for server in owners.values():
        await server.aclose()
