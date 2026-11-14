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
from fastapi import HTTPException
from pydantic import ValidationError

import resources_servers.swe_rebench.app as app
import resources_servers.swebench.agent_network as agent_network
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef


# The sandbox created for a session is the agent's; the evaluation sandbox is created for a verifier session.
AGENT_SANDBOX = {"session_id": "session"}
EVAL_SANDBOX = {"verifier_session_id": "session"}
INSTANCE = dict(instance_id="o__r-1", repo="o/r", base_commit="abc", image_name="img")
MODEL_SERVER = ModelServerRef(type="responses_api_models", name="policy_model")


class _Started(Exception):
    """Raised by the recording sandbox once it has the spec, so a seed stops before its setup commands."""


class _RecordingAsyncSandbox:
    def __init__(self, provider_config) -> None:
        self.spec = None

    async def start(self, spec) -> None:
        self.spec = spec
        if _RecordingAsyncSandbox.raise_on_start:
            raise _Started(spec)

    async def exec(self, *args, **kwargs) -> None:
        return None

    async def stop(self) -> None:
        return None


_RecordingAsyncSandbox.raise_on_start = False


@pytest.fixture
def make_server(monkeypatch):
    def make(*, enforce=True, provider="opensandbox", model_url="http://10.1.2.3:8000", sandbox_config=None):
        monkeypatch.setattr(app, "AsyncSandbox", _RecordingAsyncSandbox)
        monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {provider: {}})
        monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr(agent_network, "resolve_provider_config", lambda *_: {provider: {}})
        monkeypatch.setattr(agent_network, "get_server_url", lambda name: model_url)
        config = app.SWERebenchResourcesServerConfig(
            host="",
            port=0,
            name="swe_rebench",
            entrypoint="",
            sandbox_provider="sandbox",
            sandbox_config=sandbox_config or {},
            enforce_agent_no_network=enforce,
            sandbox_model_server=MODEL_SERVER if enforce else None,
        )
        return app.SWERebenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    return make


async def _create(server, sandbox_kwargs):
    return await server._create_sandbox(app.SWERebenchInstanceRequest(**INSTANCE), **sandbox_kwargs)


async def test_off_by_default(make_server) -> None:
    sandbox = await _create(make_server(enforce=False), AGENT_SANDBOX)
    assert "network_policy" not in sandbox.spec.provider_options


async def test_agent_sandbox_allows_only_the_model_host(make_server) -> None:
    sandbox = await _create(make_server(), AGENT_SANDBOX)
    assert sandbox.spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.1.2.3"}],
    }


async def test_seeding_blocks_the_agent_sandbox_egress(make_server, monkeypatch) -> None:
    server = make_server()
    monkeypatch.setattr(_RecordingAsyncSandbox, "raise_on_start", True)
    with pytest.raises(_Started) as started:
        await server._start_task_sandbox("session", app.SWERebenchInstanceRequest(**INSTANCE))
    assert started.value.args[0].provider_options["network_policy"]["defaultAction"] == "deny"


async def test_eval_sandbox_keeps_its_network(make_server) -> None:
    sandbox = await _create(make_server(), EVAL_SANDBOX)
    assert "network_policy" not in sandbox.spec.provider_options


async def test_other_provider_options_are_kept(make_server) -> None:
    sandbox_config = {"provider_options": {"platform": "linux/amd64"}}
    sandbox = await _create(make_server(sandbox_config=sandbox_config), AGENT_SANDBOX)
    assert sandbox.spec.provider_options["platform"] == "linux/amd64"
    assert sandbox.spec.provider_options["network_policy"]["defaultAction"] == "deny"
    assert "network_policy" not in sandbox_config["provider_options"], "the shared config must not be mutated"


async def test_unreachable_model_host_refuses_the_seed(make_server) -> None:
    server = make_server(model_url="http://127.0.0.1:8000")
    with pytest.raises(HTTPException) as refused:
        await _create(server, AGENT_SANDBOX)
    assert refused.value.status_code == 422


def test_startup_refuses_a_provider_without_network_policies(make_server) -> None:
    with pytest.raises(ValueError, match="OpenSandbox"):
        make_server(provider="docker")


def test_config_refuses_the_flag_without_a_model_server_or_with_a_set_policy() -> None:
    base = dict(host="", port=0, name="swe_rebench", entrypoint="", sandbox_provider="sandbox")
    with pytest.raises(ValidationError, match="needs sandbox_model_server"):
        app.SWERebenchResourcesServerConfig(**base, sandbox_config={}, enforce_agent_no_network=True)
    with pytest.raises(ValidationError, match="not both"):
        app.SWERebenchResourcesServerConfig(
            **base,
            sandbox_config={"provider_options": {"network_policy": {"defaultAction": "deny"}}},
            enforce_agent_no_network=True,
            sandbox_model_server=MODEL_SERVER,
        )


def test_config_fields_default_to_open_egress() -> None:
    fields = app.SWERebenchResourcesServerConfig.model_fields
    assert fields["enforce_agent_no_network"].default is False
    assert fields["sandbox_model_server"].default is None
