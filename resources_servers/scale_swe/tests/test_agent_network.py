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
from types import SimpleNamespace

import pytest


class _RecordingAsyncSandbox:
    def __init__(self, provider_config) -> None:
        self.spec = None

    async def start(self, spec) -> None:
        self.spec = spec

    async def exec(self, *args, **kwargs) -> None:
        # scale_swe's _create_sandbox may run a setup command right after start.
        return None


class TestAgentNetworkPolicy:
    """enforce_agent_no_network denies all egress from the agent's sandbox except the policy
    model server's host; the evaluation sandbox keeps its network."""

    @staticmethod
    async def _create(
        monkeypatch,
        *,
        enforce: bool,
        for_agent: bool,
        provider: str = "opensandbox",
        model_url: str = "http://10.1.2.3:8000",
        sandbox_config: dict | None = None,
    ):
        import nemo_gym.sandbox.agent_tools as agent_tools
        import resources_servers.scale_swe.app as app
        from nemo_gym.config_types import ModelServerRef

        monkeypatch.setattr(app, "AsyncSandbox", _RecordingAsyncSandbox)
        monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
        monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {provider: {}})
        monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
        monkeypatch.setattr(agent_tools, "get_server_url", lambda name: model_url)
        server = SimpleNamespace(
            config=SimpleNamespace(
                sandbox_config=sandbox_config or {},
                sandbox_provider="sandbox",
                enforce_agent_no_network=enforce,
                sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
                name="scale_swe",
            )
        )
        body = app.ScaleSWEInstanceRequest(instance_id="o__r-1", workdir="/workspace/r", image_url="img")
        return await app.ScaleSWEResourcesServer._create_sandbox(server, body, for_agent=for_agent)

    async def test_off_by_default(self, monkeypatch) -> None:
        sandbox = await self._create(monkeypatch, enforce=False, for_agent=True)
        assert "network_policy" not in sandbox.spec.provider_options

    async def test_agent_sandbox_allows_only_the_model_host(self, monkeypatch) -> None:
        sandbox = await self._create(monkeypatch, enforce=True, for_agent=True)
        assert sandbox.spec.provider_options["network_policy"] == {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "10.1.2.3"}],
        }

    async def test_eval_sandbox_keeps_its_network(self, monkeypatch) -> None:
        sandbox = await self._create(monkeypatch, enforce=True, for_agent=False)
        assert "network_policy" not in sandbox.spec.provider_options

    async def test_other_provider_options_are_kept(self, monkeypatch) -> None:
        sandbox = await self._create(
            monkeypatch, enforce=True, for_agent=True, sandbox_config={"provider_options": {"platform": "linux/amd64"}}
        )
        assert sandbox.spec.provider_options["platform"] == "linux/amd64"
        assert sandbox.spec.provider_options["network_policy"]["defaultAction"] == "deny"

    @pytest.mark.parametrize(
        ("provider", "model_url"),
        [("opensandbox", "http://127.0.0.1:8000"), ("docker", "http://10.1.2.3:8000")],
    )
    async def test_fails_closed(self, monkeypatch, provider: str, model_url: str) -> None:
        with pytest.raises(ValueError):
            await self._create(monkeypatch, enforce=True, for_agent=True, provider=provider, model_url=model_url)

    async def test_config_fields_exist_and_default_to_open(self) -> None:
        import resources_servers.scale_swe.app as app

        fields = app.ScaleSWEResourcesServerConfig.model_fields
        assert fields["enforce_agent_no_network"].default is False
        assert fields["sandbox_model_server"].default is None
