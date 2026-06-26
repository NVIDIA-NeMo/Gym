# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""wiring tests for the harness-agnostic, provider-neutral cvdp agent, pure logic."""

from unittest.mock import MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.cvdp_agent.agentic_generic import (
    CvdpGenericAgent,
    CvdpGenericAgentConfig,
    _safe_rel,
)
from responses_api_agents.cvdp_agent.in_sandbox_runner import render_runner


def _config(**kw) -> CvdpGenericAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="agentic_generic.py",
        name="cvdp_generic",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="cvdp"),
    )
    base.update(kw)
    return CvdpGenericAgentConfig(**base)


def _agent(**kw) -> CvdpGenericAgent:
    return CvdpGenericAgent(config=_config(**kw), server_client=MagicMock(spec=ServerClient))


class TestConfigDefaults:
    def test_provider_neutral_defaults(self) -> None:
        cfg = _config()
        assert cfg.sandbox_provider == {"apptainer": {}}  # swap one key for opensandbox/docker
        assert cfg.deps_provision == "bind"
        assert cfg.agent_server_class == "ClaudeCodeAgent"

    def test_harness_is_config_only(self) -> None:
        hermes = _config(
            agent_server_module="responses_api_agents.hermes_agent.app",
            agent_server_class="HermesAgent",
            agent_config_class="HermesAgentConfig",
        )
        runner = render_runner(hermes.agent_server_module, hermes.agent_server_class, hermes.agent_config_class)
        assert "HermesAgent(config=config" in runner

    def test_harvest_globs_cover_hdl(self) -> None:
        globs = _config().harvest_globs
        assert any(g.endswith(".sv") for g in globs) and any(g.endswith(".v") for g in globs)


class TestSafeRel:
    def test_rejects_escapes_and_absolute(self) -> None:
        assert not _safe_rel("/etc/passwd")
        assert not _safe_rel("../outside.sv")
        assert _safe_rel("rtl/adder.sv")


class TestSeedFiles:
    def test_maps_under_workdir_and_filters(self) -> None:
        agent = _agent()
        seeded = agent._seed_files(
            "/code",
            {"docs/spec.md": "S", "rtl/given.sv": "G", "../evil.sv": "x"},
            harness_files={"rtl/given.sv": "G"},  # declared harness file -> skipped
        )
        # spec kept under /code, harness file and unsafe path dropped
        assert seeded == {"/code/docs/spec.md": "S"}


class TestBuildSpec:
    def test_provider_neutral_spec(self, tmp_path) -> None:
        agent = _agent(deps_provision="bind")
        agent._model_url = lambda: "http://model:8000"  # avoid touching global config
        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="do it", model="m")
        spec = agent._build_spec(body, "do it", tmp_path, {"/code/docs/spec.md": "S"})

        assert spec.image == "nvidia/cvdp-sim:v1.0.0"
        assert spec.workdir == "/code"
        assert spec.env["NGTB_MODEL_URL"] == "http://model:8000"
        # runner + instruction + seeded context all delivered via provider-neutral spec.files,
        # under the workdir mount (not a separate /trajectories_mount) so upload actually lands.
        assert "/code/.ngtb/agent_runner.py" in spec.files
        assert "/code/.ngtb/instruction.txt" in spec.files
        assert spec.env["NGTB_TRAJ_DIR"] == "/code/.ngtb"
        assert spec.files["/code/docs/spec.md"] == "S"
        # bind provisioning of nemo_gym + deps
        binds = spec.provider_options["binds"]
        assert any(b.endswith(":/nemo_gym_mount:ro") for b in binds)
        assert any(b.endswith(":/agent_deps_mount:ro") for b in binds)

    def test_baked_provision_skips_binds(self, tmp_path) -> None:
        agent = _agent(deps_provision="baked")
        agent._model_url = lambda: "http://model:8000"
        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="x", model="m")
        spec = agent._build_spec(body, "x", tmp_path, {})
        assert spec.provider_options["binds"] == []  # nothing bound, deps assumed baked into image


class TestConstruct:
    def test_builds_and_inits_semaphore(self) -> None:
        agent = _agent(concurrency=3)
        assert agent.sem._value == 3
        assert agent._deps_dir is None  # lazy
