# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServerConfig
from nemo_gym.sandbox import SandboxResources
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.tests.test_app import make_server
from responses_api_agents.opencode_video_agent.app import OpenCodeVideoAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgentConfig


@pytest.mark.parametrize("benchmark", ["videomme_v1", "lvbench"])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_actual_composed_configs_resolve_and_validate(tmp_path, benchmark, setting) -> None:
    server, _, _ = make_server(tmp_path, benchmark=benchmark, setting=setting)
    root = Path(__file__).parents[2]
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"invented capsule, not executed")
    values = {
        f"{benchmark}_cohort_manifest": str(server.config.cohort_manifest),
        f"{benchmark}_agentic_cohort_manifest": str(server.config.cohort_manifest),
        "video_context_tokens": 524288,
        "video_output_tokens": 65536,
        "video_media_profile": {"max_context_tokens": 524288, "reserved_output_tokens": 65536},
        "video_runtime_archive": str(archive),
        "video_runtime_sha256": "b" * 64,
        "video_source_roots": [str(tmp_path)],
        "video_agent_artifacts_dir": str(tmp_path / "artifacts"),
        "video_sandbox_image": "example.invalid/test@sha256:" + "b" * 64,
    }
    path = root / benchmark / "configs" / f"{benchmark}_{setting}.yaml"
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path), OmegaConf.create(values)), resolve=True)
    resource_name = benchmark if setting == "single_turn" else benchmark + "_agentic_resources"
    parsed = VideoMCQAResourcesServerConfig(
        name=resource_name, host="127.0.0.1", port=8001, **config[resource_name]["resources_servers"][benchmark]
    )
    assert parsed.cohort_manifest == server.config.cohort_manifest
    agent_name = f"{benchmark}_{setting}"
    component = "simple_agent" if setting == "single_turn" else "opencode_video_agent"
    cls = SimpleAgentConfig if setting == "single_turn" else OpenCodeVideoAgentConfig
    agent = cls(name=agent_name, host="127.0.0.1", port=8002, **config[agent_name]["responses_api_agents"][component])
    assert agent.resources_server.name == resource_name
    if setting == "single_turn":
        assert agent.max_steps == 1
    else:
        assert agent.opencode_max_context_window == 524288 and agent.max_output_tokens == 65536
        # The sandbox resources are the provider-neutral ones every provider accepts.
        assert SandboxResources.from_mapping(agent.sandbox_config["resources"]) == SandboxResources(
            cpu=2, memory_mib=8192
        )
    environment = LegacyAgentEnvironmentServerConfig(
        name=agent_name + "_environment",
        host="127.0.0.1",
        port=8003,
        **config[agent_name + "_environment"]["environment_servers"]["legacy_agent"],
    )
    assert environment.agent_server.name == agent_name
