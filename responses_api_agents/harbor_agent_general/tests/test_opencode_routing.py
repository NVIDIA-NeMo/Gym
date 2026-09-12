# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig

from responses_api_agents.harbor_agent_general import app


def subject(model_server=None, steps=None):
    return SimpleNamespace(
        config=SimpleNamespace(model_server=model_server, opencode_max_steps=steps, opencode_context_window=262144),
        base_url_for_run=lambda url, body: url + "/ng-rollout/0-0",
    )


def test_route_preserves_overlay_and_does_not_mutate_shared_config(monkeypatch):
    monkeypatch.setattr(app, "get_server_url", lambda name: "http://compute:63000")
    original = AgentConfig(
        name="opencode", model_name="direct/model", kwargs={"opencode_config": {"mcp": {"task": {}}}}
    )
    job = JobConfig(agents=[original])
    app.HarborAgent.configure_opencode(subject(SimpleNamespace(name="policy_model"), 3), job, {})
    agent = job.agents[0]
    overlay = agent.kwargs["opencode_config"]
    assert agent.model_name == "nemo_gym/dummy_model"
    assert overlay["mcp"] == {"task": {}}
    assert overlay["provider"]["nemo_gym"]["options"]["baseURL"] == "http://compute:63000/ng-rollout/0-0/v1"
    assert all(config["steps"] == 3 for config in overlay["agent"].values())
    assert original.model_name == "direct/model"
    assert original.kwargs == {"opencode_config": {"mcp": {"task": {}}}}


def test_full_run_is_uncapped_and_direct_provider_is_unchanged(monkeypatch):
    monkeypatch.setattr(app, "get_server_url", lambda name: "http://compute:63000")
    job = JobConfig(agents=[AgentConfig(name="opencode", model_name="direct/model")])
    app.HarborAgent.configure_opencode(subject(), job, {})
    assert job.agents[0].model_name == "direct/model" and job.agents[0].kwargs == {}
    app.HarborAgent.configure_opencode(subject(SimpleNamespace(name="policy_model")), job, {})
    assert "agent" not in job.agents[0].kwargs["opencode_config"]


def test_step_limit_works_for_direct_provider_and_rejects_other_harnesses():
    job = JobConfig(agents=[AgentConfig(name="opencode", model_name="direct/model")])
    app.HarborAgent.configure_opencode(subject(steps=2), job, {})
    assert job.agents[0].model_name == "direct/model"
    job.agents = [AgentConfig(name="terminus-2")]
    with pytest.raises(ValueError, match="OpenCode"):
        app.HarborAgent.configure_opencode(subject(steps=2), job, {})
