# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import agentdojo.agent_pipeline.agent_pipeline as agentdyn_pipeline
import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.agentdojo_family.app import AgentDojoFamilyAgent
from responses_api_agents.agentdyn_agent.app import (
    AGENTDYN_DEFENSES,
    AGENTDYN_SUITES,
    PIGUARD_UPSTREAM_MODEL,
    PROMPT_GUARD_2_UPSTREAM_MODEL,
    AgentDynAgent,
    AgentDynAgentConfig,
    AgentDynRunRequest,
)


def _agent(*, default_defense: str | None = None, rollout_timeout_seconds: float | None = None) -> AgentDynAgent:
    config = AgentDynAgentConfig(
        rollout_timeout_seconds=rollout_timeout_seconds,
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentdyn",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        default_defense=default_defense,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    return AgentDynAgent(config=config, server_client=server_client)


def _request() -> AgentDynRunRequest:
    return AgentDynRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "selector"}]},
            "suite": "shopping",
            "user_task_id": "user_task_0",
            "injection_task_id": None,
            "attack": None,
            "defense": None,
            "benchmark_version": "v1.2.2",
        }
    )


def test_contract_exposes_only_agentdyn_suites_and_upstream_defenses() -> None:
    assert AGENTDYN_SUITES == ("shopping", "github", "dailylife")
    # Upstream's full DEFENSES list, nothing invented and nothing dropped.
    assert set(AGENTDYN_DEFENSES) == {
        "tool_filter",
        "transformers_pi_detector",
        "piguard_detector",
        "prompt_guard_2_detector",
        "spotlighting_with_delimiting",
        "repeat_user_prompt",
        "camel",
        "drift",
        "progent",
    }
    with pytest.raises(ValueError):
        AgentDynRunRequest.model_validate({**_request().model_dump(), "suite": "banking"})


async def test_configured_defense_is_applied_to_request() -> None:
    agent = _agent(default_defense="piguard_detector")
    agent._run_agentdojo = MagicMock(return_value=(True, True))

    result = await agent.run(_request())

    assert result.defense == "piguard_detector"
    assert result.reward == 1.0


async def test_prompt_guard_source_is_recorded_on_rollout() -> None:
    agent = _agent(default_defense="prompt_guard_2_detector")
    agent.config.prompt_guard_2_model_name = "mirror/prompt-guard-2"
    agent.config.prompt_guard_2_model_revision = "abc123"
    agent._run_agentdojo = MagicMock(return_value=(True, True))

    result = await agent.run(_request())

    assert result.detector_model_name == "mirror/prompt-guard-2"
    assert result.detector_model_revision == "abc123"


def test_prompt_guard_local_cache_replaces_only_upstream_detector() -> None:
    agent = _agent(default_defense="prompt_guard_2_detector")
    agent.config.prompt_guard_2_local_path = "/tmp/prompt-guard-2"
    body = _request().model_copy(
        update={
            "defense": "prompt_guard_2_detector",
            "detector_model_name": agent.config.prompt_guard_2_model_name,
            "detector_model_revision": agent.config.prompt_guard_2_model_revision,
        }
    )
    constructed_model_names: list[str] = []

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            constructed_model_names.append(kwargs["model_name"])

    def exercise_detector(*args, **kwargs):
        agentdyn_pipeline.TransformersBasedPIDetector(
            model_name=PROMPT_GUARD_2_UPSTREAM_MODEL,
            safe_label="LABEL_0",
        )
        return True, True

    with (
        patch.object(agentdyn_pipeline, "TransformersBasedPIDetector", FakeDetector),
        patch.object(AgentDojoFamilyAgent, "_run_agentdojo", side_effect=exercise_detector),
    ):
        assert agent._run_agentdojo(body, MagicMock(), "v1.2.2") == (True, True)

    assert constructed_model_names == ["/tmp/prompt-guard-2"]


def test_piguard_detector_source_is_pinned_and_recorded() -> None:
    agent = _agent(default_defense="piguard_detector")
    agent.config.piguard_model_name = "leolee99/PIGuard"
    agent.config.piguard_model_revision = "dd78b24"
    agent.config.piguard_local_path = "/tmp/piguard"
    body = _request().model_copy(update={"defense": "piguard_detector"})
    constructed_model_names: list[str] = []

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            constructed_model_names.append(kwargs["model_name"])

    def exercise_detector(*args, **kwargs):
        agentdyn_pipeline.TransformersBasedPIDetector(
            model_name=PIGUARD_UPSTREAM_MODEL,
            safe_label="benign",
        )
        return True, True

    with (
        patch.object(agentdyn_pipeline, "TransformersBasedPIDetector", FakeDetector),
        patch.object(AgentDojoFamilyAgent, "_run_agentdojo", side_effect=exercise_detector),
    ):
        assert agent._run_agentdojo(body, MagicMock(), "v1.2.2") == (True, True)

    # The pinned local snapshot replaces upstream's moving-main repo id, and the rollout
    # records which revision actually classified the tool outputs.
    assert constructed_model_names == ["/tmp/piguard"]
    assert agent.config.detector_source("piguard_detector") == ("leolee99/PIGuard", "dd78b24", "/tmp/piguard")


async def test_piguard_revision_is_recorded_on_rollout() -> None:
    agent = _agent(default_defense="piguard_detector")
    agent.config.piguard_model_revision = "dd78b24"
    agent._run_agentdojo = MagicMock(return_value=(True, True))

    result = await agent.run(_request())

    assert result.detector_model_name == "leolee99/PIGuard"
    assert result.detector_model_revision == "dd78b24"


def test_agentdyn_metrics_use_separate_namespace() -> None:
    agent = _agent()
    metrics = agent.compute_metrics(
        [
            [{"utility": True, "security": True, "attack_success": False, "injection_task_id": None}],
            [{"utility": False, "security": False, "attack_success": True, "injection_task_id": "i0"}],
        ]
    )
    assert metrics["agentdyn/benign_utility"] == 1.0
    assert metrics["agentdyn/utility_under_attack"] == 0.0
    assert metrics["agentdyn/attack_success_rate"] == 1.0


async def test_rollout_that_never_finishes_is_masked_not_scored() -> None:
    # CaMeL interprets model-generated Python with no step or time budget, so a
    # non-terminating program would stall the run rather than cost one row. The row is masked:
    # nothing ran to completion, so calling it secure would credit the defense for a hang.
    agent = _agent(default_defense="camel", rollout_timeout_seconds=0.01)
    agent.config.max_abandoned_rollouts = 99  # keep the process-exit guard out of this test

    def never_finishes(*args, **kwargs):
        time.sleep(5)
        raise AssertionError("should have been abandoned")

    agent._run_agentdojo = never_finishes
    with patch.object(AgentDynAgent, "resolve_model_base_url", return_value="http://127.0.0.1:1/v1"):
        result = await agent.run(_request())

    assert result.mask_sample is True
    assert result.adapter_error is not None and result.adapter_error.startswith("RolloutTimeout")
    assert result.utility is False and result.security is False
    assert result.reward == 0.0


async def test_a_finished_rollout_is_untouched_by_the_timeout() -> None:
    agent = _agent(default_defense="piguard_detector", rollout_timeout_seconds=30)
    agent._run_agentdojo = MagicMock(return_value=(True, True))

    result = await agent.run(_request())

    assert result.mask_sample is False
    assert result.reward == 1.0


def test_bridge_satisfies_the_tool_filter_contract_without_leaving_gym() -> None:
    # tool_filter refuses to build unless the pipeline's llm is an OpenAILLM, and then calls
    # `llm.client.chat.completions.create`. Standing in for OpenAILLM is what lets the defense
    # run; pointing its client back at the bridge is what keeps the call on the model server
    # rather than going straight at the serving endpoint, which would lose token accounting
    # and skip the developer->system rewrite some OpenAI-compatible servers require.
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    from responses_api_agents.agentdojo_family.model_bridge import NeMoGymAgentDojoLLM, NeMoGymOpenAIProxy

    bridge = object.__new__(NeMoGymAgentDojoLLM)
    NeMoGymAgentDojoLLM.__init__(
        bridge,
        pipeline_name="policy-alias",
        event_loop=MagicMock(),
        server_client=MagicMock(),
        model_server_name="policy_model",
        model_url_path="/v1/chat/completions",
        request_options={},
    )

    assert isinstance(bridge, OpenAILLM)
    assert isinstance(bridge.client, NeMoGymOpenAIProxy)
    assert bridge.client.chat.completions.create == bridge.create_chat_completion
    assert bridge.model == "policy-alias"


def test_transformers_pi_detector_source_is_pinned() -> None:
    agent = _agent(default_defense="transformers_pi_detector")
    agent.config.transformers_pi_model_revision = "90c9989b"
    repo, revision, _ = agent.config.detector_source("transformers_pi_detector")

    assert repo == "protectai/deberta-v3-base-prompt-injection-v2"
    assert revision == "90c9989b"
