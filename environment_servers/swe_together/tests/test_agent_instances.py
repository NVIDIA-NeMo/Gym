# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The candidate and the judge are separate agent instances of one episode."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from environment_servers.swe_together import app
from environment_servers.swe_together.tests.test_environment import activation, request, server, state
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


TAG = "/training-token-capture"


def _global_config(capturing_agent: str) -> dict:
    return {
        "token_id_capture": {
            "enabled": True,
            "dir": "/capture",
            "training_agent_instances": ["assistant", "judge"],
        },
        "policy": {"responses_api_agents": {"opencode_agent": {"token_id_capture": capturing_agent == "policy"}}},
        "judge": {"responses_api_agents": {"claude_code_agent": {"token_id_capture": capturing_agent == "judge"}}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("capturing_agent, assistant_tag, judge_tag", [("policy", TAG, ""), ("judge", "", TAG)])
async def test_verify_activates_the_judge_agent_as_judge_with_that_agent_s_capture_setting(
    tmp_path, monkeypatch, capturing_agent, assistant_tag, judge_tag
):
    environment = server(tmp_path)
    environment.server_client = SimpleNamespace(global_config_dict=_global_config(capturing_agent))
    episode, _ = state(tmp_path, [])
    episode.snapshots.final_patch = "diff --git a/x b/x\n" + "+line\n" * 40
    episode.snapshots.baseline = {}
    episode.sandbox = SimpleNamespace()
    episode.task_metadata = {"judge_timeout": 10}
    judge = AsyncMock(return_value={"judge_score": 0.9})
    monkeypatch.setattr(app.SWETEnvironmentServer, "_new_sandbox", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(app, "replicate_submission", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "run_judge", judge)
    monkeypatch.setattr(app, "interaction_metrics", AsyncMock(return_value={}))
    key = episode.request.episode_id.capture_key

    await environment._verify(episode, NeMoGymResponseCreateParamsNonStreaming(input="task"), [activation()])

    sent = judge.await_args.kwargs
    assert sent["agent"].name == "judge"
    assert sent["episode"] == episode.request.episode_id
    assert sent["activation_path"] == f"/ng-rollout/{key}/ng-agent-instance/judge{judge_tag}/v1/responses"
    # The candidate's path reads the candidate agent's capture setting, not the judge's.
    assert (
        environment._activation_path(request(), "policy", app.ASSISTANT_AGENT_INSTANCE)
        == f"/ng-rollout/{key}/ng-agent-instance/assistant{assistant_tag}/v1/responses"
    )
