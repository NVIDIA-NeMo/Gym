# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent's runtime knobs must reach a client and agent that accept them.

``_run_stirrup_agent`` forwards the config fields to
``DynamicMaxTokensChatCompletionsClient`` and ``NeMoAgent``. A client or agent
without those parameters raises ``TypeError`` only when a rollout starts, so
build both for real here and stop before any model call.
"""

import pytest

from responses_api_agents.stirrup_agent import nemo_agent
from responses_api_agents.stirrup_agent.app import _run_stirrup_agent


pytest.importorskip("stirrup")


class _Built(Exception):
    pass


@pytest.mark.asyncio
async def test_runtime_knobs_reach_the_client_and_agent(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    built = {}
    real_agent = nemo_agent.NeMoAgent

    def build_then_stop(**kwargs):
        built["agent"] = real_agent(**kwargs)
        raise _Built

    monkeypatch.setattr(nemo_agent, "NeMoAgent", build_then_stop)

    with pytest.raises(_Built):
        await _run_stirrup_agent(
            task_prompt="t",
            system_prompt="",
            model_base_url="http://127.0.0.1:9/v1",
            model_name="policy",
            context_window_tokens=131072,
            min_completion_tokens=8192,
            prompt_estimator_truncate_history_thinking=True,
            truncation_recovery=True,
            min_compaction_summary_words=50,
        )

    agent = built["agent"]
    client = agent._client
    assert client.max_tokens == 131072
    assert client._min_completion_tokens == 8192
    assert client._prompt_estimator_truncate_history_thinking is True
    assert client._truncation_recovery is True
    assert agent._min_compaction_summary_words == 50
