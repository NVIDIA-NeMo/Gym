# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from nooa.interactive import SummarizationConfig
from nooa.unifiedllm import FakeLLMClient
from nooa_bench.bench_agent import TaskResult

from responses_api_agents.nooa_agent.bench_agent_no_compaction import BenchAgentWithoutCompaction


async def test_delegation_preserves_disabled_compaction(tmp_path, monkeypatch):
    parent = BenchAgentWithoutCompaction(llm=FakeLLMClient(), working_dir=str(tmp_path))
    children = []

    async def solve(child, *args, **kwargs):
        children.append(child)
        return TaskResult(solution_description="done", evidence="observed", how_to_verify="inspect")

    monkeypatch.setattr(BenchAgentWithoutCompaction, "_solve_task", solve)
    try:
        await parent.delegate("inspect files")
        assert len(children) == 1
        assert parent.render_config.block_formatter is not children[0].render_config.block_formatter
        for agent in [parent, *children]:
            assert agent._summarization.policy == "none"
            assert agent._truncation.max_context_tokens is None
            assert agent._truncation.max_event_tokens is None
            assert agent._truncation.response_reserve_tokens == 0
            before = list(agent.event_manager.keys())
            error = ValueError("context window exceeded")
            with pytest.raises(ValueError) as raised:
                agent.runtime._archive_on_context_error(196608, error)
            assert raised.value is error
            assert list(agent.event_manager.keys()) == before
    finally:
        await parent.aclose()


def test_conflicting_compaction_settings_fail():
    with pytest.raises(ValueError, match="policy='none'"):
        BenchAgentWithoutCompaction(summarization=SummarizationConfig())
    with pytest.raises(ValueError, match="owns history"):
        BenchAgentWithoutCompaction(truncation=object())
