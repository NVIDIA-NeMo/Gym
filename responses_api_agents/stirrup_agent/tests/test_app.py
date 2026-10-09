# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf
from stirrup.core.models import AssistantMessage, TokenUsage, ToolCall

from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.stirrup_agent.app import StirrupAgentWrapper, StirrupAgentWrapperConfig, _observations
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent, NeMoUserMessage
from responses_api_agents.stirrup_agent.nemo_client import DynamicMaxTokensChatCompletionsClient
from responses_api_agents.stirrup_agent.stirrup_utils import convert_stirrup_history_to_output_items


STIRRUP_AGENT_DIR = Path(__file__).resolve().parent.parent


def _make_config(**fields) -> StirrupAgentWrapperConfig:
    return StirrupAgentWrapperConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="stirrup_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **fields,
    )


async def _runner_payload(config: StirrupAgentWrapperConfig, body: NeMoGymResponseCreateParamsNonStreaming) -> dict:
    """The payload the agent server hands the sandbox runner for one episode."""
    wrapper = StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))
    state = SimpleNamespace(
        request=SimpleNamespace(episode_id=EpisodeId(rollout_id="rollout", attempt=0)),
        session=SimpleNamespace(workdir="/root"),
        tool_access=DirectHTTPToolAccess(name="tools", required=True, base_url="http://resources:8000"),
        execute=AsyncMock(return_value={"input_items": [], "output_items": [], "elapsed_seconds": 0, "usages": []}),
    )
    with patch.object(StirrupAgentWrapper, "resolve_model_base_url", return_value="http://policy.invalid/v1"):
        await wrapper._run_sandbox_episode(body, state)
    return state.execute.await_args.args[0]


class TestApp:
    def test_sanity(self) -> None:
        """Config instantiation + wrapper construction should not raise."""
        config = _make_config()
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        # Generic Stirrup users retain the historical behavior. GDPVal opts in
        # to its larger floor and matching template semantics in its benchmark
        # config below.
        assert config.min_completion_tokens == 1024
        assert config.context_window_tokens == 262144
        assert config.prompt_estimator_truncate_history_thinking is None
        assert config.min_compaction_summary_words == 1
        assert config.truncation_recovery is False

    def test_gdpval_config_opts_into_safe_context_budget(self) -> None:
        repo_root = STIRRUP_AGENT_DIR.parents[1]
        config = OmegaConf.load(repo_root / "benchmarks" / "gdpval" / "config.yaml")
        agent = config.gdpval_stirrup_agent.responses_api_agents.stirrup_agent

        assert agent.min_completion_tokens == 8192
        assert agent.context_window_tokens == 262144
        assert "prompt_estimator_truncate_history_thinking" not in agent
        assert agent.min_compaction_summary_words == 50
        assert agent.truncation_recovery is True

    @pytest.mark.asyncio
    async def test_request_output_cap_does_not_replace_model_context_window(self) -> None:
        config = _make_config(context_window_tokens=262144, max_completion_tokens_cap=64000)
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy", max_output_tokens=8192)

        payload = await _runner_payload(config, body)

        assert payload["client"]["max_tokens"] == 262144
        assert payload["client"]["max_completion_tokens_cap"] == 8192

    @pytest.mark.parametrize("truncation_recovery", [None, False, True])
    @pytest.mark.asyncio
    async def test_truncation_recovery_is_forwarded_to_the_rollout(self, truncation_recovery) -> None:
        fields = {} if truncation_recovery is None else {"truncation_recovery": truncation_recovery}
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy")

        payload = await _runner_payload(_make_config(**fields), body)

        # Unset means off: only configs that opt in (GDPVal's benchmark config) get recovery nudges.
        assert payload["client"]["truncation_recovery"] is bool(truncation_recovery)

    @pytest.mark.asyncio
    async def test_runtime_knobs_reach_the_client_and_agent(self) -> None:
        """The runner builds its client from ``payload["client"]``; a field the client lacks fails only in a rollout."""
        config = _make_config(
            context_window_tokens=131072,
            min_completion_tokens=8192,
            prompt_estimator_truncate_history_thinking=True,
            truncation_recovery=True,
            min_compaction_summary_words=50,
        )
        payload = await _runner_payload(config, NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"))

        client = DynamicMaxTokensChatCompletionsClient(api_key="gym", **payload["client"])
        agent = NeMoAgent(
            client=client,
            name="stirrup_agent",
            max_turns=payload["max_turns"],
            min_compaction_summary_words=payload["min_compaction_summary_words"],
        )

        assert client.max_tokens == 131072
        assert client._min_completion_tokens == 8192
        assert client._prompt_estimator_truncate_history_thinking is True
        assert client._truncation_recovery is True
        assert agent._min_compaction_summary_words == 50

    def test_output_history_preserves_nemo_user_tool_results(self) -> None:
        """Run-history export should keep NeMo user-role tool results as tool outputs."""
        history = [
            [
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"true"}')],
                    token_usage=TokenUsage(input=1, answer=1, reasoning=0),
                ),
                NeMoUserMessage(content="ok", name="code_exec", success=True, tool_call_id="call_1"),
            ]
        ]

        input_items, output_items = convert_stirrup_history_to_output_items(history)

        assert input_items == []
        assert len(output_items) == 2
        assert output_items[0]["type"] == "function_call"
        assert output_items[0]["call_id"] == "call_1"
        assert output_items[1]["type"] == "function_call_output"
        assert output_items[1]["call_id"] == "call_1"
        assert output_items[1]["output"] == "ok"


class TestObservations:
    MODEL = ModelServerRef(type="responses_api_models", name="policy_model")
    TOOL = {"kind": "tool_call", "invocation_id": "root", "tool_call_id": "c1", "status": "completed"}

    def _output(self, **observations):
        raw = {"invocation_id": "root", "status": "completed", "model_response_ids": ["r1"], "tool_calls": [self.TOOL]}
        raw |= observations
        return {"input_items": [{"role": "user", "content": "hi", "type": "message"}], "observations": raw}

    def test_a_part_that_fails_validation_becomes_a_gap_and_the_rest_is_kept(self) -> None:
        bundle = _observations(self._output(tool_calls=[self.TOOL, {**self.TOOL, "status": "exploded"}]), self.MODEL)

        assert [record.kind for record in bundle.records] == ["agent_invocation", "tool_call"]
        assert [(gap.code, gap.detail) for gap in bundle.gaps] == [
            ("observation_capture_failed", "tool_calls[1]: ValidationError")
        ]

    def test_an_invalid_conversation_keeps_the_model_call_references(self) -> None:
        output = self._output()
        output["input_items"] = [{"type": "no_such_item"}]

        bundle = _observations(output, self.MODEL)

        invocation = bundle.records[0]
        assert [ref.response_id for ref in invocation.model_calls] == ["r1"]
        assert invocation.conversation == []
        assert [gap.detail for gap in bundle.gaps] == ["conversation: ValidationError"]

    def test_no_runner_output_is_one_gap(self) -> None:
        bundle = _observations(None, self.MODEL)

        assert bundle.records == []
        assert [gap.code for gap in bundle.gaps] == ["observation_capture_failed"]
