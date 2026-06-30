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
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from stirrup.core.models import AssistantMessage, TokenUsage, ToolCall

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.stirrup_agent.app import (
    StirrupAgentWrapper,
    StirrupAgentWrapperConfig,
    StirrupRunRequest,
    _load_task_registry,
    get_task_strategy,
)
from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage
from responses_api_agents.stirrup_agent.stirrup_utils import convert_stirrup_history_to_output_items
from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


STIRRUP_AGENT_DIR = Path(__file__).resolve().parent.parent


def _make_config(
    *, execute_only: bool = False, judge_only: bool = False, persist_deliverables_dir=None
) -> StirrupAgentWrapperConfig:
    return StirrupAgentWrapperConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="stirrup_agent",
        task="gdpval",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="gdpval_resources_server"),
        execute_only=execute_only,
        judge_only=judge_only,
        persist_deliverables_dir=persist_deliverables_dir,
    )


class TestTaskRegistry:
    def test_registry_includes_gdpval(self) -> None:
        registry = _load_task_registry()
        assert "gdpval" in registry

    def test_get_task_strategy_returns_instance(self) -> None:
        strategy = get_task_strategy("gdpval")
        assert isinstance(strategy, TaskStrategy)

    def test_get_task_strategy_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_strategy("this_task_does_not_exist")


class TestApp:
    def test_sanity(self) -> None:
        """Config instantiation + wrapper construction should not raise."""
        config = StirrupAgentWrapperConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="stirrup_agent",
            task="gdpval",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="gdpval_resources_server",
            ),
        )
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

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
        assert output_items[0].type == "function_call"
        assert output_items[0].call_id == "call_1"
        assert output_items[1].type == "function_call_output"
        assert output_items[1].call_id == "call_1"
        assert output_items[1].output == "ok"


class TestExecuteOnlyMode:
    def test_execute_only_requires_persist_dir(self) -> None:
        """execute_only without a persist dir is useless — nothing is saved."""
        config = _make_config(execute_only=True, persist_deliverables_dir=None)
        with pytest.raises(ValueError, match="execute_only=True requires persist_deliverables_dir"):
            StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_run_execute_only_skips_verify(self, tmp_path) -> None:
        """In execute_only mode, run() must not POST /verify and must return a
        judgement-free payload (no reward / judge_response) carrying the
        response + deliverables_dir."""
        config = _make_config(execute_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        # seed_session is the only legitimate POST; make it fail so the
        # non-fatal except branch is exercised and we'd notice any /verify POST.
        server_client.post = AsyncMock(side_effect=RuntimeError("no server in unit test"))
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        fake_response = NeMoGymResponse(
            id="gdpval-task-1",
            created_at=0,
            model="policy",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg-1",
                    content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={"elapsed_seconds": "12.5"},
        )

        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "task-1", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        body = StirrupRunRequest(responses_create_params=params, task_id="task-1", prompt="do the thing")
        request = MagicMock()
        request.cookies = {}

        # ``responses`` is a pydantic-model method, so patch it on the class.
        with patch.object(StirrupAgentWrapper, "responses", AsyncMock(return_value=fake_response)):
            result = await wrapper.run(request, body)

        # No /verify (or any non-seed) POST should have been issued.
        for call in server_client.post.await_args_list:
            assert call.kwargs.get("url_path") == "/seed_session"

        assert result["execute_only"] is True
        assert "reward" not in result
        assert "judge_response" not in result
        assert result["response"]["id"] == "gdpval-task-1"
        assert result["deliverables_dir"].endswith(str(Path("task_task-1") / "repeat_0"))
        assert result["elapsed_seconds"] == 12.5


class TestJudgeOnlyMode:
    def test_judge_only_requires_persist_dir(self) -> None:
        """judge_only without a persist dir has no cached deliverables to score."""
        config = _make_config(judge_only=True, persist_deliverables_dir=None)
        with pytest.raises(ValueError, match="judge_only=True requires persist_deliverables_dir"):
            StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.asyncio
    async def test_run_judge_only_scores_cached_deliverables(self, tmp_path) -> None:
        """When cached deliverables exist, run() must NOT execute the agent and
        must POST /verify with the cached deliverables_dir."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "answer.txt").write_text("cached deliverable")

        config = _make_config(judge_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "task-1", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        body = StirrupRunRequest(responses_create_params=params, task_id="task-1", prompt="do the thing")
        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.9, "judge_response": "ok"}),
            ),
        ):
            result = await wrapper.run(request, body)

        # The agent task must never run in judge-only mode.
        responses_mock.assert_not_awaited()

        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        verify_json = verify_calls[0].kwargs["json"]
        assert verify_json["deliverables_dir"].endswith(str(Path("task_task-1") / "repeat_0"))
        assert result == {"reward": 0.9, "judge_response": "ok"}

    @pytest.mark.asyncio
    async def test_run_judge_only_missing_deliverables_is_skipped(self, tmp_path) -> None:
        """A task with no cached deliverable dir is reported skipped and never
        reaches /verify."""
        config = _make_config(judge_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "missing-task", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        body = StirrupRunRequest(responses_create_params=params, task_id="missing-task", prompt="do the thing")
        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
        ):
            result = await wrapper.run(request, body)

        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert verify_calls == []
        assert result["skipped"] is True
        assert result["reward"] == 0.0


class TestReuseCachedDeliverable:
    """Per-request ``reuse_cached_deliverable`` (used by multi-stage ELO): reuse a
    deliverable produced by an earlier stage instead of re-running the policy."""

    @pytest.mark.asyncio
    async def test_reuse_skips_policy_when_cached(self, tmp_path) -> None:
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "answer.txt").write_text("cached deliverable")

        # NOT judge_only: this is a normal (produce) server that opts into reuse
        # per request.
        config = _make_config(persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "task-1", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        body = StirrupRunRequest(
            responses_create_params=params,
            task_id="task-1",
            prompt="do the thing",
            reuse_cached_deliverable=True,
        )
        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.7}),
            ),
        ):
            result = await wrapper.run(request, body)

        # Cached deliverable ⇒ policy is NOT run, but /verify still scores it.
        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert verify_calls[0].kwargs["json"]["deliverables_dir"].endswith(str(Path("task_task-1") / "repeat_0"))
        assert result == {"reward": 0.7}

    @pytest.mark.asyncio
    async def test_reuse_falls_back_to_policy_when_cold(self, tmp_path) -> None:
        # No cached deliverable on disk ⇒ reuse request must run the policy.
        config = _make_config(persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        fake_response = NeMoGymResponse(
            id="gdpval-task-1",
            created_at=0,
            model="policy",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg-1",
                    content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            metadata={"elapsed_seconds": "1.0"},
        )

        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "task-1", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        body = StirrupRunRequest(
            responses_create_params=params,
            task_id="task-1",
            prompt="do the thing",
            reuse_cached_deliverable=True,
        )
        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock(return_value=fake_response)
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.5}),
            ),
        ):
            result = await wrapper.run(request, body)

        # Cold cache ⇒ the policy runs to produce the deliverable.
        responses_mock.assert_awaited_once()
        assert result == {"reward": 0.5}


class TestExampleDataset:
    def test_example_jsonl_is_valid(self) -> None:
        """The shipped example dataset should parse and contain the GDPVal schema."""
        example_path = STIRRUP_AGENT_DIR / "data" / "example.jsonl"
        assert example_path.is_file(), f"missing {example_path}"

        lines = example_path.read_text().strip().splitlines()
        assert len(lines) >= 1

        for line in lines:
            record = json.loads(line)
            params = record["responses_create_params"]
            metadata = params["metadata"]
            # Schema contract required by GDPValTask.extract_task_info.
            assert "task_id" in metadata
            assert "prompt" in metadata
            # Metadata must be all strings (OpenAI Metadata type constraint).
            for key, value in metadata.items():
                assert isinstance(value, str), f"metadata['{key}'] is {type(value).__name__}, not str"
            # reference_files / rubric_json are JSON-encoded strings.
            json.loads(metadata["reference_files"])
            json.loads(metadata["rubric_json"])
