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
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    StirrupAgentWrapper,
    StirrupAgentWrapperConfig,
    StirrupRunRequest,
    _load_task_registry,
    _task_finished,
    _verify_cache_path,
    get_task_strategy,
)
from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage
from responses_api_agents.stirrup_agent.stirrup_utils import convert_stirrup_history_to_output_items
from responses_api_agents.stirrup_agent.task_strategy import TaskStrategy


STIRRUP_AGENT_DIR = Path(__file__).resolve().parent.parent


def _make_config(
    *,
    execute_only: bool = False,
    judge_only: bool = False,
    rerun_incomplete: bool = False,
    persist_deliverables_dir=None,
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
        rerun_incomplete=rerun_incomplete,
        persist_deliverables_dir=persist_deliverables_dir,
    )


def _fake_response(response_id: str = "gdpval-task-1") -> NeMoGymResponse:
    """A minimal completed NeMoGymResponse for patching ``responses()``."""
    return NeMoGymResponse(
        id=response_id,
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
        metadata={"elapsed_seconds": "5.0"},
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


class TestTaskFinished:
    def test_none_dir_is_unfinished(self) -> None:
        assert _task_finished(None) is False

    def test_missing_dir_is_unfinished(self, tmp_path) -> None:
        assert _task_finished(str(tmp_path / "does-not-exist")) is False

    def test_empty_dir_is_unfinished(self, tmp_path) -> None:
        assert _task_finished(str(tmp_path)) is False

    def test_finish_params_alone_counts_as_finished(self, tmp_path) -> None:
        """A finish marker means the rollout completed even with NO deliverable —
        the model just couldn't make one, so the task is finished, not re-run."""
        (tmp_path / "finish_params.json").write_text("{}")
        assert _task_finished(str(tmp_path)) is True

    def test_history_file_alone_is_unfinished(self, tmp_path) -> None:
        """Only finish_params.json marks completion; history/metadata files alone
        (e.g. partial persist, or a crash mid-persist) do not."""
        (tmp_path / "history.json").write_text("[]")
        (tmp_path / "history.pkl").write_bytes(b"")
        assert _task_finished(str(tmp_path)) is False

    def test_metadata_file_alone_is_unfinished(self, tmp_path) -> None:
        (tmp_path / "metadata.json").write_text("{}")
        assert _task_finished(str(tmp_path)) is False

    def test_finished_with_deliverable_is_finished(self, tmp_path) -> None:
        (tmp_path / "finish_params.json").write_text("{}")
        (tmp_path / "report.docx").write_text("the deliverable")
        assert _task_finished(str(tmp_path)) is True

    def test_deliverable_without_finish_marker_is_unfinished(self, tmp_path) -> None:
        """Deliverable files but no finish marker (e.g. persist interrupted) means
        the rollout did not actually finish."""
        (tmp_path / "report.docx").write_text("the deliverable")
        assert _task_finished(str(tmp_path)) is False


class TestRerunIncompleteMode:
    def test_rerun_incomplete_requires_persist_dir(self) -> None:
        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=None)
        with pytest.raises(ValueError, match="rerun_incomplete=True requires persist_deliverables_dir"):
            StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

    def test_rerun_incomplete_and_judge_only_is_allowed(self, tmp_path) -> None:
        """rerun_incomplete + judge_only is a valid combination (resume judging)."""
        config = _make_config(rerun_incomplete=True, judge_only=True, persist_deliverables_dir=str(tmp_path))
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

    def _make_body(self, task_id: str = "task-1") -> StirrupRunRequest:
        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": task_id, "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        return StirrupRunRequest(responses_create_params=params, task_id=task_id, prompt="do the thing")

    @pytest.mark.asyncio
    async def test_full_mode_finished_task_skips_rollout_and_verifies(self, tmp_path) -> None:
        """A finished task (finish marker cached) must NOT run the agent and must
        re-score the cached deliverable via /verify."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text("{}")
        (deliverables_root / "report.docx").write_text("cached deliverable")

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.7, "judge_response": "ok"}),
            ),
        ):
            result = await wrapper.run(request, self._make_body())

        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert verify_calls[0].kwargs["json"]["deliverables_dir"].endswith(str(Path("task_task-1") / "repeat_0"))
        assert result == {"reward": 0.7, "judge_response": "ok"}

        # The judgement must be cached as a sibling file (NOT inside the
        # deliverables dir, so it can't leak into the judge's input).
        cache_path = _verify_cache_path(str(deliverables_root))
        assert cache_path == tmp_path / "task_task-1" / "repeat_0_verify_response.json"
        assert json.loads(cache_path.read_text()) == {"reward": 0.7, "judge_response": "ok"}

    @pytest.mark.asyncio
    async def test_full_mode_already_judged_returns_cached_judgement(self, tmp_path) -> None:
        """A finished task AND a cached /verify result must return that judgement
        directly — no rollout and no /verify call."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text("{}")
        (deliverables_root / "report.docx").write_text("cached deliverable")
        cache_path = _verify_cache_path(str(deliverables_root))
        cache_path.write_text(json.dumps({"reward": 0.55, "judge_response": "prior"}))

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with patch.object(StirrupAgentWrapper, "responses", responses_mock):
            result = await wrapper.run(request, self._make_body())

        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert verify_calls == []
        assert result == {"reward": 0.55, "judge_response": "prior"}

    @pytest.mark.asyncio
    async def test_full_mode_no_finish_marker_routes_incomplete(self, tmp_path) -> None:
        """When the fresh rollout persists no finish marker, run() returns a
        retryable 'incomplete' failure routed to the sidecar (not terminal, not
        no-persist) and never reaches /verify."""
        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("no server in unit test"))
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        # responses() returns but writes nothing to disk (no finish marker persisted).
        with patch.object(StirrupAgentWrapper, "responses", AsyncMock(return_value=_fake_response())):
            result = await wrapper.run(request, self._make_body())

        for call in server_client.post.await_args_list:
            assert call.kwargs.get("url_path") == "/seed_session"
        assert result[NG_FAILURE_CLASS_KEY] == "incomplete"
        assert result["error_class"] == "incomplete"
        assert result["reward"] == 0.0
        assert result["skipped"] is False
        assert NG_TERMINAL_KEY not in result
        assert NG_NO_PERSIST_KEY not in result

    @pytest.mark.asyncio
    async def test_full_mode_fresh_finished_rollout_verifies(self, tmp_path) -> None:
        """When the fresh rollout finishes (persists a finish marker), the normal
        /verify path runs (no 'incomplete' routing)."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        async def _responses_side_effect(*_args, **_kwargs):
            # Mimic the Ray worker persisting a finish marker + deliverable.
            deliverables_root.mkdir(parents=True, exist_ok=True)
            (deliverables_root / "finish_params.json").write_text("{}")
            (deliverables_root / "report.docx").write_text("fresh deliverable")
            return _fake_response()

        with (
            patch.object(StirrupAgentWrapper, "responses", AsyncMock(side_effect=_responses_side_effect)),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.4}),
            ),
        ):
            result = await wrapper.run(request, self._make_body())

        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert result == {"reward": 0.4}

    @pytest.mark.asyncio
    async def test_full_mode_finished_without_deliverable_is_judged_not_rerun(self, tmp_path) -> None:
        """The model finished but made no deliverable: the task is NOT re-run; the
        fresh rollout's result is judged via /verify like any finished task."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        async def _responses_side_effect(*_args, **_kwargs):
            # Finished (finish marker persisted) but produced NO deliverable file.
            deliverables_root.mkdir(parents=True, exist_ok=True)
            (deliverables_root / "finish_params.json").write_text('{"reason": "could not produce file"}')
            (deliverables_root / "history.json").write_text("[]")
            return _fake_response()

        with (
            patch.object(StirrupAgentWrapper, "responses", AsyncMock(side_effect=_responses_side_effect)),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.0, "judge_response": "no deliverable"}),
            ),
        ):
            result = await wrapper.run(request, self._make_body())

        # Not routed as 'incomplete' — it finished, so it is judged normally.
        assert NG_FAILURE_CLASS_KEY not in result
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert result == {"reward": 0.0, "judge_response": "no deliverable"}

    @pytest.mark.asyncio
    async def test_full_mode_finished_without_deliverable_cache_reused(self, tmp_path) -> None:
        """A previously finished task with only a finish marker (no deliverable)
        skips the rollout and is judged from cache — it is not re-run."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text('{"reason": "abandoned"}')

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.0}),
            ),
        ):
            result = await wrapper.run(request, self._make_body())

        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert result == {"reward": 0.0}

    @pytest.mark.asyncio
    async def test_execute_only_finished_skips_rollout(self, tmp_path) -> None:
        """rerun_incomplete + execute_only: a finished task (finish marker cached)
        returns the payload without running the agent or judging."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text("{}")
        (deliverables_root / "report.docx").write_text("cached deliverable")

        config = _make_config(rerun_incomplete=True, execute_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("no server in unit test"))
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with patch.object(StirrupAgentWrapper, "responses", responses_mock):
            result = await wrapper.run(request, self._make_body())

        responses_mock.assert_not_awaited()
        for call in server_client.post.await_args_list:
            assert call.kwargs.get("url_path") == "/seed_session"
        assert result["execute_only"] is True
        assert "reward" not in result
        assert result["deliverables_dir"].endswith(str(Path("task_task-1") / "repeat_0"))

    @pytest.mark.asyncio
    async def test_execute_only_no_finish_marker_routes_incomplete(self, tmp_path) -> None:
        """rerun_incomplete + execute_only: a fresh rollout that persists no finish
        marker is routed as 'incomplete' rather than a success payload."""
        config = _make_config(rerun_incomplete=True, execute_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("no server in unit test"))
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        with patch.object(StirrupAgentWrapper, "responses", AsyncMock(return_value=_fake_response())):
            result = await wrapper.run(request, self._make_body())

        assert result.get("execute_only") is not True
        assert result[NG_FAILURE_CLASS_KEY] == "incomplete"
        assert result["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_judge_only_already_judged_returns_cached_judgement(self, tmp_path) -> None:
        """rerun_incomplete + judge_only: a task whose judgement is already cached
        is returned as-is — the judge is NOT re-run."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "report.docx").write_text("cached deliverable")
        _verify_cache_path(str(deliverables_root)).write_text(json.dumps({"reward": 0.8, "judge_response": "prior"}))

        config = _make_config(rerun_incomplete=True, judge_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        with patch.object(StirrupAgentWrapper, "responses", AsyncMock()):
            result = await wrapper.run(request, self._make_body())

        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert verify_calls == []
        assert result == {"reward": 0.8, "judge_response": "prior"}

    @pytest.mark.asyncio
    async def test_judge_only_unjudged_task_is_judged_and_cached(self, tmp_path) -> None:
        """rerun_incomplete + judge_only: a task with deliverables but no cached
        judgement is scored via /verify, and that judgement is cached."""
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "report.docx").write_text("cached deliverable")

        config = _make_config(rerun_incomplete=True, judge_only=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        with (
            patch.object(StirrupAgentWrapper, "responses", AsyncMock()),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value={"reward": 0.3, "judge_response": "fresh"}),
            ),
        ):
            result = await wrapper.run(request, self._make_body())

        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert result == {"reward": 0.3, "judge_response": "fresh"}
        cache_path = _verify_cache_path(str(deliverables_root))
        assert json.loads(cache_path.read_text()) == {"reward": 0.3, "judge_response": "fresh"}


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


class TestReferenceKeyedVerifyCache:
    """rerun_incomplete + multi-stage ELO: the cached judgement is keyed by the
    stage's reference subset, so each stage reuses only a judgement produced
    against the SAME references (and re-judges otherwise)."""

    def _body(self, reference_ids):
        params = NeMoGymResponseCreateParamsNonStreaming(
            input="ignored",
            metadata={"task_id": "task-1", "prompt": "do the thing", "_ng_rollout_index": "0"},
        )
        return StirrupRunRequest(
            responses_create_params=params,
            task_id="task-1",
            prompt="do the thing",
            reuse_cached_deliverable=True,
            reference_ids=list(reference_ids),
        )

    def test_cache_path_is_reference_set_keyed_and_order_independent(self, tmp_path) -> None:
        d = str(tmp_path / "task_task-1" / "repeat_0")
        unkeyed = _verify_cache_path(d)
        bc = _verify_cache_path(d, ["ref_b", "ref_c"])
        cb = _verify_cache_path(d, ["ref_c", "ref_b"])
        bd = _verify_cache_path(d, ["ref_b", "ref_d"])
        # No references ⇒ the single unkeyed slot.
        assert unkeyed.name == "repeat_0_verify_response.json"
        # Reference sets get distinct, order-independent slots.
        assert bc == cb
        assert bc != unkeyed
        assert bc != bd

    @pytest.mark.asyncio
    async def test_same_reference_set_reuses_cached_judgement(self, tmp_path) -> None:
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text("{}")
        (deliverables_root / "report.docx").write_text("cached deliverable")
        # A judgement for the {b,c} reference set is already cached.
        cached = {"reward": 0.9, "judge_response": "bc"}
        _verify_cache_path(str(deliverables_root), ["ref_b", "ref_c"]).write_text(json.dumps(cached))

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        with patch.object(StirrupAgentWrapper, "responses", responses_mock):
            result = await wrapper.run(request, self._body(["ref_c", "ref_b"]))

        # Same reference set (order-independent) ⇒ cached judgement returned; no
        # rollout and no /verify call.
        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert verify_calls == []
        assert result == cached

    @pytest.mark.asyncio
    async def test_different_reference_set_rejudges_and_caches_separately(self, tmp_path) -> None:
        deliverables_root = tmp_path / "task_task-1" / "repeat_0"
        deliverables_root.mkdir(parents=True)
        (deliverables_root / "finish_params.json").write_text("{}")
        (deliverables_root / "report.docx").write_text("cached deliverable")
        # Cached judgement is for {b,c}; this request judges against {a,b}.
        bc_cached = {"reward": 0.9, "judge_response": "bc"}
        bc_path = _verify_cache_path(str(deliverables_root), ["ref_b", "ref_c"])
        bc_path.write_text(json.dumps(bc_cached))

        config = _make_config(rerun_incomplete=True, persist_deliverables_dir=str(tmp_path))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=MagicMock())
        wrapper = StirrupAgentWrapper(config=config, server_client=server_client)

        request = MagicMock()
        request.cookies = {}

        responses_mock = AsyncMock()
        fresh = {"reward": 0.3, "judge_response": "ab"}
        with (
            patch.object(StirrupAgentWrapper, "responses", responses_mock),
            patch("responses_api_agents.stirrup_agent.app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.stirrup_agent.app.get_response_json",
                AsyncMock(return_value=fresh),
            ),
        ):
            result = await wrapper.run(request, self._body(["ref_a", "ref_b"]))

        # Different reference set ⇒ policy still skipped (deliverable cached) but
        # the deliverable is re-judged against {a,b}...
        responses_mock.assert_not_awaited()
        verify_calls = [c for c in server_client.post.await_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 1
        assert verify_calls[0].kwargs["json"]["reference_ids"] == ["ref_a", "ref_b"]
        assert result == fresh
        # ...the {b,c} cache is untouched and the {a,b} judgement is cached separately.
        assert json.loads(bc_path.read_text()) == bc_cached
        ab_path = _verify_cache_path(str(deliverables_root), ["ref_a", "ref_b"])
        assert ab_path != bc_path
        assert json.loads(ab_path.read_text()) == fresh


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
