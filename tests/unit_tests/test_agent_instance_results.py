# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Episode results report each agent instance's terminal call, reward and rebuilt response."""

import orjson
import pytest
from fastapi import Body
from omegaconf import OmegaConf
from pydantic import BaseModel

import nemo_gym.server_utils
from nemo_gym.base_environment_server import BaseEnvironmentServer, BaseEnvironmentServerConfig, CleanupContext
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BaseServerConfig
from nemo_gym.episode_types import (
    AgentInstanceResult,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.rollout_collection import _episode_record
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.delivery import MASK_SAMPLE_KEY, TOKEN_CAPTURE_KEY, finalize_rollout_token_capture
from nemo_gym.token_id_capture.lineage import stamp_continuation
from nemo_gym.token_id_capture.records import CallIdentity, ModelRef, ParentResolutionStatus, TokenEntry, stamp_lineage
from nemo_gym.token_id_capture.store import TokenCaptureStore


class _TaskInput(BaseModel):
    fail: bool = False


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[dict]):
    pass


class _Reply:
    def __init__(self, payload: dict) -> None:
        self.ok = True
        self._body = orjson.dumps(payload)

    async def read(self) -> bytes:
        return self._body


class _DebateServer(BaseEnvironmentServer[_Request, _Response]):
    """Activates two agent sessions by name and scores only the assistant."""

    request_model = _Request
    response_model = _Response

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
        key = request.episode_id.capture_key
        for name in ("assistant", "user", "assistant"):
            await self.server_client.post("agent", f"/ng-rollout/{key}/ng-agent-instance/{name}/v1/responses", json={})
        # Unnamed and non-activation calls are not agent instance terminals.
        await self.server_client.post("agent", f"/ng-rollout/{key}/v1/responses", json={})
        await self.server_client.post("resources", f"/ng-rollout/{key}/ng-agent-instance/judge/v1/responses", json={})
        if request.task.task_input.fail:
            return _Response(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                failure=EpisodeFailure(failure_reason="lost", terminal=False),
            )
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result={"reward": 1.0},
            agent_instances={"assistant": AgentInstanceResult(reward=1.0)},
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


@pytest.fixture
def debate(monkeypatch):
    replies = iter(["resp_a1", "resp_u1", "resp_a2", "resp_unnamed", "resp_resources"])

    async def transport(method, url, **kwargs):
        return _Reply({"id": next(replies)})

    monkeypatch.setattr(nemo_gym.server_utils, "request", transport)
    client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"agent": {"host": "agent", "port": 1}}},
                "resources": {"resources_servers": {"resources": {"host": "resources", "port": 1}}},
            }
        ),
    )
    config = BaseEnvironmentServerConfig(
        name="environment", host="127.0.0.1", port=1, entrypoint="app.py", cleanup_timeout_seconds=1
    )
    return _DebateServer(config=config, server_client=client)


def _request(*, fail: bool = False) -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="r"),
        task=MaterializedTask(task_id=TaskId(taskset="t", task_id="x"), task_input=_TaskInput(fail=fail)),
    )


async def test_each_agent_instance_s_terminal_is_its_last_activation_and_rewards_are_kept(debate):
    response = await debate.run_request(_request())

    assert response.agent_instances == {
        "assistant": AgentInstanceResult(terminal_response_id="resp_a2", reward=1.0),
        "user": AgentInstanceResult(terminal_response_id="resp_u1"),
    }
    record = _episode_record(response.model_dump(mode="json"))
    assert record["reward"] == 1.0
    assert record["agent_instances"] == {
        "assistant": {"terminal_response_id": "resp_a2", "reward": 1.0},
        "user": {"terminal_response_id": "resp_u1"},
    }


async def test_a_failed_episode_reports_no_agent_instances(debate):
    response = await debate.run_request(_request(fail=True))

    assert response.agent_instances is None
    assert "agent_instances" not in response.model_dump(mode="json")


def test_a_single_agent_result_row_is_unchanged():
    response = _Response(
        episode_id=EpisodeId(rollout_id="r"), task_id=TaskId(taskset="t", task_id="x"), result={"reward": 0.5}
    )

    assert "agent_instances" not in _episode_record(response.model_dump(mode="json"))


def test_agent_instance_names_in_a_result_are_validated():
    with pytest.raises(ValueError, match="invalid agent instance name"):
        _Response(
            episode_id=EpisodeId(rollout_id="r"),
            task_id=TaskId(taskset="t", task_id="x"),
            result={},
            agent_instances={"Judge": AgentInstanceResult()},
        )


# --- delivery rebuilds every captured agent instance ---


POLICY = ModelRef(name="policy")


def _entry(model_call_id: str, agent_instance: str, prompt: list[int], generation: list[int]) -> TokenEntry:
    identity = CallIdentity(agent_instance=agent_instance, model_ref=POLICY)
    entry = TokenEntry(
        rollout_id="r",
        model_call_id=model_call_id,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": model_call_id}],
        response_id=f"resp_{model_call_id}",
        agent_instance=identity.agent_instance,
        agent_kind=identity.agent_kind,
        model_ref=POLICY,
    )
    stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
    return stamp_continuation(entry, [{"role": "user", "content": model_call_id}])


@pytest.mark.parametrize(
    "judge_terminal, judge_masked, conflict",
    [("resp_j1", False, None), ("resp_minted_by_the_agent", False, None), ("resp_a1", True, "assistant")],
)
async def test_each_agent_instance_gets_its_own_rebuilt_response_and_mask(
    tmp_path, judge_terminal, judge_masked, conflict
):
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("a1", "assistant", [1, 2], [3]))
    store.append(_entry("j1", "judge", [7, 8], [9]))
    result = {
        "_ng_rollout_id": "r",
        "reward": 1.0,
        "response": {"output": []},
        "agent_instances": {
            "assistant": {"terminal_response_id": "resp_a1", "reward": 1.0},
            "judge": {"terminal_response_id": judge_terminal},
        },
    }

    await finalize_rollout_token_capture(result, store)

    assistant, judge = result["agent_instances"]["assistant"], result["agent_instances"]["judge"]
    assert result.get(MASK_SAMPLE_KEY) is not True
    assert [item["generation_token_ids"] for item in result["response"]["output"]] == [[3]]
    assert assistant["reward"] == 1.0 and assistant[MASK_SAMPLE_KEY] is False
    assert [item["generation_token_ids"] for item in assistant["response"]["output"]] == [[3]]
    assert judge[MASK_SAMPLE_KEY] is judge_masked
    assert judge[TOKEN_CAPTURE_KEY].get("terminal_conflict") == conflict
    assert [item["generation_token_ids"] for item in judge["response"]["output"]] == [[9]]
