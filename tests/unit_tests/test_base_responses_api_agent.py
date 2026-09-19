# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import time
from unittest.mock import MagicMock

import pytest

from nemo_gym._checkpoint import AgentBoundaryRecord
from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.rollout_correlation import (
    MODEL_CALL_CAPTURE_OUTCOME_HEADER,
    MODEL_CALL_ID_HEADER,
    ModelCallCaptureOutcome,
)
from nemo_gym.server_utils import ServerClient


class TestBaseResponsesAPIAgent:
    def test_BaseResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")
        BaseResponsesAPIAgent(config=config)

    def test_SimpleResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent.setup_webserver()

    async def test_aggregate_metrics_skip_verification_warns_and_returns_empty_metrics(self) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            skip_verification=True,
        )

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        body = AggregateMetricsRequest(verify_responses=[])

        with pytest.warns(RuntimeWarning, match="skip_verification=True"):
            result = await agent.aggregate_metrics(body)

        assert result.group_level_metrics == []
        assert result.agent_metrics == {}
        assert result.key_metrics == {}

    def _agent(self, global_config: dict, *, token_id_capture: bool = False) -> SimpleResponsesAPIAgent:
        config = BaseResponsesAPIAgentConfig(
            host="", port=0, entrypoint="", name="agent", token_id_capture=token_id_capture
        )

        class _Agent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        client = MagicMock(spec=ServerClient)
        client.global_config_dict = global_config
        return _Agent(config=config, server_client=client)

    def test_eval_capture_prefix_applies_to_every_agent(self) -> None:
        # Evaluation capture correlates every agent.
        # It does not depend on the agent's training-token opt-in.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        assert self._agent({}).rollout_id_from_run(body) is None
        assert self._agent({"observability_enabled": True}).rollout_id_from_run(body) == "0-0"

    def test_token_capture_prefix_is_scoped_to_participating_agents(self) -> None:
        # Training-token capture requires both run-level enablement and agent opt-in.
        # Correlated calls preserve ``/ng-rollout/<id>/training-token-capture``.
        # Native agents carry token ids inline and do not opt in.
        body = {"_ng_task_index": 0, "_ng_rollout_index": 0}
        gc = {"token_id_capture": {"enabled": True}}
        assert self._agent(gc, token_id_capture=False).rollout_id_from_run(body) is None
        assert self._agent(gc, token_id_capture=True).rollout_id_from_run(body) == "0-0"
        # Agent opt-in alone does not enable capture.
        assert self._agent({}, token_id_capture=True).rollout_id_from_run(body) is None

    def test_capture_result_is_gated_by_token_capture_not_checkpoint_participation(self) -> None:
        agent = self._agent(
            {"token_id_capture": {"enabled": True}},
            token_id_capture=True,
        )
        result = agent.model_call_capture_result(
            {
                MODEL_CALL_CAPTURE_OUTCOME_HEADER: "captured",
                MODEL_CALL_ID_HEADER: "call-1",
            }
        )

        assert result is not None
        assert result.outcome == ModelCallCaptureOutcome.CAPTURED
        assert result.model_call_id == "call-1"

    def test_capture_result_is_absent_when_token_capture_and_header_are_absent(self) -> None:
        assert self._agent({}).model_call_capture_result(None) is None

    def test_explicit_capture_outcome_is_honored_when_config_is_unavailable(self) -> None:
        result = self._agent({}).model_call_capture_result({MODEL_CALL_CAPTURE_OUTCOME_HEADER: "no_generation"})

        assert result is not None
        assert result.outcome == ModelCallCaptureOutcome.NO_GENERATION
        assert result.model_call_id is None

    def test_checkpoint_participant_preserves_legacy_model_call_id(self) -> None:
        agent = self._agent({"observability_enabled": True})
        agent._checkpoint_participant = MagicMock()

        result = agent.model_call_capture_result({MODEL_CALL_ID_HEADER: "call-1"})

        assert result is not None
        assert result.outcome == ModelCallCaptureOutcome.CAPTURED
        assert result.model_call_id == "call-1"

    async def test_checkpoint_refusal_after_resume_retries_without_parking(self) -> None:
        agent = self._agent({})
        participant = agent.checkpoint_participant()
        execution = await participant.begin("rollout-a", 0, task=asyncio.current_task())
        await participant.commit_boundary(
            execution,
            AgentBoundaryRecord(
                rollout_id="rollout-a",
                attempt_index=0,
                boundary_index=1,
                output_items=[],
            ),
        )
        response_ready = asyncio.Event()
        return_refusal = asyncio.Event()
        calls = 0

        class _Response:
            def __init__(self, status: int, body: bytes) -> None:
                self.status = status
                self._body = body

            async def read(self) -> bytes:
                return self._body

        async def operation() -> _Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                response_ready.set()
                await return_refusal.wait()
                return _Response(409, b'{"error":{"code":"checkpoint_parked"}}')
            return _Response(200, b"{}")

        token = participant.bind(execution)
        try:
            retried = asyncio.create_task(
                agent.retry_checkpoint_refusal(
                    operation,
                    checkpointable_model_wait=True,
                )
            )
            await response_ready.wait()
            report = await participant.prepare(
                time.time() + 2,
                allow_model_wait_boundary=True,
            )
            assert report["ready_to_commit"] is True

            return_refusal.set()
            await asyncio.sleep(0)
            assert not retried.done()

            await participant.resume()
            response = await asyncio.wait_for(retried, timeout=1)
            assert response.status == 200
            assert calls == 2
        finally:
            participant.unbind(token)
            await participant.finish(execution, outcome="failed")
