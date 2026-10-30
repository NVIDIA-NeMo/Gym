# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import SimpleServer
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.proof_jury_judge.app import (
    VERIFIER_FIXTURE,
    ProofJuryJudgeConfig,
    ProofJuryJudgeServer,
    ProofJuryVerifyRequest,
    parse_score,
    response_text,
)


JUDGES = ["judge_a", "judge_b", "judge_c"]


def make_response(text, reasoning=None, incomplete=None):
    output = []
    if reasoning is not None:
        output.append(
            NeMoGymResponseReasoningItem(
                id="rs", summary=[NeMoGymSummary(text=reasoning, type="summary_text")], type="reasoning"
            )
        )
    if text is not None:
        output.append(
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        )
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        status="incomplete" if incomplete else "completed",
        incomplete_details={"reason": incomplete} if incomplete else None,
    )


def make_server(replies):
    """A server whose judge `name` answers its round-r call with `replies[name][r]` (text or a response)."""
    calls = []

    async def post(server_name, url_path, json):
        prompt = json.input[0].content
        calls.append((server_name, prompt))
        round_idx = sum(name == server_name for name, _ in calls) - 1
        http_response = AsyncMock()
        reply = replies[server_name][round_idx]
        payload = (reply if isinstance(reply, NeMoGymResponse) else make_response(reply)).model_dump(mode="json")
        http_response.read = AsyncMock(return_value=orjson.dumps(payload))
        return http_response

    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(side_effect=post)
    config = ProofJuryJudgeConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judges=[ModelServerRef(type="responses_api_models", name=name) for name in JUDGES],
    )
    return ProofJuryJudgeServer(config=config, server_client=server_client), calls


def verify_request(response):
    return ProofJuryVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        problem="Prove that 1 + 1 = 2.",
        problem_idx="A1",
        tier="EASY",
    )


def scored(*scores):
    return [f"Here is my evaluation.\n\\boxed{{{s}}}" for s in scores]


async def test_verifier_fixture():
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert [(result.kind, result.observed_rewards) for result in results] == [
        ("full_reward", (1.0,)),
        ("zero_reward", (0.0,)),
        ("malformed", ()),
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("milestone \\boxed{2} ... final \\boxed{6}", 6),
        ("\\boxed{\\text{5}}", 5),
        ("\\boxed{9}", 7),
        ("\\boxed{-1}", 0),
        ("the score is 7", None),
        ("\\boxed{seven}", None),
        ("\\boxed{7", None),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


def test_response_text_drops_reasoning():
    assert response_text(make_response("Proof.", reasoning="thinking")) == "Proof."
    assert response_text(make_response("thinking</think>\nProof.")) == "Proof."


class TestVerify:
    async def test_spread_of_two_takes_the_minimum_without_reconciling(self):
        server, calls = make_server(dict(zip(JUDGES, map(lambda s: [s], scored(7, 6, 5)))))
        result = await server.verify(verify_request(make_response("My proof.")))

        assert result.judge_score == 5
        assert result.reward == pytest.approx(5 / 7)
        assert [(j.round, j.score) for j in result.judgments] == [(0, 7), (0, 6), (0, 5)]
        assert len(calls) == 3
        assert all("## Solution\nMy proof." in prompt and "Prove that 1 + 1 = 2." in prompt for _, prompt in calls)

    async def test_wide_spread_reconciles_once_and_takes_the_last_round_minimum(self):
        first, second = scored(7, 7, 2), scored(6, 4, 7)
        server, calls = make_server({name: [first[i], second[i]] for i, name in enumerate(JUDGES)})
        result = await server.verify(verify_request(make_response("My proof.")))

        assert result.judge_score == 4
        assert [(j.round, j.score) for j in result.judgments] == [(0, 7), (0, 7), (0, 2), (1, 6), (1, 4), (1, 7)]
        assert len(calls) == 6
        reconcile_prompt = calls[3][1]
        assert "additionally given the following judgments" in reconcile_prompt
        assert "### Output from judge 3 (judge_c) ###\nHere is my evaluation.\n\\boxed{2}" in reconcile_prompt

    async def test_reply_without_score_is_a_judge_failure(self):
        replies = scored(7, 7, 7)
        replies[1] = "I cannot grade this."
        server, _ = make_server({name: [reply] for name, reply in zip(JUDGES, replies)})

        with pytest.raises(JudgeError, match="judge_b returned no"):
            await server.verify(verify_request(make_response("My proof.")))

    async def test_cut_off_judge_reply_is_a_judge_failure(self):
        replies = {name: [reply] for name, reply in zip(JUDGES, scored(7, 7, 7))}
        replies["judge_c"] = [make_response("Quoting the solution: \\boxed{7}", incomplete="max_output_tokens")]
        server, _ = make_server(replies)

        with pytest.raises(JudgeError, match="judge_c reply was cut off"):
            await server.verify(verify_request(make_response("My proof.")))

    @pytest.mark.parametrize(
        "response, truncated",
        [
            (make_response("Partial proof", incomplete="max_output_tokens"), True),
            (make_response("Partial proof", incomplete="content_filter"), False),
            (make_response(None, reasoning="thinking"), False),
        ],
    )
    async def test_unjudgeable_generation_scores_zero_without_judging(self, response, truncated):
        server, calls = make_server({})
        result = await server.verify(verify_request(response))

        assert (result.judge_score, result.reward, result.truncated) == (0, 0.0, truncated)
        assert calls == []


def test_metrics():
    server, _ = make_server({})

    def rollout(score, tier, truncated=False):
        return {"judge_score": score, "truncated": truncated, "tier": tier, "reward": score / 7}

    imputed = {"reward": 0.0}
    tasks = [
        [rollout(7, "EASY"), rollout(0, "EASY", truncated=True)],
        [imputed, rollout(0, "HARD")],
    ]
    metrics = server.compute_metrics(tasks)

    assert metrics["pass@1[avg-of-2]/score"] == pytest.approx(25.0)
    assert metrics["pass@1[avg-of-2]/full_credit"] == pytest.approx(25.0)
    assert metrics["pass@1[avg-of-2]/truncated"] == pytest.approx(25.0)
    assert metrics["EASY/pass@1[avg-of-2]/score"] == pytest.approx(50.0)
    assert metrics["HARD/pass@1[avg-of-2]/score"] == pytest.approx(0.0)

    key = server.get_key_metrics({**metrics, "mean/output_tokens": 10.0})
    assert key["pass@1[avg-of-2]/score"] == pytest.approx(25.0)
    assert key["mean/output_tokens"] == 10.0


class TestJuryRoundCancellation:
    """A failed panel must not leave its other judge calls running."""

    @staticmethod
    async def _until(condition):
        while not condition():
            await asyncio.sleep(0)

    @staticmethod
    def _server_with_blocked_judges(failing_round):
        started, cancelled = [], []
        server, _ = make_server({})

        async def judge(judge_ref, prompt, round_idx):
            if judge_ref.name == "judge_c" and round_idx == failing_round:
                await TestJuryRoundCancellation._until(lambda: len(started) == 2)
                raise JudgeError("judge_c failed")
            started.append(judge_ref.name)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(judge_ref.name)
                raise

        server._judge = judge
        return server, started, cancelled

    @pytest.mark.parametrize("round_idx", [0, 1])
    async def test_failing_judge_cancels_the_others_before_the_round_raises(self, round_idx):
        server, started, cancelled = self._server_with_blocked_judges(round_idx)

        with pytest.raises(JudgeError, match="judge_c failed"):
            await asyncio.wait_for(server._round("prompt", round_idx), timeout=5)

        assert sorted(started) == sorted(cancelled) == ["judge_a", "judge_b"]

    async def test_cancelling_the_caller_cancels_every_judge(self):
        server, started, cancelled = self._server_with_blocked_judges(failing_round=None)
        round_task = asyncio.create_task(server._round("prompt", 0))
        await asyncio.wait_for(self._until(lambda: len(started) == 3), timeout=5)

        round_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(round_task, timeout=5)

        assert sorted(cancelled) == sorted(JUDGES)


def test_module_exports_app_for_multi_worker_import(monkeypatch):
    app_path = Path(__file__).parents[1] / "app.py"
    worker_app = object()
    calls = []
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda cls: calls.append(cls) or worker_app))

    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: False)
    assert "app" not in runpy.run_path(str(app_path), run_name="proof_jury_judge.library_import")
    assert calls == []

    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    assert runpy.run_path(str(app_path), run_name="proof_jury_judge.worker_import")["app"] is worker_app
