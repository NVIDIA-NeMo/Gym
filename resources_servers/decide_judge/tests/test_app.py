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
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.decide_judge.app import (
    VERIFIER_FIXTURE,
    ChecklistItem,
    DecideJudgeResourcesServer,
    DecideJudgeResourcesServerConfig,
    DecideJudgeVerifyRequest,
    score_checklist,
)
from resources_servers.decide_judge.decide_client import (
    DecideQuestion,
    NemotronDecideClient,
    NemotronDecideEndpointConfig,
    NemotronDecideError,
    clip,
    render_items,
    user_and_system_text,
)


def _response(output: list[dict]) -> dict:
    return {
        "id": "resp",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _message(text: str) -> dict:
    return {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class FakeEndpoint:
    """Records payloads and answers with fixed yes-probabilities, in either API shape."""

    def __init__(self, api: str, p_yes: dict[str, float], replies: list | None = None):
        self.api = api
        self.p_yes = p_yes
        self.replies = list(replies or [])
        self.payloads: list[dict] = []

    async def __call__(self, payload: dict):
        self.payloads.append(payload)
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        if self.api == "systemone":
            answers = {}
            for name, q in payload["questions"].items():
                if q["type"] == "noul":
                    answers[name] = {"type": "noul", "noul": self.p_yes[name]}
                else:
                    probs = self.p_yes[name]
                    top = max(probs, key=probs.get)
                    answers[name] = {"type": "choice", "probabilities": probs, "choice": top, "confidence": probs[top]}
            return 200, {"answers": answers, "usage": {"input_tokens": 42}, "metrics": {"timing": {"total": 0.01}}}
        answers = []
        for q in payload["questions"]:
            if q["type"] == "predicate":
                answers.append({"type": "predicate", "name": q["name"], "probability": self.p_yes[q["name"]]})
            else:
                probs = self.p_yes[q["name"]]
                answers.append(
                    {
                        "type": "choice",
                        "name": q["name"],
                        "choice": max(probs, key=probs.get),
                        "probabilities": [{"value": v, "probability": p} for v, p in probs.items()],
                        "confidence": max(probs.values()),
                    }
                )
        return 200, {"object": "decision", "answers": answers, "usage": {"input_tokens": 42}}


def _server(fake: FakeEndpoint, **overrides) -> DecideJudgeResourcesServer:
    config = DecideJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="decide_judge",
        decide=NemotronDecideEndpointConfig(api=fake.api, max_retries=2, retry_backoff_s=0),
        checklist=[
            ChecklistItem(name="correct", question="Is it correct?", weight=2.0),
            ChecklistItem(name="complete", question="Is it complete?", weight=1.0),
            ChecklistItem(name="boxed", question="Is it boxed?", weight=0.0),
        ],
        **overrides,
    )
    server = DecideJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server._client._post = fake
    return server


def _verify_request(**extra) -> DecideJudgeVerifyRequest:
    return DecideJudgeVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Who proposed natural selection?"}]},
            "response": _response([_message("\\boxed{Darwin}")]),
            "expected_answer": "darwin",
            **extra,
        }
    )


class TestClient:
    def test_systemone_payload_and_parse(self) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig(api="systemone"))
        questions = [
            DecideQuestion(name="ok", kind="yes_no", instructions="ok?"),
            DecideQuestion(name="gate", kind="choice", instructions="what?", options={"allow": "a", "block": "b"}),
        ]
        payload = client.build_payload({"x": 1}, questions)
        assert payload == {
            "model": "nemotron-decide",
            "state": {"x": 1},
            "questions": {
                "ok": {"type": "noul", "instructions": "ok?"},
                "gate": {"type": "choice", "instructions": "what?", "criteria": {"allow": "a", "block": "b"}},
            },
        }
        answers = client.parse_answers(
            {
                "answers": {
                    "ok": {"noul": 0.8},
                    "gate": {"probabilities": {"allow": 0.3, "block": 0.7}},
                }
            },
            questions,
        )
        assert answers["ok"].p_yes == pytest.approx(0.8)
        assert answers["ok"].choice == "yes"
        assert answers["gate"].choice == "block"
        assert answers["gate"].confidence == pytest.approx(0.7)
        assert client.url == "http://localhost:8770/v1/systemone"

    def test_decisions_payload_and_parse(self) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig(api="decisions", base_url="http://localhost:8770/"))
        questions = [
            DecideQuestion(name="ok", kind="yes_no", instructions="ok?"),
            DecideQuestion(name="gate", kind="choice", instructions="what?", options={"allow": "a", "block": "b"}),
        ]
        payload = client.build_payload({"x": 1}, questions)
        assert payload["input"] == [{"role": "user", "content": [{"type": "input_json", "data": {"x": 1}}]}]
        assert payload["questions"][0] == {"type": "predicate", "name": "ok", "instructions": "ok?"}
        assert payload["questions"][1]["choices"] == [
            {"value": "allow", "description": "a"},
            {"value": "block", "description": "b"},
        ]
        assert client.url == "http://localhost:8770/v1/decisions"
        answers = client.parse_answers(
            {
                "answers": [
                    {"name": "ok", "probability": 0.2},
                    {
                        "name": "gate",
                        "probabilities": [
                            {"value": "allow", "probability": 0.9},
                            {"value": "block", "probability": 0.1},
                        ],
                    },
                ]
            },
            questions,
        )
        assert answers["ok"].choice == "no"
        assert answers["ok"].p_yes == pytest.approx(0.2)
        assert answers["gate"].choice == "allow"

    @pytest.mark.parametrize(
        "body, match",
        [
            ({}, "no answers"),
            ({"answers": {"other": {"noul": 0.5}}}, "missing answers"),
            ({"answers": {"ok": {}}}, "no probability"),
        ],
    )
    def test_parse_errors(self, body: dict, match: str) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig())
        with pytest.raises(NemotronDecideError, match=match):
            client.parse_answers(body, [DecideQuestion(name="ok", kind="yes_no", instructions="ok?")])

    def test_choice_option_mismatch(self) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig())
        q = DecideQuestion(name="g", kind="choice", instructions="?", options={"a": "a", "b": "b"})
        with pytest.raises(NemotronDecideError, match="options"):
            client.parse_answers({"answers": {"g": {"probabilities": {"a": 1.0}}}}, [q])

    def test_build_payload_validation(self) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig())
        with pytest.raises(ValueError, match="at least one"):
            client.build_payload({}, [])
        q = DecideQuestion(name="a", kind="yes_no", instructions="?")
        with pytest.raises(ValueError, match="unique"):
            client.build_payload({}, [q, q])
        with pytest.raises(ValueError, match="two options"):
            client.build_payload({}, [DecideQuestion(name="c", kind="choice", instructions="?", options={"x": "x"})])

    def test_api_key_comes_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = NemotronDecideClient(NemotronDecideEndpointConfig(api_key_env="DECIDE_TEST_KEY"))
        monkeypatch.delenv("DECIDE_TEST_KEY", raising=False)
        assert "Authorization" not in client._headers()
        monkeypatch.setenv("DECIDE_TEST_KEY", "secret-value")
        assert client._headers()["Authorization"] == "Bearer secret-value"
        assert "secret-value" not in client.config.model_dump_json()
        no_auth = NemotronDecideClient(NemotronDecideEndpointConfig(api_key_env=None))
        assert "Authorization" not in no_auth._headers()

    def test_retries_on_retryable_status_then_succeeds(self) -> None:
        fake = FakeEndpoint("systemone", {"ok": 0.9}, replies=[(429, "slow down"), (503, {"error": "busy"})])
        client = NemotronDecideClient(NemotronDecideEndpointConfig(max_retries=2, retry_backoff_s=0))
        client._post = fake
        result = asyncio.run(client.decide({}, [DecideQuestion(name="ok", kind="yes_no", instructions="?")]))
        assert result.answers["ok"].p_yes == pytest.approx(0.9)
        assert len(fake.payloads) == 3
        assert result.usage == {"input_tokens": 42}
        assert result.server_metrics == {"timing": {"total": 0.01}}

    def test_no_retry_on_client_error(self) -> None:
        fake = FakeEndpoint("systemone", {}, replies=[(401, {"error": "bad key"})])
        client = NemotronDecideClient(NemotronDecideEndpointConfig(max_retries=3, retry_backoff_s=0))
        client._post = fake
        with pytest.raises(NemotronDecideError, match="HTTP 401") as error:
            asyncio.run(client.decide({}, [DecideQuestion(name="ok", kind="yes_no", instructions="?")]))
        assert error.value.status == 401
        assert len(fake.payloads) == 1

    def test_connection_errors_exhaust_retries(self) -> None:
        fake = FakeEndpoint("systemone", {}, replies=[TimeoutError(), OSError("refused")])
        client = NemotronDecideClient(NemotronDecideEndpointConfig(max_retries=1, retry_backoff_s=0))
        client._post = fake
        with pytest.raises(NemotronDecideError, match="refused"):
            asyncio.run(client.decide({}, [DecideQuestion(name="ok", kind="yes_no", instructions="?")]))
        assert len(fake.payloads) == 2

    def test_render_items_and_trust_split(self) -> None:
        items = [
            {"role": "system", "content": "be careful"},
            {"role": "user", "content": [{"type": "input_text", "text": "read P1"}]},
            {"type": "reasoning", "id": "r", "summary": [{"type": "summary_text", "text": "secret thoughts"}]},
            {"type": "function_call", "call_id": "c1", "name": "get", "arguments": '{"id": "P1"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "x" * 50},
            {"type": "function_call", "call_id": "c2", "name": "raw", "arguments": "not json"},
            _message("done"),
        ]
        steps = render_items(items, max_item_chars=20)
        assert steps[0] == {"role": "system", "text": "be careful"}
        assert steps[1] == {"role": "user", "text": "read P1"}
        assert steps[2] == {"role": "assistant", "tool_call": {"name": "get", "arguments": {"id": "P1"}}}
        assert steps[3]["tool"] == "get" and "chars omitted" in steps[3]["output"]
        assert steps[4]["tool_call"]["arguments"] == "not json"
        assert steps[5] == {"role": "assistant", "text": "done"}
        assert all("secret" not in json.dumps(s) for s in steps)
        assert user_and_system_text(items) == ("read P1", "be careful")
        assert clip("abc", 0) == "abc"


class TestJudge:
    def test_sanity_config_validation(self) -> None:
        with pytest.raises(ValidationError, match="at least one item"):
            DecideJudgeResourcesServerConfig(host="h", port=1, entrypoint="", name="n", checklist=[])
        with pytest.raises(ValidationError, match="unique"):
            DecideJudgeResourcesServerConfig(
                host="h",
                port=1,
                entrypoint="",
                name="n",
                checklist=[ChecklistItem(name="a", question="?"), ChecklistItem(name="a", question="?")],
            )
        with pytest.raises(ValidationError, match="weight > 0"):
            DecideJudgeResourcesServerConfig(
                host="h", port=1, entrypoint="", name="n", checklist=[ChecklistItem(name="a", question="?", weight=0)]
            )

    @pytest.mark.parametrize("api", ["systemone", "decisions"])
    def test_probability_reward_and_logging(self, api: str) -> None:
        fake = FakeEndpoint(api, {"correct": 0.9, "complete": 0.6, "boxed": 0.1})
        server = _server(fake)
        result = asyncio.run(server.verify(_verify_request()))
        # (2*0.9 + 1*0.6) / 3; the weight-0 item is logged only
        assert result.reward == pytest.approx(0.8)
        assert result.decide_scores == pytest.approx({"correct": 0.9, "complete": 0.6, "boxed": 0.1})
        assert [r.passed for r in result.decide_checklist_results] == [True, True, False]
        assert result.decide_latency_ms is not None
        assert not result.mask_sample
        payload = fake.payloads[0]
        state = payload["state"] if api == "systemone" else payload["input"][0]["content"][0]["data"]
        assert state["reference"] == {"expected_answer": "darwin"}
        assert state["task"] == [{"role": "user", "text": "Who proposed natural selection?"}]
        assert state["response"] == [{"role": "assistant", "text": "\\boxed{Darwin}"}]

    def test_pass_mode_and_required_gate(self) -> None:
        fake = FakeEndpoint("systemone", {"correct": 0.9, "complete": 0.4, "boxed": 0.9})
        server = _server(fake, reward_mode="pass")
        assert asyncio.run(server.verify(_verify_request())).reward == pytest.approx(2 / 3)

        items = [
            ChecklistItem(name="correct", question="?", required=True, weight=1),
            ChecklistItem(name="complete", question="?"),
        ]
        reward, _ = score_checklist(
            items,
            {"correct": 0.2, "complete": 1.0},
            reward_mode="probability",
            pass_threshold=0.5,
            required_fail_reward=-1,
        )
        assert reward == -1
        only_required = [ChecklistItem(name="correct", question="?", required=True, weight=0)]
        reward, _ = score_checklist(
            only_required, {"correct": 0.7}, reward_mode="probability", pass_threshold=0.5, required_fail_reward=0
        )
        assert reward == 1.0

    def test_row_checklist_overrides_and_reference_from_metadata(self) -> None:
        fake = FakeEndpoint("systemone", {"apology": 1.0, "refund": 0.5})
        server = _server(fake, grading_notes="be strict")
        body = _verify_request(
            expected_answer=None,
            metadata={"expected_answer": "from-metadata"},
            decide_checklist=[
                {"name": "apology", "question": "Apologizes?"},
                {"name": "refund", "question": "Mentions refund?", "weight": 3},
            ],
        )
        result = asyncio.run(server.verify(body))
        assert result.reward == pytest.approx((1.0 + 3 * 0.5) / 4)
        assert set(fake.payloads[0]["questions"]) == {"apology", "refund"}
        assert fake.payloads[0]["state"]["reference"] == {"expected_answer": "from-metadata"}
        assert fake.payloads[0]["state"]["notes"] == "be strict"

        server_no_row = _server(
            FakeEndpoint("systemone", {"correct": 1, "complete": 1, "boxed": 1}), allow_row_checklist=False
        )
        result = asyncio.run(server_no_row.verify(body))
        assert {r.name for r in result.decide_checklist_results} == {"correct", "complete", "boxed"}

    def test_tool_calls_can_be_hidden_and_string_input(self) -> None:
        fake = FakeEndpoint("systemone", {"correct": 1, "complete": 1, "boxed": 1})
        server = _server(fake, include_tool_calls=False)
        body = DecideJudgeVerifyRequest.model_validate(
            {
                "responses_create_params": {"input": "plain string task"},
                "response": _response(
                    [
                        {
                            "type": "function_call",
                            "call_id": "c",
                            "name": "t",
                            "arguments": "{}",
                            "id": "fc",
                            "status": "completed",
                        },
                        _message("final"),
                    ]
                ),
            }
        )
        asyncio.run(server.verify(body))
        state = fake.payloads[0]["state"]
        assert state["task"] == [{"role": "user", "text": "plain string task"}]
        assert state["response"] == [{"role": "assistant", "text": "final"}]
        assert "reference" not in state

    def test_endpoint_failure_masks_sample(self) -> None:
        fake = FakeEndpoint("systemone", {}, replies=[(500, "boom")] * 3)
        server = _server(fake)
        result = asyncio.run(server.verify(_verify_request()))
        assert result.mask_sample is True
        assert result.reward == 0.0
        assert "HTTP 500" in result.decide_error

    def test_compute_metrics(self) -> None:
        server = _server(FakeEndpoint("systemone", {}))
        rollouts = [
            {
                "reward": 1,
                "decide_latency_ms": 10.0,
                "decide_checklist_results": [{"name": "correct", "p_yes": 0.8, "passed": True}],
            },
            {
                "reward": 0,
                "decide_latency_ms": 30.0,
                "decide_checklist_results": [{"name": "correct", "p_yes": 0.2, "passed": False}],
            },
            {"reward": 0, "mask_sample": True},
        ]
        metrics = server.compute_metrics([rollouts])
        assert metrics["decide/masked_rate"] == pytest.approx(1 / 3)
        assert metrics["decide/correct/mean_p_yes"] == pytest.approx(0.5)
        assert metrics["decide/correct/pass_rate"] == pytest.approx(0.5)
        assert metrics["decide/latency_ms_mean"] == pytest.approx(20.0)
        assert server.compute_metrics([]) == {}
        key = server.get_key_metrics({"mean/reward": 0.5, **metrics})
        assert key["decide/correct/pass_rate"] == pytest.approx(0.5)
        assert "decide/correct/mean_p_yes" not in key

    def test_verifier_fixture(self) -> None:
        asyncio.run(
            exercise_verifier_fixture(
                VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
            )
        )

    def test_configs_parse(self) -> None:
        import yaml

        config_dir = Path(__file__).parents[1] / "configs"
        for path in config_dir.glob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            for instance in data.values():
                server = (instance.get("resources_servers") or {}).get("decide_judge")
                if server is None:
                    continue
                fields = {
                    k: v
                    for k, v in server.items()
                    if k not in ("entrypoint", "domain", "verified", "description", "value")
                }
                fields["decide"] = {k: v for k, v in fields["decide"].items() if "${" not in str(v)}
                fields = {k: v for k, v in fields.items() if "${" not in str(v)}
                DecideJudgeResourcesServerConfig(host="h", port=1, entrypoint="app.py", name="n", **fields)
