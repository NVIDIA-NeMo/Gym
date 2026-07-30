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
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.agentif import app as agentif_app
from resources_servers.agentif.app import (
    AgentIFResourcesServer,
    AgentIFResourcesServerConfig,
    AgentIFVerifyRequest,
    _coerce_text,
    _format_judge_prompt,
    _normalize_checker_source,
    _response_text,
    _run_code_check,
    _strip_think,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _config() -> AgentIFResourcesServerConfig:
    return AgentIFResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentif",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


def _server() -> AgentIFResourcesServer:
    return AgentIFResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


def _request(constraints: List[Dict[str, Any]], text: str = "") -> AgentIFVerifyRequest:
    return AgentIFVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        verifier_metadata={"query_id": 1, "turn_id": 0, "constraints": constraints},
    )


def _constraint(evaluation: List[Dict[str, Any]], dimension: str = "unconditional", ctype: Any = "formatting"):
    return {"dimension": dimension, "type": ctype, "evaluation": evaluation}


def _verify(server: AgentIFResourcesServer, body: AgentIFVerifyRequest):
    return asyncio.run(server.verify(body))


# ── _normalize_checker_source ─────────────────────────────────────────────


def test_normalize_checker_source_adds_def() -> None:
    src = "check_following(response):\n    return True"
    out = _normalize_checker_source(src)
    assert out.startswith("def check_following(")


def test_normalize_checker_source_noop_when_def_present() -> None:
    src = "def check_following(response):\n    return True"
    assert _normalize_checker_source(src) == src


def test_normalize_checker_source_empty_input() -> None:
    assert _normalize_checker_source("") == ""
    assert _normalize_checker_source(None) is None  # type: ignore[arg-type]


# ── _format_judge_prompt ──────────────────────────────────────────────────


def test_format_judge_prompt_appends_placeholder() -> None:
    out = _format_judge_prompt("Grade this.", "MY ANSWER")
    assert "Here is model response: MY ANSWER" in out


def test_format_judge_prompt_replaces_existing() -> None:
    out = _format_judge_prompt("Grade {response} now", "X")
    assert out == "Grade X now"


# ── _run_code_check ───────────────────────────────────────────────────────


def test_run_code_check_returns_true() -> None:
    src = "def check_following(response):\n    return 'YES' in response"
    result, err = _run_code_check(src, "YES sir")
    assert result is True
    assert err is None


def test_run_code_check_returns_none_on_error() -> None:
    src = "def check_following(response):\n    raise ValueError('boom')"
    result, err = _run_code_check(src, "x")
    assert result is None
    assert err is not None and "ValueError" in err


def test_run_code_check_empty_response() -> None:
    result, err = _run_code_check("def check_following(r): return True", "")
    assert result is None
    assert err == "Empty response"


def test_run_code_check_no_check_following() -> None:
    result, err = _run_code_check("x = 1", "text")
    assert result is None
    assert err == "check_following not defined"


# ── verify: llm constraints ───────────────────────────────────────────────


def test_verify_llm_yes_reward_1() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value="YES")
    body = _request([_constraint([{"type": "llm", "exec": "Judge {response}"}])], text="hello")
    resp = _verify(server, body)
    assert resp.reward == approx(1.0)
    assert resp.n_true == 1 and resp.isr_pass == 1 and resp.isr_counted == 1


def test_verify_llm_no_reward_0() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value="NO")
    body = _request([_constraint([{"type": "llm", "exec": "Judge {response}"}])], text="hello")
    resp = _verify(server, body)
    assert resp.reward == approx(0.0)
    assert resp.n_false == 1 and resp.isr_pass == 0 and resp.isr_counted == 1


def test_verify_llm_judge_failure_null() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value=None)
    body = _request([_constraint([{"type": "llm", "exec": "Judge {response}"}])], text="hi")
    resp = _verify(server, body)
    assert resp.n_null == 1 and resp.isr_counted == 0 and resp.reward == approx(0.0)


# ── verify: code constraints ──────────────────────────────────────────────


def test_verify_code_pass() -> None:
    server = _server()
    src = "def check_following(response):\n    return len(response) > 2"
    body = _request([_constraint([{"type": "code", "exec": src}])], text="hello")
    resp = _verify(server, body)
    assert resp.n_true == 1 and resp.reward == approx(1.0)


def test_verify_code_error_null() -> None:
    server = _server()
    src = "def check_following(response):\n    raise RuntimeError('x')"
    body = _request([_constraint([{"type": "code", "exec": src}])], text="hello")
    resp = _verify(server, body)
    assert resp.n_null == 1 and resp.isr_counted == 0


# ── verify: conditional check ─────────────────────────────────────────────


def test_verify_conditional_check_yes_continues() -> None:
    server = _server()
    src = "def check_following(response):\n    return True"
    server._call_judge = AsyncMock(return_value="YES it applies")
    body = _request(
        [_constraint([{"type": "llm_conditional_check", "exec": "cond"}, {"type": "code", "exec": src}])],
        text="hello",
    )
    resp = _verify(server, body)
    assert resp.n_true == 1 and resp.reward == approx(1.0)


def test_verify_conditional_check_not_yes_null() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value="NO does not apply")
    body = _request(
        [_constraint([{"type": "llm_conditional_check", "exec": "cond"}, {"type": "code", "exec": "x"}])],
        text="hello",
    )
    resp = _verify(server, body)
    assert resp.n_null == 1 and resp.isr_counted == 0


# ── verify: misc ──────────────────────────────────────────────────────────


def test_verify_think_stripped() -> None:
    server = _server()
    captured = {}

    async def fake_judge(prompt: str):
        captured["prompt"] = prompt
        return "YES"

    server._call_judge = fake_judge
    body = _request(
        [_constraint([{"type": "llm", "exec": "Grade {response}"}])],
        text="<think>secret reasoning</think>final answer",
    )
    resp = _verify(server, body)
    assert "secret reasoning" not in captured["prompt"]
    assert "final answer" in captured["prompt"]
    assert resp.n_true == 1


def test_verify_multi_constraint_partial() -> None:
    server = _server()
    server._call_judge = AsyncMock(side_effect=["YES", "NO"])
    body = _request(
        [
            _constraint([{"type": "llm", "exec": "a {response}"}]),
            _constraint([{"type": "llm", "exec": "b {response}"}], dimension="conditional", ctype="semantic"),
        ],
        text="x",
    )
    resp = _verify(server, body)
    assert resp.n_true == 1 and resp.n_false == 1
    # Reward is ISR: one failed constraint means the whole row fails (0.0).
    assert resp.reward == approx(0.0)
    assert resp.isr_pass == 0 and resp.isr_counted == 1


def test_verify_all_null_isr_not_counted() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value=None)
    body = _request(
        [
            _constraint([{"type": "llm", "exec": "a {response}"}]),
            _constraint([{"type": "llm", "exec": "b {response}"}]),
        ],
        text="x",
    )
    resp = _verify(server, body)
    assert resp.n_null == 2 and resp.isr_counted == 0 and resp.reward == approx(0.0)


def test_verify_llm_then_code_pipeline() -> None:
    server = _server()
    # llm rewrites the working text; code then inspects the rewritten text.
    server._call_judge = AsyncMock(return_value="TRANSFORMED")
    src = "def check_following(response):\n    return response == 'TRANSFORMED'"
    body = _request(
        [_constraint([{"type": "llm", "exec": "rewrite {response}"}, {"type": "code", "exec": src}])],
        text="orig",
    )
    resp = _verify(server, body)
    assert resp.n_true == 1 and resp.reward == approx(1.0)


# ── compute_metrics ───────────────────────────────────────────────────────


def _row(**kw: Any) -> Dict[str, Any]:
    base = {
        "reward": 0.0,
        "n_true": 0,
        "n_false": 0,
        "n_null": 0,
        "isr_pass": 0,
        "isr_counted": 0,
        "by_dimension": {},
        "by_type": {},
    }
    base.update(kw)
    return base


def test_compute_metrics_csr_isr() -> None:
    server = _server()
    rows = [
        _row(reward=1.0, n_true=2, n_false=0, isr_pass=1, isr_counted=1),
        _row(reward=0.5, n_true=1, n_false=1, isr_pass=0, isr_counted=1),
    ]
    metrics = server.compute_metrics([rows])
    assert metrics["csr"] == approx(3 / 4)
    assert metrics["isr"] == approx(1 / 2)
    assert metrics["mean_reward"] == approx(0.75)
    assert metrics["count"] == 2


def test_compute_metrics_by_dimension() -> None:
    server = _server()
    rows = [
        _row(
            n_true=1,
            n_false=1,
            by_dimension={"unconditional": {"n_true": 1, "n_false": 0}, "conditional": {"n_true": 0, "n_false": 1}},
            by_type={"formatting": {"n_true": 1, "n_false": 0}, "resource": {"n_true": 0, "n_false": 1}},
        ),
    ]
    metrics = server.compute_metrics([rows])
    assert metrics["by_dimension/vanilla/accuracy"] == approx(1.0)
    assert metrics["by_dimension/condition/accuracy"] == approx(0.0)
    assert metrics["by_type/formatting/accuracy"] == approx(1.0)
    assert metrics["by_type/tool/accuracy"] == approx(0.0)


def test_compute_metrics_empty() -> None:
    server = _server()
    assert server.compute_metrics([]) == {}
    assert server.compute_metrics([[]]) == {}


def test_compute_metrics_all_null() -> None:
    server = _server()
    rows = [_row(n_null=3, isr_counted=0)]
    metrics = server.compute_metrics([rows])
    assert "csr" not in metrics
    assert "isr" not in metrics
    assert metrics["n_null_total"] == 3
    assert server.get_key_metrics(metrics) == {"mean_reward": approx(0.0)}


# ── text helpers ──────────────────────────────────────────────────────────


def test_strip_think_removes_block() -> None:
    assert _strip_think("<think>x</think>answer") == "answer"


def test_strip_think_split_fallback() -> None:
    # A dangling closing tag (no opening) still strips via the split fallback.
    assert _strip_think("reasoning</think>final") == "final"


def test_strip_think_empty() -> None:
    assert _strip_think("") == ""
    assert _strip_think(None) == ""  # type: ignore[arg-type]


def test_coerce_text_variants() -> None:
    assert _coerce_text("plain") == "plain"
    assert _coerce_text(["a", {"text": "b"}, 3]) == "ab"
    assert _coerce_text(None) == ""
    assert _coerce_text(42) == "42"


def test_response_text_none() -> None:
    assert _response_text(None) == ""


def test_response_text_output_text_fastpath() -> None:
    resp = _make_response("direct")
    assert _response_text(resp) == "direct"


# ── _run_code_check import path ───────────────────────────────────────────


def test_run_code_check_with_import() -> None:
    src = "import re\ndef check_following(response):\n    return bool(re.match('h', response))"
    result, err = _run_code_check(src, "hello")
    assert result is True and err is None


# ── verify: non yes/no string → null ──────────────────────────────────────


def test_verify_string_not_yes_no_scores_null() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value="maybe")
    body = _request([_constraint([{"type": "llm", "exec": "Judge {response}"}])], text="hi")
    resp = _verify(server, body)
    assert resp.n_null == 1 and resp.isr_counted == 0


# ── config: nullcontext when concurrency disabled ─────────────────────────


def test_server_nullcontext_when_concurrency_none() -> None:
    cfg = AgentIFResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentif",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_endpoint_max_concurrency=None,
    )
    server = AgentIFResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
    src = "def check_following(response):\n    return True"
    body = _request([_constraint([{"type": "code", "exec": src}])], text="hi")
    resp = _verify(server, body)
    assert resp.n_true == 1


# ── _call_judge: real path (server_client mocked) ─────────────────────────


def test_call_judge_success(monkeypatch) -> None:
    server = _server()
    server.server_client.post = AsyncMock(return_value=MagicMock())
    judge_payload = _make_response("YES").model_dump()

    async def fake_get_response_json(_resp):
        return judge_payload

    monkeypatch.setattr(agentif_app, "get_response_json", fake_get_response_json)
    out = asyncio.run(server._call_judge("grade this"))
    assert out == "YES"


def test_call_judge_failure_returns_none() -> None:
    server = _server()
    server.server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
    out = asyncio.run(server._call_judge("grade this"))
    assert out is None


def test_setup_webserver() -> None:
    server = _server()
    assert server.setup_webserver() is not None


# ── verify: list-valued constraint type breakdown ─────────────────────────


def test_verify_list_type_bumps_both_buckets() -> None:
    server = _server()
    src = "def check_following(response):\n    return True"
    body = _request(
        [_constraint([{"type": "code", "exec": src}], dimension="conditional", ctype=["formatting", "semantic"])],
        text="hello",
    )
    resp = _verify(server, body)
    assert resp.by_type["formatting"] == {"n_true": 1, "n_false": 0}
    assert resp.by_type["semantic"] == {"n_true": 1, "n_false": 0}
    assert resp.by_dimension["conditional"] == {"n_true": 1, "n_false": 0}


# ── verify: empty model response → reward 0.0, no raise ───────────────────


def test_verify_empty_response_reward_zero() -> None:
    server = _server()
    server._call_judge = AsyncMock(return_value="YES")
    src = "def check_following(response):\n    return True"
    body = _request(
        [
            _constraint([{"type": "llm", "exec": "Judge {response}"}]),
            _constraint([{"type": "code", "exec": src}]),
        ],
        text="",
    )
    resp = _verify(server, body)
    # Empty response → code check yields "Empty response" error → None; llm with
    # empty student text still calls the judge (upstream parity). Whatever the
    # judge returns, the call must not raise and reward stays a valid float.
    assert isinstance(resp.reward, float)
    assert 0.0 <= resp.reward <= 1.0


# ── verify: code failure then llm → terminal None (upstream parity) ───────


def test_verify_code_fail_then_llm_scores_null() -> None:
    server = _server()
    # If the judge were (wrongly) called on the failed step, it would return YES
    # and the constraint would score True. Parity requires None instead.
    judge = AsyncMock(return_value="YES")
    server._call_judge = judge
    bad_code = "def check_following(response):\n    raise ValueError('x')"
    body = _request(
        [_constraint([{"type": "code", "exec": bad_code}, {"type": "llm", "exec": "Judge {response}"}])],
        text="hello",
    )
    resp = _verify(server, body)
    assert resp.n_null == 1 and resp.isr_counted == 0
    judge.assert_not_called()
