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
"""Tests for the stem_judge server: length guards, the boxed/free answer contract,
verdict and rubric parsing, and the shipped config."""

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml
from pydantic import ValidationError
from pytest import approx, fixture, mark, raises

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.data import load_task_data_schema, validate_jsonl_rows
from nemo_gym.server_runtime import ReverifyMode, verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding
from nemo_gym.verification import judge_failsafe
from resources_servers.stem_judge.app import (
    MathVerifyConfig,
    RubricCriterion,
    StemJudgeConfig,
    StemJudgeResourcesServer,
    StemJudgeVerifyRequest,
    final_answer,
    last_boxed,
    parse_criteria,
    parse_verdict,
    render_reference,
    render_rubric,
    rubric_fraction,
)
from resources_servers.stem_judge.task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
RUBRIC = [
    {"id": "r1", "points": 3, "criterion": "Final period T = 2π√(L/g)(1 + θ0²/16)"},
    {"id": "r2", "points": 2, "criterion": "Expands the period integral to second order"},
    {"id": "r3", "points": -2, "criterion": "Gives the small-angle period as the final answer"},
]


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _reasoning(text: str = "scratch work") -> NeMoGymResponseReasoningItem:
    return NeMoGymResponseReasoningItem(id="rs", summary=[NeMoGymSummary(text=text, type="summary_text")])


def _response(*output, truncated: bool = False) -> NeMoGymResponse:
    extra = {"incomplete_details": {"reason": "max_output_tokens"}} if truncated else {}
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **extra,
    )


def _failed_response(*output) -> NeMoGymResponse:
    """A judge reply whose operation failed: status "failed" with an error, as a provider adapter passes it on."""
    error = {"code": "server_error", "message": "judge failed"}
    return NeMoGymResponse.model_validate({**_response(*output).model_dump(), "status": "failed", "error": error})


def _split(answer: str, **kwargs) -> NeMoGymResponse:
    """A response as our model server returns it: reasoning item, then the answer."""
    return _response(_reasoning(), _msg(answer), **kwargs)


@fixture
def config() -> StemJudgeConfig:
    """The shipped config, so a change to its defaults shows up here."""
    with open(SERVER_DIR / "configs" / "stem_judge.yaml") as f:
        shipped = yaml.safe_load(f)["stem_judge"]["resources_servers"]["stem_judge"]
    for key in ("answer_prompt_template_fpath", "rubric_prompt_template_fpath"):
        shipped[key] = str(SERVER_DIR / shipped[key])
    return StemJudgeConfig(host="0.0.0.0", port=8080, **shipped)


def _server(config: StemJudgeConfig, *judge_replies: str | NeMoGymResponse) -> StemJudgeResourcesServer:
    """A server whose judge returns these replies in order; a str is a complete one-message reply."""
    client = MagicMock(spec=ServerClient)
    post = MagicMock()
    replies = [_response(_msg(r)) if isinstance(r, str) else r for r in judge_replies]
    post.read = AsyncMock(side_effect=[r.model_dump_json() for r in replies])
    client.post = AsyncMock(return_value=post)
    return StemJudgeResourcesServer(config=config, server_client=client)


def _request(response: NeMoGymResponse, expected: str = "2", **row) -> StemJudgeVerifyRequest:
    return StemJudgeVerifyRequest(
        responses_create_params=deepcopy(
            NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        ),
        response=response,
        expected_answer=expected,
        **row,
    )


def _judge_prompt(server: StemJudgeResourcesServer) -> str:
    return server.server_client.post.await_args.kwargs["json"].input[-1].content


class TestAnswerOptions:
    @mark.parametrize("preset", [False, True], ids=["v11", "cross-domain-json"])
    async def test_extra_rules_only_extend_the_rendered_guidelines(self, config, preset):
        if preset:
            config.guideline_scope = "cross_domain"
            config.verdict_mode = "yes_no_unresolved"
            config.verdict_format = "json"
        reply = '{"verdict":"correct","reason_code":"equal","evidence":"2 equals 2"}' if preset else "Judgement: yes"
        req = _request(_split(r"\boxed{2}"))
        base = _server(config, reply)
        await base.verify(req)
        original = _judge_prompt(base)
        if not preset:
            template = (SERVER_DIR / "prompt_templates" / "stem_v11.txt").read_text().strip()
            assert original == template.format(question="Q: 1+1?", expected_answer="2", generated_answer=r"\boxed{2}")

        rules = [r"Compare \frac{a}{b} exactly.", 'Treat {"id": "{question}"} as literal text.']
        server = _server(config.model_copy(update={"extra_answer_rules": rules}), reply)
        result = await server.verify(req)
        end = original.rindex("</rule>") + len("</rule>")
        assert (
            _judge_prompt(server)
            == original[:end] + "".join(f"\n\n<rule>\n{r}\n</rule>" for r in rules) + original[end:]
        )
        assert result.reward == 1 and result.judge_evaluations[0].verdict_label == "Judgement: yes"

    def test_extra_rules_require_nonblank_text_and_an_insertion_point(self, config, tmp_path):
        with raises(ValidationError, match="extra_answer_rules"):
            StemJudgeConfig.model_validate(config.model_dump() | {"extra_answer_rules": ["  "]})
        template = tmp_path / "custom.txt"
        template.write_text("Question: {question}")
        config.answer_prompt_template_fpath = str(template)
        _server(config)
        config.extra_answer_rules = ["Check the unit."]
        with raises(ValueError, match="</rule>"):
            _server(config)

    async def test_rule_file_renders_the_same_prompt_as_inline_rules(self, config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "rules.yaml"
        path.write_text(
            "- 'Compare {α} literally.'\n- |-\n  Check the unit.\n  Preserve its sign.\n", encoding="utf-8"
        )
        inline = ["Compare {α} literally.", "Check the unit.\nPreserve its sign."]
        from_file = _server(
            StemJudgeConfig.model_validate(config.model_dump() | {"extra_answer_rules": path.name}), "Judgement: yes"
        )
        path.unlink()  # A running worker has already loaded its rules.
        from_list = _server(
            StemJudgeConfig.model_validate(config.model_dump() | {"extra_answer_rules": inline}), "Judgement: yes"
        )
        req = _request(_split(r"\boxed{2}"))
        await from_file.verify(req)
        await from_list.verify(req)
        assert _judge_prompt(from_file) == _judge_prompt(from_list)
        assert all(rule in _judge_prompt(from_file) for rule in inline)

    @mark.parametrize("contents", ["rules: [some rule]", "- '  '", "- 42", "- [broken", ""])
    def test_invalid_rule_file_fails_at_startup(self, config, tmp_path, contents):
        path = tmp_path / "rules.yaml"
        config.extra_answer_rules = str(path)
        path.write_text(contents, encoding="utf-8")
        with raises(ValueError, match=r"extra_answer_rules file .*rules\.yaml"):
            _server(config)

    async def test_extra_answer_rules_leave_rubric_prompt_intact(self, config):
        body = _request(_split("Proof."), answer_format="free", rubric=RUBRIC)
        reply = '{"r1":true,"r2":true,"r3":false}'
        base = _server(config, reply)
        extended = _server(config.model_copy(update={"extra_answer_rules": ["Apply the archive alias."]}), reply)
        await base.verify(body)
        await extended.verify(body)
        assert _judge_prompt(extended) == _judge_prompt(base)

    @mark.parametrize(
        "text,exact,command",
        [
            (r"\boxed{1} then \boxed {2}", "1", "2"),
            (r"\boxed{1} then \boxed {2", "1", None),
            (r"\boxed{1} then \boxed{}", None, None),
            (r"\boxed{\frac{1}{2}} and a literal \\boxed{7}", "7", r"\frac{1}{2}"),
            (r"\boxed {\{a,b\}}", None, r"\{a,b\}"),
            (r"\boxed{1} then \boxedExtra{2}", "1", "1"),
        ],
    )
    def test_box_parser(self, text, exact, command):
        assert last_boxed(text) == exact
        assert last_boxed(text, last_command=True) == command

    @mark.parametrize("scope", ["stem", "cross_domain"])
    @mark.parametrize("mode", ["binary", "yes_no_unresolved"])
    async def test_scope_and_verdict_are_independent(self, config, scope, mode):
        server = _server(
            config.model_copy(update={"guideline_scope": scope, "verdict_mode": mode}), "Judgement: 'yes'"
        )
        result = await server.verify(_request(_split(r"\boxed{2}")))
        prompt = _judge_prompt(server)
        assert ("any subject" in prompt) == (scope == "cross_domain")
        assert ("Judgement: 'unresolved'" in prompt) == (mode == "yes_no_unresolved")
        assert ("Your final judgement must be in the following format:" in prompt) == (mode == "binary")
        assert result.reward == 1.0

    @mark.parametrize(
        "reply,verdict,status,known",
        [
            ("Judgement: 'yes'\nJudgement: 'no'", "no", "judged", True),
            ("Judgement: 'no'</think>\nJudgement: 'unresolved'", "unresolved", "judge_unresolved", False),
            ("Judgement: 'yes'\nI cannot decide", "unresolved", "judge_unparsed", False),
            ("Judgement: 'yesterday'", "unresolved", "judge_unparsed", False),
        ],
    )
    async def test_saved_no_and_abstention(self, config, reply, verdict, status, known):
        server = _server(config.model_copy(update={"verdict_mode": "yes_no_unresolved"}), reply)
        result = await server.verify(
            _request(_split(r"\boxed{3}"), verdict="yes", correctness=True, evaluation_completed=True)
        )
        saved = json.loads(result.model_dump_json())
        assert saved["verdict"] == verdict and saved["grading_status"] == status
        assert saved["evaluation_completed"] is known and saved["mask_sample"] is (not known)
        assert saved["correctness"] is (False if known else None)
        assert saved["reward"] == 0.0
        assert saved["judge_evaluations"][0]["verdict_label"] == (
            None if status == "judge_unparsed" else f"Judgement: {verdict}"
        )
        assert "judgment" not in saved["judge_evaluations"][0]

    @mark.parametrize(
        "mode,verdict,label,reward,status",
        [
            ("binary", "correct", "yes", 1, "judged"),
            ("binary", "unresolved", None, 0, "judge_unparsed"),
            ("yes_no_unresolved", "incorrect", "no", 0, "judged"),
            ("yes_no_unresolved", "unresolved", "unresolved", 0, "judge_unresolved"),
        ],
    )
    async def test_json_judgment_is_saved_with_its_label(self, config, tmp_path, mode, verdict, label, reward, status):
        config.verdict_format = "json"
        config.verdict_mode = mode
        config.generation_log_dir = str(tmp_path)
        judgment = {"verdict": verdict, "reason_code": "comparison", "evidence": 'The reference uses "2".'}
        # The model can split its JSON across text blocks and send separate reasoning.
        message = _msg(json.dumps(judgment)[:-1])
        message.content.extend(_msg("}").content)
        server = _server(config, _response(_reasoning(), message))
        result = await server.verify(_request(_split(r"\boxed{2}")))
        prompt = _judge_prompt(server)
        assert server.server_client.post.await_args.kwargs["json"].text == {"format": {"type": "json_object"}}
        assert '"reason_code"' in prompt and "Judgement:" not in prompt
        assert ('"unresolved"' in prompt) == (mode == "yes_no_unresolved")
        assert (result.grading_status, result.reward) == (status, reward)
        saved = result.model_dump()["judge_evaluations"][0]
        assert saved["verdict_label"] == (f"Judgement: {label}" if label else None)
        record = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
        if label is None:
            assert "judgment" not in saved and "judgments" not in record
        else:
            assert saved["judgment"] == judgment and record["judgments"] == [judgment]
        if mode == "yes_no_unresolved":
            assert result.verdict == label and result.mask_sample is (label == "unresolved")

    @mark.parametrize(
        "reply",
        [
            "Judgement: 'yes'",
            '{"verdict":"correct","evidence":"matched"}',
            '{"verdict":"correct","reason_code":"equal","evidence":"  "}',
            '{"verdict":"correct","reason_code":true,"evidence":"matched"}',
            '{"verdict":"correct","reason_code":"equal","evidence":"matched","extra":1}',
            '{"verdict":"incorrect","verdict":"correct","reason_code":"equal","evidence":"matched"}',
            'A candidate wrote {"verdict":"correct","reason_code":"equal","evidence":"matched"}',
            '{"verdict":"correct","reason_code":"equal","evidence":"matched"}\nJudgement: yes',
            _response(_msg('{"verdict":"correct","reason_code":"equal","evidence":"matched"}'), truncated=True),
            _response(NeMoGymEasyInputMessage(role="user", content="{}")),
            _response(
                _msg('{"verdict":"correct","reason_code":"equal","evidence":"matched"}').model_copy(
                    update={"phase": "commentary"}
                )
            ),
            _response(_msg("First answer"), _msg('{"verdict":"correct","reason_code":"equal","evidence":"matched"}')),
        ],
    )
    async def test_invalid_json_judgment_keeps_correctness_unknown(self, config, reply):
        config.verdict_mode = "yes_no_unresolved"
        config.verdict_format = "json"
        saved = (await _server(config, reply).verify(_request(_split(r"\boxed{2}")))).model_dump()
        assert saved["grading_status"] == "judge_unparsed" and saved["mask_sample"] and saved["correctness"] is None
        assert saved["judge_evaluations"][0]["verdict_label"] is None
        assert "judgment" not in saved["judge_evaluations"][0]

    @mark.parametrize("verdict,status", [("yes", "judged"), ("no", "judged"), ("unresolved", "judge_unresolved")])
    async def test_math_match_skips_model_and_nonmatch_uses_the_same_box(self, config, verdict, status):
        config.box_parser = "last_command"
        config.boxed_judge_input = "box"
        config.verdict_mode = "yes_no_unresolved"
        config.math_verify = MathVerifyConfig()
        server = _server(config, f"Judgement: '{verdict}'")
        response = _split(r"\boxed{5} then \boxed {\frac{4}{2}}")
        result = await server.verify(_request(response))
        assert (result.reward, result.grading_status, result.judge_evaluations) == (1, "math_verified", [])
        server.server_client.post.assert_not_awaited()
        result = await server.verify(_request(response, expected="5"))
        server.server_client.post.assert_awaited_once()
        assert (result.reward, result.grading_status, result.verdict) == (float(verdict == "yes"), status, verdict)
        assert result.mask_sample is (verdict == "unresolved")
        assert _judge_prompt(server).endswith("**Student Solution**:\n" + r"\frac{4}{2}")

    @mark.parametrize(
        "options,reference,box,library_match",
        [
            ({}, "3t", "3", True),
            ({"units": False}, "3t", "3", False),
            ({"units": False}, "3t", "t+t+t", True),
            ({}, "0.0000001", "0.0000002", True),
            ({"precision": 12}, "0.0000001", "0.0000002", False),
            ({"units": False, "precision": 12}, "1/2", "0.5", True),
            ({}, "1", "9=1", True),
            ({"bidirectional": True}, "1", "9=1", False),
            ({"bidirectional": True}, "1", "x=1", True),
            ({"bidirectional": True}, "1", "1", True),
            ({"bidirectional": True}, "1", "2", False),
            ({"bidirectional": True}, r"\(\frac{1}{2}\)", "0.5", True),
        ],
    )
    async def test_math_options_route_boxed_answers(self, config, options, reference, box, library_match):
        config.boxed_judge_input = "box"
        config.math_verify = MathVerifyConfig(**options)
        server = _server(config, "Judgement: 'no'")
        result = await server.verify(_request(_split(r"\boxed{" + box + "}"), expected=reference))
        assert result.grading_status == ("math_verified" if library_match else "judged")
        assert result.reward == float(library_match)
        assert server.server_client.post.await_count == (0 if library_match else 1)

    @mark.parametrize("precision", [-1, 1.5, True])
    def test_math_precision_requires_nonnegative_decimal_places(self, precision):
        with raises(ValidationError, match="precision"):
            MathVerifyConfig(precision=precision)

    async def test_math_worker_unavailable_defers_to_judge(self, config, monkeypatch):
        config.boxed_judge_input = "box"
        config.math_verify = MathVerifyConfig()
        server = _server(config, "Judgement: 'yes'")
        monkeypatch.setattr(server._math_library, "matches", AsyncMock(side_effect=OSError))
        result = await server.verify(_request(_split(r"\boxed{2}")))
        server.server_client.post.assert_awaited_once()
        assert (result.reward, result.grading_status) == (1, "judged")

    async def test_math_timeout_defers_to_stem_verdict(self, config, monkeypatch):
        def slow_match(*args):
            time.sleep(1)
            return 1.0, "2"

        monkeypatch.setattr("resources_servers.stem_judge.boxed_math._run_math_verify", slow_match)
        config.boxed_judge_input = "box"
        config.math_verify = MathVerifyConfig(timeout_seconds=0.02)
        server = _server(config, "Judgement: 'yes'")
        result = await server.verify(_request(_split(r"\boxed{2}")))
        server.server_client.post.assert_awaited_once()
        assert (result.reward, result.grading_status) == (1, "judged")

    @mark.parametrize(
        "row,view,reply",
        [
            ({"answer_format": "free"}, "box", "Judgement: 'no'"),
            ({}, "final_answer", "Judgement: 'no'"),
            ({"rubric": RUBRIC}, "box", '{"r1":false,"r2":false,"r3":false}'),
        ],
    )
    async def test_math_precheck_skips_rubrics_and_whole_answers(self, config, row, view, reply):
        config.boxed_judge_input = view
        config.math_verify = MathVerifyConfig()
        server = _server(config, reply)
        result = await server.verify(_request(_split(r"\boxed{2}"), **row))
        server.server_client.post.assert_awaited_once()
        assert result.reward == 0 and result.grading_status == "judged"

    async def test_question_bank_preset_uses_json_and_the_environment_selects_math(self, config):
        preset = yaml.safe_load((SERVER_DIR / "configs/question_bank.yaml").read_text())
        config = StemJudgeConfig.model_validate(
            config.model_dump() | preset["stem_judge"]["resources_servers"]["stem_judge"]
        )
        row = _request(_split(r"\boxed{2}"), math_equivalence=True)  # Older rows may still carry the flag.
        judgment = {"verdict": "correct", "reason_code": "equal", "evidence": "2 equals 2"}
        server = _server(config, _response(NeMoGymEasyInputMessage(role="assistant", content=json.dumps(judgment))))
        result = await server.verify(row)
        server.server_client.post.assert_awaited_once()
        assert result.verdict == "yes" and result.judge_evaluations[0].judgment.model_dump() == judgment

        server = _server(config.model_copy(update={"math_verify": MathVerifyConfig()}))
        result = await server.verify(row.model_copy(update={"math_equivalence": False}))
        server.server_client.post.assert_not_awaited()
        assert result.grading_status == "math_verified" and result.reward == 1

    async def test_reverify_drops_stale_answer_evidence(self, config):
        server = _server(config, "Judgement: 'yes'")
        result = await server.verify(
            _request(
                _split(r"\boxed{2}"),
                mask_sample=True,
                evaluation_completed=False,
                correctness=None,
                verdict="unresolved",
                library_reward=0.0,
            )
        )
        saved = result.model_dump()
        assert saved["reward"] == 1 and saved["mask_sample"] is False
        assert saved["judge_evaluations"][0]["verdict_label"] == "Judgement: yes"
        assert {"evaluation_completed", "correctness", "verdict", "library_reward"}.isdisjoint(saved)

    def test_policy_template_renders_only_the_question(self):
        from nemo_gym.data import fill_prompt, load_prompt_config

        template = load_prompt_config(str(SERVER_DIR / "prompt_templates" / "boxed_final_answer.yaml"))
        messages = fill_prompt(template, {"question": "Compute 2 + 3.", "expected_answer": "private reference"})
        assert messages == [
            {
                "role": "user",
                "content": (
                    "Compute 2 + 3.\n\nSolve the problem step by step. "
                    r"Put your final answer, and only your final answer, in \boxed{}."
                ),
            }
        ]


class TestShippedConfig:
    def test_defaults(self, config: StemJudgeConfig) -> None:
        assert config.name == "stem_judge"
        assert config.default_answer_format == "boxed"
        assert config.boxed_judge_input == "final_answer"
        assert (config.max_box_chars, config.max_answer_chars) == (None, 16000)
        assert (config.rubric_reward, config.rubric_pass_fraction) == ("fraction", 1.0)

    def test_example_rows_are_valid_requests(self) -> None:
        rows = [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]
        requests = [StemJudgeVerifyRequest(**row, response=_split("\\boxed{2}")) for row in rows]
        assert [r.answer_format for r in requests] == [None, "free", "free", None, None]
        assert [r.rubric is not None for r in requests] == [False, False, True, False, False]

    def test_answer_template_is_stem_v11(self) -> None:
        # Pins the stem_v11 prompt as it was measured offline.
        digest = hashlib.sha256((SERVER_DIR / "prompt_templates" / "stem_v11.txt").read_bytes()).hexdigest()
        assert digest == "9b9dd45f94b114c0ea03c7a4bf91a41b37f6f62bbcbafea5eac0b99be8598498"  # pragma: allowlist secret


class TestFinalAnswer:
    def test_the_message_is_the_answer(self) -> None:
        assert final_answer(_split("  The answer is 2.  ")) == ("The answer is 2.", "ok")
        assert final_answer(_response(_msg("The answer is 2."))) == ("The answer is 2.", "ok")

    @mark.parametrize("answer", ["", "Answer: 2"])
    def test_truncated(self, answer: str) -> None:
        assert final_answer(_split(answer, truncated=True)) == (None, "truncated")

    @mark.parametrize("response", [_split("  "), _response(_reasoning())])
    def test_empty(self, response: NeMoGymResponse) -> None:
        assert final_answer(response) == (None, "empty_final_answer")


class TestParsing:
    @mark.parametrize(
        "text, expected",
        [
            ("x \\boxed{\\frac{4}{2}} y", "\\frac{4}{2}"),
            ("\\boxed{1} then \\boxed{2}", "2"),
            ("\\boxed{1} then \\boxed{2", None),
            ("\\boxed{ }", None),
            ("\\boxed{f=\\left\\{ x \\right.}", "f=\\left\\{ x \\right."),
            ("\\boxed{\\{1, 2\\}}", "\\{1, 2\\}"),
            ("no box", None),
        ],
    )
    def test_last_boxed(self, text: str, expected) -> None:
        assert last_boxed(text) == expected

    @mark.parametrize(
        "text, expected",
        [
            ("Judgement: 'yes'", "yes"),
            ("**Judgement:** 'no'", "no"),
            ("Judgement: yes\nOn reflection, Judgement: 'no'", "no"),
            ("Final Answer: \\boxed{yes}", "yes"),
            ("The student is right.", None),
        ],
    )
    def test_verdict(self, text: str, expected) -> None:
        assert parse_verdict(text) == expected

    @mark.parametrize(
        "text",
        # The stem_v11 patterns took 2.6 s on the first and 11.5 s on the second: a regression fails, not hangs.
        ["Answer:" + "\n" * 60 + "z", "\n" * 20000, "Verdict:" + " *" * 3000, "Judgement" + "*" * 50000],
    )
    def test_degenerate_judge_text_does_not_backtrack(self, text: str) -> None:
        # Backtracking on these runs would stall the event loop (the stem_v11 patterns took seconds on 60 newlines).
        started = time.perf_counter()
        assert parse_verdict(text) is None
        assert time.perf_counter() - started < 0.5

    @mark.parametrize(
        "text, expected",
        [
            ("Verdict = no", "no"),
            ("Judgment: yes", "yes"),
            ("**Final Judgement**: no", "no"),
            ("Answer: $\\boxed{\\text{No}}$", "no"),
            ("Conclusion:  'Yes'", "yes"),
            ("Answer: 2/3", None),
            ("Answer: yesterday", None),
        ],
    )
    def test_fallback_verdict(self, text: str, expected) -> None:
        assert parse_verdict(text) == expected

    def test_only_text_after_the_judges_think_block_counts(self) -> None:
        reasoning = 'Maybe Judgement: \'yes\' unless the sign matters. {"r1": true, "r2": true}</think>'
        assert parse_verdict(reasoning + "The sign is wrong.\n**Final Verdict:** no") == "no"
        assert parse_criteria(reasoning + "Not parseable.", ["r1", "r2"]) is None

    def test_criteria(self) -> None:
        ids = ["r1", "r2"]
        assert parse_criteria('Criteria: {"r1": true, "r2": false}', ids) == {"r1": True, "r2": False}
        assert parse_criteria('```json\n{"r1": "True", "r2": "false"}\n```', ids) == {"r1": True, "r2": False}
        # The last object that marks every id wins.
        text = '{"r1": false, "r2": false} draft {"r1": true} final {"r1": true, "r2": true}'
        assert parse_criteria(text, ids) == {"r1": True, "r2": True}

    @mark.parametrize(
        "text",
        ['{"r1": true}', '{"r1": true, "r2": "maybe"}', '{"r1": 1, "r2": 0}', "r1: met, r2: not met", ""],
    )
    def test_criteria_unparsed(self, text: str) -> None:
        assert parse_criteria(text, ["r1", "r2"]) is None

    @mark.parametrize(
        "cid, reply",
        [("r}", '{"r}": true}'), ("r{", 'Marks: {"r{": true}'), ('a"}', '{"a\\"}": true} done')],
    )
    def test_braces_and_quotes_inside_criterion_ids(self, cid: str, reply: str) -> None:
        assert parse_criteria(reply, [cid]) == {cid: True}

    def test_latex_after_the_criteria_does_not_hide_them(self) -> None:
        text = 'Criteria: {"r1": true, "r2": false}' + " $x^{2}$" * 100
        assert parse_criteria(text, ["r1", "r2"]) == {"r1": True, "r2": False}

    def test_deeply_nested_json_is_unparsed_quickly(self) -> None:
        started = time.perf_counter()
        assert parse_criteria('{"a":' * 20000 + "1" + "}" * 20000, ["r1"]) is None
        assert time.perf_counter() - started < 0.5

    @mark.parametrize(
        "expected, rendered",
        [
            ("- (π^2/9) m_t^2 m_H^2 / m_1^2", "-(π^2/9) m_t^2 m_H^2 / m_1^2"),
            ("  - g^4 N / (4π)", "  -g^4 N / (4π)"),
            ("- \\Theta(t^2 - r^2) f\n  \\times g", "-\\Theta(t^2 - r^2) f\n  \\times g"),
            ("-3.2 eV", "-3.2 eV"),
            ("x - y", "x - y"),
            (
                "- Molecular formula: C2H6O\n- Molar mass: 46 g/mol",
                "- Molecular formula: C2H6O\n- Molar mass: 46 g/mol",
            ),
            ("", ""),
        ],
    )
    def test_render_reference(self, expected: str, rendered: str) -> None:
        assert render_reference(expected) == rendered

    def test_render_reference_is_linear_in_blank_lines(self) -> None:
        started = time.perf_counter()
        assert render_reference("- 5" + "\n" * 20000) == "-5" + "\n" * 20000
        assert time.perf_counter() - started < 0.5

    def test_render_and_score(self) -> None:
        rubric = [RubricCriterion(**c) for c in RUBRIC]
        assert render_rubric(rubric).splitlines()[2] == (
            "[r3] (penalty, -2 points) Gives the small-angle period as the final answer"
        )
        assert rubric_fraction(rubric, {"r1": True, "r2": False, "r3": False}) == approx(0.6)
        assert rubric_fraction(rubric, {"r1": False, "r2": False, "r3": True}) == approx(0.0)
        assert rubric_fraction(rubric, {"r1": True, "r2": True, "r3": True}) == approx(0.6)


class TestRequestValidation:
    @mark.parametrize(
        "row",
        [
            {"rubric": []},
            {"rubric": [{"id": "r1", "points": 1, "criterion": "a"}, {"id": "r1", "points": 1, "criterion": "b"}]},
            {"rubric": [{"id": "r1", "points": 0, "criterion": "a"}, {"id": "r2", "points": 1, "criterion": "b"}]},
            {"rubric": [{"id": "r1", "points": float("nan"), "criterion": "a"}]},
            {"rubric": [{"id": "r1", "points": float("inf"), "criterion": "a"}]},
            {"rubric": [{"id": "r1", "points": -1, "criterion": "a"}]},
            {"answer_format": "boxd"},
        ],
    )
    def test_bad_rows_are_rejected(self, row: dict) -> None:
        with raises(ValidationError):
            _request(_split("x"), **row)

    @mark.parametrize(
        "expected, metadata",
        [(None, None), ("  ", None), (None, {"expected_answer": None}), (None, {"expected_answer": []})],
    )
    def test_answer_row_needs_a_reference(self, expected, metadata) -> None:
        with raises(ValidationError, match="needs a non-empty expected_answer"):
            _request(_split("x"), expected=expected, metadata=metadata)

    def test_reference_from_metadata_and_rubric_rows_without_one(self) -> None:
        assert _request(_split("x"), expected=None, metadata={"expected_answer": 0}).expected_answer is None
        _request(_split("x"), expected=None, rubric=RUBRIC)

    def test_bad_question_regex_is_rejected_at_startup(self, config: StemJudgeConfig) -> None:
        with raises(ValidationError):
            StemJudgeConfig(**{**config.model_dump(), "question_extract_regex": "(unclosed"})


class TestVerify:
    async def test_boxed_row_judges_the_final_answer(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Reasoning...\nJudgement: 'yes'")
        res = await server.verify(_request(_split("First 1+1 = 2, so \\boxed{\\frac{4}{2}}.")))
        assert (res.reward, res.grading_status, res.answer_format) == (approx(1.0), "judged", "boxed")
        assert res.judge_evaluations[0].verdict_label == "Judgement: yes"
        assert "**Student Solution**:\nFirst 1+1 = 2, so \\boxed{\\frac{4}{2}}." in _judge_prompt(server)
        assert "**Question**:\nQ: 1+1?" in _judge_prompt(server)

    async def test_boxed_row_judges_only_the_box(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"boxed_judge_input": "box"}), "Judgement: 'yes'")
        await server.verify(_request(_split("First 1+1 = 2, so \\boxed{\\frac{4}{2}}.")))
        assert _judge_prompt(server).endswith("**Student Solution**:\n\\frac{4}{2}")
        assert "First 1+1" not in _judge_prompt(server)

    async def test_free_row_length_gate(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("y" * 16001), answer_format="free"))
        assert (res.reward, res.grading_status) == (approx(0.0), "answer_too_long")
        server.server_client.post.assert_not_awaited()
        server = _server(config.model_copy(update={"max_answer_chars": None}), "Judgement: 'yes'")
        assert (await server.verify(_request(_split("y" * 16001), answer_format="free"))).reward == approx(1.0)

    @mark.parametrize(
        "response, status",
        [
            (_split("The answer is 2."), "no_boxed"),
            (_split("y" * 16001 + " \\boxed{2}"), "answer_too_long"),
            (_split("\\boxed{2}", truncated=True), "truncated"),
            (_response(_reasoning()), "empty_final_answer"),
        ],
    )
    async def test_gates_skip_the_judge(self, config: StemJudgeConfig, response, status: str) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(response))
        assert (res.reward, res.grading_status, res.judge_evaluations) == (approx(0.0), status, [])
        server.server_client.post.assert_not_awaited()

    async def test_box_cap(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{" + "x" * 2001 + "}")))
        assert res.reward == approx(1.0)
        server = _server(config.model_copy(update={"max_box_chars": 2000}), "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{" + "x" * 2001 + "}")))
        assert (res.reward, res.grading_status) == (approx(0.0), "box_too_long")
        server.server_client.post.assert_not_awaited()

    async def test_answer_cap_off_when_the_judge_sees_only_the_box(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"boxed_judge_input": "box"}), "Judgement: 'yes'")
        res = await server.verify(_request(_split("y" * 16001 + " \\boxed{2}")))
        assert res.reward == approx(1.0)

    async def test_leading_minus_reference_keeps_its_sign(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{-2}"), expected="- 2"))
        assert "**Reference Answer(s)**:\n-2\n" in _judge_prompt(server)
        assert res.expected_answer == "- 2"

    async def test_free_row_needs_no_box(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("Two, because 1+1=2."), answer_format="free"))
        assert (res.reward, res.answer_format) == (approx(1.0), "free")
        assert "**Student Solution**:\nTwo, because 1+1=2." in _judge_prompt(server)

    @mark.parametrize(
        "judge, reward, status", [("Judgement: 'no'", 0.0, "judged"), ("I agree.", 0.0, "judge_unparsed")]
    )
    async def test_judge_no_or_unparsed(self, config: StemJudgeConfig, judge: str, reward: float, status: str) -> None:
        server = _server(config, judge)
        res = await server.verify(_request(_split("\\boxed{3}")))
        assert (res.reward, res.grading_status) == (approx(reward), status)
        assert server.server_client.post.await_count == 1

    async def test_reference_from_metadata(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{0}"), expected=None, metadata={"expected_answer": 0}))
        assert "**Reference Answer(s)**:\n0\n" in _judge_prompt(server)
        assert res.expected_answer == "0"

    @mark.parametrize(
        "question_input",
        [
            [{"role": "user", "content": [{"type": "input_text", "text": "Q: 1+1?"}]}],
            "Q: 1+1?",
        ],
    )
    async def test_question_from_content_list_or_string_input(self, config: StemJudgeConfig, question_input) -> None:
        server = _server(config, "Judgement: 'yes'")
        req = _request(_split("\\boxed{2}"))
        req.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=question_input)
        await server.verify(req)
        assert "**Question**:\nQ: 1+1?\n" in _judge_prompt(server)

    async def test_reasoning_only_judge_reply_is_unparsed(self, config: StemJudgeConfig) -> None:
        server = _server(config, _response(_reasoning("Judgement: 'yes'")))
        res = await server.verify(_request(_split("\\boxed{3}")))
        assert (res.reward, res.grading_status) == (approx(0.0), "judge_unparsed")

    async def test_system_message_and_regex_precedence(self, config: StemJudgeConfig) -> None:
        update = {"judge_system_message": "Be strict.", "extract_problem_from_prompt": True}
        update["question_extract_regex"] = r"Problem:\s*(.*)"
        server = _server(config.model_copy(update=update), "Judgement: 'yes'")
        req = _request(_split("\\boxed{2}"))
        req.responses_create_params.input[0].content = "Answer: {your final answer}\nProblem: What is 1+1?"
        await server.verify(req)
        judge_input = server.server_client.post.await_args.kwargs["json"].input
        assert (judge_input[0].role, judge_input[0].content) == ("system", "Be strict.")
        assert "**Question**:\nWhat is 1+1?\n" in judge_input[-1].content

    async def test_cut_off_judge_reply_is_unparsed(self, config: StemJudgeConfig) -> None:
        cut_off = _response(_msg("The student writes Judgement: 'yes', but"), truncated=True)
        server = _server(config, cut_off.model_copy(update={"status": "incomplete"}))
        res = await server.verify(_request(_split("\\boxed{3}")))
        assert (res.reward, res.grading_status) == (approx(0.0), "judge_unparsed")

    async def test_row_fields_are_kept_and_cannot_override_the_grade(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'no'")
        row = {"id": "row-7", "subject": "physics", "reward": 1.0, "grading_status": "judged", "mask_sample": True}
        res = await server.verify(_request(_split("\\boxed{3}"), **row))
        assert (res.reward, res.grading_status, res.mask_sample) == (approx(0.0), "judged", False)
        assert (res.id, res.subject) == ("row-7", "physics")

    async def test_failed_judge_call_masks_the_row(self, config: StemJudgeConfig) -> None:
        # A judge outage is not a wrong answer: the row is masked, not scored 0.
        server = _server(config)
        server.server_client.post = AsyncMock(side_effect=RuntimeError("judge unavailable"))
        request = _request(_split("\\boxed{2}"))
        failed = json.loads((await judge_failsafe(server.verify)(request)).body)
        assert (failed["_ng_failure_class"], failed["mask_sample"]) == ("judge_failed", True)
        assert "judge unavailable" in failed["_ng_failure_judge_error"]
        assert failed["response"] == request.response.model_dump(mode="json")

    async def test_rubric_fraction(self, config: StemJudgeConfig) -> None:
        server = _server(config, 'Checked each.\nCriteria: {"r1": true, "r2": false, "r3": false}')
        res = await server.verify(_request(_split("T = 2π√(L/g)(1+θ0²/16)"), answer_format="free", rubric=RUBRIC))
        assert (res.reward, res.rubric_score, res.grading_status) == (approx(0.6), approx(0.6), "judged")
        assert res.judge_evaluations[0].criteria_met == {"r1": True, "r2": False, "r3": False}
        prompt = _judge_prompt(server)
        assert "[r1] (3 points) Final period" in prompt
        assert "**Reference Answer**:\n2" in prompt
        assert prompt.endswith("**Student Solution**:\nT = 2π√(L/g)(1+θ0²/16)")

    async def test_saved_rubric_grade_has_no_empty_verdict_label(self, config: StemJudgeConfig) -> None:
        # Gym's rollout recovery reads a saved verdict_label of None as a judge reply without a verdict.
        server = _server(config, '{"r1": true, "r2": false, "r3": false}', "no marks")
        parsed, unparsed = [
            (await server.verify(_request(_split("T"), answer_format="free", rubric=RUBRIC))).model_dump()
            for _ in range(2)
        ]
        assert "verdict_label" not in parsed["judge_evaluations"][0]
        assert parsed["judge_evaluations"][0]["criteria_met"] == {"r1": True, "r2": False, "r3": False}
        assert unparsed["judge_evaluations"][0]["verdict_label"] is None

    async def test_reverify_mode(self, config: StemJudgeConfig) -> None:
        assert await _server(config).get_reverify_mode() == ReverifyMode.STATELESS

    async def test_rubric_with_a_brace_in_a_criterion_id(self, config: StemJudgeConfig) -> None:
        server = _server(config, '{"r}": true}')
        rubric = [{"id": "r}", "points": 1, "criterion": "c"}]
        res = await server.verify(_request(_split("T"), answer_format="free", rubric=rubric))
        assert (res.grading_status, res.rubric_score, res.reward) == ("judged", approx(1.0), approx(1.0))

    @mark.parametrize("output", [[], [_msg("Judgement: 'yes'")]])
    async def test_failed_judge_response_masks_the_row(self, config: StemJudgeConfig, output) -> None:
        failed = _failed_response(*output)
        server = _server(config, failed)
        request = _request(_split("\\boxed{2}"))
        result = json.loads((await judge_failsafe(server.verify)(request)).body)
        assert (result["_ng_failure_class"], result["mask_sample"], result["reward"]) == ("judge_failed", True, 0.0)
        assert result["response"] == request.response.model_dump(mode="json")

    async def test_rubric_row_without_a_reference(self, config: StemJudgeConfig) -> None:
        server = _server(config, '{"r1": true, "r2": true, "r3": false}')
        await server.verify(_request(_split("T"), expected=None, answer_format="free", rubric=RUBRIC))
        assert "**Reference Answer**:\n(none)\n" in _judge_prompt(server)

    async def test_rubric_threshold_tolerates_float_rounding(self, config: StemJudgeConfig) -> None:
        rubric = [{"id": f"r{i}", "points": p, "criterion": "c"} for i, p in enumerate((0.7, 0.1, 0.2))]
        threshold = config.model_copy(update={"rubric_reward": "threshold", "rubric_pass_fraction": 0.8})
        server = _server(threshold, '{"r0": true, "r1": true, "r2": false}')
        res = await server.verify(_request(_split("T"), answer_format="free", rubric=rubric))
        assert res.reward == approx(1.0)

    async def test_rubric_threshold(self, config: StemJudgeConfig) -> None:
        threshold = config.model_copy(update={"rubric_reward": "threshold", "rubric_pass_fraction": 0.5})
        for marks, reward in (('"r1": true, "r2": false', 1.0), ('"r1": false, "r2": true', 0.0)):
            server = _server(threshold, '{%s, "r3": false}' % marks)
            res = await server.verify(_request(_split("T"), answer_format="free", rubric=RUBRIC))
            assert res.reward == approx(reward)

    async def test_rubric_unparsed(self, config: StemJudgeConfig) -> None:
        server = _server(config, 'Criteria: {"r1": true, "r2": false}')
        res = await server.verify(_request(_split("T"), answer_format="free", rubric=RUBRIC))
        assert (res.reward, res.grading_status, res.rubric_score) == (approx(0.0), "judge_unparsed", None)

    async def test_rubric_row_still_follows_its_answer_format(self, config: StemJudgeConfig) -> None:
        server = _server(config, '{"r1": true, "r2": true, "r3": false}')
        res = await server.verify(_request(_split("T without a box"), rubric=RUBRIC))
        assert res.grading_status == "no_boxed"

    async def test_extract_problem_from_prompt(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"extract_problem_from_prompt": True}), "Judgement: 'yes'")
        req = _request(_split("\\boxed{2}"))
        req.responses_create_params.input[0].content = "Preamble.\nAnswer: {your final answer}\nWhat is 1+1?"
        await server.verify(req)
        assert "**Question**:\nWhat is 1+1?\n" in _judge_prompt(server)

    async def test_generation_log_keeps_replies_without_a_closing_message(self, config, tmp_path) -> None:
        reasoning_only = _response(_reasoning("Still checking the units"))
        server = _server(config.model_copy(update={"generation_log_dir": str(tmp_path)}), reasoning_only)
        res = await server.verify(_request(_split("\\boxed{2}")))
        assert res.grading_status == "judge_unparsed"
        (record,) = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()]
        assert (record["grading_status"], record["judge_generations"]) == ("judge_unparsed", [""])

    def test_bad_custom_template_fails_at_startup(self, config: StemJudgeConfig, tmp_path) -> None:
        template = tmp_path / "bad.txt"
        template.write_text("Question: {question}\nReference: {reference}")
        with raises(KeyError):
            _server(config.model_copy(update={"answer_prompt_template_fpath": str(template)}))

    async def test_generation_log(self, config: StemJudgeConfig, tmp_path) -> None:
        server = _server(config.model_copy(update={"generation_log_dir": str(tmp_path)}), "Judgement: 'yes'")
        await server.verify(_request(_split("so \\boxed{2}")))
        await server.verify(_request(_split("no box")))
        records = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()]
        assert [(r["grading_status"], r["reward"], r["verdicts"]) for r in records] == [
            ("judged", 1.0, ["Judgement: yes"]),
            ("no_boxed", 0.0, []),
        ]
        assert records[0]["judged_generation"] == "so \\boxed{2}"


class TestGradingObservation:
    """The real verify and call_judge record the grading for the active verifier invocation."""

    async def _observe(
        self, config: StemJudgeConfig, server: StemJudgeResourcesServer, request: StemJudgeVerifyRequest
    ):
        invocation = VerifierInvocation(invocation_id="verify-stem")
        token = verifier_observability._active_verifier.set(invocation)
        try:
            result = await server.verify(request)
        except BaseException as error:
            result = error
        finally:
            verifier_observability._active_verifier.reset(token)
        return result, invocation

    @mark.parametrize(
        "judge, row, outcome",
        [
            ("Judgement: 'no'", {}, "completed"),
            ('{"r1": true, "r2": false, "r3": false}', {"answer_format": "free", "rubric": RUBRIC}, "completed"),
            ("I agree.", {}, "unknown"),
            ('{"r1": true}', {"answer_format": "free", "rubric": RUBRIC}, "unknown"),
        ],
    )
    async def test_parsed_grades_complete_and_unparsed_stay_unknown(self, config, judge, row, outcome) -> None:
        server = _server(config, judge)
        result, invocation = await self._observe(config, server, _request(_split("\\boxed{3}"), **row))
        (operation,) = invocation.grading_operations
        assert operation.outcome == outcome
        assert [d.response_id for d in operation.dependencies] == ["resp"]
        if outcome == "completed":
            assert operation.result_binding_sha256 == grading_result_binding(result)

    async def test_a_failed_judge_response_is_a_failed_grading(self, config: StemJudgeConfig) -> None:
        failed = _failed_response()
        result, invocation = await self._observe(config, _server(config, failed), _request(_split("\\boxed{3}")))
        assert isinstance(result, BaseException)
        (operation,) = invocation.grading_operations
        assert operation.outcome == "failed"

    async def test_a_cut_off_reply_stays_unknown(self, config: StemJudgeConfig) -> None:
        cut_off = _response(_msg("Judgement: 'yes'"), truncated=True).model_copy(update={"status": "incomplete"})
        _, invocation = await self._observe(config, _server(config, cut_off), _request(_split("\\boxed{3}")))
        (operation,) = invocation.grading_operations
        assert operation.outcome == "unknown"

    async def test_a_failed_judge_transport_is_a_failed_grading(self, config: StemJudgeConfig, monkeypatch) -> None:
        monkeypatch.setattr("nemo_gym.verification.judge._judge_retry_wait_s", lambda attempt: 0)
        server = _server(config)
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge down"))
        result, invocation = await self._observe(config, server, _request(_split("\\boxed{3}")))
        assert isinstance(result, BaseException)
        assert server.server_client.post.await_count == 3  # a lost connection is retried before it fails
        (operation,) = invocation.grading_operations
        assert (operation.outcome, operation.dependencies) == ("failed", [])

    async def test_a_gated_row_records_no_judge_grading(self, config: StemJudgeConfig) -> None:
        _, invocation = await self._observe(config, _server(config), _request(_split("no box")))
        assert not invocation.grading_operations


class TestTaskDataSchema:
    """task_data.TaskData is the row contract; the runtime request inherits it."""

    @mark.parametrize(
        "row, valid",
        [
            ({"expected_answer": "2"}, True),
            ({"metadata": {"expected_answer": 0}}, True),
            ({"rubric": RUBRIC, "answer_format": "free"}, True),
            ({"expected_answer": "2", "id": "row-1", "subject": "physics"}, True),
            ({}, False),
            ({"metadata": {"expected_answer": []}}, False),
            ({"rubric": [RUBRIC[0], RUBRIC[0]]}, False),
            ({"rubric": [{"id": "r1", "points": float("nan"), "criterion": "c"}]}, False),
        ],
    )
    def test_schema_and_runtime_agree(self, row: dict, valid: bool) -> None:
        def accepts(build) -> bool:
            try:
                build()
                return True
            except ValidationError:
                return False

        assert accepts(lambda: TaskData.model_validate(row)) is valid
        fields = {k: v for k, v in row.items() if k != "expected_answer"}
        assert accepts(lambda: _request(_split("x"), expected=row.get("expected_answer"), **fields)) is valid

    def test_example_rows_validate_against_the_schema(self) -> None:
        adapter = load_task_data_schema(SERVER_DIR)
        rows = (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()
        report = validate_jsonl_rows("stem_judge", adapter, "example.jsonl", rows)
        assert (report.rows, report.clean) == (5, True), report.summary()
