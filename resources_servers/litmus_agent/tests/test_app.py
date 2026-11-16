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
"""Unit tests for the domain-agnostic litmus_agent resources server."""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    list_providers,
    register_provider,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesParticipant,
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
    check_resources_session_contract,
)
from nemo_gym.trajectory import grading_result_binding
from resources_servers.litmus_agent.app import (
    _ANSWER_FORMAT_REGEXES,
    _CODE_EXEC_DRIVER_PATH,
    _DEFAULT_RULE,
    _SUPPORTED_ANSWER_TYPES,
    BOOL,
    FLOAT,
    REWARD_RULES,
    STRING,
    LitmusAgentConfig,
    LitmusAgentResourcesServer,
    LitmusAgentVerifyRequest,
    compute_reward,
    extract_predicted_value,
    resolve_answer_type,
    resolve_reward_rule,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {"input": [{"role": "user", "content": "test"}]}


def _make_server() -> LitmusAgentResourcesServer:
    config = LitmusAgentConfig(host="0.0.0.0", port=8080, entrypoint="", name="litmus_agent")
    return LitmusAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(text: str, **fields) -> LitmusAgentVerifyRequest:
    return LitmusAgentVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(text),
        **fields,
    )


# ---------------------------------------------------------------------------
# resolve_answer_type
# ---------------------------------------------------------------------------


class TestResolveAnswerType:
    def test_explicit_answer_type(self):
        assert resolve_answer_type(FLOAT) == FLOAT

    def test_supported_set(self):
        assert _SUPPORTED_ANSWER_TYPES == {FLOAT, BOOL, STRING}

    def test_unsupported_answer_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported answer_type='bogus'"):
            resolve_answer_type("bogus")


# ---------------------------------------------------------------------------
# extract_predicted_value
# ---------------------------------------------------------------------------


class TestExtractPredictedValue:
    def test_double_parens_integer(self):
        assert extract_predicted_value("The answer is ((42))", FLOAT) == 42.0

    def test_double_parens_last_occurrence_wins(self):
        assert extract_predicted_value("First ((3)), actually ((5))", FLOAT) == 5.0

    def test_bare_number_rejected(self):
        assert extract_predicted_value("42", FLOAT) is None

    def test_empty_capture_returns_none(self):
        assert extract_predicted_value("(())", FLOAT) is None

    def test_non_numeric_capture_returns_none(self):
        assert extract_predicted_value("((hello))", FLOAT) is None

    def test_non_string_input(self):
        assert extract_predicted_value(None, FLOAT) is None

    def test_numeric_token_inside_capture(self):
        assert extract_predicted_value("Final value is: about 12.5 g/mol", FLOAT, answer_format="fmt_25") == 12.5

    def test_string_returns_raw_capture(self):
        assert (
            extract_predicted_value("**Answer: Carboxylic Acid**", STRING, answer_format="fmt_18") == "Carboxylic Acid"
        )

    def test_bool_word_token(self):
        assert extract_predicted_value("<final_answer>yes</final_answer>", BOOL, answer_format="fmt_15") == 1.0

    def test_bool_numeric_token(self):
        assert extract_predicted_value("((0))", BOOL) == 0.0

    def test_unknown_answer_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported answer_format='fmt_99'"):
            extract_predicted_value("((42))", FLOAT, answer_format="fmt_99")

    def test_all_formats_registered(self):
        assert set(_ANSWER_FORMAT_REGEXES) == {f"fmt_{i:02d}" for i in range(31)}

    def test_output_regex_extracts_value(self):
        assert (
            extract_predicted_value("Correct Answer: [3]", FLOAT, output_regex=r"Correct Answer:\s*\[(.+?)\]") == 3.0
        )

    def test_output_regex_preferred_over_answer_format(self):
        # answer_format would pick the boxed 7; output_regex targets the label.
        text = r"\boxed{7}. Final Answer = 42"
        assert (
            extract_predicted_value(text, FLOAT, output_regex=r"Final Answer\s*=\s*(.+)", answer_format="fmt_07")
            == 42.0
        )

    def test_output_regex_recovers_dropped_wrapper(self):
        # A lenient row regex (optional bracket) scores an answer that the strict
        # fmt_06 regex would miss when the model drops the '[]' wrapper.
        text = "Correct Answer: 2"
        assert extract_predicted_value(text, FLOAT, answer_format="fmt_06") is None
        lenient = r"Correct Answer:\s*\**\[?\s*([-+]?\d*\.?\d+)"
        assert extract_predicted_value(text, FLOAT, output_regex=lenient) == 2.0

    def test_output_regex_last_match_wins(self):
        assert extract_predicted_value("val: 3 then val: 5", FLOAT, output_regex=r"val:\s*(\d+)") == 5.0

    def test_output_regex_string_returns_raw(self):
        assert extract_predicted_value("SMILES: CCO", STRING, output_regex=r"SMILES:\s*(.+)") == "CCO"

    def test_output_regex_no_match_returns_none(self):
        assert extract_predicted_value("no answer here", FLOAT, output_regex=r"Answer:\s*(\d+)") is None

    def test_output_regex_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid output_regex"):
            extract_predicted_value("x", FLOAT, output_regex=r"(unclosed")

    def test_output_regex_wrong_group_count_raises(self):
        with pytest.raises(ValueError, match="exactly one capture group"):
            extract_predicted_value("x", FLOAT, output_regex=r"(a)(b)")


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


class TestComputeReward:
    def test_none_prediction(self):
        assert compute_reward(None, 5.0, FLOAT) == 0.0

    def test_nan_prediction(self):
        assert compute_reward(float("nan"), 5.0, FLOAT) == 0.0

    def test_numeric_int_correct(self):
        assert compute_reward(5.0, "5", FLOAT) == 1.0

    def test_numeric_int_rounds(self):
        assert compute_reward(4.9, "5", FLOAT, match={"rule": "exact"}) == 1.0

    def test_numeric_int_wrong(self):
        assert compute_reward(4.0, "5", FLOAT) == 0.0

    def test_numeric_float_correct(self):
        assert compute_reward(857.833, "857.833", FLOAT) == 1.0

    def test_numeric_float_wrong(self):
        assert compute_reward(857.834, "857.833", FLOAT) == 0.0

    def test_numeric_float_tolerance_override(self):
        assert compute_reward(1.0, "1.05", FLOAT, float_abs_tol=0.1) == 1.0

    def test_bool_correct_token_expected(self):
        assert compute_reward(1.0, "true", BOOL) == 1.0

    def test_bool_wrong(self):
        assert compute_reward(0.0, "true", BOOL) == 0.0

    def test_bool_numeric_expected(self):
        assert compute_reward(1.0, 1, BOOL) == 1.0

    def test_bool_unparseable_expected(self):
        assert compute_reward(1.0, "maybe", BOOL) == 0.0

    def test_string_correct_normalized(self):
        assert compute_reward("Carboxylic   Acid", "carboxylic acid", STRING) == 1.0

    def test_string_wrong(self):
        assert compute_reward("alcohol", "carboxylic acid", STRING) == 0.0


# ---------------------------------------------------------------------------
# reward-rule registry + resolution
# ---------------------------------------------------------------------------


class TestResolveRewardRule:
    def test_default_rules_cover_all_answer_types(self):
        assert set(_DEFAULT_RULE) == _SUPPORTED_ANSWER_TYPES
        assert all(name in REWARD_RULES for name in _DEFAULT_RULE.values())

    def test_default_when_no_match(self):
        assert resolve_reward_rule(FLOAT, None) == ("isclose", {})
        assert resolve_reward_rule(BOOL, None) == ("bool_eq", {})
        assert resolve_reward_rule(STRING, None) == ("string_eq", {})

    def test_match_overrides_default(self):
        assert resolve_reward_rule(FLOAT, {"rule": "abs_window", "abs_tol": 2}) == ("abs_window", {"abs_tol": 2})

    def test_match_without_rule_raises(self):
        with pytest.raises(ValueError, match="must include a 'rule' key"):
            resolve_reward_rule(FLOAT, {"abs_tol": 2})


class TestComputeRewardWithMatch:
    def test_abs_window_within_tolerance(self):
        assert compute_reward(98.0, "100", FLOAT, match={"rule": "abs_window", "abs_tol": 2}) == 1.0

    def test_abs_window_outside_tolerance(self):
        assert compute_reward(97.0, "100", FLOAT, match={"rule": "abs_window", "abs_tol": 2}) == 0.0

    def test_rel_window_within_tolerance(self):
        assert compute_reward(18.2, "18.0", FLOAT, match={"rule": "rel_window", "rel_tol": 0.02}) == 1.0

    def test_match_forces_exact_over_default_window(self):
        # default for numeric_float is isclose; match can force a stricter/other rule
        assert compute_reward(5.0, "5.4", FLOAT, match={"rule": "exact"}) == 1.0

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unsupported reward rule='bogus'"):
            compute_reward(1.0, "1", FLOAT, match={"rule": "bogus"})

    def test_none_prediction_short_circuits_before_rule_lookup(self):
        # None returns 0.0 without resolving the (here invalid) rule
        assert compute_reward(None, "1", FLOAT, match={"rule": "bogus"}) == 0.0

    def test_custom_registered_rule(self):
        def within_five_percent(predicted, expected, **_):
            return 1.0 if abs(float(predicted) - float(expected)) <= 0.05 * abs(float(expected)) else 0.0

        REWARD_RULES["within_five_percent"] = within_five_percent
        try:
            assert compute_reward(0.62, "0.60", FLOAT, match={"rule": "within_five_percent"}) == 1.0
            assert compute_reward(0.70, "0.60", FLOAT, match={"rule": "within_five_percent"}) == 0.0
        finally:
            del REWARD_RULES["within_five_percent"]


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class TestVerify:
    def test_answer_type_is_required(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="answer_type"):
            _make_verify_request("((3))", expected_answer=3)

    async def test_unresolvable_answer_type_raises(self):
        server = _make_server()
        with pytest.raises(ValueError, match="Unsupported answer_type"):
            await server.verify(_make_verify_request("((3))", expected_answer="3", answer_type="unsupported"))

    @pytest.mark.parametrize(
        ("text", "expected_answer", "answer_type", "abs_error"),
        [
            ("((2.6))", 2.1, FLOAT, 0.5),
            ("((2.1))", 2.1, FLOAT, 0.0),
            ("((1))", 0, BOOL, None),
            ("no answer here", 2.1, FLOAT, None),
        ],
    )
    async def test_abs_error_uses_answer_type(self, text, expected_answer, answer_type, abs_error):
        server = _make_server()
        result = await server.verify(
            _make_verify_request(text, expected_answer=expected_answer, answer_type=answer_type)
        )
        if abs_error is None:
            assert result.abs_error is None
        else:
            assert result.abs_error == pytest.approx(abs_error)

    async def test_correct_numeric_int(self):
        server = _make_server()
        result = await server.verify(
            _make_verify_request("Reasoning... ((3))", expected_answer="3", answer_type=FLOAT)
        )
        assert result.reward == 1.0
        assert result.correct is True
        assert result.predicted_value == 3.0
        assert result.resolved_answer_type == FLOAT
        assert result.resolved_reward_rule == "isclose"

    async def test_output_regex_row_scores_dropped_wrapper(self):
        # End-to-end: a lenient output_regex on the row recovers an answer whose
        # strict fmt_06 wrapper the model dropped ("Correct Answer: 2", no []).
        server = _make_server()
        strict = await server.verify(
            _make_verify_request("Correct Answer: 2", expected_answer="2", answer_type=FLOAT, answer_format="fmt_06")
        )
        assert strict.reward == 0.0
        assert strict.predicted_value is None

        lenient = await server.verify(
            _make_verify_request(
                "Correct Answer: 2",
                expected_answer="2",
                answer_type=FLOAT,
                answer_format="fmt_06",
                output_regex=r"Correct Answer:\s*\**\[?\s*([-+]?\d*\.?\d+)",
            )
        )
        assert lenient.reward == 1.0
        assert lenient.predicted_value == 2.0

    async def test_per_row_match_window(self):
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "Reasoning... ((98))",
                expected_answer="100",
                answer_type=FLOAT,
                match={"rule": "abs_window", "abs_tol": 2},
            )
        )
        assert result.reward == 1.0
        assert result.resolved_reward_rule == "abs_window"

    async def test_incorrect_when_unextractable(self):
        server = _make_server()
        result = await server.verify(_make_verify_request("no answer here", expected_answer="3", answer_type=FLOAT))
        assert result.reward == 0.0
        assert result.correct is False
        assert result.predicted_value is None

    async def test_string_answer(self):
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "**Answer: Paris**", expected_answer="paris", answer_type=STRING, answer_format="fmt_18"
            )
        )
        assert result.reward == 1.0

    async def test_passthrough_fields_echoed(self):
        server = _make_server()
        result = await server.verify(
            _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT, smiles="CCO")
        )
        assert (result.model_extra or {}).get("smiles") == "CCO"

    async def test_string_content_message_extracted(self):
        # Some backends emit the assistant message content as a bare string
        # rather than a list of parts; the extractor handles both.
        server = _make_server()
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[{"id": "msg_1", "type": "message", "role": "assistant", "content": "((3))"}],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = LitmusAgentVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            expected_answer="3",
            answer_type=FLOAT,
        )
        result = await server.verify(request)
        assert result.reward == 1.0

    async def test_reserved_passthrough_field_does_not_collide(self):
        # A passthrough field named like one verify() sets explicitly (e.g.
        # "reward"/"correct") must not 500 the endpoint via a splat kwarg
        # collision -- the computed value wins and the colliding field is dropped.
        server = _make_server()
        result = await server.verify(
            _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT, reward="ignore-me", correct="nope")
        )
        assert result.reward == 1.0
        assert result.correct is True


# ---------------------------------------------------------------------------
# compute_metrics / get_key_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_mae_grouped_per_property(self):
        server = _make_server()

        def _row(prop, answer_type, abs_error, correct):
            return {
                "method": "direct",
                "resolved_answer_type": answer_type,
                "property": prop,
                "reward": 1.0 if correct else 0.0,
                "correct": correct,
                "abs_error": abs_error,
            }

        tasks = [
            [_row("MolWt", "float", 0.0, True), _row("MolWt", "float", 4.0, False)],
            [_row("HeavyAtomCount", "float", 2.0, False), _row("HeavyAtomCount", "float", None, False)],
            [_row("HasBenzene", "bool", None, True)],
        ]
        by_property = server.compute_metrics(tasks)["by_property"]

        assert by_property["MolWt"]["mae"] == pytest.approx(2.0)
        assert by_property["MolWt"]["mae_count"] == 2
        # The unparsed rollout is excluded from the mean but still counted.
        assert by_property["HeavyAtomCount"]["mae"] == pytest.approx(2.0)
        assert by_property["HeavyAtomCount"]["mae_count"] == 1
        assert by_property["HeavyAtomCount"]["count"] == 2
        # bool properties carry no abs_error, so they report no MAE.
        assert by_property["HasBenzene"]["mae"] is None
        assert by_property["HasBenzene"]["mae_count"] == 0
        assert by_property["HasBenzene"]["accuracy"] == 1.0

    def _rollout(self, method, atype, reward, correct):
        return {"method": method, "resolved_answer_type": atype, "reward": reward, "correct": correct}

    def test_grouping_by_method_and_answer_type(self):
        server = _make_server()
        tasks = [
            [
                self._rollout("direct", FLOAT, 1.0, True),
                self._rollout("direct", FLOAT, 0.0, False),
                self._rollout("direct", BOOL, 1.0, True),
            ],
            [self._rollout("mcp-python", FLOAT, 1.0, True)],
        ]
        metrics = server.compute_metrics(tasks)

        assert metrics["direct"]["count"] == 3
        assert metrics["direct"]["accuracy"] == pytest.approx(2 / 3)
        assert metrics["direct"]["mean_reward"] == pytest.approx(2 / 3)
        assert metrics["direct"]["by_answer_type"][FLOAT]["count"] == 2
        assert metrics["direct"]["by_answer_type"][BOOL]["accuracy"] == 1.0
        assert metrics["mcp-python"]["count"] == 1

    def test_defaults_for_missing_fields(self):
        server = _make_server()
        metrics = server.compute_metrics([[{"reward": 1.0}]])
        assert metrics["unknown"]["by_answer_type"]["unknown"]["count"] == 1
        assert metrics["unknown"]["accuracy"] == 0.0

    def test_pass_at_k_metrics_merged_in(self):
        server = _make_server()
        tasks = [
            [
                {**self._rollout("direct", FLOAT, 1.0, True), "predicted_value": 3.0},
                {**self._rollout("direct", FLOAT, 0.0, False), "predicted_value": 4.0},
            ],
            [
                {**self._rollout("direct", FLOAT, 0.0, False), "predicted_value": 7.0},
                {**self._rollout("direct", FLOAT, 0.0, False), "predicted_value": None},
            ],
        ]
        metrics = server.compute_metrics(tasks)

        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(25.0)
        assert metrics["pass@2/accuracy"] == pytest.approx(50.0)
        assert metrics["pass@1[avg-of-2]/no_answer"] == pytest.approx(25.0)
        assert "majority@2/accuracy" in metrics
        # The method breakdown is preserved alongside.
        assert metrics["direct"]["count"] == 4

    def test_get_key_metrics_filters(self):
        server = _make_server()
        out = server.get_key_metrics({"mean/reward": 0.5, "mean/correct": 0.5, "other": 9})
        assert out == {"mean/reward": 0.5, "mean/correct": 0.5}

    def test_get_key_metrics_selects_highest_k(self):
        server = _make_server()
        out = server.get_key_metrics(
            {
                "mean/reward": 0.5,
                "mean/output_tokens": 120.0,
                "pass@1/accuracy": 40.0,
                "pass@2/accuracy": 60.0,
                "pass@1[avg-of-2]/accuracy": 50.0,
                "pass@2/no_answer": 10.0,
            }
        )
        assert out == {
            "mean/reward": 0.5,
            "mean/output_tokens": 120.0,
            "pass@2/accuracy": 60.0,
            "pass@1[avg-of-2]/accuracy": 50.0,
        }


# ---------------------------------------------------------------------------
# Sandbox-backed code-execution tool
# ---------------------------------------------------------------------------


class _LocalFakeProvider:
    """In-process SandboxProvider that runs the driver via a local subprocess.

    Uploaded files (the driver, injected through ``spec.files``) are kept in
    memory; ``exec`` materializes the requested script and runs it with the
    given env. This exercises the real replay driver without a remote sandbox.
    """

    name = "litmus_fake"
    instances: list["_LocalFakeProvider"] = []

    def __init__(self, **_: object) -> None:
        self.created = 0
        self.closed = 0
        self.aclosed = 0
        # Set to make close fail the way a remote stop can.
        self.close_error: Exception | None = None
        self._files: dict[str, dict[str, str]] = {}
        _LocalFakeProvider.instances.append(self)

    async def create(self, spec) -> SandboxHandle:
        self.created += 1
        sandbox_id = f"{self.name}-{self.created}"
        self._files[sandbox_id] = {}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def upload_file(self, handle, source_path, target_path) -> None:
        self._files[handle.sandbox_id][target_path] = Path(source_path).read_text()

    async def download_file(self, handle, source_path, target_path) -> None:
        Path(target_path).write_text(self._files[handle.sandbox_id].get(source_path, ""))

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        script_path = command.split()[-1]
        contents = self._files[handle.sandbox_id].get(script_path, "")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(contents)
            local_script = f.name
        try:
            proc = subprocess.run(
                [sys.executable, local_script],
                capture_output=True,
                text=True,
                env={**os.environ, **(env or {})},
                timeout=timeout_s,
            )
        finally:
            os.unlink(local_script)
        return SandboxExecResult(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)

    async def status(self, handle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle) -> None:
        if self.close_error is not None:
            raise self.close_error
        self.closed += 1

    async def aclose(self) -> None:
        self.aclosed += 1


if _LocalFakeProvider.name not in list_providers():
    register_provider(_LocalFakeProvider.name, _LocalFakeProvider)


class _FakeRequest:
    """Minimal stand-in for a Starlette Request used by the tool endpoint."""

    def __init__(self, session_id: str | None = None, body: dict | None = None) -> None:
        self.session = {SESSION_ID_KEY: session_id} if session_id is not None else {}
        self._body = body or {}

    async def json(self) -> dict:
        return self._body


def _make_sandbox_server(**config_overrides) -> LitmusAgentResourcesServer:
    config_kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="litmus_agent",
        sandbox_provider={_LocalFakeProvider.name: {}},
        sandbox_spec={"image": "fake:latest"},
    )
    config_kwargs.update(config_overrides)
    config = LitmusAgentConfig(**config_kwargs)
    return LitmusAgentResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def _run_code(server: LitmusAgentResourcesServer, session_id: str, code: str) -> str:
    response = await server.execute_code(_FakeRequest(session_id=session_id, body={"code": code}))
    return response.body.decode()


class TestCodeExecTool:
    async def test_verify_failure_still_cleans_up_session_sandbox(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print(1)")
        provider = _LocalFakeProvider.instances[-1]

        with pytest.raises(ValueError, match="Unsupported answer_type"):
            await server._verify_and_cleanup(
                _FakeRequest(session_id="s1"),
                _make_verify_request("((3))", expected_answer="3", answer_type="unsupported"),
            )

        assert provider.closed == 1 and provider.aclosed == 1
        assert "s1" not in server._sessions

    async def test_single_cell_output(self):
        server = _make_sandbox_server()
        assert (await _run_code(server, "s1", "print(2 + 3)")).strip() == "5"

    async def test_state_persists_across_cells(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "x = 21")
        assert (await _run_code(server, "s1", "print(x * 2)")).strip() == "42"

    async def test_only_newest_cell_output_returned(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print('first')")
        # The prior cell's print is suppressed on replay; only the new one shows.
        assert (await _run_code(server, "s1", "print('second')")).strip() == "second"

    async def test_failing_cell_not_added_to_history(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "y = 10")
        err = await _run_code(server, "s1", "raise ValueError('boom')")
        assert "ValueError: boom" in err
        # The bad cell was not retained, and earlier state survives.
        assert (await _run_code(server, "s1", "print(y)")).strip() == "10"

    async def test_malformed_request_body_defaults_empty(self):
        # A request whose body isn't valid JSON must not crash the endpoint; the
        # code defaults to empty and the driver runs a no-op cell.
        server = _make_sandbox_server()

        class _BadRequest(_FakeRequest):
            async def json(self):
                raise ValueError("bad json")

        response = await server.execute_code(_BadRequest(session_id="s1"))
        assert response.body.decode() == ""

    async def test_replay_failure_resets_session(self, tmp_path):
        # A prior cell that succeeds once but fails when replayed (its side effect
        # already applied) leaves emulated state unrecoverable -> the driver
        # signals a reset (exit 2) and the server drops the session history.
        server = _make_sandbox_server()
        marker = tmp_path / "made"
        await _run_code(server, "s1", f"import os; os.mkdir({str(marker)!r})")
        session = server._sessions["s1"]
        assert session.cells  # retained as known-good
        # The replayed mkdir now raises FileExistsError, forcing a reset.
        out = await _run_code(server, "s1", "print('unreached')")
        assert "environment was reset" in out
        assert session.cells == []

    async def test_system_exit_reset_code_does_not_wipe_history(self):
        # A cell that exits with the driver's reserved reset code (2) must not
        # be able to spoof a session reset; prior state has to survive as an
        # ordinary cell error rather than a silent history wipe.
        server = _make_sandbox_server()
        await _run_code(server, "s1", "keep = 99")
        out = await _run_code(server, "s1", "import sys; sys.exit(2)")
        assert "SystemExit" in out
        assert (await _run_code(server, "s1", "print(keep)")).strip() == "99"

    async def test_system_exit_zero_cell_not_retained(self):
        # A cell calling sys.exit(0) terminates the interpreter before the driver
        # can signal success, so it is reported as an error and dropped -- never
        # replayed on later calls where its SystemExit would desync the session.
        server = _make_sandbox_server()
        await _run_code(server, "s1", "base = 7")
        out = await _run_code(server, "s1", "import sys; sys.exit(0)")
        assert "SystemExit" in out
        assert (await _run_code(server, "s1", "print(base)")).strip() == "7"

    async def test_sessions_are_isolated(self):
        server = _make_sandbox_server()
        await _run_code(server, "a", "secret = 1")
        out = await _run_code(server, "b", "print('secret' in dir())")
        assert out.strip() == "False"

    async def test_output_truncation(self):
        server = _make_sandbox_server(code_exec_max_output_chars=20)
        out = await _run_code(server, "s1", "print('z' * 100)")
        assert "output truncated to 20 chars" in out
        assert len(out.splitlines()[0]) <= 20

    async def test_verify_cleans_up_session_sandbox(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print(1)")
        provider = _LocalFakeProvider.instances[-1]
        assert provider.closed == 0

        await server._verify_and_cleanup(
            _FakeRequest(session_id="s1"),
            _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT),
        )
        assert provider.closed == 1 and provider.aclosed == 1
        assert "s1" not in server._sessions

    async def test_shutdown_closes_open_sessions(self):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print(1)")
        provider = _LocalFakeProvider.instances[-1]
        await server._shutdown_all_sessions()
        assert provider.closed == 1
        assert server._sessions == {}


class TestSandboxSpecAndRouting:
    def test_driver_injected_into_spec_files(self):
        server = _make_sandbox_server()
        spec = server._build_sandbox_spec()
        assert _CODE_EXEC_DRIVER_PATH in spec.files
        assert spec.image == "fake:latest"

    def test_unknown_spec_key_raises(self):
        server = _make_sandbox_server(sandbox_spec={"image": "x", "bogus": 1})
        with pytest.raises(ValueError, match="Unknown sandbox_spec keys: bogus"):
            server._build_sandbox_spec()

    def test_tool_route_registered_only_with_provider(self):
        with_tool = _make_sandbox_server().setup_webserver()
        assert "/stateful_python_code_exec" in {r.path for r in with_tool.routes}

        without_tool = _make_server().setup_webserver()
        assert not any("stateful" in r.path for r in without_tool.routes)


EPISODE_ID = EpisodeId(rollout_id="rollout", attempt=0)


def _seed(resources_session_id: str) -> dict:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EPISODE_ID,
        task_id=TaskId(taskset="litmus_agent", task_id="0"),
        task_data={"expected_answer": "3", "answer_type": FLOAT},
    ).model_dump(mode="json")


def _close(resources_session_id: str) -> dict:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=EPISODE_ID).model_dump(
        mode="json"
    )


VERIFY_BODY = _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT).model_dump(mode="json")


def _session_client(**client_options) -> tuple[LitmusAgentResourcesServer, TestClient]:
    server = _make_sandbox_server()
    return server, TestClient(server.setup_webserver(), **client_options)


class TestResourcesSessions:
    @pytest.mark.parametrize("with_provider", [False, True])
    def test_http_verify_preserves_grading_observation_and_binding(self, with_provider: bool) -> None:
        server = _make_sandbox_server() if with_provider else _make_server()
        server.server_client.global_config_dict = {"observability_enabled": True}
        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
            if with_provider:
                assert client.post("/stateful_python_code_exec", json={"code": "print(3)"}).status_code == 200
            response = client.post("/ng-rollout/observed-litmus/verify", json=VERIFY_BODY)
        assert response.status_code == 200
        payload = response.json()
        assert payload["reward"] == 1.0
        (invocation,) = payload["ng_verifier_invocations"]
        assert invocation["grading_record"] == {
            "verifier": server.config.name,
            "score": 1.0,
            "verdict": "scored",
            "result_binding_sha256": grading_result_binding(payload),
        }
        assert not server.has_session_state("rs-1")

    async def test_request_scored_verify_retries_after_scoring_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _make_sandbox_server()
        request = _FakeRequest("rs-1")
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        await _run_code(server, "rs-1", "print(3)")
        original_verify = LitmusAgentResourcesServer._score
        calls = 0

        async def fail_once_then_score(server, body):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("scoring unavailable")
            return await original_verify(server, body)

        monkeypatch.setattr(LitmusAgentResourcesServer, "_score", fail_once_then_score)
        body = LitmusAgentVerifyRequest.model_validate(VERIFY_BODY)
        with pytest.raises(ValueError, match="scoring unavailable"):
            await server._verify_and_cleanup(request, body)
        assert not server.has_session_state("rs-1")
        assert _LocalFakeProvider.instances[-1].closed == 1
        retried = await server._verify_and_cleanup(request, body)
        assert retried.reward == 1.0 and not retried.mask_sample
        assert calls == 2
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))

    def test_verifier_only_config_releases_identity_and_masks_verify_after_close(self) -> None:
        server = _make_server()
        assert not server.config.sandbox_provider
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert server.session_identity("rs-1") is not None

        verified = client.post("/verify", json=VERIFY_BODY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert server._session_episodes == {}
        assert server.session_is_verified("rs-1")
        assert not server.has_session_state("rs-1")
        wrong_close = _close("rs-1") | {"episode_id": {"rollout_id": "different", "attempt": 0}}
        assert client.post("/close_session", json=wrong_close).status_code == 409
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        late_verify = client.post("/verify", json=VERIFY_BODY)
        assert late_verify.status_code == 200
        assert late_verify.json()["mask_sample"] and late_verify.json()["failure_kind"] == "session_lost"

    async def test_verify_cleanup_prevents_late_worker_allocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _make_sandbox_server()
        request = _FakeRequest("rs-1")
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        stopped, release = asyncio.Event(), asyncio.Event()
        original_cleanup = LitmusAgentResourcesServer._cleanup_session

        async def delayed_cleanup(server, session_id):
            await original_cleanup(server, session_id)
            stopped.set()
            await asyncio.wait_for(release.wait(), timeout=5)

        monkeypatch.setattr(LitmusAgentResourcesServer, "_cleanup_session", delayed_cleanup)
        verification = asyncio.create_task(
            server._verify_and_cleanup(request, LitmusAgentVerifyRequest.model_validate(VERIFY_BODY))
        )
        await asyncio.wait_for(stopped.wait(), timeout=5)
        tool = asyncio.create_task(_run_code(server, "rs-1", "print(3)"))
        try:
            await asyncio.sleep(0.05)
            assert not tool.done() and server._sessions == {}
            release.set()
            assert (await verification).reward == 1.0
            with pytest.raises(ResourcesSessionLostError):
                await tool
            assert server._sessions == {}
        finally:
            release.set()
            await asyncio.gather(verification, tool, return_exceptions=True)

    def test_checkpoint_failed_verify_cleanup_stays_owned_until_retry(self) -> None:
        server = _make_sandbox_server()
        app = server.setup_webserver()
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app, raise_server_exceptions=False)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(3)"}).status_code == 200
        provider = _LocalFakeProvider.instances[-1]
        provider.close_error = RuntimeError("cleanup failed")
        first = client.post(prefix + "/verify", json=VERIFY_BODY, headers={"x-ng-checkpoint-close-follows": "1"})
        assert first.status_code == 200 and first.json()["reward"] == 1.0
        assert server.checkpoint_participant.is_open("rs-1")
        assert "rs-1" in server._session_episodes
        assert "rs-1" in server._sessions
        provider.close_error = None
        second = client.post(prefix + "/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert not server.checkpoint_participant.is_open("rs-1")
        assert server._session_episodes == {}
        assert server._sessions == {} and provider.closed == 1

    def test_checkpoint_verify_releases_active_identity_but_allows_score_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_gym.server_runtime.base_resources_server as base_resources

        server = _make_sandbox_server()
        app = server.setup_webserver()
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(3)"}).status_code == 200
        assert server.checkpoint_participant.is_open("rs-1")

        first = client.post(prefix + "/verify", json=VERIFY_BODY, headers={"x-ng-checkpoint-close-follows": "1"})
        assert first.status_code == 200 and first.json()["reward"] == 1.0
        # Default wait verification unbinds before the HTTP reply. A replacement may never send close.
        assert not server.checkpoint_participant.is_open("rs-1")
        assert server._session_episodes == {}
        assert server.session_identity("rs-1") == (
            EPISODE_ID,
            TaskId.model_validate(_seed("rs-1")["task_id"]),
        )
        second = client.post(prefix + "/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 409
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(3)"}).status_code == 404
        wrong_close = _close("rs-1") | {"episode_id": {"rollout_id": "different", "attempt": 0}}
        assert client.post(prefix + "/close_session", json=wrong_close).status_code == 409

        now = base_resources.monotonic()
        monkeypatch.setattr(
            base_resources, "monotonic", lambda: now + server.config.session_close_retry_window_seconds + 1
        )
        assert not server.session_is_verified("rs-1")

    async def test_checkpoint_retire_releases_owned_sandbox_and_retries_failure(self) -> None:
        server = _make_sandbox_server()
        await server.seed_session(
            ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=_FakeRequest("rs-1")
        )
        await _run_code(server, "rs-1", "print(3)")
        provider = _LocalFakeProvider.instances[-1]
        participant = ResourcesParticipant(server, "restart_only")
        participant.seeded("rs-1", EPISODE_ID)
        provider.close_error = RuntimeError("sandbox unavailable")
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            await participant.retire(EPISODE_ID)
        assert "rs-1" in server._sessions and "rs-1" in participant._sessions
        provider.close_error = None
        await participant.retire(EPISODE_ID)
        assert provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}
        assert server._session_episodes == {} and participant._sessions == {}

    def test_session_contract(self):
        server = _make_sandbox_server()

        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._session_episodes,
            verify_body=VERIFY_BODY,
        )

    def test_leaving_the_lifespan_stops_the_sessions_still_open(self):
        server, client = _session_client()
        with client:
            assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
            assert client.post("/stateful_python_code_exec", json={"code": "print(3)"}).status_code == 200
            provider = _LocalFakeProvider.instances[-1]
            assert provider.closed == 0 and "rs-1" in server._sessions

        assert provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}

    def test_close_without_verify_stops_the_sandbox(self):
        server, client = _session_client()
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        tool_call = client.post("/stateful_python_code_exec", json={"code": "x = 21\nprint(x * 2)"})
        assert tool_call.status_code == 200 and tool_call.text.strip() == "42"
        provider = _LocalFakeProvider.instances[-1]

        # An Environment Server whose seed reply was lost closes without the cookie.
        client.cookies.clear()
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        assert provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}
        assert "rs-1" not in server._session_episodes

    def test_failed_stop_leaves_session_open_for_a_retry(self):
        server, client = _session_client(raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(1)"}).status_code == 200
        provider = _LocalFakeProvider.instances[-1]

        provider.close_error = RuntimeError("sandbox API unavailable")
        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        assert not server.session_is_closed("rs-1")
        assert "rs-1" in server._sessions

        provider.close_error = None
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert provider.closed == 1
        assert server._sessions == {}
        assert server.session_is_closed("rs-1")

    def test_failed_stop_at_verify_preserves_score_and_keeps_cleanup_owned(self):
        server, client = _session_client(raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(1)"}).status_code == 200
        provider = _LocalFakeProvider.instances[-1]
        provider.close_error = RuntimeError("sandbox API unavailable")

        verified = client.post("/verify", json=VERIFY_BODY)

        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        # The sandbox stays registered, so the close retries the stop and reports its failure.
        assert "rs-1" in server._sessions
        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        provider.close_error = None
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert provider.closed == 1 and server._sessions == {}

    def test_tool_call_after_close_is_a_lost_session(self):
        server, client = _session_client()
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        providers_before = len(_LocalFakeProvider.instances)

        response = client.post("/stateful_python_code_exec", json={"code": "print(1)"})

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        # The refused call started no sandbox.
        assert len(_LocalFakeProvider.instances) == providers_before
        assert server._sessions == {}

    async def test_close_racing_a_cell_waits_for_it_then_refuses_the_next(self, monkeypatch):
        server = _make_sandbox_server()
        running = asyncio.Event()
        finish_exec = asyncio.Event()
        real_exec = _LocalFakeProvider.exec

        async def blocking_exec(provider, handle, command, **kwargs):
            running.set()
            await finish_exec.wait()
            return await real_exec(provider, handle, command, **kwargs)

        monkeypatch.setattr(_LocalFakeProvider, "exec", blocking_exec)
        await server.seed_session(
            ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=_FakeRequest("rs-1")
        )
        tool_call = asyncio.create_task(_run_code(server, "rs-1", "print(1)"))
        await running.wait()
        close = asyncio.create_task(
            server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        )
        await asyncio.sleep(0.05)
        provider = _LocalFakeProvider.instances[-1]
        # The close does not stop the sandbox under the running cell.
        assert provider.closed == 0 and not close.done()

        finish_exec.set()
        assert (await tool_call).strip() == "1"
        await close
        assert provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}
        with pytest.raises(ResourcesSessionLostError):
            await _run_code(server, "rs-1", "print(2)")

    async def test_verify_during_startup_waits_for_the_cell_and_releases_sandbox(self, monkeypatch):
        server = _make_sandbox_server()
        creating = asyncio.Event()
        finish_create = asyncio.Event()
        real_create = _LocalFakeProvider.create

        async def delayed_create(provider, spec):
            creating.set()
            await finish_create.wait()
            return await real_create(provider, spec)

        monkeypatch.setattr(_LocalFakeProvider, "create", delayed_create)
        # An Agent's /run session: its verify, not a close, stops the sandbox.
        tool_call = asyncio.create_task(_run_code(server, "s1", "print(1)"))
        await creating.wait()
        verify = asyncio.create_task(
            server._verify_and_cleanup(
                _FakeRequest("s1"), _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT)
            )
        )
        await asyncio.sleep(0.05)
        finish_create.set()

        assert (await verify).reward == 1.0
        # The accepted cell finishes before verification releases its sandbox.
        assert (await tool_call).strip() == "1"
        provider = _LocalFakeProvider.instances[-1]
        assert provider.created == 1 and provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}

    async def test_close_waits_for_startup_and_retries_a_failed_stop(self, monkeypatch):
        server = _make_sandbox_server()
        started = asyncio.Event()
        finish_start = asyncio.Event()
        real_create = _LocalFakeProvider.create

        async def delayed_create(provider, spec):
            handle = await real_create(provider, spec)
            provider.close_error = RuntimeError("sandbox API unavailable")
            started.set()
            await finish_start.wait()
            return handle

        monkeypatch.setattr(_LocalFakeProvider, "create", delayed_create)
        seed = ResourcesSeedSessionRequest.model_validate(_seed("rs-1"))
        await server.seed_session(seed, request=_FakeRequest("rs-1"))
        tool_call = asyncio.create_task(_run_code(server, "rs-1", "print(1)"))
        await started.wait()
        close = asyncio.create_task(
            server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        )
        await asyncio.sleep(0)
        finish_start.set()
        try:
            with pytest.raises(RuntimeError, match="sandbox API unavailable"):
                await close
            assert not server.session_is_closed("rs-1")
            assert "rs-1" in server._sessions
            provider = _LocalFakeProvider.instances[-1]
            provider.close_error = None
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            assert provider.closed == 1
            assert server._sessions == {}
        finally:
            await asyncio.gather(tool_call, return_exceptions=True)
            _LocalFakeProvider.instances[-1].close_error = None
            await server._shutdown_all_sessions()

    async def test_close_releases_lock_after_failed_startup(self, monkeypatch):
        server = _make_sandbox_server()

        async def failed_create(provider, spec):
            raise RuntimeError("sandbox startup failed")

        monkeypatch.setattr(_LocalFakeProvider, "create", failed_create)
        with pytest.raises(RuntimeError, match="sandbox startup failed"):
            await _run_code(server, "rs-1", "print(1)")
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        assert server._session_locks == {}
        assert server._sessions == {}

    async def test_cancelled_startup_and_close_keep_allocation_owned_until_stop(self, monkeypatch):
        server = _make_sandbox_server()
        allocated, finish_start = asyncio.Event(), asyncio.Event()
        startup_cancelled = False
        real_create = _LocalFakeProvider.create

        async def delayed_create(provider, spec):
            nonlocal startup_cancelled
            handle = await real_create(provider, spec)
            allocated.set()
            try:
                await finish_start.wait()
            except asyncio.CancelledError:
                startup_cancelled = True
                raise
            return handle

        monkeypatch.setattr(_LocalFakeProvider, "create", delayed_create)
        await server.seed_session(
            ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=_FakeRequest("rs-1")
        )
        tool_call = asyncio.create_task(_run_code(server, "rs-1", "print(1)"))
        close_body = ResourcesCloseSessionRequest.model_validate(_close("rs-1"))
        try:
            await asyncio.wait_for(allocated.wait(), timeout=5)
            tool_call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tool_call
            assert not startup_cancelled
            assert server.has_session_state("rs-1")
            provider = _LocalFakeProvider.instances[-1]

            close = asyncio.create_task(server.close_resources_session(close_body))
            await asyncio.sleep(0)
            assert not close.done()
            assert provider.closed == provider.aclosed == 0
            close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close
            assert not startup_cancelled
            assert server.has_session_state("rs-1")
            assert not server.session_is_closed("rs-1")

            provider.close_error = RuntimeError("sandbox API unavailable")
            finish_start.set()
            with pytest.raises(RuntimeError, match="sandbox API unavailable"):
                await server.close_resources_session(close_body)
            assert server.has_session_state("rs-1")
            assert not server.session_is_closed("rs-1")
            assert provider.created == 1

            provider.close_error = None
            await server.close_resources_session(close_body)
            assert provider.closed == provider.aclosed == 1
            assert server.session_is_closed("rs-1")
            assert server._sessions == server._session_locks == {}
        finally:
            finish_start.set()
            await asyncio.gather(tool_call, return_exceptions=True)
            for provider in _LocalFakeProvider.instances:
                provider.close_error = None
            await server._shutdown_all_sessions()

    async def test_failed_startup_keeps_sandbox_until_stop_succeeds(self, monkeypatch):
        server = _make_sandbox_server()
        real_upload = _LocalFakeProvider.upload_file

        async def failed_upload(provider, handle, source_path, target_path):
            provider.close_error = RuntimeError("sandbox stop failed")
            raise RuntimeError("driver upload failed")

        monkeypatch.setattr(_LocalFakeProvider, "upload_file", failed_upload)
        with pytest.raises(RuntimeError, match="sandbox stop failed"):
            await _run_code(server, "rs-1", "print(1)")
        pending = server._sessions["rs-1"]
        assert not pending.ready
        provider = _LocalFakeProvider.instances[-1]
        providers_before = len(_LocalFakeProvider.instances)

        # A tool retry must not replace the incomplete sandbox or execute a cell in it.
        with pytest.raises(RuntimeError, match="sandbox stop failed"):
            await _run_code(server, "rs-1", "print(2)")
        assert server._sessions["rs-1"] is pending
        assert len(_LocalFakeProvider.instances) == providers_before
        assert pending.cells == []
        with pytest.raises(RuntimeError, match="sandbox stop failed"):
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        assert not server.session_is_closed("rs-1")

        provider.close_error = None
        monkeypatch.setattr(_LocalFakeProvider, "upload_file", real_upload)
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        assert provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}

    def test_verify_then_close_stops_the_sandbox_once(self):
        server, client = _session_client()
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(1)"}).status_code == 200
        provider = _LocalFakeProvider.instances[-1]

        verified = client.post("/verify", json=VERIFY_BODY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        assert provider.closed == 1
        assert server._sessions == {}

    def test_typed_seed_cannot_take_over_existing_cookie_sandbox(self) -> None:
        server, client = _session_client()
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "value = 42"}).status_code == 200
        [session_id] = server._sessions
        session = server._sessions[session_id]
        provider = _LocalFakeProvider.instances[-1]
        try:
            assert client.post("/seed_session", json=_seed(session_id)).status_code == 409
            assert server._session_episodes == {}
            assert server._sessions[session_id] is session
            assert provider.created == 1 and provider.closed == 0
            assert client.post("/stateful_python_code_exec", json={"code": "print(value)"}).text.strip() == "42"
        finally:
            assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        assert server._sessions == {} and provider.closed == 1

    def test_cookie_seed_keeps_working(self):
        server, client = _session_client()

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY)
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/stateful_python_code_exec", json={"code": "print(2)"}).text.strip() == "2"
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        assert server._sessions == {}

    def test_cookie_reseed_retries_failed_sandbox_stop_before_reset(self) -> None:
        server, client = _session_client(raise_server_exceptions=False)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "value = 42"}).status_code == 200
        [session_id] = server._sessions
        session = server._sessions[session_id]
        provider = _LocalFakeProvider.instances[-1]
        provider.close_error = RuntimeError("sandbox still running")
        try:
            assert client.post("/seed_session", json=VERIFY_BODY).status_code == 500
            assert server._sessions[session_id] is session and provider.closed == 0
            assert client.post("/stateful_python_code_exec", json={"code": "print(value)"}).text.strip() == "42"
        finally:
            provider.close_error = None
            assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert server._sessions == {} and provider.closed == 1
        try:
            response = client.post("/stateful_python_code_exec", json={"code": "print('value' in globals())"})
            assert response.text.strip() == "False"
            assert list(server._sessions) == [session_id]
        finally:
            assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0

    async def test_shutdown_stops_the_rest_after_a_failed_stop(self, caplog):
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print(1)")
        failing = _LocalFakeProvider.instances[-1]
        await _run_code(server, "s2", "print(2)")
        healthy = _LocalFakeProvider.instances[-1]
        failing.close_error = RuntimeError("sandbox API unavailable")

        await server._shutdown_all_sessions()

        assert healthy.closed == 1
        assert list(server._sessions) == ["s1"]
        assert "litmus_agent session s1 at shutdown" in caplog.text
        assert "session s2" not in caplog.text

    async def test_shutdown_stops_sandboxes_concurrently(self, monkeypatch):
        # The server supervisor kills the process about a second after asking it to stop, so stops cannot queue.
        server = _make_sandbox_server()
        await _run_code(server, "s1", "print(1)")
        await _run_code(server, "s2", "print(2)")
        stopping = 0
        both_stopping = asyncio.Event()
        real_close = _LocalFakeProvider.close

        async def close_once_both_are_stopping(provider, handle):
            nonlocal stopping
            stopping += 1
            if stopping == 2:
                both_stopping.set()
            await asyncio.wait_for(both_stopping.wait(), timeout=1)
            await real_close(provider, handle)

        monkeypatch.setattr(_LocalFakeProvider, "close", close_once_both_are_stopping)
        await server._shutdown_all_sessions()

        assert server._sessions == {}
        assert sum(provider.closed for provider in _LocalFakeProvider.instances[-2:]) == 2

    async def test_shutdown_during_startup_waits_then_stops_the_sandbox(self, monkeypatch):
        server = _make_sandbox_server()
        creating = asyncio.Event()
        finish_create = asyncio.Event()
        real_create = _LocalFakeProvider.create

        async def delayed_create(provider, spec):
            creating.set()
            await finish_create.wait()
            return await real_create(provider, spec)

        monkeypatch.setattr(_LocalFakeProvider, "create", delayed_create)
        tool_call = asyncio.create_task(_run_code(server, "s1", "print(1)"))
        await creating.wait()
        shutdown = asyncio.create_task(server._shutdown_all_sessions())
        await asyncio.sleep(0.05)
        # Stopping now would only release the client, and the container the start then creates would leak.
        assert not shutdown.done()
        finish_create.set()

        await shutdown
        provider = _LocalFakeProvider.instances[-1]
        assert provider.created == 1 and provider.closed == 1
        assert server._sessions == {} and server._session_locks == {}
        with pytest.raises(ResourcesSessionLostError):
            await tool_call


def test_base_verify_route_masks_judge_failure_and_releases_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.verification import JudgeError

    async def fail_score(self, body):
        raise JudgeError("judge unavailable")

    monkeypatch.setattr(LitmusAgentResourcesServer, "_score", fail_score)
    server = _make_server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        response = client.post("/verify", json=VERIFY_BODY)
        assert response.status_code == 200
        assert response.json()["mask_sample"] is True
        assert response.json()["failure_kind"] == "judge_failed"
        assert server._session_episodes == {}
        assert server.session_is_verified("rs-1")
