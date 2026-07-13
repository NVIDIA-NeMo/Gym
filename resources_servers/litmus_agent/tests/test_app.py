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

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    list_providers,
    register_provider,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
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
    def test_supported_set(self):
        assert _SUPPORTED_ANSWER_TYPES == {FLOAT, BOOL, STRING}

    def test_explicit_answer_type_wins(self):
        assert resolve_answer_type(FLOAT, {"property_type": "float"}) == FLOAT

    def test_unsupported_answer_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported answer_type='bogus'"):
            resolve_answer_type("bogus", {})

    @pytest.mark.parametrize(
        ("property_type", "expected"),
        [
            ("float", FLOAT),
            ("count", FLOAT),
            ("fragment", FLOAT),
            ("bool", BOOL),
            ("presence", BOOL),
        ],
    )
    def test_legacy_property_type_mapping(self, property_type, expected):
        assert resolve_answer_type(None, {"property_type": property_type}) == expected

    def test_missing_and_unmappable_raises(self):
        with pytest.raises(ValueError, match="not mappable"):
            resolve_answer_type(None, {"property_type": "mystery"})

    def test_missing_entirely_raises(self):
        with pytest.raises(ValueError, match="not mappable"):
            resolve_answer_type(None, {})


# ---------------------------------------------------------------------------
# extract_predicted_value
# ---------------------------------------------------------------------------


class TestExtractPredictedValue:
    def test_double_parens_integer(self):
        assert extract_predicted_value("The answer is ((42))", FLOAT) == 42.0

    def test_double_parens_last_occurrence_wins(self):
        assert extract_predicted_value("First ((3)), actually ((5))", FLOAT) == 5.0

    def test_boxed_via_use_box_format(self):
        assert extract_predicted_value(r"\boxed{12}", FLOAT, use_box_format=True) == 12.0

    def test_default_double_parens_when_no_format(self):
        assert extract_predicted_value("((7))", FLOAT, use_box_format=False) == 7.0

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

    def test_answer_format_overrides_use_box_format(self):
        text = r"Ignore \boxed{7}. Final Answer = 42"
        assert extract_predicted_value(text, FLOAT, answer_format="fmt_28", use_box_format=True) == 42.0

    def test_all_formats_registered(self):
        assert set(_ANSWER_FORMAT_REGEXES) == {f"fmt_{i:02d}" for i in range(31)}


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

    async def test_legacy_property_type_passthrough(self):
        server = _make_server()
        result = await server.verify(_make_verify_request("((1))", expected_answer="1", property_type="fragment"))
        assert result.reward == 1.0
        assert result.resolved_answer_type == FLOAT

    async def test_passthrough_fields_echoed(self):
        server = _make_server()
        result = await server.verify(
            _make_verify_request("((3))", expected_answer="3", answer_type=FLOAT, smiles="CCO")
        )
        assert (result.model_extra or {}).get("smiles") == "CCO"

    async def test_unresolvable_answer_type_raises(self):
        server = _make_server()
        with pytest.raises(ValueError, match="not mappable"):
            await server.verify(_make_verify_request("((3))", expected_answer="3"))

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

    def test_get_key_metrics_filters(self):
        server = _make_server()
        out = server.get_key_metrics({"mean/reward": 0.5, "mean/correct": 0.5, "other": 9})
        assert out == {"mean/reward": 0.5, "mean/correct": 0.5}


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
