# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit + smoke tests for the VCQA agent.

The interesting bits:

- `test_judge_grade_*` covers the rubric-to-reward math and the YES/NO parser.
- `test_dispatch_tool_*` exercises the tool dispatcher's error-handling paths
  with a fake `ApptainerSandbox` whose `exec` is monkeypatched; apptainer
  does not need to be on PATH for these tests.
- `test_path_resolution_*` locks down the path-containment helper so a model
  trying `/etc/passwd` or `../../etc` always sees an error rather than a leak.
- `test_run_end_to_end_*` boots a real Apptainer instance against a tiny
  fixture working tree, mocks the model server (via ServerClient) and the
  judge HTTP call (via the global aiohttp request fn), and verifies the full
  /run pipeline produces a `reward` in [0, 1] with no exceptions. This is
  skipped when `apptainer` is not on PATH.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from responses_api_agents.vcqa_agent import judge as judge_mod
from responses_api_agents.vcqa_agent import sandbox as sandbox_mod
from responses_api_agents.vcqa_agent import tools as tools_mod
from responses_api_agents.vcqa_agent.app import (
    VcqaAgent,
    VcqaAgentConfig,
    _extract_final_assistant_text,
    _extract_problem_statement,
)
from responses_api_agents.vcqa_agent.sandbox import (
    CODEBASE_PATH,
    ApptainerDirectSandbox,
    ApptainerExecResult,
    ApptainerSandbox,
    ExecResult,
)


########################################
# Helpers
########################################


def _make_config(**overrides) -> VcqaAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="vcqa_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        container_image="docker://debian:bookworm-slim",
        artifact_url_prefix="https://example.invalid/vcqa-artifacts",
        judge_base_url="https://example.invalid/v1",
        judge_api_key="dummy",  # pragma: allowlist secret
        judge_model_name="judge-model",
        max_turns=4,
    )
    base.update(overrides)
    return VcqaAgentConfig(**base)


class FakeSandbox:
    """Stand-in for ApptainerSandbox that records exec calls and replays canned results."""

    codebase_path: str = "/codebase"
    scratch_path: str = "/tmp/scratch"
    todos_path: str = "/tmp/scratch/todos.md"

    def __init__(self, results: dict[str, ApptainerExecResult] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, int | None]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def exec(self, command: str, timeout_s: int | None = None) -> ApptainerExecResult:
        self.calls.append((command, timeout_s))
        for key, value in self.results.items():
            if key in command:
                return value
        return ApptainerExecResult(exit_code=0, stdout="", stderr="")


########################################
# Config hygiene
########################################


def test_config_uses_package_paths_and_hf_artifacts() -> None:
    import yaml

    config_path = Path(__file__).parents[1] / "configs" / "vcqa_agent.yaml"
    data = yaml.safe_load(config_path.read_text())
    config = data["vcqa_agent"]["responses_api_agents"]["vcqa_agent"]

    assert config["entrypoint"] == "app.py"
    assert config["artifact_url_prefix"] == "https://huggingface.co/datasets/appliedcompute/vcqa-v1/resolve/v1.0.0"
    assert config["artifact_request_headers"] == {"Authorization": "Bearer ${oc.env:HF_TOKEN}"}

    jsonl_paths = [dataset["jsonl_fpath"] for dataset in config["datasets"]]
    assert jsonl_paths == [
        "responses_api_agents/vcqa_agent/data/example.jsonl",
        "responses_api_agents/vcqa_agent/data/train.jsonl",
        "responses_api_agents/vcqa_agent/data/test.jsonl",
    ]


########################################
# Path resolution
########################################


class TestPathResolution:
    def test_relative_path_resolves_under_codebase(self) -> None:
        assert tools_mod._resolve_under_codebase("src/foo.py") == f"{CODEBASE_PATH}/src/foo.py"

    def test_empty_path_returns_codebase_root(self) -> None:
        assert tools_mod._resolve_under_codebase("") == CODEBASE_PATH

    def test_dot_segments_are_normalized(self) -> None:
        assert tools_mod._resolve_under_codebase("./src/./foo.py") == f"{CODEBASE_PATH}/src/foo.py"

    def test_double_dot_within_codebase_is_allowed(self) -> None:
        assert tools_mod._resolve_under_codebase("src/utils/../foo.py") == f"{CODEBASE_PATH}/src/foo.py"

    def test_double_dot_escaping_returns_none(self) -> None:
        assert tools_mod._resolve_under_codebase("../../etc/passwd") is None

    def test_relative_path_to_sibling_prefix_returns_none(self) -> None:
        assert tools_mod._resolve_under_codebase("../codebase_evil/secret.txt") is None

    def test_absolute_path_outside_codebase_returns_none(self) -> None:
        assert tools_mod._resolve_under_codebase("/etc/passwd") is None

    def test_absolute_path_inside_codebase_is_kept(self) -> None:
        assert tools_mod._resolve_under_codebase("/codebase/src/foo.py") == "/codebase/src/foo.py"

    def test_resolution_under_arbitrary_codebase_path(self) -> None:
        # LocalSandbox uses an absolute working-tree path as `codebase_path`.
        root = "/tmp/vcqa_local/working_tree"
        assert tools_mod._resolve_under_codebase("src/foo.py", root) == f"{root}/src/foo.py"
        assert tools_mod._resolve_under_codebase("../../etc/passwd", root) is None
        assert tools_mod._resolve_under_codebase("../working_tree_evil/secret.txt", root) is None
        assert tools_mod._resolve_under_codebase("/etc/passwd", root) is None
        assert tools_mod._resolve_under_codebase(f"{root}/src/foo.py", root) == f"{root}/src/foo.py"


########################################
# Local sandbox
########################################


class TestLocalSandbox:
    async def test_local_sandbox_runs_commands_in_working_tree(self, tmp_path: Path) -> None:
        from responses_api_agents.vcqa_agent.sandbox import LocalSandbox

        working_tree = tmp_path / "wt"
        working_tree.mkdir()
        (working_tree / "answer.txt").write_text("forty-two\n")
        scratch = tmp_path / "scratch"

        sandbox = LocalSandbox(working_tree=working_tree, scratch_dir=scratch, exec_timeout_s=10)
        await sandbox.start()
        try:
            res = await sandbox.exec("cat -- answer.txt")
            assert res.exit_code == 0
            assert "forty-two" in res.stdout
            assert sandbox.codebase_path == str(working_tree)
            assert sandbox.todos_path == str(scratch / "todos.md")
        finally:
            await sandbox.stop()


########################################
# Direct Apptainer sandbox
########################################


class TestApptainerDirectSandbox:
    async def test_direct_sandbox_binds_codebase_and_scratch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        working_tree = tmp_path / "repo"
        scratch = tmp_path / "scratch"
        working_tree.mkdir()

        calls: list[tuple[str, list[str], int]] = []

        async def fake_run_subprocess(cmd: list[str], timeout_s: int) -> ExecResult:
            calls.append(("run", cmd, timeout_s))
            return ExecResult(exit_code=0, stdout="", stderr="")

        async def fake_exec_with_timeout(cmd: list[str], *, timeout_s: int, cwd: str | None) -> ExecResult:
            assert cwd is None
            calls.append(("exec", cmd, timeout_s))
            return ExecResult(exit_code=0, stdout="ok", stderr="")

        monkeypatch.setattr(sandbox_mod, "_run_subprocess", fake_run_subprocess)
        monkeypatch.setattr(sandbox_mod, "_exec_with_timeout", fake_exec_with_timeout)

        sandbox = ApptainerDirectSandbox(
            container_image="/images/vcqa-tools.sif",
            working_tree=working_tree,
            scratch_dir=scratch,
            exec_timeout_s=17,
        )

        await sandbox.start()
        result = await sandbox.exec("git status")

        assert result.stdout == "ok"
        assert calls[0][0] == "run"
        assert calls[0][2] == 60
        assert calls[1][0] == "exec"
        assert calls[1][2] == 17

        exec_cmd = calls[1][1]
        assert exec_cmd[:2] == ["apptainer", "exec"]
        assert "--writable-tmpfs" in exec_cmd
        assert f"{working_tree}:/codebase:ro" in exec_cmd
        assert f"{scratch}:/tmp/scratch:rw" in exec_cmd
        assert "/images/vcqa-tools.sif" in exec_cmd
        assert exec_cmd[-3:] == ["bash", "-lc", "git status"]

    async def test_direct_sandbox_rejects_setup_command(self, tmp_path: Path) -> None:
        sandbox = ApptainerDirectSandbox(
            container_image="/images/vcqa-tools.sif",
            working_tree=tmp_path,
            scratch_dir=tmp_path / "scratch",
            extra_setup_command="apt-get update",
        )

        with pytest.raises(RuntimeError, match="preinstalled"):
            await sandbox.start()


########################################
# Tool dispatch
########################################


class TestDispatchTool:
    async def test_unknown_tool_returns_error(self) -> None:
        sandbox = FakeSandbox()
        result = await tools_mod.dispatch_tool("not_a_tool", {}, sandbox)  # type: ignore[arg-type]
        assert result["result"] is None
        assert "unknown tool" in result["error"]

    async def test_invalid_args_returns_error(self) -> None:
        sandbox = FakeSandbox()
        result = await tools_mod.dispatch_tool("read_file", {}, sandbox)  # type: ignore[arg-type]
        assert result["result"] is None
        assert "invalid arguments" in result["error"]

    async def test_path_escape_returns_error(self) -> None:
        sandbox = FakeSandbox()
        result = await tools_mod.dispatch_tool(
            "read_file",
            {"path": "/etc/passwd"},
            sandbox,  # type: ignore[arg-type]
        )
        assert result["result"] is None
        assert "outside the working tree" in result["error"]
        assert sandbox.calls == []

    async def test_read_file_returns_truncated_output(self) -> None:
        big = "X" * (tools_mod.MAX_TOOL_OUTPUT_BYTES + 100)
        sandbox = FakeSandbox(
            results={
                "head -c": ApptainerExecResult(exit_code=0, stdout=big, stderr=""),
            }
        )
        result = await tools_mod.dispatch_tool(
            "read_file",
            {"path": "src/foo.py"},
            sandbox,  # type: ignore[arg-type]
        )
        assert result["error"] is None
        assert result["result"]["truncated"] is True
        assert len(result["result"]["text"]) <= tools_mod.MAX_TOOL_OUTPUT_BYTES

    async def test_grep_handles_no_matches(self) -> None:
        # rg returns exit code 1 when there are no matches, which is not an error here.
        sandbox = FakeSandbox(results={"rg ": ApptainerExecResult(exit_code=1, stdout="", stderr="")})
        result = await tools_mod.dispatch_tool(
            "grep",
            {"pattern": "needle"},
            sandbox,  # type: ignore[arg-type]
        )
        assert result["error"] is None
        assert result["result"]["text"] == ""

    async def test_grep_propagates_real_errors(self) -> None:
        sandbox = FakeSandbox(
            results={
                "rg ": ApptainerExecResult(exit_code=2, stdout="", stderr="rg: bad regex"),
            }
        )
        result = await tools_mod.dispatch_tool(
            "grep",
            {"pattern": "[invalid"},
            sandbox,  # type: ignore[arg-type]
        )
        assert result["result"] is None
        assert "grep failed" in result["error"]

    async def test_bash_timeout_surfaces_in_result(self) -> None:
        sandbox = FakeSandbox(
            results={"sleep": ApptainerExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)}
        )
        result = await tools_mod.dispatch_tool(
            "bash",
            {"command": "sleep 100", "timeout": 1},
            sandbox,  # type: ignore[arg-type]
        )
        # bash never sets `error` on timeout; it surfaces it via the structured result.
        assert result["error"] is None
        assert result["result"]["timed_out"] is True
        assert result["result"]["exit_code"] == -1

    async def test_handler_exception_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("kaboom")

        monkeypatch.setitem(tools_mod._DISPATCH, "read_file", boom)  # type: ignore[arg-type]
        result = await tools_mod.dispatch_tool(
            "read_file",
            {"path": "src/foo.py"},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["result"] is None
        assert "RuntimeError" in result["error"]
        assert "kaboom" in result["error"]


########################################
# Judge
########################################


class TestJudgeParse:
    def test_yes_no_parser_handles_punctuation_and_case(self) -> None:
        assert judge_mod.parse_yes_no("YES") is True
        assert judge_mod.parse_yes_no("no") is False
        assert judge_mod.parse_yes_no("Yes.") is True
        assert judge_mod.parse_yes_no("\nNO\nbecause...") is False

    def test_yes_no_parser_returns_none_on_garbage(self) -> None:
        assert judge_mod.parse_yes_no("maybe") is None
        assert judge_mod.parse_yes_no("") is None

    def test_extract_must_have_only(self) -> None:
        rubric = {
            "judge": [
                {"category": "must_have", "description": "A"},
                {"category": "good_to_have", "description": "B"},
                {"category": "MUST_HAVE", "description": "C"},
                {"category": "must_have", "description": "  "},  # skipped: empty desc
                {"category": "must_have"},  # skipped: no desc
            ]
        }
        items = judge_mod.extract_must_have_items(rubric)
        assert [i.description for i in items] == ["A", "C"]

    def test_extract_vcqa_v1_schema(self) -> None:
        # Production rows use `importance: "must have"` + `title` + optional
        # `evidence_required`, and lack the legacy `category` field. This
        # regression-tests against that schema.
        rubric = {
            "judge": [
                {
                    "id": "j01",
                    "title": "Mentions OpenOptionsExt imports.",
                    "importance": "must have",
                    "evidence_required": ["tokio/src/fs/open_options.rs", "lines 19-22"],
                },
                {
                    "id": "j02",
                    "title": "Lower-priority detail",
                    "importance": "good to have",
                },
                {
                    "id": "j03",
                    "title": "  ",  # skipped: blank
                    "importance": "must have",
                },
            ]
        }
        items = judge_mod.extract_must_have_items(rubric)
        assert len(items) == 1
        assert items[0].description.startswith("Mentions OpenOptionsExt imports.")
        # Evidence must show up in the criterion so the judge sees it.
        assert "tokio/src/fs/open_options.rs" in items[0].description
        assert "lines 19-22" in items[0].description

    def test_extract_must_have_handles_missing_rubric(self) -> None:
        assert judge_mod.extract_must_have_items(None) == []
        assert judge_mod.extract_must_have_items({}) == []
        assert judge_mod.extract_must_have_items({"judge": []}) == []


class TestJudgeGrade:
    async def test_no_must_have_items_returns_zero_with_error(self) -> None:
        result = await judge_mod.grade(
            rubric={"judge": []},
            problem_statement="",
            model_answer="anything",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
        )
        assert result.reward == 0.0
        assert result.must_total == 0
        assert "no must_have" in (result.error or "")

    async def test_empty_answer_short_circuits(self) -> None:
        result = await judge_mod.grade(
            rubric={"judge": [{"category": "must_have", "description": "X"}]},
            problem_statement="",
            model_answer="   ",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
        )
        assert result.reward == 0.0
        assert result.must_total == 1
        assert result.must_pass == 0
        assert result.error == "empty model answer"

    async def test_partial_pass_reward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        replies = ["YES", "NO", "YES"]
        call_count = {"i": 0}

        async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            i = call_count["i"]
            call_count["i"] += 1
            text = replies[i]
            response = MagicMock()
            response.ok = True
            response.status = 200
            payload = {"choices": [{"message": {"content": text}}]}
            response.read = AsyncMock(return_value=json.dumps(payload).encode())
            return response

        monkeypatch.setattr(judge_mod, "request", fake_request)

        rubric = {
            "judge": [
                {"category": "must_have", "description": "A"},
                {"category": "must_have", "description": "B"},
                {"category": "must_have", "description": "C"},
            ]
        }
        result = await judge_mod.grade(
            rubric=rubric,
            problem_statement="problem",
            model_answer="answer",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
        )
        assert result.must_total == 3
        assert result.must_pass == 2
        assert result.reward == pytest.approx(2 / 3)
        assert all(r.get("error") is None for r in result.per_item)

    async def test_judge_http_error_marks_item_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            response.ok = False
            response.status = 503
            response.content.read = AsyncMock(return_value=b"upstream error")
            return response

        monkeypatch.setattr(judge_mod, "request", fake_request)

        result = await judge_mod.grade(
            rubric={"judge": [{"category": "must_have", "description": "X"}]},
            problem_statement="problem",
            model_answer="answer",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
        )
        assert result.reward == 0.0
        assert result.must_pass == 0
        assert "HTTP 503" in (result.per_item[0]["error"] or "")


########################################
# Response extraction helpers
########################################


class TestResponseExtractionHelpers:
    def test_extract_final_assistant_text(self) -> None:
        from nemo_gym.core import (
            NeMoGymResponse,
            NeMoGymResponseFunctionToolCall,
            NeMoGymResponseOutputMessage,
            NeMoGymResponseOutputText,
        )

        resp = NeMoGymResponse(
            id="resp",
            created_at=0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseFunctionToolCall(arguments="{}", call_id="c1", name="read_file", type="function_call"),
                NeMoGymResponseOutputMessage(
                    id="m1",
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(annotations=[], text="The answer is 42."),
                    ],
                ),
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        assert _extract_final_assistant_text(resp) == "The answer is 42."

    def test_extract_problem_statement_string_input(self) -> None:
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

        params = NeMoGymResponseCreateParamsNonStreaming(input="hello world")
        assert _extract_problem_statement(params) == "hello world"


########################################
# End-to-end /run smoke (Apptainer-backed, skipped when missing)
########################################


REQUIRES_APPTAINER = pytest.mark.skipif(
    shutil.which("apptainer") is None,
    reason="apptainer is not installed on this host",
)


@REQUIRES_APPTAINER
class TestRunEndToEnd:
    """Boots a real Apptainer instance and drives /run with mocked model+judge HTTP."""

    async def test_run_against_local_fixture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import Request

        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.vcqa_agent.app import VcqaAgentRunRequest

        fixture = tmp_path / "fixture_repo"
        fixture.mkdir()
        (fixture / "answer.txt").write_text("forty-two\n")

        async def fake_materialize(**kwargs):  # type: ignore[no-untyped-def]
            return fixture

        monkeypatch.setattr(
            "responses_api_agents.vcqa_agent.app.materialize_working_tree",
            fake_materialize,
        )

        mock_model_response = {
            "id": "resp_1",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "output": [
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [
                        {
                            "annotations": [],
                            "text": "After reading answer.txt the answer is forty-two.",
                            "type": "output_text",
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        post_response = AsyncMock()
        post_response.ok = True
        post_response.read = AsyncMock(return_value=json.dumps(mock_model_response).encode())
        post_response.cookies = {}

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=post_response)

        async def fake_judge_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            response.ok = True
            response.status = 200
            payload = {"choices": [{"message": {"content": "YES"}}]}
            response.read = AsyncMock(return_value=json.dumps(payload).encode())
            return response

        monkeypatch.setattr(judge_mod, "request", fake_judge_request)

        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        body = VcqaAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "What is the answer?"}]
            ),
            verifier_metadata={
                "task_id": "smoke-1",
                "dataset_kind": "fileset",
                "artifact_key": "ignored.tar.gz",
                "rubric": {"judge": [{"category": "must_have", "description": "Says forty-two."}]},
                "max_turns": 2,
            },
        )

        request = MagicMock(spec=Request)
        request.cookies = {}
        response = MagicMock()

        result = await agent.run(request=request, response=response, body=body)
        assert 0.0 <= result.reward <= 1.0
        assert result.reward == 1.0
        assert result.must_pass == 1
        assert result.must_total == 1
        assert result.error is None
        assert "forty-two" in (result.final_answer or "").lower()


########################################
# Failure-isolation
########################################


class TestRunFailureIsolation:
    async def test_run_returns_zero_on_materialize_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import Request

        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.vcqa_agent.app import VcqaAgentRunRequest

        async def boom(**kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("download failed")

        monkeypatch.setattr("responses_api_agents.vcqa_agent.app.materialize_working_tree", boom)

        agent = VcqaAgent(
            config=_make_config(),
            server_client=MagicMock(spec=ServerClient),
        )

        body = VcqaAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "anything"}]
            ),
            verifier_metadata={
                "task_id": "isolation-1",
                "dataset_kind": "fileset",
                "artifact_key": "missing.tar.gz",
                "rubric": {"judge": [{"category": "must_have", "description": "X"}]},
            },
        )

        request = MagicMock(spec=Request)
        request.cookies = {}
        response = MagicMock()

        result = await agent.run(request=request, response=response, body=body)
        assert result.reward == 0.0
        assert "RuntimeError" in (result.error or "")
        assert "download failed" in (result.error or "")
        assert result.must_total is None or result.must_total == 0


########################################
# Dispatcher synchronous bits
########################################


_REAL_TOOL_NAMES = ["read_file", "grep", "glob", "list_dir", "write_todos", "bash"]
_DISTRACTOR_TOOL_NAMES = ["install_package", "send_pr_review", "websearch", "ask_user"]


def test_build_tool_definitions_includes_distractors_by_default() -> None:
    defs = tools_mod.build_tool_definitions()
    names = sorted(d["name"] for d in defs)
    assert names == sorted(_REAL_TOOL_NAMES + _DISTRACTOR_TOOL_NAMES)
    for d in defs:
        assert d["type"] == "function"
        assert "description" in d
        assert "parameters" in d
        # OpenAI Responses API requires `strict` on every function tool.
        assert d["strict"] is False


def test_build_tool_definitions_can_drop_distractors() -> None:
    defs = tools_mod.build_tool_definitions(include_distractors=False)
    names = sorted(d["name"] for d in defs)
    assert names == sorted(_REAL_TOOL_NAMES)
    assert not any(d["name"] in _DISTRACTOR_TOOL_NAMES for d in defs)


def test_distractor_tool_names_constant_matches_registered_set() -> None:
    # Defensive: the public `DISTRACTOR_TOOL_NAMES` constant is what other
    # call sites (metrics, A/B harnesses) key off; keep it in sync with
    # both the arg-model registry and the dispatcher.
    assert tools_mod.DISTRACTOR_TOOL_NAMES == frozenset(_DISTRACTOR_TOOL_NAMES)
    for name in _DISTRACTOR_TOOL_NAMES:
        assert name in tools_mod._TOOL_ARG_MODELS
        assert name in tools_mod._DISPATCH


class TestDistractorHandlers:
    """Distractors must never crash and must always return a non-empty
    text payload so the model has something to read."""

    async def test_install_package_returns_offline_error_text(self) -> None:
        result = await tools_mod.dispatch_tool(
            "install_package",
            {"name": "numpy", "manager": "pip"},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        assert "not available" in result["result"]["text"]

    async def test_send_pr_review_returns_offline_error_text(self) -> None:
        result = await tools_mod.dispatch_tool(
            "send_pr_review",
            {"comments": "LGTM", "verdict": "approve"},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        assert "offline" in result["result"]["text"]

    async def test_websearch_echoes_query_in_ungrounded_blurb(self) -> None:
        result = await tools_mod.dispatch_tool(
            "websearch",
            {"query": "rust lifetimes"},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        text = result["result"]["text"]
        # Tier-2 distractor: the response should look plausible (mention the
        # query) but offer nothing concrete that grounds the model's answer.
        assert "rust lifetimes" in text
        assert "Web search" in text

    async def test_websearch_handles_empty_query(self) -> None:
        result = await tools_mod.dispatch_tool(
            "websearch",
            {"query": "   "},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        assert "(empty query)" in result["result"]["text"]

    async def test_ask_user_reflects_question_and_signals_no_response(self) -> None:
        question = "What is the expected output here?"
        result = await tools_mod.dispatch_tool(
            "ask_user",
            {"question": question},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        text = result["result"]["text"]
        assert question in text
        assert "user is not available" in text

    async def test_ask_user_truncates_long_questions(self) -> None:
        question = "Why does " + ("X" * 200) + " misbehave?"
        result = await tools_mod.dispatch_tool(
            "ask_user",
            {"question": question},
            FakeSandbox(),  # type: ignore[arg-type]
        )
        assert result["error"] is None
        text = result["result"]["text"]
        # Preview cap is 80 chars + ellipsis.
        assert "..." in text
        assert len(question) > 80

    async def test_distractor_does_not_touch_sandbox(self) -> None:
        # All four distractors are sandbox-independent. If they ever start
        # `exec`ing things, this test will flag it.
        sandbox = FakeSandbox()
        for name, args in [
            ("install_package", {"name": "numpy"}),
            ("send_pr_review", {"comments": "x"}),
            ("websearch", {"query": "y"}),
            ("ask_user", {"question": "z"}),
        ]:
            await tools_mod.dispatch_tool(name, args, sandbox)  # type: ignore[arg-type]
        assert sandbox.calls == []

    async def test_distractor_rejects_invalid_args(self) -> None:
        # Pydantic gating still applies; the dispatcher won't pass through
        # malformed arguments to the handler.
        result = await tools_mod.dispatch_tool(
            "install_package",
            {},
            FakeSandbox(),  # missing required `name`
        )  # type: ignore[arg-type]
        assert result["result"] is None
        assert "invalid arguments" in result["error"]


def test_apptainer_sandbox_make_instance_name_is_unique() -> None:
    from responses_api_agents.vcqa_agent.sandbox import make_instance_name

    a, b = make_instance_name(), make_instance_name()
    assert a != b
    assert a.startswith("vcqa-")


async def test_apptainer_sandbox_cleans_up_instance_when_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_calls: list[list[str]] = []
    exec_commands: list[str] = []

    async def fake_run_subprocess(cmd: list[str], timeout_s: int) -> ExecResult:
        run_calls.append(cmd)
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def fake_exec_with_timeout(cmd: list[str], *, timeout_s: int, cwd: str | None) -> ExecResult:
        assert cwd is None
        exec_commands.append(cmd[-1])
        if cmd[-1] == "setup fails":
            return ExecResult(exit_code=1, stdout="", stderr="boom")
        return ExecResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(sandbox_mod, "_exec_with_timeout", fake_exec_with_timeout)

    sandbox = ApptainerSandbox(
        instance_name="vcqa-test",
        container_image="docker://debian:bookworm-slim",
        working_tree=tmp_path,
        scratch_dir=tmp_path / "scratch",
        extra_setup_command="setup fails",
    )

    with pytest.raises(RuntimeError, match="sandbox command failed"):
        await sandbox.start()

    assert exec_commands[-1] == "setup fails"
    assert any(cmd == ["apptainer", "instance", "stop", "vcqa-test"] for cmd in run_calls)
    assert sandbox.cleaned_up is True
    assert sandbox.started is False


def test_sandbox_stop_is_idempotent_when_never_started(tmp_path: Path) -> None:
    sandbox = ApptainerSandbox(
        instance_name="vcqa-test",
        container_image="docker://debian:bookworm-slim",
        working_tree=tmp_path,
        scratch_dir=tmp_path / "scratch",
    )
    asyncio.run(sandbox.stop())
    asyncio.run(sandbox.stop())


def test_fetch_artifact_threads_auth_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`fetch_artifact(headers=...)` must reach the underlying aiohttp `request`.

    Regression guard for the private-HF-Hub auth path: if the header dict
    isn't propagated, every artifact fetch from a private dataset repo will
    401 and turn into reward=0.0.
    """
    import responses_api_agents.vcqa_agent.materialize as materialize_mod

    captured: dict[str, object] = {}

    class FakeContent:
        async def iter_chunked(self, _n: int):
            for chunk in (b"hello ", b"world"):
                yield chunk

    class FakeResponse:
        content = FakeContent()

        def release(self) -> None:
            pass

    async def fake_request(**kwargs: object):
        captured.update(kwargs)
        return FakeResponse()

    async def fake_raise_for_status(_resp: object) -> None:
        pass

    monkeypatch.setattr(materialize_mod, "request", fake_request)
    monkeypatch.setattr(materialize_mod, "raise_for_status", fake_raise_for_status)

    dest = tmp_path / "out.bin"
    asyncio.run(
        materialize_mod.fetch_artifact(
            url="https://example.test/file",
            dest_path=dest,
            timeout_s=5,
            headers={"Authorization": "Bearer abc123"},  # pragma: allowlist secret
        )
    )
    assert dest.read_bytes() == b"hello world"
    assert captured["method"] == "GET"
    assert captured["url"] == "https://example.test/file"
    assert captured["headers"] == {"Authorization": "Bearer abc123"}  # pragma: allowlist secret


def test_fetch_artifact_omits_headers_when_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no headers are passed, no `headers=` kwarg is forwarded to the underlying request."""
    import responses_api_agents.vcqa_agent.materialize as materialize_mod

    captured: dict[str, object] = {}

    class FakeContent:
        async def iter_chunked(self, _n: int):
            for chunk in (b"x",):
                yield chunk

    class FakeResponse:
        content = FakeContent()

        def release(self) -> None:
            pass

    async def fake_request(**kwargs: object):
        captured.update(kwargs)
        return FakeResponse()

    async def fake_raise_for_status(_resp: object) -> None:
        pass

    monkeypatch.setattr(materialize_mod, "request", fake_request)
    monkeypatch.setattr(materialize_mod, "raise_for_status", fake_raise_for_status)

    asyncio.run(
        materialize_mod.fetch_artifact(
            url="https://example.test/file",
            dest_path=tmp_path / "out.bin",
            timeout_s=5,
        )
    )
    assert "headers" not in captured


########################################
# RL hardening: capture path, masking, judge retries, line ranges, containment
########################################


def _hardening_run_body(text_rubric: str = "Says forty-two.", **body_extra):  # type: ignore[no-untyped-def]
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.vcqa_agent.app import VcqaAgentRunRequest

    return VcqaAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is the answer?"}]
        ),
        verifier_metadata={
            "task_id": "hardening-1",
            "dataset_kind": "fileset",
            "artifact_key": "a.tar.gz",
            "rubric": {"judge": [{"category": "must_have", "description": text_rubric}]},
            "max_turns": 2,
        },
        **body_extra,
    )


def _answering_server_client(answer: str = "The answer is forty-two.") -> MagicMock:
    model_response = {
        "id": "resp_1",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [{"annotations": [], "text": answer, "type": "output_text"}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    post_response = AsyncMock()
    post_response.ok = True
    post_response.read = AsyncMock(return_value=json.dumps(model_response).encode())
    post_response.cookies = {}
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=post_response)
    return server_client


def _model_reply(output: list[dict]) -> AsyncMock:
    reply = AsyncMock()
    reply.ok, reply.status, reply.cookies = True, 200, {}
    payload = {
        "id": "resp_turn",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    reply.read = AsyncMock(return_value=json.dumps(payload).encode())
    return reply


def _model_error_reply(status: int, error: dict) -> AsyncMock:
    reply = AsyncMock()
    reply.ok, reply.status, reply.cookies = False, status, {}
    reply.read = AsyncMock(return_value=json.dumps({"error": {"type": "invalid_request_error", **error}}).encode())
    reply.raise_for_status = MagicMock(side_effect=RuntimeError(f"HTTP {status}"))
    return reply


def _scripted_judge(monkeypatch: pytest.MonkeyPatch, replies: list[tuple[int, str]]) -> list[dict]:
    """Judge replies in order (the last one repeats); returns the request bodies sent."""
    sent: list[dict] = []

    async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
        sent.append(json.loads(kwargs["data"]))
        status, content = replies[min(len(sent) - 1, len(replies) - 1)]
        response = MagicMock()
        response.ok = status == 200
        response.status = status
        payload = {"choices": [{"message": {"content": content}}]}
        response.read = AsyncMock(return_value=json.dumps(payload).encode())
        response.content = MagicMock()
        response.content.read = AsyncMock(return_value=content.encode())
        return response

    monkeypatch.setattr(judge_mod, "request", fake_request)
    return sent


async def _run_with_fake_sandbox(agent: VcqaAgent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body):  # type: ignore[no-untyped-def]
    from fastapi import Request

    async def fake_materialize(**kwargs):  # type: ignore[no-untyped-def]
        return tmp_path

    monkeypatch.setattr("responses_api_agents.vcqa_agent.app.materialize_working_tree", fake_materialize)
    monkeypatch.setattr(VcqaAgent, "_build_sandbox", lambda self, **kwargs: FakeSandbox())
    request = MagicMock(spec=Request)
    request.cookies = {}
    return await agent.run(request=request, response=MagicMock(), body=body)


async def test_local_sandboxes_do_not_share_writable_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import Request

    built: list[Path] = []

    async def fake_materialize(**kwargs):  # type: ignore[no-untyped-def]
        built.append(kwargs["scratch_dir"])
        await asyncio.sleep(0.05)  # both rollouts hold their trees at once
        return kwargs["scratch_dir"]

    monkeypatch.setattr("responses_api_agents.vcqa_agent.app.materialize_working_tree", fake_materialize)
    monkeypatch.setattr(VcqaAgent, "_build_sandbox", lambda self, **kwargs: FakeSandbox())
    _scripted_judge(monkeypatch, [(200, "YES")])
    agent = VcqaAgent(
        config=_make_config(sandbox_backend="local", share_working_trees=True),
        server_client=_answering_server_client(),
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    await asyncio.gather(
        *(agent.run(request=request, response=MagicMock(), body=_hardening_run_body()) for _ in range(2))
    )

    assert len(built) == 2 and built[0] != built[1]


class TestRunCapturePathAndMasking:
    async def test_model_calls_carry_the_training_capture_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server_client = _answering_server_client()
        server_client.global_config_dict = {"token_id_capture": {"enabled": True, "all_agents": True}}
        _scripted_judge(monkeypatch, [(200, "YES")])
        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body(_ng_rollout_id="r-7"))

        url_paths = [call.kwargs["url_path"] for call in server_client.post.await_args_list]
        assert url_paths == ["/ng-rollout/r-7/training-token-capture/v1/responses"]
        assert result.reward == 1.0
        assert result.evaluation_completed is True
        assert result.mask_sample is False

    async def test_model_calls_are_unprefixed_without_capture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server_client = _answering_server_client()
        _scripted_judge(monkeypatch, [(200, "YES")])
        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert server_client.post.await_args.kwargs["url_path"] == "/v1/responses"

    async def test_materialize_failure_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import Request

        async def boom(**kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("download failed")

        monkeypatch.setattr("responses_api_agents.vcqa_agent.app.materialize_working_tree", boom)
        agent = VcqaAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        request = MagicMock(spec=Request)
        request.cookies = {}

        result = await agent.run(request=request, response=MagicMock(), body=_hardening_run_body())

        assert result.reward == 0.0
        assert result.evaluation_completed is False
        assert result.mask_sample is True
        assert result.failure_stage == "materialize"
        assert result.failure_kind == "vcqa_agent:artifact_unavailable"
        assert "download failed" in (result.failure_reason or "")

    async def test_model_server_error_is_masked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("model server down"))
        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert result.evaluation_completed is False
        assert result.mask_sample is True
        assert result.failure_stage == "model"
        assert result.failure_kind == "agent_request_failed"

    @pytest.mark.parametrize("first_call", [False, True])
    async def test_context_refusal_ends_the_episode_as_a_measurement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_call: bool
    ) -> None:
        tool_turn = _model_reply(
            [{"id": "fc_1", "call_id": "c1", "type": "function_call", "name": "list_dir", "arguments": "{}"}]
        )
        refusal = _model_error_reply(
            400, {"message": "maximum context length is 32768 tokens", "code": "context_length_exceeded"}
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=[refusal] if first_call else [tool_turn, refusal])
        sent = _scripted_judge(monkeypatch, [(200, "YES")])
        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert result.evaluation_completed is True and result.mask_sample is False
        assert result.reward == 0.0 and sent == []  # no answer, so no judge call
        assert result.response.status == "incomplete"
        assert result.response.metadata["nemo_gym_context_window_exceeded"] == "true"
        expected = [] if first_call else ["function_call", "function_call_output"]
        assert [item.type for item in result.response.output] == expected

    async def test_other_bad_requests_stay_masked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            return_value=_model_error_reply(400, {"message": "bad tool", "code": "invalid_value"})
        )
        agent = VcqaAgent(config=_make_config(), server_client=server_client)

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert result.mask_sample is True and result.failure_stage == "model"

    async def test_judge_outage_is_masked_not_scored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(503, "unavailable")])
        agent = VcqaAgent(
            config=_make_config(judge_max_attempts=3, judge_retry_backoff_s=0.0),
            server_client=_answering_server_client(),
        )

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert len(sent) == 3
        assert result.reward == 0.0
        assert result.judge_errors == 1
        assert result.evaluation_completed is False
        assert result.mask_sample is True
        assert result.failure_kind == "judge_failed"

    async def test_empty_answer_is_a_valid_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(200, "YES")])
        agent = VcqaAgent(config=_make_config(), server_client=_answering_server_client(answer="   "))

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert sent == []
        assert result.reward == 0.0
        assert result.evaluation_completed is True
        assert result.mask_sample is False


class TestJudgeRetries:
    _RUBRIC = {"judge": [{"category": "must_have", "description": "A"}]}

    async def _grade(self, **kwargs):  # type: ignore[no-untyped-def]
        return await judge_mod.grade(
            rubric=self._RUBRIC,
            problem_statement="problem",
            model_answer="answer",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
            retry_backoff_s=0.0,
            **kwargs,
        )

    async def test_rate_limit_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(429, "slow down"), (200, "YES")])
        result = await self._grade(max_attempts=3)
        assert len(sent) == 2
        assert result.reward == 1.0
        assert result.judge_errors == 0
        assert result.per_item[0]["attempts"] == 2

    async def test_unparseable_reply_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _scripted_judge(monkeypatch, [(200, "Let me think about this."), (200, "NO")])
        result = await self._grade(max_attempts=2)
        assert result.reward == 0.0
        assert result.judge_errors == 0
        assert result.error is None

    async def test_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(400, "bad model name")])
        result = await self._grade(max_attempts=4)
        assert len(sent) == 1
        assert result.judge_errors == 1
        assert "HTTP 400" in result.per_item[0]["error"]

    async def test_any_server_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(520, "origin error"), (200, "YES")])
        result = await self._grade(max_attempts=2)
        assert len(sent) == 2
        assert result.reward == 1.0 and result.judge_errors == 0
        assert result.per_item[0]["attempts"] == 2

    @pytest.mark.parametrize("status", [200, 503])
    async def test_stalled_reply_body_times_out_and_frees_its_slot(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        never = asyncio.Event()
        responses: list[MagicMock] = []

        async def stall() -> bytes:
            await never.wait()
            return b""

        async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            if not responses:  # the first criterion's reply arrives but its body never does
                response.ok, response.status = status == 200, status
                response.read = stall
                response.content.read = stall
            else:
                response.ok, response.status = True, 200
                response.read = AsyncMock(
                    return_value=json.dumps({"choices": [{"message": {"content": "YES"}}]}).encode()
                )
            responses.append(response)
            return response

        monkeypatch.setattr(judge_mod, "request", fake_request)
        result = await judge_mod.grade(
            rubric={"judge": [{"category": "must_have", "description": d} for d in "AB"]},
            problem_statement="p",
            model_answer="a",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
            timeout_s=0.2,
            semaphore=asyncio.Semaphore(1),
        )
        assert [item["error"] for item in result.per_item] == ["judge timed out", None]
        assert result.judge_errors == 1 and result.must_pass == 1
        responses[0].close.assert_called_once()
        responses[1].release.assert_called_once()

    async def test_extra_body_is_merged_into_the_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = _scripted_judge(monkeypatch, [(200, "YES")])
        await self._grade(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent[0]["model"] == "judge-model"

    async def test_semaphore_bounds_concurrent_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = {"in_flight": 0, "peak": 0}

        async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            response = MagicMock()
            response.ok = True
            response.status = 200
            response.read = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "YES"}}]}).encode())
            return response

        monkeypatch.setattr(judge_mod, "request", fake_request)
        rubric = {"judge": [{"category": "must_have", "description": d} for d in "ABCDEF"]}
        result = await judge_mod.grade(
            rubric=rubric,
            problem_statement="p",
            model_answer="a",
            base_url="https://example.invalid/v1",
            api_key="dummy",  # pragma: allowlist secret
            model_name="judge-model",
            semaphore=asyncio.Semaphore(2),
        )
        assert result.reward == 1.0
        assert state["peak"] == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("YES", True),
        ("NO", False),
        ("yes.", True),
        ("**NO**", False),
        ("\n\nYES\nbecause it cites foo.go", True),
        ("NO - the answer never names the file", False),
        ("<think>the answer cites foo.go</think>\nYES", True),
        ("YESTERDAY", None),
        ("Maybe", None),
        ("", None),
    ],
)
def test_parse_yes_no_variants(text: str, expected: bool | None) -> None:
    assert judge_mod.parse_yes_no(text) is expected


class TestReadFileLineRanges:
    @pytest.fixture
    def sandbox(self, tmp_path: Path) -> sandbox_mod.LocalSandbox:
        (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 6)))
        return sandbox_mod.LocalSandbox(working_tree=tmp_path, scratch_dir=tmp_path / ".scratch")

    async def test_range_is_numbered_with_file_line_numbers(self, sandbox: sandbox_mod.LocalSandbox) -> None:
        out = await tools_mod.dispatch_tool("read_file", {"path": "f.txt", "start_line": 2, "end_line": 3}, sandbox)
        assert out["error"] is None
        text = out["result"]["text"]
        assert text.splitlines() == ["     2\tline2", "     3\tline3"]

    async def test_open_ended_range_reads_to_eof(self, sandbox: sandbox_mod.LocalSandbox) -> None:
        out = await tools_mod.dispatch_tool("read_file", {"path": "f.txt", "start_line": 4}, sandbox)
        assert out["result"]["text"].splitlines() == ["     4\tline4", "     5\tline5"]

    async def test_whole_file_read_is_unchanged(self, sandbox: sandbox_mod.LocalSandbox) -> None:
        out = await tools_mod.dispatch_tool("read_file", {"path": "f.txt"}, sandbox)
        assert out["result"]["text"].startswith("line1\nline2\n")

    async def test_inverted_range_is_an_error(self, sandbox: sandbox_mod.LocalSandbox) -> None:
        out = await tools_mod.dispatch_tool("read_file", {"path": "f.txt", "start_line": 4, "end_line": 2}, sandbox)
        assert out["result"] is None
        assert "before start_line" in out["error"]

    async def test_missing_file_with_range_is_an_error(self, sandbox: sandbox_mod.LocalSandbox) -> None:
        out = await tools_mod.dispatch_tool("read_file", {"path": "nope.txt", "start_line": 1}, sandbox)
        assert out["result"] is None
        assert "read_file failed" in out["error"]

    @pytest.mark.skipif(shutil.which("dash") is None, reason="needs dash, the /bin/sh of Debian sandbox images")
    async def test_range_runs_under_posix_sh(self, tmp_path: Path) -> None:
        # Provider sandboxes run a non-root user's command through `su -s /bin/sh`, which is dash on Debian images.
        (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 6)))
        sandbox = _DashSandbox(tmp_path)
        out = await tools_mod.dispatch_tool("read_file", {"path": "f.txt", "start_line": 2, "end_line": 3}, sandbox)
        assert out["error"] is None
        assert out["result"]["text"].splitlines() == ["     2\tline2", "     3\tline3"]
        missing = await tools_mod.dispatch_tool("read_file", {"path": "nope.txt", "start_line": 1}, sandbox)
        assert missing["result"] is None and "no such readable file" in missing["error"]


class _DashSandbox:
    """Runs tool commands with dash in a directory, as a provider sandbox runs them for its non-root user."""

    def __init__(self, root: Path) -> None:
        self.codebase_path = str(root)

    async def exec(self, command: str, timeout_s: int | None = None) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            "dash",
            "-c",
            command,
            cwd=self.codebase_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return ExecResult(exit_code=proc.returncode or 0, stdout=stdout.decode(), stderr=stderr.decode())


async def test_grep_include_becomes_a_ripgrep_glob() -> None:
    sandbox = FakeSandbox()
    out = await tools_mod.dispatch_tool("grep", {"pattern": "Foo", "include": "*.go"}, sandbox)
    assert out["error"] is None
    assert "--glob=*.go" in sandbox.calls[0][0]


async def test_apptainer_instance_is_contained_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: list[list[str]] = []
    exec_calls: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], timeout_s: int) -> ExecResult:
        run_calls.append(cmd)
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def fake_exec_with_timeout(cmd: list[str], *, timeout_s: int, cwd: str | None) -> ExecResult:
        exec_calls.append(cmd)
        return ExecResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(sandbox_mod, "_exec_with_timeout", fake_exec_with_timeout)

    contained = ApptainerSandbox(
        instance_name="vcqa-c", container_image="img.sif", working_tree=tmp_path, scratch_dir=tmp_path / "s"
    )
    await contained.start()
    assert "--containall" in run_calls[-1]
    assert "--cleanenv" in exec_calls[-1]

    open_box = ApptainerSandbox(
        instance_name="vcqa-o",
        container_image="img.sif",
        working_tree=tmp_path,
        scratch_dir=tmp_path / "s2",
        contain=False,
    )
    await open_box.start()
    assert "--containall" not in run_calls[-1]
    assert "--cleanenv" not in exec_calls[-1]

    direct = ApptainerDirectSandbox(container_image="img.sif", working_tree=tmp_path, scratch_dir=tmp_path / "s3")
    assert "--containall" in direct._exec_cmd("true")


class TestSharedWorkingTrees:
    async def test_concurrent_rollouts_build_once_and_clean_up(self, tmp_path: Path) -> None:
        from responses_api_agents.vcqa_agent.materialize import SharedWorkingTrees

        trees = SharedWorkingTrees(tmp_path / "trees")
        builds: list[Path] = []
        seen: list[Path] = []

        async def build(directory: Path) -> Path:
            builds.append(directory)
            await asyncio.sleep(0.02)
            (directory / "working_tree").mkdir()
            return directory / "working_tree"

        async def rollout() -> None:
            async with trees.acquire("fileset:a.tar.gz:", build) as path:
                seen.append(path)
                await asyncio.sleep(0.01)

        await asyncio.gather(*(rollout() for _ in range(5)))
        assert len(builds) == 1
        assert len(set(seen)) == 1
        assert not builds[0].exists()

    async def test_failed_build_fails_waiters_and_next_acquire_retries(self, tmp_path: Path) -> None:
        from responses_api_agents.vcqa_agent.materialize import SharedWorkingTrees

        trees = SharedWorkingTrees(tmp_path / "trees")
        attempts = {"n": 0}

        async def build(directory: Path) -> Path:
            attempts["n"] += 1
            await asyncio.sleep(0.01)
            if attempts["n"] == 1:
                raise RuntimeError("tar failed")
            return directory

        async def rollout() -> Path:
            async with trees.acquire("k", build) as path:
                return path

        results = await asyncio.gather(rollout(), rollout(), return_exceptions=True)
        assert all(isinstance(r, Exception) for r in results)
        assert isinstance(await rollout(), Path)
        assert attempts["n"] == 2


class TestLocalArtifacts:
    async def test_fileset_tarball_is_extracted_in_place(self, tmp_path: Path) -> None:
        import tarfile

        from responses_api_agents.vcqa_agent.materialize import materialize_working_tree

        src = tmp_path / "repo-abc"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "mod.go").write_text("package pkg\n")
        archive = tmp_path / "mirror" / "abc.tar.gz"
        archive.parent.mkdir()
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(src, arcname="repo-abc")

        tree = await materialize_working_tree(
            dataset_kind="fileset", artifact_url=str(archive), head_sha=None, scratch_dir=tmp_path / "scratch"
        )
        assert (tree / "pkg" / "mod.go").read_text() == "package pkg\n"
        assert archive.exists(), "the mirror's tarball must never be deleted"

    async def test_git_archive_tarball_keeps_the_repo_root(self, tmp_path: Path) -> None:
        """VCQA tarballs are `git archive` output: files at the archive root, no wrapping directory."""
        import tarfile

        from responses_api_agents.vcqa_agent.materialize import materialize_working_tree

        src = tmp_path / "src_tree"
        for rel, text in {"README.md": "top\n", "src/pkg/mod.py": "x = 1\n", "tests/pkg/test_mod.py": "t\n"}.items():
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_text(text)
        archive = tmp_path / "flat.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for child in sorted(src.iterdir()):
                tar.add(child, arcname=child.name)

        tree = await materialize_working_tree(
            dataset_kind="fileset", artifact_url=str(archive), head_sha=None, scratch_dir=tmp_path / "scratch"
        )
        assert (tree / "README.md").read_text() == "top\n"
        assert (tree / "src" / "pkg" / "mod.py").exists()
        assert (tree / "tests" / "pkg" / "test_mod.py").exists()
        assert not (tree / "pkg").exists(), "top-level directories must not be flattened into the root"

    async def test_githistory_bundle_is_cloned_at_head_sha(self, tmp_path: Path) -> None:
        import subprocess

        from responses_api_agents.vcqa_agent.materialize import materialize_working_tree

        repo = tmp_path / "repo"
        repo.mkdir()
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "f.txt").write_text("v1\n")
        subprocess.run([*git, "add", "f.txt"], check=True)
        subprocess.run([*git, "commit", "-qm", "v1"], check=True)
        first = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        (repo / "f.txt").write_text("v2\n")
        subprocess.run([*git, "commit", "-qam", "v2"], check=True)
        bundle = tmp_path / "mirror" / "repo.bundle"
        bundle.parent.mkdir()
        subprocess.run([*git, "bundle", "create", "-q", str(bundle), "--all"], check=True)

        tree = await materialize_working_tree(
            dataset_kind="githistory", artifact_url=f"file://{bundle}", head_sha=first, scratch_dir=tmp_path / "s"
        )
        assert (tree / "f.txt").read_text() == "v1\n"
        assert bundle.exists(), "the mirror's bundle must never be deleted"


class TestJudgeApiKeyEnv:
    async def test_key_is_read_from_the_named_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[str] = []

        async def fake_request(method, url, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs["headers"]["Authorization"])
            response = MagicMock()
            response.ok = True
            response.status = 200
            response.read = AsyncMock(return_value=json.dumps({"choices": [{"message": {"content": "YES"}}]}).encode())
            return response

        monkeypatch.setattr(judge_mod, "request", fake_request)
        monkeypatch.setenv("VCQA_TEST_JUDGE_KEY", "from-env")
        agent = VcqaAgent(
            config=_make_config(judge_api_key_env="VCQA_TEST_JUDGE_KEY"), server_client=_answering_server_client()
        )

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert captured == ["Bearer from-env"]
        assert result.reward == 1.0

    async def test_missing_env_var_masks_the_sample(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VCQA_TEST_JUDGE_KEY_MISSING", raising=False)
        agent = VcqaAgent(
            config=_make_config(judge_api_key_env="VCQA_TEST_JUDGE_KEY_MISSING"),
            server_client=_answering_server_client(),
        )

        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, _hardening_run_body())

        assert result.mask_sample is True
        assert result.failure_stage == "judge"
        assert "VCQA_TEST_JUDGE_KEY_MISSING" in (result.failure_reason or "")

    async def test_missing_key_is_not_needed_without_a_judge_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VCQA_TEST_JUDGE_KEY_MISSING", raising=False)
        sent = _scripted_judge(monkeypatch, [(200, "YES")])
        config = _make_config(judge_api_key_env="VCQA_TEST_JUDGE_KEY_MISSING")

        empty = VcqaAgent(config=config, server_client=_answering_server_client(answer="   "))
        result = await _run_with_fake_sandbox(empty, monkeypatch, tmp_path, _hardening_run_body())
        assert result.reward == 0.0 and result.evaluation_completed is True and result.mask_sample is False

        no_criteria = _hardening_run_body()
        no_criteria.verifier_metadata.rubric = {"judge": [{"category": "nice_to_have", "description": "Extra."}]}
        agent = VcqaAgent(config=config, server_client=_answering_server_client())
        result = await _run_with_fake_sandbox(agent, monkeypatch, tmp_path, no_criteria)
        assert result.mask_sample is True and result.failure_kind == "verifier_error"
        assert sent == []


async def test_apptainer_network_none_adds_an_empty_network_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_calls: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], timeout_s: int) -> ExecResult:
        run_calls.append(cmd)
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def fake_exec_with_timeout(cmd: list[str], *, timeout_s: int, cwd: str | None) -> ExecResult:
        return ExecResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(sandbox_mod, "_exec_with_timeout", fake_exec_with_timeout)

    offline = ApptainerSandbox(
        instance_name="vcqa-n",
        container_image="img.sif",
        working_tree=tmp_path,
        scratch_dir=tmp_path / "s",
        network_none=True,
    )
    await offline.start()
    start = run_calls[-1]
    assert start[start.index("--net") : start.index("--net") + 3] == ["--net", "--network", "none"]

    direct = ApptainerDirectSandbox(
        container_image="img.sif", working_tree=tmp_path, scratch_dir=tmp_path / "s2", network_none=True
    )
    cmd = direct._exec_cmd("true")
    assert cmd[cmd.index("--net") : cmd.index("--net") + 3] == ["--net", "--network", "none"]
    assert "--net" not in ApptainerDirectSandbox(
        container_image="img.sif", working_tree=tmp_path, scratch_dir=tmp_path / "s3"
    )._exec_cmd("true")


def test_config_wires_network_none_into_the_exec_sandbox(tmp_path: Path) -> None:
    agent = VcqaAgent(
        config=_make_config(sandbox_backend="apptainer_exec", apptainer_setup_command="", apptainer_network_none=True),
        server_client=MagicMock(spec=ServerClient),
    )
    sandbox = agent._build_sandbox(working_tree=tmp_path, scratch_dir=tmp_path / "s")
    assert isinstance(sandbox, ApptainerDirectSandbox)
    assert sandbox.network_none is True


########################################
# Sandbox-provider backend (OpenSandbox, ...)
########################################


class FakeProvider:
    """Records the provider calls AsyncSandbox makes; replays canned exec results."""

    name = "vcqa_fake"
    instances: list["FakeProvider"] = []

    def __init__(
        self,
        fail_exec_containing: str | None = None,
        timeout_on: str | None = None,
        lost_on: str | None = None,
        **_: object,
    ) -> None:
        self.fail_exec_containing = fail_exec_containing
        self.timeout_on = timeout_on
        self.lost_on = lost_on
        self.specs: list = []
        self.uploads: list[tuple[str, str]] = []
        self.execs: list[dict] = []
        self.closed: list[str] = []
        FakeProvider.instances.append(self)

    async def create(self, spec):  # type: ignore[no-untyped-def]
        from nemo_gym.sandbox import SandboxHandle

        self.specs.append(spec)
        return SandboxHandle(sandbox_id=f"sb-{len(self.specs)}", provider_name=self.name, raw=None)

    async def upload_file(self, handle, local_path, remote_path):  # type: ignore[no-untyped-def]
        self.uploads.append((str(local_path), remote_path))

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):  # type: ignore[no-untyped-def]
        from nemo_gym.sandbox import SandboxExecResult

        self.execs.append({"command": command, "cwd": cwd, "user": user, "timeout_s": timeout_s})
        if self.fail_exec_containing and self.fail_exec_containing in command:
            raise ConnectionError("sandbox pod evicted")
        if self.timeout_on and self.timeout_on in command:
            return SandboxExecResult(stdout="", stderr="", return_code=124, error_type="timeout")
        if self.lost_on and self.lost_on in command:
            return SandboxExecResult(stdout="", stderr="stream ended", return_code=125, error_type="sandbox")
        return SandboxExecResult(stdout="ok", stderr="", return_code=0)

    async def close(self, handle):  # type: ignore[no-untyped-def]
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_provider_global_config() -> dict:
    from nemo_gym.sandbox import register_provider

    register_provider("vcqa_fake", FakeProvider, override=True)
    FakeProvider.instances.clear()
    return {"sandbox": {"default_metadata": {"team": "t"}, "vcqa_fake": {}}}


def test_provider_setup_commands_build_the_tree_inside_the_sandbox() -> None:
    fileset = sandbox_mod.provider_setup_commands(
        dataset_kind="fileset",
        artifact_path="/tmp/vcqa-artifact",
        head_sha=None,
        tools_command="install-tools",
        run_as="vcqa",
        remove_artifact=True,
    )
    assert fileset[0] == "install-tools"
    assert "safe.directory" in fileset[1]
    assert "useradd -m vcqa" in fileset[2] and "chown -R vcqa /tmp/scratch" in fileset[2]
    assert "tar -xzf /tmp/vcqa-artifact -C /tmp/vcqa-x" in fileset[3] and "strip-components" not in fileset[3]
    assert fileset[-1] == "rm -f /tmp/vcqa-artifact"

    history = sandbox_mod.provider_setup_commands(
        dataset_kind="githistory",
        artifact_path="/mnt/data/b.bundle",
        head_sha="abc123",
        tools_command=None,
        run_as=None,
        remove_artifact=False,
    )
    assert "git clone --quiet --no-checkout /mnt/data/b.bundle /codebase" in history[-1]
    assert "checkout --quiet abc123" in history[-1]
    assert not any("useradd" in c for c in history) and not any(c.startswith("rm -f") for c in history)
    with pytest.raises(ValueError):
        sandbox_mod.provider_setup_commands(
            dataset_kind="githistory",
            artifact_path="x",
            head_sha=None,
            tools_command=None,
            run_as=None,
            remove_artifact=False,
        )


def _setup_kwargs() -> dict:
    return dict(
        dataset_kind="fileset", artifact_path="/a.tgz", head_sha=None, tools_command=None, remove_artifact=True
    )


def test_hidden_paths_must_be_roots_and_fail_closed(tmp_path: Path) -> None:
    import os
    import subprocess

    as_root = sandbox_mod.provider_setup_commands(run_as=None, hidden_paths=["/mnt"], **_setup_kwargs())
    assert not any("chmod 700" in c or "chown -hR" in c for c in as_root)
    no_hidden = sandbox_mod.provider_setup_commands(run_as="vcqa", hidden_paths=[], **_setup_kwargs())
    assert not any("chmod 700" in c for c in no_hidden)

    def hide(*paths: str) -> int:
        command = sandbox_mod.provider_setup_commands(run_as="vcqa", hidden_paths=list(paths), **_setup_kwargs())[-1]
        return subprocess.run(["sh", "-c", command], capture_output=True).returncode

    assert hide(str(tmp_path / "absent")) == 0
    owned = tmp_path / "vol x"
    owned.mkdir(mode=0o755)
    if os.geteuid() == 0:
        assert hide(str(owned)) == 0 and owned.stat().st_mode & 0o777 == 0o700
    else:
        # Mode 700 leaves a path readable by its owner, so one that is not root's fails the setup instead.
        assert hide(str(owned)) == 1 and owned.stat().st_mode & 0o777 == 0o755
        # Root's, but this user cannot restrict it: also a failed setup, never a readable path.
        assert hide("/") == 1


def test_provider_tree_is_made_root_owned_and_read_only(tmp_path: Path) -> None:
    import stat
    import subprocess

    commands = sandbox_mod.provider_setup_commands(run_as="vcqa", hidden_paths=[], **_setup_kwargs())
    extract = next(i for i, c in enumerate(commands) if "tar -xzf" in c)
    normalize = next(i for i, c in enumerate(commands) if c.startswith("chown -hR 0:0 /codebase"))
    remove = next(i for i, c in enumerate(commands) if c.startswith("rm -f"))
    assert extract < normalize < remove
    assert not any("chown -hR" in c for c in sandbox_mod.provider_setup_commands(run_as=None, **_setup_kwargs()))

    # The mode half on a tree like an archive can leave it (chown needs root).
    tree = tmp_path / "codebase"
    (tree / "bin").mkdir(parents=True)
    (tree / "answer.txt").write_text("x")
    (tree / "answer.txt").chmod(0o666)
    (tree / "bin" / "run.sh").write_text("#!/bin/sh\n")
    (tree / "bin" / "run.sh").chmod(0o4777)
    (tree / "link").symlink_to("answer.txt")
    chmod = commands[normalize].split(" && ")[1].replace("/codebase", str(tree))
    assert subprocess.run(["sh", "-c", chmod]).returncode == 0
    assert stat.S_IMODE((tree / "answer.txt").stat().st_mode) == 0o644
    assert stat.S_IMODE((tree / "bin" / "run.sh").stat().st_mode) == 0o755
    assert (tree / "link").is_symlink() and stat.S_IMODE((tree / "bin").stat().st_mode) == 0o755


class TestProviderSandbox:
    def _sandbox(self, provider: FakeProvider, **kwargs):  # type: ignore[no-untyped-def]
        from nemo_gym.sandbox import SandboxSpec

        return sandbox_mod.ProviderSandbox(
            provider=provider,
            spec=SandboxSpec(image="debian:bookworm-slim"),
            uploads=[(Path("/mirror/a.tar.gz"), "/tmp/vcqa-artifact")],
            setup_commands=["setup-1", "setup-2"],
            run_as="vcqa",
            **kwargs,
        )

    async def test_start_uploads_then_sets_up_as_root_and_tools_run_as_the_user(self) -> None:
        provider = FakeProvider()
        sb = self._sandbox(provider)
        await sb.start()
        assert provider.uploads == [("/mirror/a.tar.gz", "/tmp/vcqa-artifact")]
        assert [e["command"] for e in provider.execs] == ["setup-1", "setup-2"]
        assert all(e["user"] == "root" for e in provider.execs)
        result = await sb.exec("rg foo", timeout_s=7)
        assert result.exit_code == 0 and result.stdout == "ok" and not result.timed_out
        assert provider.execs[-1] == {
            "command": "bash -lc 'rg foo'",
            "cwd": "/codebase",
            "user": "vcqa",
            "timeout_s": 7,
        }
        await sb.stop()
        await sb.stop()
        assert provider.closed == ["sb-1"]

    async def test_deadline_is_reported_as_a_timeout(self) -> None:
        sb = self._sandbox(FakeProvider(timeout_on="sleep"))
        await sb.start()
        result = await sb.exec("sleep 99", timeout_s=1)
        assert result.timed_out and result.exit_code == 124

    async def test_command_without_an_exit_code_raises_sandbox_infra_error(self) -> None:
        sb = self._sandbox(FakeProvider(lost_on="git log"))
        await sb.start()
        with pytest.raises(sandbox_mod.SandboxInfraError, match="sandbox"):
            await sb.exec("git log")

    async def test_provider_failure_raises_sandbox_infra_error(self) -> None:
        sb = self._sandbox(FakeProvider(fail_exec_containing="git log"))
        await sb.start()
        with pytest.raises(sandbox_mod.SandboxInfraError):
            await sb.exec("git log")

    @pytest.mark.skipif(shutil.which("dash") is None, reason="needs dash, the /bin/sh of Debian sandbox images")
    async def test_tool_commands_keep_bash_semantics_under_the_providers_sh(self) -> None:
        import subprocess

        provider = FakeProvider()
        sb = self._sandbox(provider)
        await sb.start()
        await sb.exec("a=(one two); printf '%s\\n' \"${a[1]}\"")
        # The provider runs a non-root user's command through `su -s /bin/sh`, dash on Debian images.
        sent = provider.execs[-1]["command"]
        assert subprocess.run(["dash", "-c", sent], capture_output=True, text=True).stdout == "two\n"

    async def test_failed_stop_stays_retryable(self) -> None:
        provider = FakeProvider()
        closes = {"n": 0}

        async def flaky_close(handle):  # type: ignore[no-untyped-def]
            closes["n"] += 1
            if closes["n"] == 1:
                raise ConnectionError("close timed out")
            provider.closed.append(handle.sandbox_id)

        provider.close = flaky_close  # type: ignore[method-assign]
        sb = self._sandbox(provider)
        await sb.start()
        with pytest.raises(ConnectionError):
            await sb.stop()
        await sb.stop()
        await sb.stop()
        assert closes["n"] == 2 and provider.closed == ["sb-1"]

    async def test_failed_setup_stops_the_sandbox(self) -> None:
        from nemo_gym.sandbox import SandboxExecResult

        provider = FakeProvider()

        async def failing_exec(handle, command, **kwargs):  # type: ignore[no-untyped-def]
            return SandboxExecResult(stdout="", stderr="tar: bad archive", return_code=2)

        provider.exec = failing_exec  # type: ignore[method-assign]
        sb = self._sandbox(provider)
        with pytest.raises(RuntimeError, match="setup command failed"):
            await sb.start()
        assert provider.closed == ["sb-1"]


async def test_run_retries_a_failed_sandbox_stop() -> None:
    from responses_api_agents.vcqa_agent.app import _stop_sandbox

    sandbox = MagicMock()
    sandbox.stop = AsyncMock(side_effect=[ConnectionError("close timed out"), None])
    await _stop_sandbox(sandbox)
    assert sandbox.stop.await_count == 2


async def test_dispatch_tool_lets_sandbox_infra_errors_abort_the_rollout() -> None:
    class DeadSandbox(FakeSandbox):
        async def exec(self, command: str, timeout_s: int | None = None) -> ApptainerExecResult:
            raise sandbox_mod.SandboxInfraError("gone")

    with pytest.raises(sandbox_mod.SandboxInfraError):
        await tools_mod.dispatch_tool("bash", {"command": "ls"}, DeadSandbox())


class TestRunWithProviderBackend:
    def _agent(self, global_config: dict, server_client: MagicMock, tmp_path: Path, **overrides):  # type: ignore[no-untyped-def]
        mirror = tmp_path / "mirror"
        (mirror / "fileset").mkdir(parents=True)
        (mirror / "fileset" / "a.tar.gz").write_bytes(b"x")
        tools = tmp_path / "tools.tar.gz"
        tools.write_bytes(b"t")
        server_client.global_config_dict = global_config
        config = _make_config(
            sandbox_backend="provider",
            artifact_url_prefix=str(mirror),
            sandbox_tools_archive=str(tools),
            judge_max_attempts=1,
            **overrides,
        )
        return VcqaAgent(config=config, server_client=server_client)

    def _body(self):  # type: ignore[no-untyped-def]
        from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
        from responses_api_agents.vcqa_agent.app import VcqaAgentRunRequest

        return VcqaAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}]),
            verifier_metadata={
                "task_id": "p-1",
                "dataset_kind": "fileset",
                "artifact_key": "fileset/a.tar.gz",
                "rubric": {"judge": [{"category": "must_have", "description": "Says forty-two."}]},
                "max_turns": 2,
            },
        )

    async def test_tree_is_built_in_the_sandbox_not_on_the_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider_global_config: dict
    ) -> None:
        from fastapi import Request

        async def host_materialize(**kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("the provider backend must not extract on the agent host")

        monkeypatch.setattr("responses_api_agents.vcqa_agent.app.materialize_working_tree", host_materialize)
        _scripted_judge(monkeypatch, [(200, "YES")])
        agent = self._agent(fake_provider_global_config, _answering_server_client(), tmp_path)
        request = MagicMock(spec=Request)
        request.cookies = {}

        result = await agent.run(request=request, response=MagicMock(), body=self._body())

        provider = FakeProvider.instances[-1]
        spec = provider.specs[0]
        assert spec.image == "docker.io/library/debian:bookworm-slim"
        assert spec.provider_options == {"network_policy": {"defaultAction": "deny"}}
        assert spec.metadata["team"] == "t" and spec.metadata["vcqa-task"] == "p-1"
        assert [u[1] for u in provider.uploads] == ["/tmp/vcqa-tools.tar.gz", "/tmp/vcqa-artifact"]
        setup = [e["command"] for e in provider.execs if e["user"] == "root"]
        assert "--keep-directory-symlink" in setup[0]
        assert any("tar -xzf /tmp/vcqa-artifact" in c for c in setup)
        assert provider.closed == ["sb-1"]
        assert result.reward == 1.0 and result.evaluation_completed is True

    async def test_mounted_artifacts_are_not_uploaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider_global_config: dict
    ) -> None:
        from fastapi import Request

        _scripted_judge(monkeypatch, [(200, "YES")])
        agent = self._agent(
            fake_provider_global_config,
            _answering_server_client(),
            tmp_path,
            sandbox_artifact_root="/mnt/s3-data/vcqa",
        )
        request = MagicMock(spec=Request)
        request.cookies = {}
        await agent.run(request=request, response=MagicMock(), body=self._body())
        provider = FakeProvider.instances[-1]
        assert [u[1] for u in provider.uploads] == ["/tmp/vcqa-tools.tar.gz"]
        root = [e["command"] for e in provider.execs if e["user"] == "root"]
        built = next(i for i, c in enumerate(root) if "tar -xzf /mnt/s3-data/vcqa/fileset/a.tar.gz" in c)
        # The mount holds every task's artifacts: root builds the tree from it, then hides it from the tool user.
        hidden = next(i for i, c in enumerate(root) if "chmod 700" in c)
        assert built < hidden and root[hidden].startswith("for p in /mnt;")

    async def test_sandbox_dying_mid_rollout_is_masked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider_global_config: dict
    ) -> None:
        from fastapi import Request

        tool_call = {
            "id": "resp_t",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "bash",
                    "arguments": '{"command": "uname -a"}',
                    "id": "fc_1",
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        post_response = AsyncMock()
        post_response.ok = True
        post_response.read = AsyncMock(return_value=json.dumps(tool_call).encode())
        post_response.cookies = {}
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=post_response)
        fake_provider_global_config["sandbox"]["vcqa_fake"] = {"fail_exec_containing": "uname"}
        agent = self._agent(fake_provider_global_config, server_client, tmp_path)
        request = MagicMock(spec=Request)
        request.cookies = {}

        result = await agent.run(request=request, response=MagicMock(), body=self._body())

        assert result.mask_sample is True and result.evaluation_completed is False
        assert result.failure_stage == "sandbox" and result.failure_kind == "provider_unavailable"
        assert FakeProvider.instances[-1].closed == ["sb-1"]


########################################
# Token usage across turns
########################################


def _usage(input_tokens: int, cached: int | None, output_tokens: int, reasoning: int | None) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def _model_turn(output: list[dict], usage: dict) -> MagicMock:
    payload = {
        "id": "resp_turn",
        "created_at": 0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }
    turn = MagicMock()
    turn.ok = True
    turn.read = AsyncMock(return_value=json.dumps(payload).encode())
    turn.cookies = {}
    return turn


@pytest.mark.parametrize(
    ("second_usage", "expected"),
    [
        (_usage(200, 80, 20, 6), _usage(300, 120, 30, 10)),
        # A turn whose provider reported no details makes the summed details unknown, not 0.
        (_usage(200, None, 20, None), _usage(300, None, 30, None)),
    ],
)
async def test_tool_loop_sums_usage_and_details_across_turns(second_usage: dict, expected: dict) -> None:
    tool_call = {
        "id": "call_1",
        "call_id": "call_1",
        "type": "function_call",
        "name": "unknown_tool",
        "arguments": "{}",
        "status": "completed",
    }
    answer = {
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "type": "message",
        "content": [{"annotations": [], "text": "done", "type": "output_text"}],
    }
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(
        side_effect=[_model_turn([tool_call], _usage(100, 40, 10, 4)), _model_turn([answer], second_usage)]
    )
    agent = VcqaAgent(config=_make_config(inject_tools=False), server_client=server_client)

    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    result, steps = await agent._tool_loop(
        request=MagicMock(cookies={}),
        response=MagicMock(),
        body=NeMoGymResponseCreateParamsNonStreaming(input="What is the answer?"),
        sandbox=FakeSandbox(),
        max_turns=4,
    )

    assert steps == 2
    assert result.usage.model_dump(mode="json") == expected
