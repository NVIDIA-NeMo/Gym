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

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
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
        from nemo_gym.openai_utils import (
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
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

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

        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
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

        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
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
