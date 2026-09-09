# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shlex import split as shlex_split
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21SeedSessionRequest,
    TerminalBench21VerifyRequest,
    uploaded_subdirectories,
)


SESSION_ID = "session-1"
AGENT_USER_ERROR = "agent_user must be an account name, a uid, or null"
NORMALIZED_AGENT_USERS = [
    ("1000", 1000),
    ("0", 0),
    ("agent", "agent"),
    ("root", "root"),
    (1000, 1000),
    (0, 0),
    (None, None),
    # isdecimal, not isdigit: a superscript digit is not a uid and must not make int() raise.
    ("\u00b2", "\u00b2"),
]
# Booleans (pydantic lax mode would coerce `true` to uid 1) and empty / option-like names (`su` would parse them
# as options; shlex.quote leaves "-m" unquoted).
REJECTED_AGENT_USERS = [True, False, "", "-m", "--login", "-"]


def _make_server(**config_overrides) -> TerminalBench21ResourcesServer:
    config = TerminalBench21ResourcesServerConfig(
        sandbox_provider="",
        sandbox_config=dict(),
        host="",
        port=0,
        entrypoint="",
        name="",
        **config_overrides,
    )
    return TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request() -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: SESSION_ID})


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text="done", type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _verify_request(task_folder: Path, **extra) -> TerminalBench21VerifyRequest:
    return TerminalBench21VerifyRequest(
        task_name="terminal-bench/some-task",
        docker_image="example.invalid/some-task:latest",
        task_folder=str(task_folder),
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response(),
        **extra,
    )


class FakeSandbox:
    """Records exec/upload/download/stop calls; the reward download writes a parseable reward file."""

    def __init__(self, cwd: str = "/app", reward: str = "1\n"):
        self._handle = SimpleNamespace(sandbox_id="sb-1")
        self._cwd = cwd
        self._reward = reward
        self.execs: list[tuple[str, dict]] = []
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.stopped = False

    async def exec(self, command, **kwargs):
        self.execs.append((command, kwargs))
        stdout = self._cwd if command == "pwd" else ""
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0)

    async def upload(self, local_path, remote_path):
        self.uploads.append((str(local_path), remote_path))

    async def download(self, remote_path, local_path):
        self.downloads.append((remote_path, str(local_path)))
        Path(local_path).write_text(self._reward)

    async def stop(self):
        self.stopped = True


def _exec_commands(sandbox: FakeSandbox) -> list[str]:
    return [command for command, _ in sandbox.execs]


def _exec_user(sandbox: FakeSandbox, command: str):
    matches = [kwargs for cmd, kwargs in sandbox.execs if cmd == command]
    assert len(matches) == 1, (command, sandbox.execs)
    return matches[0].get("user")


def _make_task_folder(tmp_path: Path, with_solution: bool = False, nested_solution: bool = False) -> Path:
    task_folder = tmp_path / "task"
    (task_folder / "tests").mkdir(parents=True)
    (task_folder / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    if with_solution or nested_solution:
        (task_folder / "solution").mkdir()
        (task_folder / "solution" / "solve.sh").write_text("#!/bin/bash\ntouch /app/done\n")
    if nested_solution:
        (task_folder / "solution" / "pkg").mkdir()
        (task_folder / "solution" / "pkg" / "helper.py").write_text("VALUE = 1\n")
    return task_folder


class TestApp:
    def test_sanity(self) -> None:
        _make_server()


class TestAgentUserNormalization:
    # The normalizer's own table lives in tests/unit_tests/test_sandbox_agent_user.py; these cover the schema wiring.
    @pytest.mark.parametrize(("raw", "expected"), NORMALIZED_AGENT_USERS)
    def test_request_normalizes_agent_user(self, raw, expected) -> None:
        body = TerminalBench21SeedSessionRequest(
            task_name="t", docker_image="img", task_folder="folder", agent_user=raw
        )
        assert body.agent_user == expected
        assert type(body.agent_user) is type(expected)

    @pytest.mark.parametrize("raw", REJECTED_AGENT_USERS)
    def test_request_rejects_bools_and_option_like_agent_user(self, raw) -> None:
        with pytest.raises(ValidationError, match=AGENT_USER_ERROR):
            TerminalBench21SeedSessionRequest(task_name="t", docker_image="img", task_folder="folder", agent_user=raw)

    def test_omitted_is_none(self) -> None:
        body = TerminalBench21SeedSessionRequest(task_name="t", docker_image="img", task_folder="folder")
        assert body.agent_user is None


class TestUploadedSubdirectories:
    def test_flat_upload_has_no_subdirectories(self) -> None:
        assert uploaded_subdirectories("/app", ["/app/solve.sh"]) == []

    def test_nested_uploads_yield_deduplicated_parents_first(self) -> None:
        remote_paths = [
            "/app/pkg/sub/deep.py",
            "/app/solve.sh",
            "/app/pkg/helper.py",
            "/app/pkg/sub/other.py",
            "/app/data/x.txt",
        ]
        assert uploaded_subdirectories("/app", remote_paths) == ["/app/pkg", "/app/pkg/sub", "/app/data"]

    def test_never_includes_the_upload_root(self) -> None:
        assert "/app" not in uploaded_subdirectories("/app", ["/app/pkg/helper.py", "/app/solve.sh"])


class TestSeedSession:
    @pytest.mark.parametrize(
        ("row_extra", "expected"),
        [({"agent_user": "agent"}, "agent"), ({}, None), ({"agent_user": 1000}, 1000), ({"agent_user": "1000"}, 1000)],
    )
    async def test_seed_session_echoes_agent_user(self, monkeypatch, row_extra, expected) -> None:
        server = _make_server()
        sandbox = FakeSandbox()
        created_for = []

        async def fake_create_sandbox(verify_request):
            created_for.append(verify_request)
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", fake_create_sandbox)
        body = TerminalBench21SeedSessionRequest(task_name="t", docker_image="img", task_folder="folder", **row_extra)

        result = await server.seed_session(_request(), body)

        assert result.sandbox_handle == "sb-1"
        assert result.agent_user == expected
        assert type(result.agent_user) is type(expected)
        assert created_for == [body]
        assert server._session_id_to_sandbox[SESSION_ID] is sandbox


class TestVerify:
    async def test_verify_reuses_session_sandbox_and_runs_tests_as_image_default(self, tmp_path) -> None:
        server = _make_server()
        sandbox = FakeSandbox()
        server._session_id_to_sandbox[SESSION_ID] = sandbox
        task_folder = _make_task_folder(tmp_path)

        result = await server.verify(_request(), _verify_request(task_folder, agent_user="agent"))

        assert _exec_commands(sandbox) == ["mkdir -p /tests", "bash /tests/test.sh"]
        # Verifier commands never carry a user override: they run as the image default even when the
        # agent ran as `agent`.
        assert _exec_user(sandbox, "mkdir -p /tests") is None
        assert _exec_user(sandbox, "bash /tests/test.sh") is None
        assert sandbox.uploads == [(str(task_folder / "tests" / "test.sh"), "/tests/test.sh")]
        assert [remote for remote, _ in sandbox.downloads] == ["/logs/verifier/reward.txt"]
        assert result.reward == 1.0
        assert result.evaluation_completed is True
        assert result.golden_patch_output is None
        assert result.agent_user == "agent"
        assert sandbox.stopped is True
        assert SESSION_ID not in server._session_id_to_sandbox

    async def test_verify_records_reward_zero_when_reward_file_unparseable(self, tmp_path) -> None:
        server = _make_server()
        sandbox = FakeSandbox(reward="not-a-number")
        server._session_id_to_sandbox[SESSION_ID] = sandbox

        result = await server.verify(_request(), _verify_request(_make_task_folder(tmp_path)))

        assert result.reward == 0.0
        assert result.evaluation_completed is False
        assert result.agent_user is None
        assert sandbox.stopped is True

    async def test_golden_mode_runs_solve_sh_as_agent_user_after_chown(self, tmp_path, monkeypatch) -> None:
        server = _make_server(is_verifying_golden_patch=True)
        sandbox = FakeSandbox(cwd="/app")

        async def fake_create_sandbox(verify_request):
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", fake_create_sandbox)
        task_folder = _make_task_folder(tmp_path, with_solution=True)

        result = await server.verify(_request(), _verify_request(task_folder, agent_user="agent"))

        assert _exec_commands(sandbox) == [
            "pwd",
            "mkdir -p /app",
            "chown agent /app/solve.sh",
            "bash /app/solve.sh",
            "mkdir -p /tests",
            "bash /tests/test.sh",
        ]
        assert _exec_user(sandbox, "chown agent /app/solve.sh") is None
        assert _exec_user(sandbox, "bash /app/solve.sh") == "agent"
        assert _exec_user(sandbox, "bash /tests/test.sh") is None
        assert _exec_user(sandbox, "mkdir -p /tests") is None
        assert sandbox.uploads == [
            (str(task_folder / "solution" / "solve.sh"), "/app/solve.sh"),
            (str(task_folder / "tests" / "test.sh"), "/tests/test.sh"),
        ]
        assert result.reward == 1.0
        assert result.golden_patch_output == ""
        assert result.agent_user == "agent"
        assert sandbox.stopped is True

    async def test_golden_mode_chowns_nested_solution_directories_and_files(self, tmp_path, monkeypatch) -> None:
        server = _make_server(is_verifying_golden_patch=True)
        sandbox = FakeSandbox(cwd="/app")

        async def fake_create_sandbox(verify_request):
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", fake_create_sandbox)
        task_folder = _make_task_folder(tmp_path, nested_solution=True)

        result = await server.verify(_request(), _verify_request(task_folder, agent_user="agent"))

        commands = _exec_commands(sandbox)
        # glob order is filesystem-dependent, so the two solution mkdirs are compared as a set.
        assert commands[0] == "pwd"
        assert sorted(commands[1:3]) == ["mkdir -p /app", "mkdir -p /app/pkg"]
        chown_command = commands[3]
        assert commands[4:] == ["bash /app/solve.sh", "mkdir -p /tests", "bash /tests/test.sh"]
        chown_argv = shlex_split(chown_command)
        # One non-recursive chown as the image default: the nested directory first, then both files.
        assert chown_argv[:3] == ["chown", "agent", "/app/pkg"]
        assert sorted(chown_argv[3:]) == ["/app/pkg/helper.py", "/app/solve.sh"]
        assert "-R" not in chown_argv
        assert _exec_user(sandbox, chown_command) is None
        assert _exec_user(sandbox, "bash /app/solve.sh") == "agent"
        assert _exec_user(sandbox, "bash /tests/test.sh") is None
        assert sorted(sandbox.uploads) == [
            (str(task_folder / "solution" / "pkg" / "helper.py"), "/app/pkg/helper.py"),
            (str(task_folder / "solution" / "solve.sh"), "/app/solve.sh"),
            (str(task_folder / "tests" / "test.sh"), "/tests/test.sh"),
        ]
        assert result.reward == 1.0
        assert result.agent_user == "agent"
        assert sandbox.stopped is True

    async def test_golden_mode_without_agent_user_runs_solve_sh_as_image_default(self, tmp_path, monkeypatch) -> None:
        server = _make_server(is_verifying_golden_patch=True)
        sandbox = FakeSandbox(cwd="/app")

        async def fake_create_sandbox(verify_request):
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", fake_create_sandbox)
        task_folder = _make_task_folder(tmp_path, with_solution=True)

        result = await server.verify(_request(), _verify_request(task_folder))

        assert _exec_commands(sandbox) == [
            "pwd",
            "mkdir -p /app",
            "bash /app/solve.sh",
            "mkdir -p /tests",
            "bash /tests/test.sh",
        ]
        assert not any(cmd.startswith("chown") for cmd in _exec_commands(sandbox))
        assert _exec_user(sandbox, "bash /app/solve.sh") is None
        assert _exec_user(sandbox, "bash /tests/test.sh") is None
        assert result.reward == 1.0
        assert result.agent_user is None
        assert sandbox.stopped is True

    @pytest.mark.parametrize("agent_user", ["root", 0])
    async def test_golden_mode_root_escape_hatch_skips_chown(self, tmp_path, monkeypatch, agent_user) -> None:
        server = _make_server(is_verifying_golden_patch=True)
        sandbox = FakeSandbox(cwd="/app")

        async def fake_create_sandbox(verify_request):
            return sandbox

        monkeypatch.setattr(server, "_create_sandbox", fake_create_sandbox)
        task_folder = _make_task_folder(tmp_path, with_solution=True)

        result = await server.verify(_request(), _verify_request(task_folder, agent_user=agent_user))

        assert not any(cmd.startswith("chown") for cmd in _exec_commands(sandbox))
        assert _exec_user(sandbox, "bash /app/solve.sh") == agent_user
        assert result.agent_user == agent_user
