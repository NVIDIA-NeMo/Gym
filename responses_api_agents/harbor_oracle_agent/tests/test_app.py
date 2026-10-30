# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.transfer import TRANSFER_USER
from nemo_gym.server_utils import ServerClient
from nemo_gym.tasks.harbor.publish import answer_write_command
from responses_api_agents.harbor_oracle_agent import app as module
from responses_api_agents.harbor_oracle_agent.app import OracleAgent, OracleAgentConfig


@dataclass
class FakeSandbox:
    """The sandbox the environment server seeded, with the solution already at ``/solution`` when ``solution`` is set.

    ``result`` is what solve.sh produced: its stdout and stderr both land in the log the oracle tails.
    ``log_present=False`` models the guard exiting before the redirection created the log (no ``timeout``
    or ``bash`` in the image). ``stdout_rejoin`` models a provider that rebuilds exec stdout from per-line
    messages (OpenSandbox's foreground exec): every command's stdout goes through it.
    """

    result: SandboxExecResult = SandboxExecResult(stdout="Done!\n", stderr="", return_code=0)
    solution: bool = True
    # The bytes of /solution/answer.md when the task ships one instead of solve.sh; None when it ships neither.
    answer: str | bytes | None = None
    # Whether the image has `base64` (GNU coreutils and BusyBox both do).
    base64_present: bool = True
    log_present: bool = True
    # Whether the working directory accepts the oracle's answer.md write.
    workdir_writable: bool = True
    stdout_rejoin: Callable[[str], str] | None = None
    execs: list[dict] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    # Files the oracle's answer write left, by absolute path (resolved against the exec's cwd).
    files: dict[str, str] = field(default_factory=dict)
    disconnected: bool = False

    async def exec(
        self, command, *, cwd=None, env=None, timeout_s=None, user=None, preserve_background_services=False
    ):
        self.execs.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "user": user,
                "preserve_background_services": preserve_background_services,
            }
        )
        result = await self._exec(command, cwd)
        if self.stdout_rejoin is not None and result.stdout:
            result = SandboxExecResult(
                stdout=self.stdout_rejoin(result.stdout),
                stderr=result.stderr,
                return_code=result.return_code,
                error_type=result.error_type,
            )
        return result

    async def _exec(self, command, cwd):
        if command == "test -f /solution/solve.sh":
            return SandboxExecResult(stdout="", stderr="", return_code=0 if self.solution else 1)
        if command == "test -f /solution/answer.md":
            return SandboxExecResult(stdout="", stderr="", return_code=0 if self.answer is not None else 1)
        if command == "base64 /solution/answer.md":
            if not self.base64_present:
                return SandboxExecResult(stdout="", stderr="sh: 1: base64: not found", return_code=127)
            if self.answer is None:
                return SandboxExecResult(stdout="", stderr="base64: /solution/answer.md: No such file", return_code=1)
            raw = self.answer.encode() if isinstance(self.answer, str) else self.answer
            # GNU base64 wraps at 76 columns and ends with a newline.
            return SandboxExecResult(stdout=base64.encodebytes(raw).decode(), stderr="", return_code=0)
        if command == module.ANSWER_WRITE_COMMAND:
            # `cat /solution/answer.md > answer.md`: the file's bytes land in the exec's working directory.
            assert self.answer is not None, command
            if not self.workdir_writable:
                return SandboxExecResult(
                    stdout="", stderr="sh: 1: cannot create answer.md: Permission denied", return_code=1
                )
            raw = self.answer.encode() if isinstance(self.answer, str) else self.answer
            self.files[f"{cwd}/answer.md"] = raw.decode("utf-8")
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if "solve.sh" in command:
            # solve.sh output is redirected to a file in the sandbox, so the exec itself carries only the
            # exit code and the provider's own error type.
            return SandboxExecResult(
                stdout="", stderr="", return_code=self.result.return_code, error_type=self.result.error_type
            )
        if command.startswith("tail -c"):
            if not self.log_present:
                return SandboxExecResult(stdout="", stderr="tail: cannot open", return_code=1)
            log = (self.result.stdout or "") + (self.result.stderr or "")
            return SandboxExecResult(stdout=log, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload(self, local_path, remote_path):
        self.uploads.append(remote_path)

    async def disconnect(self):
        self.disconnected = True


def make_agent(monkeypatch: MonkeyPatch, *, sandbox: FakeSandbox | None = None):
    # The oracle needs the run config for one thing: the sandbox provider block its connection names.
    global_config = {"sandbox": {"opensandbox": {"connection": {"domain": "example.invalid"}}}}
    monkeypatch.setattr(module, "get_global_config_dict", lambda: global_config)
    sandbox = sandbox or FakeSandbox()
    provider = MagicMock()
    monkeypatch.setattr(module, "create_provider", lambda config: provider)

    async def connect(descriptor, *, provider, owns_provider=True):
        connect.descriptors.append(descriptor)
        return sandbox

    connect.descriptors = []
    monkeypatch.setattr(module.AsyncSandbox, "connect", connect)
    config = OracleAgentConfig(
        host="0.0.0.0",
        port=8081,
        entrypoint="",
        name="harbor_oracle_agent",
    )
    agent = OracleAgent(config=config, server_client=MagicMock(spec=ServerClient))
    return agent, sandbox, connect


def seed_body(
    *,
    session="ag-1",
    rollout="r1",
    attempt=0,
    task="hello",
    with_sandbox=True,
    timeout: float | None = 120.0,
    user: str | None = "runner",
    env: dict[str, str] | None = None,
):
    """What the harbor environment server posts: the sandbox it seeded (the solution already in it), the task
    user, the `[solution].env` it resolved on its host, and the task's `[agent].timeout_sec`."""
    body = {
        "agent_session_id": session,
        "episode_id": {"rollout_id": rollout, "attempt": attempt},
        "task_id": {"taskset": "ds", "task_id": task},
        "tool_accesses": [],
    }
    if with_sandbox:
        body["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb"}},
            "workdir": "/app",
            "user": user,
            "env": {"GREETING": "hi"} if env is None else env,
        }
    if timeout is not None:
        body["agent_timeout_sec"] = timeout
    return body


RESPONSES_BODY = {"input": [{"role": "user", "content": "Create hello.txt"}]}


def solve_run(sandbox: FakeSandbox) -> dict:
    return next(call for call in sandbox.execs if "bash /solution/solve.sh" in call["command"])


def test_oracle_runs_the_solution_the_server_placed_in_the_borrowed_sandbox(monkeypatch):
    agent, sandbox, connect = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())

    seed = client.post("/v1/agent_sessions", json=seed_body())
    assert seed.status_code == 200, seed.text
    assert connect.descriptors == [{"sandbox_id": "sb"}]

    response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["metadata"]["oracle"] == "solved"
    assert payload["metadata"]["oracle_return_code"] == "0"
    assert payload["output"][0]["content"][0]["text"] == "Done!\n"
    # The oracle uploads nothing: the environment server put /solution in place before the session opened.
    assert sandbox.uploads == []
    # It looks for the script as the task user, where Harbor's oracle copies the solution.
    probe = sandbox.execs[0]
    assert (probe["command"], probe["user"]) == ("test -f /solution/solve.sh", "runner")
    run = solve_run(sandbox)
    # The portable short form (`--signal=` fails on BusyBox), guarded so a missing wrapper exits 127 before
    # the log is created; the log carries Harbor's oracle name.
    assert run["command"] == (
        f"{module.TIMEOUT_WRAPPER_GUARD}; timeout -s KILL 120 bash /solution/solve.sh > /logs/agent/oracle.txt 2>&1"
    )
    assert "--signal" not in run["command"]
    # The budget is the seed's agent_timeout_sec; the exec outlives the in-container `timeout` by the grace.
    assert run["timeout_s"] == 120 + 60
    assert (run["cwd"], run["user"]) == ("/app", "runner")
    # The `[solution].env` the server resolved travels in the sandbox access, under Harbor's oracle default.
    assert run["env"] == {"DEBIAN_FRONTEND": "noninteractive", "GREETING": "hi"}
    assert run["preserve_background_services"] is True

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200 and sandbox.disconnected


def test_oracle_reads_no_task_folder():
    """The task, its digest and its solution are the environment server's; the oracle has no path to them."""
    assert "environment_server" not in OracleAgentConfig.model_fields
    assert set(OracleAgentConfig.model_fields) - set(module.BaseResponsesAPIAgentConfig.model_fields) == {
        "max_solve_timeout_s",
        "output_tail_chars",
    }
    assert not any(name in dir(module) for name in ("load_task", "HarborTask", "upload_dir", "Path"))


def test_oracle_declares_task_tool_context_support(monkeypatch):
    """`gym dataset validate` must not refuse tasks with MCP servers or skills: solve.sh reaches them by itself."""
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())

    seed = client.post("/v1/agent_sessions", json=seed_body())
    assert seed.status_code == 200, seed.text
    # The harbor environment server reads this declaration and lets the task through to the oracle.
    assert seed.json() == {"agent_session_id": "ag-1", "supports_task_tool_context": True}


def test_task_without_solution_is_unvalidated_not_failed(monkeypatch):
    """The server uploaded nothing for a task without `solution/`, so the script is absent from the sandbox."""
    sandbox = FakeSandbox(solution=False)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "unvalidated"}
    assert "solve.sh" in payload["output"][0]["content"][0]["text"]
    assert "answer.md" in payload["output"][0]["content"][0]["text"]
    # Only the two probes ran: no log folder was prepared, nothing was read, executed or uploaded.
    assert [call["command"] for call in sandbox.execs] == ["test -f /solution/solve.sh", "test -f /solution/answer.md"]
    assert sandbox.uploads == []


def test_answer_md_is_written_to_the_workdir_and_submitted_as_the_response(monkeypatch):
    """A task verified by its answer ships solution/answer.md: the oracle copies it to answer.md in the working
    directory, as the task user (the file the published solve.sh writes there under Harbor's oracle), and submits it."""
    sandbox = FakeSandbox(solution=False, answer="it's 72\n")
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "solved", "oracle_solution": "answer", "oracle_return_code": "0"}
    assert payload["output"][0]["content"][0]["text"] == "it's 72\n"
    calls = [(call["command"], call["cwd"], call["user"]) for call in sandbox.execs]
    assert calls == [
        ("test -f /solution/solve.sh", "/", "runner"),
        ("test -f /solution/answer.md", "/", "runner"),
        # The read is base64 text decoded on the host, never the exec's raw stdout (`cat`).
        ("base64 /solution/answer.md", "/", "runner"),
        ("cat /solution/answer.md > answer.md", "/app", "runner"),
    ]
    # The file a workspace-reading verifier grades, in Gym as under Harbor: byte for byte the answer.
    assert sandbox.files == {"/app/answer.md": "it's 72\n"}
    assert sandbox.uploads == []


def _joined_without_newlines(text: str) -> str:
    """OpenSandbox's foreground exec as observed live: per-line messages joined with "\n", the trailing newline and
    every CR gone ("72\n" -> "72", "r1\r\nr2\r\n" -> "r1\nr2")."""
    return "\n".join(text.splitlines())


def _joined_with_newlines(text: str) -> str:
    """The provider's join over messages that still carry their newline (what its code literally does): every
    newline doubled ("a\nb\n" -> "a\n\nb\n")."""
    return "\n".join(text.splitlines(keepends=True))


REJOINED_ANSWERS = ["72\n", "first\n\n\nlast line\r\nend\n", "it's 4", "x\n\n\ny\n"]


@pytest.mark.parametrize("rejoin", [_joined_without_newlines, _joined_with_newlines])
@pytest.mark.parametrize("answer", REJOINED_ANSWERS)
def test_the_answer_survives_a_provider_that_rebuilds_stdout_line_by_line(monkeypatch, rejoin, answer):
    """A provider that rebuilds exec stdout from per-line messages (OpenSandbox's foreground exec, its default) hands
    `cat` output back with the trailing newline dropped, blank lines doubled or CRs lost. The oracle reads the answer as
    base64 and decodes it on the host, so the file it writes and the text it submits are the answer byte for byte."""
    # The re-join is lossy over the set (a `cat` read would have altered most of these answers; the live run did).
    assert sum(rejoin(text) != text for text in REJOINED_ANSWERS) >= 2
    sandbox = FakeSandbox(solution=False, answer=answer, stdout_rejoin=rejoin)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "solved", "oracle_solution": "answer", "oracle_return_code": "0"}
    assert payload["output"][0]["content"][0]["text"] == answer
    assert sandbox.files == {"/app/answer.md": answer}
    # The base64 payload itself came back re-joined (the fake applies the join to every command's stdout).
    read = next(call for call in sandbox.execs if call["command"] == module.ANSWER_READ_COMMAND)
    assert read["user"] == "runner" and module.ANSWER_READ_COMMAND == "base64 /solution/answer.md"
    assert not any(call["command"] == f"cat {module.ANSWER_FILE}" for call in sandbox.execs)


def test_a_long_answer_is_copied_in_the_sandbox_not_passed_as_an_exec_argument(monkeypatch):
    """The Docker provider hands an exec's command to the sandbox's shell as one argument (`docker exec ... sh -c
    <command>`), which Linux caps at MAX_ARG_STRLEN (128 KiB on 4 KiB pages). A write command carrying the answer
    failed with E2BIG for a 200 KiB answer where the published solve.sh, run from a file, wrote every byte. The oracle
    copies the file inside the sandbox instead, as the task user in the working directory, whatever the answer's size."""
    answer = "x" * (200 * 1024) + "\n"
    sandbox = FakeSandbox(solution=False, answer=answer)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {"oracle": "solved", "oracle_solution": "answer", "oracle_return_code": "0"}
    assert payload["output"][0]["content"][0]["text"] == answer
    (write,) = [call for call in sandbox.execs if call["cwd"] == "/app"]
    assert write["command"] == module.ANSWER_WRITE_COMMAND == "cat /solution/answer.md > answer.md"
    assert write["user"] == "runner" and answer not in write["command"]
    assert sandbox.files == {"/app/answer.md": answer}
    # The command the published solve.sh runs, from a file, would not fit one exec argument on a standard kernel.
    assert len(answer_write_command(answer).encode()) > 128 * 1024


def test_an_image_without_base64_is_a_failed_oracle_naming_it(monkeypatch):
    sandbox = FakeSandbox(solution=False, answer="72\n", base64_present=False)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {
        "oracle": "failed",
        "oracle_solution": "answer",
        "oracle_return_code": "127",
        "oracle_error_type": "answer",
    }
    text = payload["output"][0]["content"][0]["text"]
    assert "the image has no `base64`" in text and "base64: not found" in text
    # Nothing was written: the row is scored against the task's own state, with the reason visible.
    assert sandbox.files == {}


def test_an_answer_that_is_not_utf8_text_is_a_failed_oracle(monkeypatch):
    sandbox = FakeSandbox(solution=False, answer=b"\xff\xfe not text")
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {
        "oracle": "failed",
        "oracle_solution": "answer",
        "oracle_return_code": "0",
        "oracle_error_type": "answer",
    }
    assert "is not UTF-8 text" in payload["output"][0]["content"][0]["text"]
    assert sandbox.files == {}


def test_an_answer_the_workdir_refuses_is_a_failed_oracle_not_a_solved_one(monkeypatch):
    sandbox = FakeSandbox(solution=False, answer="72\n", workdir_writable=False)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"] == {
        "oracle": "failed",
        "oracle_solution": "answer",
        "oracle_return_code": "1",
        "oracle_error_type": "answer",
    }
    assert "Permission denied" in payload["output"][0]["content"][0]["text"]
    assert sandbox.files == {}


def test_a_lost_sandbox_on_the_answer_write_is_not_scored(monkeypatch):
    class LostOnWrite(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            if command == module.ANSWER_WRITE_COMMAND:
                return SandboxExecResult(stdout="", stderr="gone", return_code=1, error_type="timeout")
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)

    sandbox = LostOnWrite(solution=False, answer="72\n")
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    with pytest.raises(module.SandboxUnavailableError, match="could not write answer.md in /app"):
        client.post("/v1/responses", json=RESPONSES_BODY)


@pytest.mark.parametrize(
    ("lost_command", "message"),
    [
        ("test -f /solution/answer.md", "could not probe /solution/answer.md"),
        ("base64 /solution/answer.md", "could not read /solution/answer.md"),
    ],
)
def test_a_lost_sandbox_answering_the_answer_probe_or_read_is_not_scored(monkeypatch, lost_command, message):
    class LostOnAnswer(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            if command == lost_command:
                return SandboxExecResult(stdout="", stderr="gone", return_code=1, error_type="timeout")
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)

    sandbox = LostOnAnswer(solution=False, answer="72\n")
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    with pytest.raises(module.SandboxUnavailableError, match=message):
        client.post("/v1/responses", json=RESPONSES_BODY)
    assert sandbox.files == {}


def test_sandbox_default_user_runs_the_solution_when_the_task_names_none(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(user=None)).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).json()["metadata"]["oracle"] == "solved"
    assert sandbox.execs[0]["user"] is None and solve_run(sandbox)["user"] is None


def test_no_solution_env_means_only_the_oracle_default(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(env={})).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200
    assert solve_run(sandbox)["env"] == {"DEBIAN_FRONTEND": "noninteractive"}


def test_failing_solution_is_reported(monkeypatch):
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="boom", return_code=2))
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert payload["metadata"]["oracle_return_code"] == "2"
    assert "oracle_error_type" not in payload["metadata"]
    # stderr is redirected into the log, so the failure's own diagnostics reach the report.
    assert "boom" in payload["output"][0]["content"][0]["text"]


def test_provider_error_type_is_reported(monkeypatch):
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="", return_code=137, error_type="timeout"))
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert payload["metadata"]["oracle_error_type"] == "timeout"
    assert "solve.sh exited with 137" in payload["output"][0]["content"][0]["text"]


def test_missing_timeout_wrapper_is_a_wrapper_error_not_a_solution_failure(monkeypatch):
    """The guard exits 127 before the log exists: the image lacks `timeout` or `bash`, solve.sh never ran."""
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="", return_code=127), log_present=False)
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert payload["metadata"]["oracle_error_type"] == "wrapper"
    assert "could not be started" in payload["output"][0]["content"][0]["text"]


def test_solution_exiting_127_itself_is_a_solution_failure(monkeypatch):
    sandbox = FakeSandbox(result=SandboxExecResult(stdout="", stderr="foo: not found", return_code=127))
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    payload = client.post("/v1/responses", json=RESPONSES_BODY).json()

    assert payload["metadata"]["oracle"] == "failed"
    assert "oracle_error_type" not in payload["metadata"]
    assert "foo: not found" in payload["output"][0]["content"][0]["text"]


def test_seed_requires_sandbox_access(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(with_sandbox=False)).status_code == 422


def test_seed_requires_the_task_agent_timeout(monkeypatch):
    """The solve budget is the seed's agent_timeout_sec; without it the oracle has no task to read one from."""
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    response = client.post("/v1/agent_sessions", json=seed_body(timeout=None))
    assert response.status_code == 422
    assert "agent_timeout_sec" in response.text


def test_responses_requires_seeded_session(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    assert TestClient(agent.setup_webserver()).post("/v1/responses", json=RESPONSES_BODY).status_code == 404


def test_legacy_run_route_is_not_supported(monkeypatch):
    agent, _, _ = make_agent(monkeypatch)
    response = TestClient(agent.setup_webserver()).post("/run", json={"responses_create_params": {"input": []}})
    assert response.status_code == 501


@pytest.mark.parametrize("cap,expected", [(None, 7200.0), (8000.0, 7200.0), (60.0, 60.0), (0.5, 0.5)])
def test_solve_uses_the_seeds_timeout_unless_explicitly_capped(monkeypatch, cap, expected):
    agent, sandbox, _ = make_agent(monkeypatch)
    if cap is not None:
        agent.config.max_solve_timeout_s = cap
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(timeout=7200.0)).status_code == 200
    assert client.post("/v1/responses", json=RESPONSES_BODY).status_code == 200
    # The exec outlives the in-container `timeout` by the grace period.
    run = solve_run(sandbox)
    assert run["timeout_s"] == expected + 60.0
    # The in-container budget is a whole number of seconds, never 0 (which would mean no limit).
    assert f"timeout -s KILL {max(1, math.ceil(expected))} bash" in run["command"]


def test_transient_disconnect_is_retried_by_base_session_lifecycle(monkeypatch):
    agent, sandbox, _ = make_agent(monkeypatch)
    attempts = []

    async def disconnect():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient disconnect")
        sandbox.disconnected = True

    monkeypatch.setattr(sandbox, "disconnect", disconnect)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    body = {"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}}
    assert client.post("/v1/agent_sessions/close", json=body).status_code == 500
    assert client.post("/v1/agent_sessions/close", json=body).status_code == 200
    assert len(attempts) == 2 and sandbox.disconnected


def test_unwritable_log_dir_is_an_oracle_failure_not_a_500(monkeypatch):
    """An image whose user cannot make /logs/agent writable: scored as a failed oracle run, with the reason."""

    class ReadOnlyLogs(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)
            if command.startswith("mkdir -p /logs/agent"):
                return SandboxExecResult(stdout="", stderr="operation not permitted", return_code=1)
            return SandboxExecResult(stdout="", stderr="", return_code=0)

    sandbox = ReadOnlyLogs()
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    response = client.post("/v1/responses", json=RESPONSES_BODY)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["metadata"] == {"oracle": "failed", "oracle_error_type": "transfer"}
    assert "operation not permitted" in payload["output"][0]["content"][0]["text"]
    assert not any("bash /solution/solve.sh" in call["command"] for call in sandbox.execs)


def test_provider_error_in_the_probe_is_not_scored(monkeypatch):
    """A lost sandbox answering the solve.sh probe propagates (a retryable failed episode), never `unvalidated`."""

    class Lost(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)
            return SandboxExecResult(stdout="", stderr="container gone", return_code=1, error_type="sandbox")

    sandbox = Lost()
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 500, response.text
    assert [call["command"] for call in sandbox.execs] == ["test -f /solution/solve.sh"]


def test_provider_error_in_the_log_dir_setup_is_not_scored_either(monkeypatch):
    """`exec_as_root` turns a provider `error_type` on the /logs/agent preparation into the same propagating
    failure. The fake fails only that command, so the probe passes and the request is proven to reach this step."""

    class Lost(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None, **kwargs):
            await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user, **kwargs)
            if command.startswith("mkdir -p /logs/agent"):
                return SandboxExecResult(stdout="", stderr="container gone", return_code=125, error_type="sandbox")
            return SandboxExecResult(stdout="", stderr="", return_code=0)

    sandbox = Lost()
    agent, _, _ = make_agent(monkeypatch, sandbox=sandbox)
    client = TestClient(agent.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 500, response.text
    # The probe found the solution, the log-directory preparation ran as root and was the step that failed, and
    # nothing ran after it: no fallback to the default user (the failure is the provider's, not a refused root
    # switch) and no solve.sh.
    assert sandbox.execs[0]["command"] == "test -f /solution/solve.sh"
    assert sandbox.execs[1]["command"].startswith(f"mkdir -p {module.SOLVE_LOG_DIR}")
    assert sandbox.execs[1]["user"] == TRANSFER_USER
    assert len(sandbox.execs) == 2


def test_verbose_logging_traces_the_solve_run_at_debug_level_only(monkeypatch, caplog):
    import logging

    agent, sandbox, _ = make_agent(
        monkeypatch, sandbox=FakeSandbox(result=SandboxExecResult(stdout="", stderr="boom", return_code=3))
    )
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        response = client.post("/v1/responses", json=RESPONSES_BODY)
    assert response.status_code == 200 and response.json()["metadata"]["oracle"] == "failed"
    records = [record for record in caplog.records if record.name == module.__name__]
    # Each line names the task and the episode (its capture key) the session was seeded for, as the environment
    # server's trace does, so concurrent reference solutions, all at the same script path, can be told apart.
    assert [record.getMessage() for record in records] == [
        "hello: episode r1: running /solution/solve.sh as runner in /app with a 120s budget",
        "hello: episode r1: /solution/solve.sh exited 3 (failed)",
    ]
    # Only `--verbose` (`verbose: true`, which raises the root level) shows the trace.
    assert {record.levelno for record in records} == {logging.DEBUG}
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=module.__name__):
        client.post("/v1/responses", json=RESPONSES_BODY)
    assert not [record for record in caplog.records if record.name == module.__name__]

    # A task that names no [agent].user (the common case) runs as the sandbox's default user, and says so.
    agent, _, _ = make_agent(monkeypatch)
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_body(session="ag-2", user=None)).status_code == 200
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        assert client.post("/v1/responses", json=RESPONSES_BODY).json()["metadata"]["oracle"] == "solved"
    assert [record.getMessage() for record in caplog.records if record.name == module.__name__] == [
        "hello: episode r1: running /solution/solve.sh as the sandbox default user in /app with a 120s budget",
        "hello: episode r1: /solution/solve.sh exited 0 (solved)",
    ]


def test_concurrent_oracle_traces_name_their_task_and_episode(monkeypatch, caplog):
    """Review (round 38, non-blocking): concurrent reference solutions run the same script path with possibly
    identical settings, so the start and exit lines carry the task and episode identifiers from the session's seed
    request, correlating with each other and with the environment server's `<task>: episode <key>` traces."""
    import logging

    agent, _, _ = make_agent(
        monkeypatch, sandbox=FakeSandbox(result=SandboxExecResult(stdout="", stderr="boom", return_code=3))
    )
    app = agent.setup_webserver()
    # One client per session: the session rides on the seed's cookie.
    first, second = TestClient(app), TestClient(app)
    assert first.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert (
        second.post(
            "/v1/agent_sessions", json=seed_body(session="ag-2", task="world", rollout="r2", attempt=1, user=None)
        ).status_code
        == 200
    )
    with caplog.at_level(logging.DEBUG, logger=module.__name__):
        assert first.post("/v1/responses", json=RESPONSES_BODY).json()["metadata"]["oracle"] == "failed"
        assert second.post("/v1/responses", json=RESPONSES_BODY).json()["metadata"]["oracle"] == "failed"
    assert [record.getMessage() for record in caplog.records if record.name == module.__name__] == [
        "hello: episode r1: running /solution/solve.sh as runner in /app with a 120s budget",
        "hello: episode r1: /solution/solve.sh exited 3 (failed)",
        "world: episode r2-a1: running /solution/solve.sh as the sandbox default user in /app with a 120s budget",
        "world: episode r2-a1: /solution/solve.sh exited 3 (failed)",
    ]
