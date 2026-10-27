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
import sqlite3
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.goose_agent import app
from responses_api_agents.goose_agent.app import (
    GOOSE_ARCHIVE_SHA256,
    GooseAgent,
    GooseAgentConfig,
    GooseAgentRunRequest,
    goose_archive_name,
    goose_messages_to_output,
    goose_stop_reason,
    max_concurrent,
)


THINK = {"type": "thinking", "thinking": "Fan out.", "signature": ""}


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "toolRequest",
        "id": call_id,
        "toolCall": {"status": "success", "value": {"name": name, "arguments": arguments}},
    }


def _result(call_id: str, text: str) -> dict:
    return {
        "type": "toolResponse",
        "id": call_id,
        "toolResult": {"status": "success", "value": {"content": [{"type": "text", "text": text}], "isError": False}},
    }


def write_sessions_archive(path: Path) -> None:
    """A Goose session store shaped like a real 1.54.0 run: a root that delegates to two children."""
    db = path.parent / "sessions" / "sessions.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute(
        "create table sessions (id text primary key, session_type text, parent_session_id text, created_at timestamp)"
    )
    con.execute(
        "create table messages (id integer primary key autoincrement, message_id text, session_id text, role text, "
        "content_json text, created_timestamp integer, metadata_json text)"
    )
    con.execute(
        "create table usage_ledger (id integer primary key autoincrement, session_id text, input_tokens integer, "
        "output_tokens integer, cache_read_tokens integer)"
    )
    sessions = [("s1", "user", None, "2026-10-13 06:07:02")]
    sessions += [(f"s{i}", "sub_agent", "s1", "2026-10-13 06:07:30") for i in (2, 3)]
    con.executemany("insert into sessions values (?, ?, ?, ?)", sessions)
    messages = [
        ("s1", "user", [{"type": "text", "text": "Weather for two cities"}], 100),
        ("s1", "user", [{"type": "text", "text": "<turn-context>...</turn-context>"}], 100),
        (
            "s1",
            "assistant",
            [THINK, {"type": "text", "text": "Delegating."}, _call("c1", "delegate", {"instructions": "Paris"})],
            101,
        ),
        ("s1", "user", [_result("c1", "Task s2 started")], 101),
        # Goose splits one completion into one message per tool call and repeats the thinking.
        ("s1", "assistant", [THINK, _call("c2", "delegate", {"instructions": "Tokyo"})], 101),
        ("s1", "user", [_result("c2", "Task s3 started")], 101),
        ("s2", "assistant", [_call("w1", "get_weather", {"city": "Paris"})], 102),
        ("s3", "assistant", [_call("w2", "get_weather", {"city": "Tokyo"})], 103),
        ("s2", "assistant", [{"type": "text", "text": "Sunny in Paris."}], 106),
        ("s3", "assistant", [{"type": "text", "text": "Sunny in Tokyo."}], 105),
        ("s1", "assistant", [{"type": "text", "text": "Paris and Tokyo are sunny."}], 110),
    ]
    con.executemany(
        "insert into messages (message_id, session_id, role, content_json, created_timestamp) values (?, ?, ?, ?, ?)",
        [(f"m{i}", s, r, json.dumps(c), t) for i, (s, r, c, t) in enumerate(messages)],
    )
    con.executemany(
        "insert into usage_ledger (session_id, input_tokens, output_tokens, cache_read_tokens) values (?, ?, ?, ?)",
        [("s1", 100, 10, 5), ("s2", 50, 5, 0), ("s3", 60, 6, None)],
    )
    con.commit()
    con.close()
    with tarfile.open(path, "w:gz") as archive:
        archive.add(db.parent, arcname="sessions")


def response(value: dict) -> SimpleNamespace:
    return SimpleNamespace(cookies={}, ok=True, read=AsyncMock(return_value=json.dumps(value).encode()))


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "sandbox_server_url", lambda _, **kwargs: "http://model.example:8000")
    monkeypatch.setattr(
        app,
        "seed_mcp_servers",
        AsyncMock(return_value={"weather": {"url": "http://tools/mcp", "headers": {"X-Token": "scoped"}}}),
    )
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        if kwargs["url_path"] != "/verify":
            return response({})
        final = [i for i in kwargs["json"]["response"]["output"] if i.get("type") == "message"]
        return response(kwargs["json"] | {"reward": float(bool(final) and "sunny" in json.dumps(final[-1]))})

    client.post = AsyncMock(side_effect=post)
    archive_source = tmp_path / "archive" / "sessions.tgz"
    archive_source.parent.mkdir()
    write_sessions_archive(archive_source)
    config = GooseAgentConfig(
        name="goose",
        host="127.0.0.1",
        port=9000,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "weather"},
        model_server={"type": "responses_api_models", "name": "model"},
        tool_servers=[{"type": "resources_servers", "name": "weather"}],
        sandbox_provider="sandbox",
        subagent_model="child-model",
        system_prompt="Be brief.",
        env={"GOOSE_CONTEXT_LIMIT": "1000"},
        artifacts_dir=str(tmp_path / "artifacts"),
        execution_failure_reward_zero=True,
    )
    server = GooseAgent(config=config, server_client=client)
    staged = {}

    async def upload(local, remote):
        staged[remote] = Path(local).read_text() if Path(local).suffix != ".gz" else "archive"

    async def download(remote, local):
        Path(local).write_bytes(archive_source.read_bytes() if remote.endswith(".tgz") else b"")

    async def exec_(command, **kwargs):
        return SimpleNamespace(return_code=0, error_type=None, stdout="x86_64\n" if command == "uname -m" else "")

    sandbox = SimpleNamespace(
        upload=AsyncMock(side_effect=upload),
        download=AsyncMock(side_effect=download),
        exec=AsyncMock(side_effect=exec_),
        stop=AsyncMock(),
    )
    server._start_sandbox = AsyncMock(return_value=sandbox)
    return server, sandbox, staged


def run_body() -> GooseAgentRunRequest:
    return GooseAgentRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Use tools."},
                    {"role": "user", "content": "Weather for two cities"},
                ],
                "temperature": 0.6,
            }
        }
    )


async def test_run_configures_goose_and_exports_root_and_children(agent):
    server, sandbox, staged = agent
    result = await server.run(SimpleNamespace(cookies={}), run_body())

    assert result.reward == 1 and not result.goose_failed and result.goose_exit_code == 0
    assert result.goose_subagents == 2 and result.goose_max_concurrent_subagents == 2
    # Root output only: one reasoning item despite the split messages, children stay out of the root transcript.
    output = result.response.output
    assert [item.type for item in output] == [
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert json.loads(output[4].arguments) == {"instructions": "Tokyo"} and output[5].output == "Task s3 started"
    assert output[-1].content[0].text == "Paris and Tokyo are sunny."
    # Usage covers the root and both delegates.
    assert result.response.usage.input_tokens == 210 and result.response.usage.output_tokens == 21
    assert result.response.usage.input_tokens_details.cached_tokens == 5

    config = json.loads(next(v for k, v in staged.items() if k.endswith("config/config.yaml")))
    assert config["extensions"]["summon"] == {"enabled": True}
    assert config["extensions"]["weather"]["type"] == "streamable_http"
    assert config["extensions"]["weather"]["headers"] == {"X-Token": "scoped"}
    assert next(v for k, v in staged.items() if k.endswith("prompt.txt")) == "Weather for two cities"
    assert next(v for k, v in staged.items() if k.endswith("system.txt")) == "Be brief.\n\nUse tools."

    commands = [c.kwargs for c in sandbox.exec.await_args_list]
    install = next(c for c in commands if "sha256sum" in c["command"])
    run = next(c for c in commands if "goose run" in c["command"])
    assert GOOSE_ARCHIVE_SHA256["x86_64"] in install["command"]
    assert "goose-x86_64-unknown-linux-musl.tar.gz" in install["command"] and "sha256sum -c" in install["command"]
    env = run["env"]
    assert env["OPENAI_BASE_URL"] == "http://model.example:8000/v1"
    assert env["GOOSE_SUBAGENT_MODEL"] == "child-model" and env["GOOSE_MODE"] == "auto"
    assert env["GOOSE_TEMPERATURE"] == "0.6" and env["GOOSE_CONTEXT_LIMIT"] == "1000"
    assert "--max-turns 1000" in run["command"] and "--system" in run["command"]
    assert (Path(result.goose_results_dir) / "generation.json").exists()
    sandbox.stop.assert_awaited_once()


async def test_solo_control_disables_delegate(agent):
    server, _, staged = agent
    server.config.delegate = False
    await server.run(SimpleNamespace(cookies={}), run_body())
    config = json.loads(next(v for k, v in staged.items() if k.endswith("config/config.yaml")))
    assert config["extensions"]["summon"] == {"enabled": False}


async def test_local_archive_is_uploaded_instead_of_downloaded(agent, tmp_path):
    server, sandbox, staged = agent
    server.config.goose_archive_dir = str(tmp_path)
    (tmp_path / "goose-x86_64-unknown-linux-musl.tar.gz").write_bytes(b"\0")
    await server.run(SimpleNamespace(cookies={}), run_body())
    assert any(k.endswith("goose-x86_64-unknown-linux-musl.tar.gz") for k in staged)
    assert not any("curl" in c.kwargs["command"] for c in sandbox.exec.await_args_list)


@pytest.mark.parametrize("failure", ["exit", "returned_timeout", "server_timeout"])
async def test_execution_failure_scores_zero_and_keeps_transcript(agent, failure):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def failing(command, **kwargs):
        if "goose run" not in command:
            return await run_exec(command, **kwargs)
        if failure == "returned_timeout":
            # The Docker provider returns its timeout instead of raising it.
            return SimpleNamespace(return_code=125, error_type="timeout", stdout="")
        if failure == "server_timeout":
            # OpenSandbox enforces the timeout server-side and reports exit 124, before the wrapper's own tar.
            return SimpleNamespace(return_code=124, error_type=None, stdout="")
        return SimpleNamespace(return_code=1, error_type=None, stdout="")

    sandbox.exec.side_effect = failing
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.reward == 0 and result.goose_failed
    assert result.goose_error_type == (None if failure == "exit" else "timeout")
    assert server.server_client.post.await_args.kwargs["json"]["response"]["output"] == []
    assert result.response.output  # The original generation is retained for inspection.
    if failure != "exit":
        assert any(
            c.args == () and "goose run" not in c.kwargs["command"] and "tar czf" in c.kwargs["command"]
            for c in sandbox.exec.await_args_list
        )
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("failure", ["install", "cancel", "disconnect", "poll_timeout", "lost_status"])
async def test_infrastructure_failures_propagate_and_stop_sandbox(agent, failure):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def failing(command, **kwargs):
        if failure == "install" and "sha256sum" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="checksum mismatch")
        if "goose run" in command:
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "disconnect":
                raise ConnectionResetError()
            if failure == "poll_timeout":
                # A status poll that keeps timing out raises, while Goose may still be running.
                raise TimeoutError()
            if failure == "lost_status":
                return SimpleNamespace(return_code=125, error_type="sandbox", stdout="")
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = failing
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await server.run(SimpleNamespace(cookies={}), run_body())
    assert not any(c.kwargs["url_path"] == "/verify" for c in server.server_client.post.await_args_list)
    sandbox.stop.assert_awaited_once()


async def test_sandbox_stays_up_until_verification_returns(agent):
    server, sandbox, _ = agent
    post = server.server_client.post.side_effect

    async def checked(**kwargs):
        if kwargs["url_path"] == "/verify":
            # A resource-owned sandbox is graded in place.
            sandbox.stop.assert_not_awaited()
        return await post(**kwargs)

    server.server_client.post.side_effect = checked
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.reward == 1
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("error", [RuntimeError("verifier down"), asyncio.CancelledError()])
async def test_verify_failure_or_cancellation_still_stops_the_sandbox(agent, error):
    server, sandbox, _ = agent
    post = server.server_client.post.side_effect

    async def failing(**kwargs):
        if kwargs["url_path"] == "/verify":
            raise error
        return await post(**kwargs)

    server.server_client.post.side_effect = failing
    with pytest.raises(type(error)):
        await server.run(SimpleNamespace(cookies={}), run_body())
    sandbox.stop.assert_awaited_once()


async def test_request_instructions_reach_the_system_prompt(agent):
    server, sandbox, staged = agent
    body = run_body()
    body.responses_create_params.instructions = "Return only the digit."
    await server.run(SimpleNamespace(cookies={}), body)
    assert next(v for k, v in staged.items() if k.endswith("system.txt")) == (
        "Be brief.\n\nReturn only the digit.\n\nUse tools."
    )


@pytest.mark.parametrize("base_url", ["https://inference.example/v1", None])
async def test_restricted_network_allows_the_endpoint_goose_calls(agent, monkeypatch, base_url):
    server, _, _ = agent
    server.config.network_access = "model_only"
    server.config.base_url = base_url
    allowed = {}
    monkeypatch.setattr(app, "create_provider", lambda _: SimpleNamespace(name="opensandbox"))
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "restricted_network_policy", lambda provider, urls: allowed.setdefault("urls", urls))
    started = AsyncMock()
    monkeypatch.setattr(app, "AsyncSandbox", lambda provider: SimpleNamespace(start=started))
    await type(server)._start_sandbox(server, {})
    assert allowed["urls"] == [base_url or "http://model.example:8000"]
    started.assert_awaited_once()


def test_archive_names_and_concurrency():
    assert goose_archive_name("arm64") == "goose-aarch64-unknown-linux-musl.tar.gz"
    with pytest.raises(ValueError, match="riscv64"):
        goose_archive_name("riscv64")
    assert max_concurrent([(0, 5), (1, 2), (6, 7)]) == 2
    assert max_concurrent([(0, 1), (1, 2)]) == 2  # Second-resolution timestamps: touching counts as overlap.
    assert max_concurrent([]) == 0


def test_thinking_dedup_is_per_turn():
    think = {"type": "thinking", "thinking": "plan"}

    def assistant(text):
        return {"role": "assistant", "content": [think, {"type": "text", "text": text}]}

    def user(text):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    items = goose_messages_to_output([user("task"), assistant("a"), assistant("b"), user("next"), assistant("c")])
    assert [item.type for item in items] == ["reasoning", "message", "message", "reasoning", "message"]


@pytest.mark.parametrize("fault", ["lost_status", "disconnect"])
async def test_recorded_exit_code_wins_over_a_provider_fault(agent, fault):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        if "goose run" in command:
            if fault == "disconnect":
                raise ConnectionResetError()
            return SimpleNamespace(return_code=125, error_type="sandbox", stdout="")
        if command.startswith("cat ") and command.endswith("/exit_code"):
            return SimpleNamespace(return_code=0, error_type=None, stdout="0\n")  # Goose finished
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = exec_
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.reward == 1 and not result.goose_failed
    assert result.goose_exit_code == 0 and result.goose_error_type is None


# Notices verbatim from Goose 1.54.0 runs against a stub model server (500, 429, connection refused, context overflow).
NOTICES = {
    "server_error": "Ran into this error: Server error: Server error (500 Internal Server Error) at "
    "http://127.0.0.1:18991/v1/chat/completions: boom.\n\nPlease retry if you think this is a transient or recoverable error.",
    "rate_limit": "Ran into this error: Rate limit exceeded: rate limited.\n\nPlease retry if you think this is a transient "
    "or recoverable error.",
    "network": "Network error: Could not connect to 127.0.0.1:9 \u2014 check your network connection and try again.\n\n"
    "Please resend your message to try again.",
    "not_found": "Ran into this error: Request failed: Resource not found (404) at "
    "http://127.0.0.1:18983/v1/chat/completions: model not found.\n\nPlease retry if you think this is a transient or "
    "recoverable error.",
    "model_error": "Ran into this error trying to compact: Failed to compact: the base prompt (system prompt, tool schemas, "
    "and conversation) exceeds the model's effective context window.",
    "max_turns": "I've reached the maximum number of actions I can do without user input. Would you like me to continue?",
}


def _stream(last_id: str, text: str) -> str:
    # A banner precedes the stream-json events. Model messages are persisted under their streamed id (m10 is the root's
    # final message in the fixture store), Goose's own notice is not.
    events = [
        {"type": "message", "message": {"id": i, "role": "assistant", "content": [{"type": "text", "text": t}]}}
        for i, t in (("m10", "Paris and Tokyo are sunny."), (last_id, text))
    ]
    return "    goose is ready\n" + "\n".join(json.dumps(e) for e in events + [{"type": "complete"}])


def _serve_stdout(sandbox, stdout: str) -> None:
    download = sandbox.download.side_effect

    async def download_(remote, local):
        if remote.endswith("stdout.jsonl"):
            Path(local).write_text(stdout)
        else:
            await download(remote, local)

    sandbox.download.side_effect = download_


@pytest.mark.parametrize("reason", ["model_error", "max_turns"])
async def test_goose_stopping_early_with_exit_zero_is_a_failure(agent, reason):
    server, sandbox, _ = agent
    _serve_stdout(sandbox, _stream("msg_notice", NOTICES[reason]))
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.goose_exit_code == 0 and result.goose_failed and result.goose_error_type == reason
    assert result.reward == 0
    sessions = [{"messages": [{"message_id": "m10"}]}]
    assert goose_stop_reason(_stream("m10", "Paris and Tokyo are sunny."), sessions) is None


@pytest.mark.parametrize("notice", ["server_error", "rate_limit", "network", "not_found"])
async def test_model_server_failure_fails_the_request(agent, notice):
    # The model server failed, not the policy: the request fails so it is retried, never a scored zero.
    server, sandbox, _ = agent
    _serve_stdout(sandbox, _stream("msg_notice", NOTICES[notice]))
    with pytest.raises(RuntimeError, match="model server"):
        await server.run(SimpleNamespace(cookies={}), run_body())
    assert not any(c.kwargs["url_path"] == "/verify" for c in server.server_client.post.await_args_list)
    sandbox.stop.assert_awaited_once()


async def test_rejected_credentials_fail_the_request(agent):
    # Goose 1.54.0 on a 401 persists an error block (under the id it streams) and exits 1.
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect
    message = {
        "id": "m10",
        "role": "assistant",
        "content": [
            {
                "type": "error",
                "kind": "authentication",
                "message": "Ran into this error: Authentication error: Authentication failed for "
                "http://127.0.0.1:18981/v1/chat/completions. Status: 401 Unauthorized. Response: invalid api key.",
            }
        ],
    }
    _serve_stdout(sandbox, json.dumps({"type": "message", "message": message}))

    async def exec_(command, **kwargs):
        if "goose run" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="")
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = exec_
    with pytest.raises(RuntimeError, match="model server"):
        await server.run(SimpleNamespace(cookies={}), run_body())
    assert not any(c.kwargs["url_path"] == "/verify" for c in server.server_client.post.await_args_list)


def test_persisted_goose_notices_are_error_blocks():
    # Goose persists a notice only as an error block (rejected credentials). Persisted text is the policy's answer.
    sessions = [{"messages": [{"message_id": "m1"}]}]

    def last(content):
        return json.dumps({"type": "message", "message": {"id": "m1", "role": "assistant", "content": content}})

    compaction = {"type": "error", "message": NOTICES["model_error"]}
    assert goose_stop_reason(last([compaction]), sessions) == "model_error"
    assert goose_stop_reason(last([{"type": "text", "text": NOTICES["model_error"]}]), sessions) is None
    assert goose_stop_reason(last([{"type": "text", "text": NOTICES["max_turns"]}]), sessions) is None


async def test_task_goes_in_on_stdin_so_the_system_prompt_is_kept(agent):
    # Goose 1.54.0 drops --system when the task comes from `-i <file>`; only `-i -` (stdin) keeps it.
    server, sandbox, _ = agent
    await server.run(SimpleNamespace(cookies={}), run_body())
    run = next(c.kwargs["command"] for c in sandbox.exec.await_args_list if "goose run" in c.kwargs["command"])
    assert " -i - " in run and "/prompt.txt > " in run and " -i /" not in run
    assert '--system="$(cat ' in run


async def test_output_limit_stop_is_a_failed_incomplete_run(agent):
    # vLLM answers a context overflow with an empty completion cut at the length limit; Goose persists it and exits 0.
    server, sandbox, _ = agent
    cut = {"id": "m10", "role": "assistant", "content": [], "metadata": {"outputTokenLimitReached": True}}
    _serve_stdout(sandbox, json.dumps({"type": "message", "message": cut}))
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.goose_failed and result.goose_error_type == "length" and result.reward == 0
    assert result.response.status == "incomplete"
    assert result.response.incomplete_details.reason == "max_output_tokens"


async def test_persisted_policy_text_is_never_read_as_a_goose_notice(agent):
    # The policy's own final answer (persisted under its streamed id) may start like a notice.
    server, sandbox, _ = agent
    _serve_stdout(sandbox, _stream("m10", "Network error: the service in the task is down, so I could not finish."))
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert not result.goose_failed and result.goose_error_type is None


async def test_compaction_that_fails_at_the_model_server_fails_the_request(agent):
    server, sandbox, _ = agent
    notice = "Ran into this error trying to compact: Server error: Server error (502 Bad Gateway) at http://m/v1."
    _serve_stdout(sandbox, _stream("msg_notice", notice))
    with pytest.raises(RuntimeError, match="model server"):
        await server.run(SimpleNamespace(cookies={}), run_body())


async def test_kill_before_the_deadline_is_a_scored_failure(agent):
    # 137 reaches the agent only for a kill before the deadline (the wrapper records a deadline kill as 124). The sandbox
    # cgroup is shared with the policy's commands, so an out-of-memory kill is the episode's failure, not a retry.
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        if "goose run" in command:
            assert "-eq 137 ] && [ $(($(date +%s) - start)) -ge " in command
            return SimpleNamespace(return_code=137, error_type=None, stdout="")
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = exec_
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.goose_failed and result.goose_exit_code == 137 and result.goose_error_type == "killed"
    assert result.reward == 0


async def test_goose_runs_under_an_in_sandbox_deadline(agent):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        if "goose run" in command:
            # The deadline ended Goose; the wrapper archived its sessions and recorded the exit status.
            return SimpleNamespace(return_code=124, error_type=None, stdout="")
        if command.startswith("cat ") and command.endswith("/exit_code"):
            return SimpleNamespace(return_code=0, error_type=None, stdout="124\n")
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = exec_
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    run = next(c.kwargs for c in sandbox.exec.await_args_list if "goose run" in c.kwargs["command"])
    deadline = int(run["command"].split("timeout --kill-after=10 ", 1)[1].split()[0])
    # It fires before the provider's own deadline, with room to archive the sessions.
    assert deadline + 10 < run["timeout_s"] == server.config.timeout
    assert result.goose_failed and result.goose_error_type == "timeout" and result.goose_exit_code == 124
    assert result.reward == 0 and result.response.output


async def test_extension_that_fails_to_start_is_an_infrastructure_failure(agent):
    server, sandbox, _ = agent
    download = sandbox.download.side_effect

    async def download_(remote, local):
        if remote.endswith("stderr.log"):
            Path(local).write_text(
                "  Failed to start extension 'weather' (connection refused), continuing without it\n"
            )
        else:
            await download(remote, local)

    sandbox.download.side_effect = download_
    with pytest.raises(RuntimeError, match="weather"):
        await server.run(SimpleNamespace(cookies={}), run_body())
    assert not any(c.kwargs["url_path"] == "/verify" for c in server.server_client.post.await_args_list)
    sandbox.stop.assert_awaited_once()


async def test_failed_run_in_a_resource_owned_sandbox_is_graded_in_place(agent):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def exec_(command, **kwargs):
        if "goose run" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="")
        return await run_exec(command, **kwargs)

    async def post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return response({"sandbox_handle": "sb-1"})
        # The grader scores the patch left in the sandbox, whatever the response holds.
        return response(kwargs["json"] | {"reward": 1.0})

    sandbox.exec.side_effect = exec_
    server.server_client.post.side_effect = post
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.goose_failed and result.reward == 1
    assert server.server_client.post.await_args.kwargs["json"]["response"]["output"]


@pytest.mark.parametrize("system", ["- Answer in one line.", "x" * (128 * 1024)])
async def test_system_prompt_is_passed_as_one_option_value(agent, system):
    server, sandbox, _ = agent
    server.config.system_prompt = system
    if len(system) >= 128 * 1024:
        # Linux caps a single argv value at 128 KiB, so Goose could never start.
        with pytest.raises(ValueError, match="argument limit"):
            await server.run(SimpleNamespace(cookies={}), run_body())
        assert not any("goose run" in c.kwargs["command"] for c in sandbox.exec.await_args_list)
        return
    await server.run(SimpleNamespace(cookies={}), run_body())
    run = next(c.kwargs["command"] for c in sandbox.exec.await_args_list if "goose run" in c.kwargs["command"])
    assert '--system="$(cat ' in run


def test_goose_only_messages_are_not_output():
    def text(role, value, visible=True):
        return {"role": role, "content": [{"type": "text", "text": value}], "user_visible": visible}

    # Compaction hides earlier messages from the agent but keeps them visible, and adds Goose-only ones.
    messages = [
        {"role": "assistant", "content": [_call("c1", "shell", {"command": "ls"})], "user_visible": True},
        {"role": "user", "content": [_result("c1", "a.txt")], "user_visible": True},
        text("user", "SUMMARY", visible=False),
        text("assistant", "Your context was compacted.", visible=False),
        text("assistant", "Done."),
    ]
    items = goose_messages_to_output(messages)
    assert [item.type for item in items] == ["function_call", "function_call_output", "message"]
    assert items[-1].content[0].text == "Done."


async def test_delegate_model_server_failure_fails_the_request(agent, tmp_path):
    # summon returns a delegate's (unpersisted) provider-error notice to the root as a successful tool result.
    server, sandbox, _ = agent
    archive = tmp_path / "delegate" / "sessions.tgz"
    archive.parent.mkdir()
    write_sessions_archive(archive)
    db = archive.parent / "sessions" / "sessions.db"
    con = sqlite3.connect(db)
    notice = "Ran into this error: Server error: Server error (503 Service Unavailable) at http://m/v1: refit."
    con.execute("update messages set content_json = ? where message_id = 'm3'", (json.dumps([_result("c1", notice)]),))
    con.commit()
    con.close()
    archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(db.parent, arcname="sessions")

    async def download(remote, local):
        Path(local).write_bytes(archive.read_bytes() if remote.endswith(".tgz") else b"")

    sandbox.download.side_effect = download
    with pytest.raises(RuntimeError, match="model server"):
        await server.run(SimpleNamespace(cookies={}), run_body())


def test_system_prompt_limit_counts_the_option_and_nul(agent):
    # The argv value is "--system=<text>" plus its terminating NUL, capped at 128 KiB.
    server, _, _ = agent
    body = run_body()
    body.responses_create_params.input = [{"role": "user", "content": "hi"}]
    fits = "x" * (128 * 1024 - len("--system=") - 1)
    server.config.system_prompt = fits + "x"
    with pytest.raises(ValueError, match="argument limit"):
        asyncio.run(server._run_goose(None, body, {}, Path("/nonexistent")))
    server.config.system_prompt = fits
    with pytest.raises(AttributeError):  # Past the limit check, at the first use of the (absent) sandbox.
        asyncio.run(server._run_goose(None, body, {}, Path("/nonexistent")))
