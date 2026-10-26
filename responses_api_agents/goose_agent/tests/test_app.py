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
        "create table messages (id integer primary key autoincrement, session_id text, role text, "
        "content_json text, created_timestamp integer)"
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
        "insert into messages (session_id, role, content_json, created_timestamp) values (?, ?, ?, ?)",
        [(s, r, json.dumps(c), t) for s, r, c, t in messages],
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


@pytest.mark.parametrize("failure", ["exit", "timeout", "returned_timeout"])
async def test_execution_failure_scores_zero_and_keeps_transcript(agent, failure):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def failing(command, **kwargs):
        if "goose run" not in command:
            return await run_exec(command, **kwargs)
        if failure == "timeout":
            raise TimeoutError()
        if failure == "returned_timeout":
            # The Docker provider returns its timeout instead of raising it.
            return SimpleNamespace(return_code=125, error_type="timeout", stdout="")
        return SimpleNamespace(return_code=1, error_type=None, stdout="")

    sandbox.exec.side_effect = failing
    result = await server.run(SimpleNamespace(cookies={}), run_body())
    assert result.reward == 0 and result.goose_failed
    assert result.goose_error_type == {"timeout": "TimeoutError", "returned_timeout": "timeout"}.get(failure)
    assert server.server_client.post.await_args.kwargs["json"]["response"]["output"] == []
    assert result.response.output  # The original generation is retained for inspection.
    if failure != "exit":
        assert any(
            c.args == () and "goose run" not in c.kwargs["command"] and "tar czf" in c.kwargs["command"]
            for c in sandbox.exec.await_args_list
        )
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("failure", ["install", "cancel"])
async def test_infrastructure_failures_propagate_and_stop_sandbox(agent, failure):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def failing(command, **kwargs):
        if failure == "install" and "sha256sum" in command:
            return SimpleNamespace(return_code=1, error_type=None, stdout="", stderr="checksum mismatch")
        if failure == "cancel" and "goose run" in command:
            raise asyncio.CancelledError()
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
