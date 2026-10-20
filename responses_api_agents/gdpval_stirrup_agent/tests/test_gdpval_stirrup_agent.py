# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDPVal Stirrup training agent (no sandbox, no policy)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.gdpval_stirrup_agent import app as agent_app
from responses_api_agents.gdpval_stirrup_agent.opensandbox_provider import SandboxCodeExecToolProvider
from responses_api_agents.gdpval_stirrup_agent.policy_client import (
    GDPValPolicyClient,
    PolicyContextLimitError,
    _message_chars,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_matches_aa_v2_template_with_working_dir():
    text = agent_app.build_user_prompt(
        agent_app.build_task_text("Write a memo.", "Finance", "Analysts"),
        ["reference_files/data.xlsx", "reference_files/brief.pdf"],
        working_dir="/workspace",
    )
    assert "Your working directory is `/workspace`" in text
    assert "cd /workspace/work && python build.py" in text
    assert "/root" not in text
    # reference list is sorted, one per line, inside the tag
    assert "<reference_files>\n- reference_files/brief.pdf\n- reference_files/data.xlsx\n\n</reference_files>" in text  # upstream keeps the trailing newline
    assert "<task>\nSector: Finance\nOccupation: Analysts\n\nWrite a memo.\n</task>" in text
    assert "`code_exec` tool" in text and "`finish` tool" in text and "abandon_task_finish" in text


def test_prompt_without_references_says_none():
    text = agent_app.build_user_prompt("Sector: \nOccupation: \n\nx", [], working_dir="/workspace")
    assert "<reference_files>\nNone\n</reference_files>" in text


def test_prompt_differs_from_upstream_only_by_working_dir():
    ours = Path(agent_app.PROMPT_TEMPLATE_PATH).read_text()
    upstream = ours.replace("{working_dir}", "/root")
    assert "{working_dir}" not in upstream
    assert upstream.count("/root") == 2


# ---------------------------------------------------------------------------
# Task fields
# ---------------------------------------------------------------------------


def _body(**extra):
    params = SimpleNamespace(metadata={"task_id": "task_1"}, temperature=None, top_p=None, max_output_tokens=None)
    body = SimpleNamespace(responses_create_params=params, model_extra=extra)
    return body


def test_task_fields_prefer_seed_then_row_then_metadata():
    body = _body(prompt="row prompt", sector="Row sector", reference_files='["a.pdf"]')
    seed = {"task": {"task_id": "task_1", "prompt": "seed prompt", "reference_files": ["b.pdf", "c/d.csv"]}}
    fields = agent_app.task_fields_from_request(body, seed)
    assert fields["prompt"] == "seed prompt"
    assert fields["reference_files"] == ["b.pdf", "c/d.csv"]
    assert fields["sector"] == "Row sector"  # not in seed -> row
    fields = agent_app.task_fields_from_request(body, None)
    assert fields["prompt"] == "row prompt"
    assert fields["reference_files"] == ["a.pdf"]  # JSON string decoded
    assert fields["task_id"] == "task_1"


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def test_classify_outcome():
    finish = SimpleNamespace(reason="done", paths=["/workspace/out.pdf"])
    abandon = SimpleNamespace(reason="no inputs")
    assert agent_app.classify_outcome(finish_params=finish, error=None, timed_out=False) == (None, None, None, "finish")
    assert agent_app.classify_outcome(finish_params=abandon, error=None, timed_out=False) == (None, None, None, "abandon")
    assert agent_app.classify_outcome(finish_params=None, error=None, timed_out=False) == (
        "agent_iteration_limit", None, None, "max_turns",
    )
    assert agent_app.classify_outcome(finish_params=None, error=None, timed_out=True) == (
        None, "agent_exec_timeout", None, "timeout",
    )
    assert agent_app.classify_outcome(finish_params=None, error=PolicyContextLimitError("x"), timed_out=False) == (
        "agent_context_limit", None, None, "context_limit",
    )
    from stirrup.core.exceptions import ContextOverflowError

    assert agent_app.classify_outcome(finish_params=None, error=ContextOverflowError("y"), timed_out=False)[0] == (
        "agent_context_limit"
    )

    class APITimeoutError(Exception):
        pass

    reason, incomplete, err, kind = agent_app.classify_outcome(
        finish_params=None, error=APITimeoutError("slow"), timed_out=False
    )
    assert (reason, incomplete, kind) == (None, "model_response_timeout", "model_timeout")
    assert "APITimeoutError" in err
    reason, incomplete, err, kind = agent_app.classify_outcome(finish_params=None, error=ValueError("boom"), timed_out=False)
    assert (reason, incomplete, kind) == (None, "agent_exec_incomplete", "error")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_response_splits_input_and_output_and_keeps_served_id():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "code_exec", "arguments": '{"cmd": "ls"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "code_exec", "content": "<shell_results>ok</shell_results>"},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "finish", "arguments": '{"reason": "ok", "paths": ["/workspace/a.txt"]}'}}],
        },
    ]
    tools = [{"type": "function", "function": {"name": "code_exec", "description": "d", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}, "strict": False}}]
    params, response = agent_app.export_response(
        messages=messages, tools=tools, response_id="chatcmpl-served-42", model="gdpval-policy", status="completed",
        metadata={"finish_kind": "finish", "turns": 2},
    )
    assert response.id == "chatcmpl-served-42"
    assert [getattr(i, "role", None) for i in params.input] == ["system", "user"]
    types = [i.type for i in response.output]
    assert types == ["function_call", "function_call_output", "message", "function_call"]
    assert response.output[0].call_id == "call_1" and response.output[-1].name == "finish"
    assert response.metadata == {"finish_kind": "finish", "turns": "2"}
    dumped = response.model_dump(mode="json")
    assert "prompt_token_ids" not in json.dumps(dumped)  # capture is external


# ---------------------------------------------------------------------------
# Policy client
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _reply(content="", tool_calls=None, finish="stop", prompt_tokens=100, completion_tokens=10, rid="chatcmpl-1"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=[SimpleNamespace(id=t["id"], function=SimpleNamespace(name=t["name"], arguments=t["arguments"])) for t in (tool_calls or [])] or None,
        reasoning_content=None,
    )
    msg.model_dump = lambda exclude_none=True: {
        "role": "assistant",
        "content": content,
        **({"tool_calls": [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}} for t in tool_calls]} if tool_calls else {}),
    }
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, completion_tokens_details=None)
    usage.model_dump = lambda exclude_none=True: {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return SimpleNamespace(id=rid, choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage)


def _client(replies, **kw):
    client = GDPValPolicyClient(
        model="gdpval-policy", base_url="http://policy/v1", context_window=kw.pop("context_window", 10_000),
        max_completion_tokens_cap=kw.pop("cap", 4_000), min_completion_tokens=kw.pop("floor", 512),
        completion_token_buffer=kw.pop("buffer", 100), **kw,
    )
    fake = _FakeCompletions(replies)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_budget_uses_usage_after_first_call():
    from stirrup.core.models import SystemMessage, UserMessage

    client, fake = _client([_reply("hi", prompt_tokens=3000, completion_tokens=200, rid="r1"), _reply("again", prompt_tokens=3300, completion_tokens=50, rid="r2")])
    history = [SystemMessage(content="s" * 300), UserMessage(content="u" * 600)]
    first = asyncio.run(client.generate(history, {}))
    assert first.content == "hi"
    # first call: chars // 3 = 300, budget = min(cap, ctx - 300 - 100) = 4000
    assert fake.requests[0]["max_completion_tokens"] == 4000
    history.append(first)
    history.append(UserMessage(content="x" * 3000))
    asyncio.run(client.generate(history, {}))
    # second: last prompt 3000 + completion 200 + 3000 new chars // 3 = 4200 -> ctx 10000 - 4200 - 100 = 5700 -> cap 4000
    assert fake.requests[1]["max_completion_tokens"] == 4000
    assert client.last_response_id == "r2"
    assert client.total_prompt_tokens == 6300
    msgs = client.final_messages()
    assert msgs[-1] == {"role": "assistant", "content": "again"}
    assert msgs[-2]["role"] == "user"


def test_budget_floor_and_context_limit_detection():
    from stirrup.core.models import UserMessage

    # Prompt estimate pushes the budget to the floor; empty length stop -> context limit.
    client, fake = _client([_reply("", finish="length", prompt_tokens=9_800, completion_tokens=0)], context_window=10_000, floor=512, buffer=100)
    with pytest.raises(PolicyContextLimitError):
        asyncio.run(client.generate([UserMessage(content="q" * 30_000)], {}))
    assert fake.requests[0]["max_completion_tokens"] == 512
    assert client.length_finishes == 1


def test_server_context_error_is_typed():
    from stirrup.core.models import UserMessage

    class BadRequest(Exception):
        status_code = 400

    client, _ = _client([BadRequest("This model's maximum context length is 10000 tokens. However, you requested 10500 tokens")])
    with pytest.raises(PolicyContextLimitError):
        asyncio.run(client.generate([UserMessage(content="q")], {}))
    assert client.calls and client.calls[0]["response"] is None and "maximum context" in client.calls[0]["error"]


def test_non_empty_length_finish_is_not_an_error():
    from stirrup.core.models import UserMessage

    client, _ = _client([_reply("partial answer", finish="length", prompt_tokens=100, completion_tokens=4000)])
    out = asyncio.run(client.generate([UserMessage(content="q")], {}))
    assert out.content == "partial answer" and client.length_finishes == 1


def test_message_chars_counts_tool_calls_and_reasoning():
    assert _message_chars([
        {"role": "assistant", "content": "ab", "reasoning_content": "cd", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
        {"role": "user", "content": [{"type": "text", "text": "xyz"}]},
    ]) == 2 + 2 + 1 + 2 + 3


# ---------------------------------------------------------------------------
# Sandbox provider
# ---------------------------------------------------------------------------


class _FakeSandbox:
    def __init__(self):
        self.files = {"/workspace/reference_files/a.txt": b"hello", "/workspace/out/b.bin": b"\x00" * 10}
        self.execs = []

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append((command, cwd, timeout_s))
        assert command.startswith("bash -c ")
        inner = command[len("bash -c "):]
        import shlex

        inner = shlex.split(inner)[0]
        if inner.startswith("test -f "):
            path = shlex.split(inner)[2]
            return SimpleNamespace(stdout="", stderr="", return_code=0 if path in self.files else 1, error_type=None)
        if inner.startswith("test -d "):
            path = shlex.split(inner)[2].rstrip("/")
            return SimpleNamespace(stdout="", stderr="", return_code=0 if any(f.startswith(path + "/") for f in self.files) else 1, error_type=None)
        if inner.startswith("find "):
            path = shlex.split(inner)[1].rstrip("/")
            listed = "\n".join(sorted(f for f in self.files if f.startswith(path + "/")))
            return SimpleNamespace(stdout=listed + "\n", stderr="", return_code=0, error_type=None)
        if inner.startswith("stat -c %s "):
            path = shlex.split(inner)[3]
            return SimpleNamespace(stdout=f"{len(self.files[path])}\n", stderr="", return_code=0, error_type=None)
        if inner.startswith("mkdir -p "):
            return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
        if inner == "sleep 100":
            raise TimeoutError("exec timed out")
        if inner == "exit 3":
            return SimpleNamespace(stdout="", stderr="bad", return_code=3, error_type=None)
        return SimpleNamespace(stdout=f"ran:{inner}", stderr="", return_code=0, error_type=None)

    async def download(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    async def upload(self, local, remote):
        self.files[remote] = Path(local).read_bytes()


def test_provider_runs_commands_in_working_dir_and_formats_timeouts():
    sandbox = _FakeSandbox()
    provider = SandboxCodeExecToolProvider(sandbox, working_dir="/workspace", shell_timeout=7)
    tool = asyncio.run(provider.__aenter__())
    assert tool.name == "code_exec"
    result = asyncio.run(provider.run_command("echo hi"))
    assert result.exit_code == 0 and result.stdout == "ran:echo hi"
    assert sandbox.execs[-1][1] == "/workspace" and sandbox.execs[-1][2] == 7
    timed = asyncio.run(provider.run_command("sleep 100"))
    assert timed.error_kind == "timeout" and "7 seconds" in timed.stderr
    failed = asyncio.run(provider.run_command("exit 3"))
    assert failed.exit_code == 3 and failed.stderr == "bad" and failed.error_kind is None
    assert provider.commands_run == 3 and provider.commands_timed_out == 1


def test_provider_file_primitives():
    sandbox = _FakeSandbox()
    provider = SandboxCodeExecToolProvider(sandbox)
    assert asyncio.run(provider.file_exists("reference_files/a.txt"))
    assert not asyncio.run(provider.file_exists("/workspace/missing"))
    assert asyncio.run(provider.is_directory("/workspace/out"))
    assert asyncio.run(provider.list_files("/workspace")) == ["out/b.bin", "reference_files/a.txt"]
    assert asyncio.run(provider.read_file_bytes("reference_files/a.txt")) == b"hello"
    with pytest.raises(FileNotFoundError):
        asyncio.run(provider.read_file_bytes("nope.txt"))
    asyncio.run(provider.write_file_bytes("new/c.txt", b"data"))
    assert sandbox.files["/workspace/new/c.txt"] == b"data"
    # save_output_files (base implementation) flattens by basename into a local dir
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        saved = asyncio.run(provider.save_output_files(["/workspace/out/b.bin", "/workspace/missing"], tmp))
        assert [s.output_path.name for s in saved.saved] == ["b.bin"] and "/workspace/missing" in saved.failed


def test_quiet_logger_satisfies_stirrup_interface():
    from stirrup.utils.logging import AgentLoggerBase

    abstract = {name for name in getattr(AgentLoggerBase, "__abstractmethods__", set())}
    missing = [m for m in abstract if not callable(getattr(agent_app._QuietLogger, m, None))]
    assert not missing, missing


def test_capture_store_root_reads_the_gym_block(monkeypatch, tmp_path):
    from responses_api_agents.gdpval_stirrup_agent import app as agent_app

    agent = object.__new__(agent_app.GDPValStirrupAgent)
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
    assert agent._capture_store_root() == tmp_path
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {"token_id_capture": {"enabled": True, "sink": "x:Y", "dir": str(tmp_path)}})
    assert agent._capture_store_root() is None
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {})
    assert agent._capture_store_root() is None


def test_capture_cleanup_keeps_records_another_handler_is_appending_to(monkeypatch, tmp_path):
    """Two /run handlers can share a rollout id; cleanup must not delete the other one's records."""
    import asyncio

    from nemo_gym.token_id_capture.lineage import stamp_continuation
    from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage
    from nemo_gym.token_id_capture.store import TokenCaptureStore
    from responses_api_agents.gdpval_stirrup_agent import app as agent_app

    def put(call_id, response_id):
        entry = TokenEntry(
            rollout_id="t0-g1", model_call_id=call_id, prompt_token_ids=[1], generation_token_ids=[2],
            generation_log_probs=[-0.1], output_items=[{"role": "assistant", "content": call_id}], response_id=response_id,
        )
        stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        stamp_continuation(entry, [{"role": "user", "content": "hello"}])
        TokenCaptureStore(tmp_path).append(entry)

    agent = object.__new__(agent_app.GDPValStirrupAgent)
    monkeypatch.setattr(agent_app, "get_global_config_dict", lambda: {"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
    monkeypatch.setattr(agent_app.GDPValStirrupAgent, "rollout_id_from_run", lambda self, body: "t0-g1")
    store = TokenCaptureStore(tmp_path)

    put("c1", "chatcmpl-ours")
    put("c2", "chatcmpl-theirs")  # the duplicate handler's later call
    asyncio.run(agent._cleanup_capture_records(object(), "chatcmpl-ours"))
    assert store.path_for("t0-g1").exists()

    asyncio.run(agent._cleanup_capture_records(object(), "chatcmpl-theirs"))
    assert not store.path_for("t0-g1").exists()

    put("c3", "chatcmpl-x")
    asyncio.run(agent._cleanup_capture_records(object(), None))  # no policy call of our own: legacy unconditional delete
    assert not store.path_for("t0-g1").exists()
