# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the real Stirrup agent loop with a scripted policy and an in-memory sandbox.

Exercises what the unit tests cannot: NeMoAgent + finish tools + the sandbox
exec provider + the policy client + the trajectory export, end to end, without a
GPU, a model server or OpenSandbox.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

from responses_api_agents.gdpval_stirrup_agent import app as agent_app
from responses_api_agents.gdpval_stirrup_agent.opensandbox_provider import SandboxCodeExecToolProvider
from responses_api_agents.gdpval_stirrup_agent.policy_client import GDPValPolicyClient
from responses_api_agents.stirrup_agent.finish_tool_coercing import ABANDON_FINISH_TOOL, COERCING_FINISH_TOOL
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent


class _Sandbox:
    """Tiny in-memory sandbox: a few commands, a file dict."""

    def __init__(self):
        self.files = {"/workspace/reference_files/brief.txt": b"Q3 numbers"}
        self.commands = []

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        inner = shlex.split(command[len("bash -c "):])[0] if command.startswith("bash -c ") else command
        self.commands.append(inner)
        if inner.startswith("cat "):
            path = shlex.split(inner)[1]
            data = self.files.get(path)
            return SimpleNamespace(stdout=(data or b"").decode(), stderr="" if data else "No such file", return_code=0 if data else 1, error_type=None)
        if inner.startswith("printf ") and " > " in inner:
            body, target = inner.split(" > ", 1)
            self.files[target.strip()] = shlex.split(body)[1].encode()
            return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
        if inner.startswith("python3 -c "):  # the deliverable collector
            argv = shlex.split(inner)
            paths, dest, wd = json.loads(argv[3]), argv[4], argv[5]
            copied, missing = [], []
            for p in paths:
                src = p if p.startswith("/") else f"{wd}/{p}"
                if src in self.files:
                    self.files[f"{dest}/{Path(src).name}"] = self.files[src]
                    copied.append(p)
                else:
                    missing.append(p)
            return SimpleNamespace(stdout=json.dumps({"copied": copied, "missing": missing, "skipped_dirs": []}), stderr="", return_code=0, error_type=None)
        if inner.startswith("mkdir -p") or inner.startswith("test ") or inner.startswith("find "):
            return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
        return SimpleNamespace(stdout="", stderr=f"unknown: {inner}", return_code=127, error_type=None)

    async def download(self, remote, local):
        Path(local).write_bytes(self.files[remote])

    async def upload(self, local, remote):
        self.files[remote] = Path(local).read_bytes()


class _ScriptedPolicy:
    """Replies in order; each reply is (content, [tool calls as (name, args dict)])."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        content, calls = self.script.pop(0)
        n = len(self.requests)
        tool_calls = [
            SimpleNamespace(id=f"call_{n}_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(calls)
        ] or None
        msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
        msg.model_dump = lambda exclude_none=True, _c=content, _t=tool_calls: {
            "role": "assistant",
            "content": _c,
            **({"tool_calls": [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in _t]} if _t else {}),
        }
        prompt_tokens = sum(len(str(m.get("content") or "")) for m in kwargs["messages"]) // 4
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=20, completion_tokens_details=None)
        usage.model_dump = lambda exclude_none=True, _p=prompt_tokens: {"prompt_tokens": _p, "completion_tokens": 20}
        return SimpleNamespace(id=f"chatcmpl-{n}", choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def _text(content):
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


class _FakeExaHTTP:
    """Stand-in for the provider's httpx client: canned Exa search JSON, canned page HTML."""

    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def post(self, url, json=None, headers=None):
        self.posts.append((url, json, dict(headers or {})))
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"title": "Q3 results", "url": "https://example.com/q3", "text": "Revenue grew 12%."}]},
        )

    async def get(self, url, headers=None):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, text="<html><body><p>Page body text here.</p></body></html>")


def _run(script, max_turns=10, web=False):
    sandbox = _Sandbox()
    policy = _ScriptedPolicy(script)
    web_provider = None
    if web:
        from responses_api_agents.gdpval_stirrup_agent import exa_search

        web_provider = exa_search.ExaToolProvider(api_keys=["test-key-1", "test-key-2"])
        web_provider._fake_http = _FakeExaHTTP()
    client = GDPValPolicyClient(
        model="gdpval-policy", base_url="http://policy/v1", context_window=200_000, max_completion_tokens_cap=32_768
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=policy))
    provider = SandboxCodeExecToolProvider(sandbox, working_dir="/workspace", shell_timeout=300)
    quiet = agent_app._QuietLogger()
    agent = NeMoAgent(
        client=client,
        name=agent_app.AGENT_NAME,
        max_turns=max_turns,
        tools=[provider] + ([web_provider] if web_provider is not None else []),
        finish_tool=[COERCING_FINISH_TOOL, ABANDON_FINISH_TOOL],
        tool_response_as_user=True,
        skip_input_file_listing=True,
        context_summarization_cutoff=2.0,
        recover_from_context_overflow=False,
        turns_remaining_warning_threshold=2,
        logger=quiet,
    )
    agent._cache_on_interrupt = False
    agent._web_provider = web_provider
    prompt = agent_app.build_user_prompt(
        agent_app.build_task_text("Summarize the brief into summary.txt", "Finance", "Analyst"),
        ["reference_files/brief.txt"],
        working_dir="/workspace",
    )

    async def go():
        async with agent.session() as session:
            return await session.run(prompt)

    if web_provider is not None:
        # Stirrup enters the provider through the class, so stub the HTTP client class
        # the provider instantiates (no network in tests).
        from responses_api_agents.gdpval_stirrup_agent import exa_search

        real_client_cls = exa_search.httpx.AsyncClient
        exa_search.httpx.AsyncClient = lambda **kwargs: web_provider._fake_http
        try:
            finish_params, history, metadata = asyncio.run(go())
        finally:
            exa_search.httpx.AsyncClient = real_client_cls
    else:
        finish_params, history, metadata = asyncio.run(go())
    if web_provider is not None:
        return sandbox, policy, client, provider, quiet, finish_params, web_provider
    return sandbox, policy, client, provider, quiet, finish_params


def test_loop_with_web_tools_searches_then_finishes():
    sandbox, policy, client, provider, quiet, finish, web = _run(
        [
            ("Let me look this up.", [("web_search", {"query": "example corp Q3 results"})]),
            ("Fetching the page.", [("fetch_web_page", {"url": "https://example.com/q3"})]),
            ("Writing.", [("code_exec", {"cmd": "printf 'Revenue grew 12 percent' > /workspace/summary.txt"})]),
            ("", [("finish", {"reason": "done", "paths": ["/workspace/summary.txt"]})]),
        ],
        web=True,
    )
    assert finish is not None and finish.paths == ["/workspace/summary.txt"]
    # The model saw the eval tool set: code_exec, web_search, fetch_web_page + the two finish tools.
    tool_names = sorted(t["function"]["name"] for t in policy.requests[0]["tools"])
    assert tool_names == ["abandon_task_finish", "code_exec", "fetch_web_page", "finish", "web_search"]
    web_search_tool = next(t for t in policy.requests[0]["tools"] if t["function"]["name"] == "web_search")
    assert list(web_search_tool["function"]["parameters"]["properties"]) == ["query"]
    # The search hit Exa with a key and its XML result reached the model as the next tool message.
    assert web.searches == 1 and web.fetches == 1
    url, body, headers = web._fake_http.posts[0]
    assert url.startswith("https://api.exa.ai/") and body["query"] == "example corp Q3 results" and headers["x-api-key"] == "test-key-1"
    second = policy.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and "<title>Q3 results</title>" in _text(second[-1]["content"])
    assert "Revenue grew 12%" in _text(second[-1]["content"])
    third = policy.requests[2]["messages"]
    assert "<web_fetch><url>https://example.com/q3</url>" in _text(third[-1]["content"])
    # Keys never appear in anything the model sees or we export.
    blob = json.dumps(policy.requests, default=str) + json.dumps(client.final_messages())
    assert "test-key-1" not in blob and "test-key-2" not in blob
    assert sandbox.files["/workspace/summary.txt"] == b"Revenue grew 12 percent"


def test_build_web_tools_flag_and_keys(tmp_path, monkeypatch):
    cfg = SimpleNamespace(web_tools_enabled=False, web_search_api_keys_file=None, web_search_api_keys_env="EXA_API_KEYS", web_search_max_sweeps=1)
    assert agent_app.build_web_tools(cfg) is None
    cfg.web_tools_enabled = True
    monkeypatch.delenv("EXA_API_KEYS", raising=False)
    import pytest

    with pytest.raises(ValueError, match="no web search keys"):
        agent_app.build_web_tools(cfg)
    monkeypatch.setenv("EXA_API_KEYS", "e1,e2")
    assert agent_app.build_web_tools(cfg).num_keys == 2
    keys = tmp_path / "keys.txt"
    keys.write_text("f1\nf2\ne1\n")
    cfg.web_search_api_keys_file = str(keys)
    assert agent_app.build_web_tools(cfg).num_keys == 4


def test_loop_reads_reference_writes_deliverable_and_finishes():
    sandbox, policy, client, provider, quiet, finish = _run(
        [
            ("Let me read the brief.", [("code_exec", {"cmd": "cat /workspace/reference_files/brief.txt"})]),
            ("Writing the summary.", [("code_exec", {"cmd": "printf 'Q3 numbers: good' > /workspace/summary.txt"})]),
            ("", [("finish", {"reason": "Summary written.", "paths": ["/workspace/summary.txt"]})]),
        ]
    )
    assert finish is not None and finish.paths == ["/workspace/summary.txt"]
    assert sandbox.files["/workspace/summary.txt"] == b"Q3 numbers: good"
    assert provider.commands_run == 2
    # The model saw the AA v2 tool set: code_exec + the two finish tools, nothing else.
    tool_names = sorted(t["function"]["name"] for t in policy.requests[0]["tools"])
    assert tool_names == ["abandon_task_finish", "code_exec", "finish"]
    # enable_thinking travels as metadata.chat_template_kwargs (Gym's model server forbids a
    # top-level chat_template_kwargs body field), not as extra_body.
    assert json.loads(policy.requests[0]["metadata"]["chat_template_kwargs"]) == {"enable_thinking": True}
    assert "extra_body" not in policy.requests[0]
    # Stirrup's base system prompt (eval parity) is the system message; our user prompt follows.
    first = policy.requests[0]["messages"]
    assert first[0]["role"] == "system" and "10 steps" in _text(first[0]["content"])
    assert first[1]["role"] == "user" and "<task>" in _text(first[1]["content"])
    # Tool results are re-sent to the provider as tool-role messages matching the call ids,
    # in the strict OpenAI shapes Gym's model server validates (no Stirrup-internal keys).
    second = policy.requests[1]["messages"]
    assistant_turn = next(m for m in second if m["role"] == "assistant")
    assert set(assistant_turn) <= {"role", "content", "reasoning_content", "tool_calls"}
    assert all(set(tc) == {"id", "type", "function"} and set(tc["function"]) == {"name", "arguments"} for tc in assistant_turn["tool_calls"])
    assert all(set(m) == {"role", "content", "tool_call_id"} for m in second if m["role"] == "tool")
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1_0"
    assert "Q3 numbers" in _text(second[-1]["content"])
    # The exported history is plain-string content (Stirrup sends list parts on the wire).
    assert all(isinstance(m["content"], str) for m in client.final_messages())
    # Export: the whole conversation, served id of the last call, no token arrays.
    messages = client.final_messages()
    assert messages[-1]["tool_calls"][0]["function"]["name"] == "finish"
    params, response = agent_app.export_response(
        messages=messages, tools=client.calls[-1]["tools"], response_id=client.last_response_id,
        model="gdpval-policy", status="completed",
    )
    assert response.id == "chatcmpl-3"
    assert [i.type for i in response.output].count("function_call") == 3
    assert [i.type for i in response.output].count("function_call_output") == 2
    assert agent_app.classify_outcome(finish_params=finish, error=None, timed_out=False)[3] == "finish"


def test_loop_abandon_and_max_turns():
    _, _, _, _, _, finish = _run([("Cannot do this.", [("abandon_task_finish", {"reason": "inputs missing"})])])
    assert finish is not None and not hasattr(finish, "paths")
    assert agent_app.classify_outcome(finish_params=finish, error=None, timed_out=False)[3] == "abandon"

    chatter = [("still thinking", [])] * 3
    _, policy, client, _, quiet, finish = _run(chatter, max_turns=3)
    assert finish is None
    assert len(policy.requests) == 3
    # Stirrup injects "Please continue the task" / turn warnings as user turns; history stays contiguous.
    roles = [m["role"] for m in policy.requests[-1]["messages"]]
    assert roles[0] == "system" and all(r in ("user", "assistant") for r in roles[1:])
    assert agent_app.classify_outcome(finish_params=None, error=None, timed_out=False)[0] == "agent_iteration_limit"


# ---------------------------------------------------------------------------
# Context compaction: segments for NeMo RL multi-trace training
# ---------------------------------------------------------------------------


class _EchoingPolicy(_ScriptedPolicy):
    """Scripted policy whose replies carry the model server's echoed token ids."""

    async def create(self, **kwargs):
        response = await super().create(**kwargs)
        n = len(self.requests)
        text = "".join(_text(m.get("content")) for m in kwargs["messages"])
        prompt_ids = [ord(c) % 251 + 1 for c in text[: 40 + 5 * n]]
        gen_ids = [1000 + n] * 3
        msg = response.choices[0].message
        base = msg.model_dump()
        msg.model_dump = lambda exclude_none=True, _b=base, _p=prompt_ids, _g=gen_ids: {
            **_b,
            "prompt_token_ids": _p,
            "generation_token_ids": _g,
            "generation_log_probs": [-0.1] * len(_g),
        }
        # Reported usage drives Stirrup's compaction trigger: the first reply sits at 400
        # prompt tokens, everything afterwards (summary call, continuation) at 100.
        prompt_tokens = 400 if n == 1 else 100
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=20, completion_tokens_details=None)
        usage.model_dump = lambda exclude_none=True, _p=prompt_tokens: {"prompt_tokens": _p, "completion_tokens": 20}
        response.usage = usage
        return response


def _run_compacting(script, *, context_window, cutoff, max_turns=10):
    sandbox = _Sandbox()
    policy = _EchoingPolicy(script)
    client = GDPValPolicyClient(
        model="gdpval-policy", base_url="http://policy/v1", context_window=context_window,
        max_completion_tokens_cap=32_768, min_completion_tokens=8, completion_token_buffer=0,
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=policy))
    provider = SandboxCodeExecToolProvider(sandbox, working_dir="/workspace", shell_timeout=300)
    quiet = agent_app._QuietLogger()
    agent = NeMoAgent(
        client=client, name=agent_app.AGENT_NAME, max_turns=max_turns, tools=[provider],
        finish_tool=[COERCING_FINISH_TOOL, ABANDON_FINISH_TOOL], tool_response_as_user=True,
        skip_input_file_listing=True, context_summarization_cutoff=cutoff,
        recover_from_context_overflow=False, turns_remaining_warning_threshold=2, logger=quiet,
    )
    agent._cache_on_interrupt = False
    prompt = agent_app.build_user_prompt(
        agent_app.build_task_text("Summarize the brief into summary.txt", "Finance", "Analyst"),
        ["reference_files/brief.txt"], working_dir="/workspace",
    )

    async def go():
        async with agent.session() as session:
            return await session.run(prompt)

    finish_params, history, metadata = asyncio.run(go())
    return sandbox, policy, client, quiet, finish_params


def test_compaction_segments_are_exported_as_separate_traces():
    # The echoing policy reports 400 prompt tokens on the first reply and 100 afterwards,
    # so with a 600-token window and cutoff 0.5 only the first reply triggers Stirrup's
    # summarization, whose single policy call is the next script entry. The continuation
    # then writes the file and finishes.
    sandbox, policy, client, quiet, finish = _run_compacting(
        [
            ("Let me read the brief.", [("code_exec", {"cmd": "cat /workspace/reference_files/brief.txt"})]),
            ("SUMMARY: task is to summarize brief.txt; brief read.", []),
            ("Writing the summary.", [("code_exec", {"cmd": "printf 'Q3 numbers: good' > /workspace/summary.txt"})]),
            ("", [("finish", {"reason": "Summary written.", "paths": ["/workspace/summary.txt"]})]),
        ],
        context_window=600, cutoff=0.5,
    )
    assert finish is not None and finish.paths == ["/workspace/summary.txt"]
    assert quiet.compactions == 1
    from stirrup.prompts import MESSAGE_SUMMARIZER
    summary_request = policy.requests[1]["messages"]
    assert _text(summary_request[-1]["content"]).strip() == MESSAGE_SUMMARIZER.strip()

    segments = client.segments()
    assert [(s["index"], s["reason"], len(s["calls"])) for s in segments] == [
        (0, "", 1), (1, "compaction", 1), (2, "post_compaction", 2),
    ]
    messages0, prefix0 = client.segment_messages(segments[0])
    assert prefix0 == 2 and [m["role"] for m in messages0[prefix0:]] == ["assistant"]
    messages1, prefix1 = client.segment_messages(segments[1])
    assert messages1[prefix1 - 1]["role"] == "user" and messages1[prefix1]["role"] == "assistant"
    assert messages1[prefix1]["generation_token_ids"] == [1002, 1002, 1002]
    messages2, prefix2 = client.segment_messages(segments[2])
    # post-compaction context: system, task, summary bridge, acknowledgement
    assert [m["role"] for m in messages2[:prefix2]] == ["system", "user", "user", "user"]
    assistants2 = [m for m in messages2[prefix2:] if m["role"] == "assistant"]
    assert [m["generation_token_ids"][0] for m in assistants2] == [1003, 1004]

    responses = agent_app.export_segment_responses(client, model="gdpval-policy", session_id="task-1")
    assert [r.metadata["segment_boundary_reason"] for r in responses] == ["", "compaction", "post_compaction"]
    assert [r.metadata["segment_index"] for r in responses] == ["0", "1", "2"]
    assert all(r.metadata["session_id"] == "task-1" and r.metadata["parent_session_id"] == "" for r in responses)
    # NeMo RL session-trace validator contract (training_mode=compaction)
    owned = []
    for r in responses:
        assert r.metadata["training_mode"] == "compaction" and r.metadata["root_session_id"] == "task-1"
        assert r.metadata["session_role"] == "parent" and r.metadata["spawn_call_id"] == ""
        ids = json.loads(r.metadata["request_ids"])
        assert ids and len(ids) == int(r.metadata["policy_calls"])
        owned += ids
    assert len(owned) == len(set(owned)) == 4  # every policy call owned exactly once
    for response in responses:
        trained = [i for i in (item.model_dump() for item in response.output) if i.get("generation_token_ids")]
        assert trained, response.id
        for item in trained:
            assert len(item["generation_token_ids"]) == len(item["generation_log_probs"]) == 3
            assert item["prompt_token_ids"]
    assert len([i for i in (x.model_dump() for x in responses[1].output) if i.get("generation_token_ids")]) == 1
    assert len([i for i in (x.model_dump() for x in responses[2].output) if i.get("generation_token_ids")]) == 2


def test_no_compaction_yields_one_plain_segment():
    sandbox, policy, client, quiet, finish = _run_compacting(
        [
            ("Writing the summary.", [("code_exec", {"cmd": "printf 'Q3' > /workspace/summary.txt"})]),
            ("", [("finish", {"reason": "done", "paths": ["/workspace/summary.txt"]})]),
        ],
        context_window=200_000, cutoff=0.7,
    )
    assert quiet.compactions == 0
    segments = client.segments()
    assert [(s["index"], s["reason"], len(s["calls"])) for s in segments] == [(0, "", 2)]
    responses = agent_app.export_segment_responses(client, model="gdpval-policy", session_id="t")
    assert len(responses) == 1 and responses[0].metadata["segment_boundary_reason"] == ""


def test_segments_without_token_echo_export_text_only():
    """Profiling/eval servers do not echo token ids; segments still export (no token fields)."""
    sandbox, policy, client, provider, quiet, finish = _run(
        [
            ("Writing the summary.", [("code_exec", {"cmd": "printf 'Q3' > /workspace/summary.txt"})]),
            ("", [("finish", {"reason": "done", "paths": ["/workspace/summary.txt"]})]),
        ]
    )
    responses = agent_app.export_segment_responses(client, model="gdpval-policy", session_id="t")
    assert len(responses) == 1
    items = [item.model_dump() for item in responses[0].output]
    assert items and all("prompt_token_ids" not in item for item in items)


def test_stirrup_disk_cache_is_disabled():
    from stirrup.core import agent as stirrup_agent_module

    manager = stirrup_agent_module.CacheManager()
    assert manager.save_state("hash", None, None) is None and manager.load_state("hash") is None
    assert not (Path.home() / ".cache" / "stirrup" / "hash").exists()
