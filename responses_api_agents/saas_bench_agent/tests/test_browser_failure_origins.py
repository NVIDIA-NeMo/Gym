# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native catch-all behavior without requiring Chrome in unit CI."""

import ast
import asyncio
import contextvars
import json
import os
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT


def native_functions():
    path = BENCHMARK_ROOT / "saas_bench/agent.py"
    tree = ast.parse(path.read_text())
    names = {
        "ModelActionError",
        "TrajectoryNoProgressError",
        "_track_browser_runtime",
        "_is_context_limit_error",
        "run_task",
    }
    selected = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_CleanOutputChatOpenAI")
    invoke = next(n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "ainvoke")
    namespace = {"Any": Any, "Path": Path, "asyncio": asyncio, "os": os, "re": re, "shutil": shutil, "time": time}
    exec(compile(ast.Module(body=selected + [invoke], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("malformed_tool_call", "model"),
        ("context_limit", "model"),
        ("timeout", "evaluation"),
        ("provider_503", "evaluation"),
    ],
)
async def test_model_call_boundary_only_attributes_confirmed_model_errors(kind, expected):
    ns = native_functions()
    if kind == "malformed_tool_call":
        error = ns["ModelActionError"](kind)
    elif kind == "context_limit":
        error = RuntimeError("maximum context length exceeded")
        error.status_code = 400
    elif kind == "provider_503":
        error = RuntimeError("maximum context length service unavailable")
        error.status_code = 503
    else:
        error = TimeoutError("endpoint stalled")

    async def invoke(*args, **kwargs):
        raise error

    owner = SimpleNamespace(_invoke_clean_output=invoke)
    with pytest.raises(type(error)):
        await ns["ainvoke"](owner, [])
    assert owner.last_actor_failure["failure_origin"] == expected
    assert owner.last_actor_exception is error


async def test_browser_fault_is_cleared_only_after_observation_recovers():
    ns = native_functions()
    calls = 0
    value = object()

    async def observation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("screenshot unavailable")
        return value

    async def start():
        return None

    browser = SimpleNamespace(start=start, get_browser_state_summary=observation)
    faults = {}
    ns["_track_browser_runtime"](browser, faults)
    with pytest.raises(OSError):
        await browser.get_browser_state_summary()
    assert faults == {"get_browser_state_summary": "OSError"}
    assert await browser.get_browser_state_summary() is value
    assert not faults and calls == 2


@pytest.mark.parametrize(
    "scenario,origin",
    [
        ("cdp_start", "evaluation"),
        ("screenshot", "evaluation"),
        ("swallowed_screenshot", "evaluation"),
        ("model_parse", "model"),
        ("unknown_after_turn", "evaluation"),
    ],
)
async def test_native_run_task_catch_all_preserves_failure_origin(tmp_path, monkeypatch, scenario, origin):
    ns = native_functions()
    monkeypatch.setenv("SAAS_SAVE_SCREENSHOTS", "0")
    history = SimpleNamespace(history=[object()], final_result=lambda: "", is_done=lambda: False)
    llm = SimpleNamespace(call_records=[], request_ids=[], telemetry_retry_records=[], context_budget_events=[])

    async def start():
        return None

    async def observation(*args, **kwargs):
        raise OSError("screenshot unavailable")

    browser = SimpleNamespace(start=start, get_browser_state_summary=observation)

    class Agent:
        def __init__(self, **kwargs):
            self.history = history

        async def run(self, **kwargs):
            if scenario in {"screenshot", "swallowed_screenshot"}:
                try:
                    await browser.get_browser_state_summary()
                except OSError:
                    if scenario == "screenshot":
                        raise
                return history
            if scenario == "model_parse":
                error = ns["ModelActionError"]("malformed_tool_call")
                llm.last_actor_failure = {"failure_origin": "model", "failure_kind": "malformed_tool_call"}
                llm.last_actor_exception = error
                raise error
            raise RuntimeError("browser/runtime failed after a real turn")

    class Playwright:
        async def __aenter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(executable_path="unused"))

        async def __aexit__(self, *args):
            return None

    def chrome(*args):
        if scenario == "cdp_start":
            raise RuntimeError("Chrome CDP not ready")
        return None, None

    async def kill(*args):
        return None

    ns.update(
        _TMP_BASE=str(tmp_path),
        _free_port=lambda: 9000,
        _dom_refresh_enabled=lambda: False,
        _sft_capture_enabled=lambda: False,
        _ALLOWED_LOCAL_ORIGINS=contextvars.ContextVar("test_origins"),
        async_playwright=Playwright,
        _start_chrome=chrome,
        _build_llm=lambda _: llm,
        Browser=lambda **kwargs: browser,
        Tools=lambda **kwargs: None,
        Agent=Agent,
        _positive_timeout=lambda name, default: default,
        TASK_G_RULES="",
        _checkpoint_history=lambda **kwargs: {"trajectory": [{"actions": [{"click": {"index": 1}}]}]},
        _extract_trajectory=lambda *args, **kwargs: [{"actions": [{"click": {"index": 1}}]}],
        _kill=kill,
        _atomic_write_json=lambda path, value: path.write_text(json.dumps(value)),
    )
    result = await ns["run_task"]({"task_id": "fixture"}, "model", "goal", str(tmp_path / "out"))
    assert result["status"] == "error" and result["failure_origin"] == origin
    if scenario != "cdp_start":
        assert result["trajectory"]  # The catch-all must not lose partial model evidence.


@pytest.mark.parametrize(
    "completion,finish,origin,kind",
    [
        ("not json", "stop", "model", "malformed_tool_call"),
        ("not json", "length", "model", "output_token_limit"),
        ("", "stop", "evaluation", "runtime_error"),
    ],
)
async def test_raw_response_parsing_separates_model_text_from_missing_response(completion, finish, origin, kind):
    from pydantic import BaseModel

    class Output(BaseModel):
        action: str

    class Parent:
        async def ainvoke(self, messages, output_format=None, **kwargs):
            if output_format is not None:
                return output_format.model_validate_json(completion)
            return SimpleNamespace(completion=completion, stop_reason=finish)

    source = BENCHMARK_ROOT / "saas_bench/agent.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_CleanOutputChatOpenAI")
    owner.decorator_list = []
    owner.body = [
        n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name in {"ainvoke", "_invoke_clean_output"}
    ]
    ns = native_functions()
    ns.update(
        ChatOpenAI=Parent,
        _current_request_id=contextvars.ContextVar("request"),
        _current_raw_response_usage=contextvars.ContextVar("usage"),
        _strip_tool_call_wrapper=lambda text: text,
        _rewrite_banned_actions=lambda text: text,
    )
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(source), "exec"), ns)
    client = ns["_CleanOutputChatOpenAI"]()
    with pytest.raises(RuntimeError):
        await client.ainvoke([], output_format=Output)
    assert client.last_actor_failure == {"failure_origin": origin, "failure_kind": kind}
