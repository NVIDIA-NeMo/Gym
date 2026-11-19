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
"""The server against a stand-in worker (the real one needs the harness: ``test_worker_live.py``), and the pieces the
worker is made of."""

import copy
import hashlib
import json
import os
import sys
import threading
import time
import types
from collections import Counter
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import ENVIRONMENT_SERVER_FAILED, MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR, VERIFIER_ERROR
from nemo_gym.core.failure_kinds import ENVIRONMENT_PREPARE_FAILED
from resources_servers.arc_agi_3_pivot import app as app_module
from resources_servers.arc_agi_3_pivot import create_dataset, oracle, recording, setup_worker
from resources_servers.arc_agi_3_pivot.app import ArcAgi3PivotConfig, ArcAgi3PivotResourcesServer, RestoreError


HERE = Path(__file__).resolve().parents[1]
POOL = HERE / "data" / "example_pool"


class FakeWorker:
    """d0 = 5; each tool call is one action and lowers d by 1. Code "win" completes the level, "helper" makes the
    harness's next request a summary request, "trim" makes it a rewritten context, "fail" fails the restore."""

    instances: list["FakeWorker"] = []

    def __init__(self, python: str, env: dict[str, str], cwd: str, timeout_s: float = 3600.0):
        self.python, self.env, self.cwd = python, env, cwd
        self.d, self.levels, self.closed, self.replies = 5, 0, False, []
        self.messages: list[dict[str, Any]] = []
        FakeWorker.instances.append(self)

    def _request(self) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(self.messages),
            "tools": [{"type": "function", "function": {"name": "python"}}],
            "chat_template_kwargs": {"enable_thinking": True},
        }

    def call(self, op: str, *args: Any) -> Any:
        if op == "start":
            spec = args[0]
            if spec.get("fail"):
                error = RestoreError("request differs from the recording")
                error.info = {"reason": "request differs from the recording", "request": 3}
                raise error
            self.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "board"}]
            return {
                "request": self._request(),
                "role": "gameplay",
                "continuation": False,
                "ended": None,
                "level_index": 0,
                "levels_completed": 0,
                "action_count": 3,
                "masked": 0,
                "replayed": 5,
                "reasoning_effort": {"reasoning_effort": spec["reasoning_effort"]}
                if spec.get("reasoning_effort")
                else {},
            }
        if op == "distance":
            return self.d
        if op == "reply":
            message, finish_reason, usage = args
            self.replies.append((message, finish_reason, usage))
            self.messages.append(message)
            actions, role = [], "gameplay"
            for tool_call in message.get("tool_calls") or []:
                code = json.loads(tool_call["function"]["arguments"])["code"]
                self.messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": f"ran {code}"})
                if code == "win":
                    self.levels = 1
                elif code == "helper":
                    role = "summary"
                elif code == "trim":
                    self.messages = self.messages[:1] + self.messages[2:]
                # "loop": the oracle finds no completion (its search cap or a revisited state); "zero": the oracle
                # sees the level complete but the game does not
                self.d = None if code == "loop" else 0 if code == "zero" else self.d - 1
                actions.append(
                    {"name": "ACTION1", "data": {}, "state": "NOT_FINISHED", "levels_completed": self.levels}
                )
            self.messages.append({"role": "user", "content": "next opener"})
            return {
                "request": self._request(),
                "role": role,
                "continuation": False,
                "ended": None,
                "levels_completed": self.levels,
                "actions": actions,
            }
        raise AssertionError(op)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def worker(monkeypatch):
    FakeWorker.instances = []
    monkeypatch.setattr(app_module, "HarnessWorker", FakeWorker)

    def fake_spec(self, row, workdir, overrides):
        return {
            "env": {"ARC3_X": "1"},
            "fail": json.loads(row["pivot_json"]).get("fail", False),
            "reasoning_effort": row.get("reasoning_effort") or self.config.reasoning_effort,
        }

    monkeypatch.setattr(ArcAgi3PivotResourcesServer, "_spec", fake_spec)
    return FakeWorker.instances


def meta(pivot_id: str, **pivot: Any) -> dict:
    return {
        "pivot_id": pivot_id,
        "game_id": "ls20-9607627b",
        "pivot": {"pivot_json": json.dumps({"pivot_id": pivot_id, **pivot}), "harness": "30d2cc8b"},
    }


def make_client(tmp_path, **overrides):
    config = ArcAgi3PivotConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        pool_dir=str(POOL),
        **{"harness_root": str(tmp_path), "worker_python": sys.executable, **overrides},
    )
    server = ArcAgi3PivotResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    class StatelessCookieJar(CookieJar):
        def extract_cookies(self, response: object, request: object) -> None:
            pass

    return server, TestClient(server.setup_webserver(), cookies=StatelessCookieJar())


def reply(code: str, call_id: str = "model_call_1") -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "think",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }
        ],
    }


def verify_body(m: dict, **extra) -> dict:
    response = {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "tool_choice": "auto",
        "tools": [],
        "parallel_tool_calls": False,
    }
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
        "verifier_metadata": m,
        "response": response,
        **extra,
    }


def seed(client, m):
    response = client.post("/seed_session", json={"verifier_metadata": m})
    assert response.status_code == 200, response.text
    return response.json(), response.cookies


def test_window_follows_the_harness_requests_and_scores_progress(worker, tmp_path):
    _, client = make_client(tmp_path, max_calls=3)
    prompt, cookies = seed(client, meta("p1"))
    assert [m["role"] for m in prompt["messages"]] == ["system", "user"] and prompt["restore_error"] is None
    assert prompt["chat_template_kwargs"] == {"enable_thinking": True}
    assert worker[0].env["ARC3_X"] == "1"

    step = client.post(
        "/step", json={"message": reply("a = 1"), "gen_tokens": 7, "prompt_tokens": 50}, cookies=cookies
    )
    step = step.json()
    assert step["done"] is False
    # the harness's next request, and the part after the reply: the tool result (the model's call id) and the opener
    assert [m["role"] for m in step["request_messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert step["messages"] == step["request_messages"][3:]
    assert step["messages"][0]["tool_call_id"] == "model_call_1"
    # the harness got the reply as a chat completion message, and the call's generated tokens
    message, finish_reason, usage = worker[0].replies[0]
    assert message["reasoning_content"] == "think" and usage == {"completion_tokens": 7, "prompt_tokens": 50}
    client.post("/step", json={"message": reply("b = 2"), "gen_tokens": 3}, cookies=cookies)
    last = client.post("/step", json={"message": reply("c = 3")}, cookies=cookies).json()
    assert last["done"] is True and last["request_messages"] is None and last["call"]["done_reason"] == "max_calls"
    assert worker[0].closed  # the worker exits when the window ends

    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["reward"] == 3.0 and result["progress"] == 3.0  # d 5 -> 2
    assert result["d0"] == 5 and result["d_end"] == 2 and result["level_completed"] == 0.0
    assert result["actions"] == 3.0 and result["calls"] == 3.0 and result["gen_tokens"] == 10.0
    assert result["mask_sample"] is False and result["reasoning_effort"] is None


@pytest.mark.parametrize("config_effort, row_effort, want", [("high", None, "high"), ("high", "low", "low")])
def test_live_reasoning_effort_from_the_config_or_the_row(worker, tmp_path, config_effort, row_effort, want):
    _, client = make_client(tmp_path, max_calls=1, reasoning_effort=config_effort)
    m = dict(meta("p1"), **({"reasoning_effort": row_effort} if row_effort else {}))
    _, cookies = seed(client, m)
    client.post("/step", json={"message": reply("a = 1")}, cookies=cookies)
    result = client.post("/verify", json=verify_body(m), cookies=cookies).json()
    assert result["reasoning_effort"] == want and result["mask_sample"] is False


def test_level_completion_scores_d0(worker, tmp_path):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    step = client.post("/step", json={"message": reply("win")}, cookies=cookies).json()
    assert (
        step["done"] is True and step["call"]["level_completed"] and step["call"]["done_reason"] == "level_completed"
    )
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["reward"] == 5.0 and result["level_completed"] == 1.0 and result["d_end"] == 0


def test_action_budget_ends_the_window(worker, tmp_path):
    _, client = make_client(tmp_path, max_actions=2)
    _, cookies = seed(client, meta("p1"))
    assert client.post("/step", json={"message": reply("a")}, cookies=cookies).json()["done"] is False
    step = client.post("/step", json={"message": reply("b")}, cookies=cookies).json()
    assert step["done"] is True and step["call"]["done_reason"] == "action_budget"


@pytest.mark.parametrize("code", ["loop", "zero"])
def test_an_unmeasurable_window_is_masked_not_scored(worker, tmp_path, code):
    # an unfinished window whose end the oracle cannot measure must not earn the completion reward (d0)
    _, client = make_client(tmp_path, max_calls=1)
    _, cookies = seed(client, meta("p1"))
    step = client.post("/step", json={"message": reply(code)}, cookies=cookies).json()
    assert step["done"] is True and step["call"]["level_completed"] is False
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == VERIFIER_ERROR
    assert result["reward"] == 0.0 and result["level_completed"] == 0.0 and result["d0"] == 5


def test_restore_error_is_masked_not_scored(worker, tmp_path):
    _, client = make_client(tmp_path)
    prompt, cookies = seed(client, meta("p1", fail=True))
    assert "request differs from the recording" in prompt["restore_error"] and prompt["messages"] == []
    assert worker[0].closed
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == ENVIRONMENT_PREPARE_FAILED


def test_oracle_distance_must_match_the_recording(worker, tmp_path):
    _, client = make_client(tmp_path)
    prompt, _ = seed(client, meta("p1", po_d0=4))
    assert "oracle distance 5 != recorded 4" in prompt["restore_error"]


@pytest.mark.parametrize("code,reason", [("helper", "helper_request"), ("trim", "context_rewrite")])
def test_a_request_that_is_not_the_chain_ends_and_masks_the_window(worker, tmp_path, code, reason):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    step = client.post("/step", json={"message": reply(code)}, cookies=cookies).json()
    assert step["done"] is True and step["call"]["done_reason"] == reason
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == MODEL_CONTEXT_LIMIT


def test_unfinished_window_is_masked(worker, tmp_path):
    _, client = make_client(tmp_path)
    _, cookies = seed(client, meta("p1"))
    client.post("/step", json={"message": reply("a = 1")}, cookies=cookies)
    result = client.post("/verify", json=verify_body(meta("p1"), agent_error="TimeoutError"), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == MODEL_REQUEST_ERROR


def test_dollar_cap_ends_the_window_after_the_call_that_reaches_it(worker, tmp_path):
    # $1 per uncached token, $0.5 per cached token, generated tokens free: dollars are easy to read.
    prices = {"in": 1e6, "cached": 0.5e6, "out": 0.0}
    _, client = make_client(tmp_path, dollar_cap=100.0, prices=prices)
    _, cookies = seed(client, meta("p1"))
    # First call: all cached (no previous call in the window) -> $50.
    first = client.post(
        "/step", json={"message": reply("a = 1"), "prompt_tokens": 100, "gen_tokens": 10}, cookies=cookies
    ).json()
    assert first["done"] is False
    # Second: 110 held (100 prompt + 10 generated), 20 new -> $55 + $20; cumulative $125 >= $100: the window ends.
    second = client.post(
        "/step", json={"message": reply("b = 2"), "prompt_tokens": 130, "gen_tokens": 10}, cookies=cookies
    ).json()
    assert second["done"] is True and second["call"]["done_reason"] == "dollar_cap"
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is False and result["reward"] == 2.0  # a capped window is scored
    assert result["dollars"] == 125.0 and result["dollar_cap_reached"] == 1.0
    assert (result["cached_tokens"], result["uncached_tokens"], result["gen_tokens"]) == (210.0, 20.0, 20.0)


def test_measured_cached_tokens_are_used(worker, tmp_path):
    _, client = make_client(tmp_path, prices={"in": 1e6, "cached": 0.0, "out": 0.0}, max_calls=3)
    _, cookies = seed(client, meta("p1"))
    for code, prompt, cached in (("a = 1", 100, None), ("b = 2", 130, 105), ("c = 3", 150, None)):
        body = {"message": reply(code), "prompt_tokens": prompt, "gen_tokens": 10, "cached_tokens": cached}
        last = client.post("/step", json=body, cookies=cookies).json()
    assert last["done"] is True
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    # uncached: 0 (first call) + 25 (measured 105 of 130) + 10 (150 - 140 held)
    assert result["uncached_tokens"] == 35.0 and result["dollars"] == 35.0 and result["dollar_cap_reached"] == 0.0


# ------------------------------------------------------------------ the worker spec from the example pool
def fake_harness_root(tmp_path: Path) -> Path:
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework"):
        (tmp_path / "harness" / "taaf" / "src" / repo / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "harness" / "public110").mkdir(exist_ok=True)
    return tmp_path


def example_rows() -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / "example.jsonl").read_text().splitlines()]


def test_spec_of_an_example_pivot(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_worker, "DEFAULT_DEST", tmp_path / "runtime")
    monkeypatch.setattr(app_module, "DEFAULT_DEST", tmp_path / "runtime")
    server, _ = make_client(tmp_path, harness_root=str(fake_harness_root(tmp_path / "h")))
    row = app_module.pool_row(example_rows()[0]["verifier_metadata"])  # cn04 c4
    spec = server._spec(row, str(tmp_path / "w"), {})
    assert spec["upto"] == 4 and spec["game_id"] == "cn04-2fe56bfb" and spec["sandbox_seed"] == 0
    assert spec["env"]["LOCAL_ANALYZER_MODEL_ID"] == "stub"  # the recorded request's model field
    assert spec["env"]["LOCAL_ANALYZER_YIELD_TOKENS"] == "2048" and "ARC3_SNAPSHOT_DIR" not in spec["env"]
    game = Path(spec["environments_dir"]) / "cn04" / "2fe56bfb"
    assert (game / "cn04.py").is_file() and json.loads((game / "metadata.json").read_text())[
        "game_id"
    ] == "cn04-2fe56bfb"
    assert spec["harness_paths"][-1].endswith("public110")
    assert spec["reasoning_effort"] is None and spec["recorded_effort"] is None  # the pivot's recorded effort
    assert server._spec(dict(row, thinking_effort="high"), str(tmp_path / "w"), {})["recorded_effort"] == "high"
    assert server._spec(dict(row, reasoning_effort="low"), str(tmp_path / "w"), {})["reasoning_effort"] == "low"

    other = dict(row, harness="278e0089")
    with pytest.raises(RestoreError, match="recorded under harness 278e0089"):
        server._spec(other, str(tmp_path / "w"), {})

    # a row pins its game's source revision: with another revision of the game in the pool (same game id, same code
    # up to the pivot), a pin that matches neither is refused, never served by the other revision
    pivot = json.loads(row["pivot_json"])
    entry = server._games[("cn04-2fe56bfb", pivot["game_source_sha256"])]
    server._games[("cn04-2fe56bfb", "f" * 64)] = dict(entry, source_sha256="f" * 64)
    assert server._spec(row, str(tmp_path / "w"), {})["game_id"] == "cn04-2fe56bfb"  # the exact pin
    pinned = dict(row, pivot_json=json.dumps(dict(pivot, game_source_sha256="e" * 64)))
    with pytest.raises(RestoreError, match="at source sha256 e+ is not in games.parquet"):
        server._spec(pinned, str(tmp_path / "w"), {})
    legacy = dict(row, pivot_json=json.dumps({k: v for k, v in pivot.items() if k != "game_source_sha256"}))
    with pytest.raises(RestoreError, match="not \\(uniquely\\)"):
        server._spec(legacy, str(tmp_path / "w"), {})
    del server._games[("cn04-2fe56bfb", "f" * 64)]
    assert server._spec(legacy, str(tmp_path / "w"), {})["game_id"] == "cn04-2fe56bfb"  # a legacy row, one revision

    # files a row names stay inside the pool
    outside = tmp_path / "outside.snap.jsonl"
    outside.write_bytes((POOL / pivot["snapshot"]["snap_file"]).read_bytes())
    escaped = dict(pivot, snapshot=dict(pivot["snapshot"], snap_file=os.path.relpath(outside, POOL)))
    with pytest.raises(RestoreError, match="outside the pool"):
        server._spec(dict(row, pivot_json=json.dumps(escaped)), str(tmp_path / "w"), {})
    with pytest.raises(RestoreError, match="outside the pool"):
        server._spec(dict(row, harness_env=str(outside)), str(tmp_path / "w"), {})
    pivot = json.loads(row["pivot_json"])
    pivot["snapshot"]["snap_sha256"] = "0" * 64
    with pytest.raises(RestoreError, match="recording sha256 differs"):
        server._spec(dict(row, pivot_json=json.dumps(pivot)), str(tmp_path / "w"), {})


def test_worker_env_is_minimal(tmp_path, monkeypatch):
    server, _ = make_client(tmp_path)
    monkeypatch.setenv("LOCAL_ANALYZER_SEED", "7")
    monkeypatch.setenv("ARC3_FORCE_ACT", "0")
    for name in ("HF_TOKEN", "POLICY_API_KEY", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "dummy-secret")
    env = server._worker_env({"ARC3_FORCE_ACT": "1"})
    assert "LOCAL_ANALYZER_SEED" not in env and env["ARC3_FORCE_ACT"] == "1" and "PATH" in env
    assert set(env) <= set(app_module.WORKER_ENV_NAMES) | {"ARC3_FORCE_ACT"}
    assert "dummy-secret" not in env.values()


def test_recorded_harness_env_keeps_harness_settings_only(tmp_path):
    toml = tmp_path / "harness_env.toml"
    toml.write_text(
        'use_priority_scheduling = true\n[notebook]\nARC3_X = "1"\nHF_TOKEN = "dummy-secret"\n'
        '[setup_env]\nLOCAL_ANALYZER_TOP_K = 20\nUSE_TF = "0"\nARC3_SNAPSHOT_DIR = "/x"\nHF_HOME = "/cache"\n'
        '[priority]\nARC3_MAX_ACTIVE_STREAMS = 96\nPOLICY_API_KEY = "dummy-secret"\n'
    )
    assert app_module.harness_env_tables(toml) == {
        "ARC3_X": "1",
        "LOCAL_ANALYZER_TOP_K": "20",
        "USE_TF": "0",
        "ARC3_MAX_ACTIVE_STREAMS": "96",
    }


# ------------------------------------------------------------------ a worker that does not answer
STUCK_WORKER = """
import json, os, subprocess, sys, time
from multiprocessing.connection import Connection
conn = Connection(int(sys.argv[1]))
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # a sandbox process
with open(os.environ["FAKE_PIDS"], "w") as f:
    json.dump([os.getpid(), child.pid], f)
request = {"messages": [{"role": "user", "content": "board"}]}
start = {"request": request, "role": "gameplay", "continuation": False, "level_index": 0, "levels_completed": 0}
while True:
    op, args = conn.recv()
    if op == os.environ["FAKE_HANG"]:
        time.sleep(600)
    conn.send((True, start if op == "start" else 5 if op == "distance" else None))
    if op == "close":
        break
os._exit(0)
"""


def alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie does not)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def stuck_worker(tmp_path, monkeypatch):
    """The real HarnessWorker over a worker that never answers the op ``hang``; returns (hang setter, pids)."""
    script = tmp_path / "stuck_worker.py"
    script.write_text(STUCK_WORKER)
    monkeypatch.setattr(app_module, "WORKER", script)
    pids_file = tmp_path / "pids.json"
    state = {"hang": "start"}

    def fake_spec(self, row, workdir, overrides):
        return {"env": {"FAKE_HANG": state["hang"], "FAKE_PIDS": str(pids_file)}}

    monkeypatch.setattr(ArcAgi3PivotResourcesServer, "_spec", fake_spec)
    return state, lambda: json.loads(pids_file.read_text())


def wait_dead(pids: list[int], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process groups and /proc")
def test_a_worker_stuck_in_the_restore_is_killed_and_masked(tmp_path, stuck_worker):
    state, pids = stuck_worker
    _, client = make_client(tmp_path, worker_timeout_s=1, clock_repasses=0)
    started = time.monotonic()
    prompt, cookies = seed(client, meta("p1"))
    assert time.monotonic() - started < 20
    assert "did not answer start within 1 s" in prompt["restore_error"]
    assert wait_dead(pids())  # the worker (reaped) and its sandbox process
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == ENVIRONMENT_PREPARE_FAILED


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process groups and /proc")
def test_a_worker_stuck_in_a_step_is_killed_and_masked(tmp_path, stuck_worker):
    state, pids = stuck_worker
    state["hang"] = "reply"
    _, client = make_client(tmp_path, worker_timeout_s=1)
    prompt, cookies = seed(client, meta("p1"))
    assert prompt["restore_error"] is None
    started = time.monotonic()
    step = client.post("/step", json={"message": reply("a = 1")}, cookies=cookies).json()
    assert time.monotonic() - started < 20
    assert step["done"] is True and step["call"]["done_reason"] == "worker_error"
    assert "did not answer reply within 1 s" in step["call"]["error"]
    assert wait_dead(pids())
    result = client.post("/verify", json=verify_body(meta("p1")), cookies=cookies).json()
    assert result["mask_sample"] is True and result["failure_kind"] == ENVIRONMENT_SERVER_FAILED


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process groups and /proc")
def test_closing_a_worker_interrupts_a_pending_call(tmp_path, stuck_worker):
    # session retirement while a call waits (its asyncio waiter cancelled, its thread still in recv)
    state, pids = stuck_worker
    state["hang"] = "reply"
    worker = app_module.HarnessWorker(
        sys.executable,
        {"PATH": os.environ.get("PATH", ""), "FAKE_HANG": "reply", "FAKE_PIDS": str(tmp_path / "pids.json")},
        str(tmp_path),
        timeout_s=600,
    )
    assert worker.call("start", {})["role"] == "gameplay"
    errors: list[Exception] = []

    def pending():
        try:
            worker.call("reply", {}, "stop", {})
        except RestoreError as exc:
            errors.append(exc)

    thread = threading.Thread(target=pending)
    thread.start()
    time.sleep(0.5)
    started = time.monotonic()
    worker.close()
    thread.join(timeout=10)
    assert time.monotonic() - started < 10 and not thread.is_alive()
    assert errors and "exited" in str(errors[0])
    assert wait_dead(pids()) and worker._proc.returncode is not None
    worker.close()  # idempotent


# ------------------------------------------------------------------ the recording
def test_recording_replies_and_yields():
    raw = (POOL / "snap-raw" / "stub-cn04-2fe56bfb.snap.jsonl").read_bytes()
    parsed = recording.parse(raw)
    calls = parsed["calls"]
    assert parsed["header"]["game_id"] == "cn04-2fe56bfb" and [c["n"] for c in calls] == list(range(13))
    # record.py keeps each answered request's reply usage and finish reason (the last request was never answered)
    assert all(c["usage"]["completion_tokens"] == 40 and c["finish_reason"] == "tool_calls" for c in calls[:-1])
    assert "usage" not in calls[-1] and [c["step"] for c in calls] == list(range(13))
    # every request extends the previous one; its reply is the assistant message after it
    for k in range(len(calls) - 1):
        r = recording.reply_of(calls, k)
        assert (
            r["role"] == "assistant" and r == calls[k + 1]["payload"]["messages"][len(calls[k]["payload"]["messages"])]
        )
    assert recording.reply_of(calls, len(calls) - 1) is None
    # no reply reached the turn's token budget (the stub's replies are short): no yield
    assert [k for k in range(len(calls)) if recording.yielded_after(calls, k)] == []
    assert recording.request_sha256(calls[5]["payload"]) == calls[5]["request_sha256"]


def _call(messages, **payload):
    return {
        "kind": "continuation" if payload.get("continue_final_message") else "policy",
        "payload": {"messages": messages, **payload},
    }


def test_recorded_replies_of_the_harness_helper_requests():
    sys_, opener = {"role": "system", "content": "s"}, {"role": "user", "content": "board"}
    tool_call = {"id": "t1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
    # force-act: request 0's reply was cut off inside its reasoning; request 1 continues it, closed by the harness
    prefix = "deep thought\n\n" + recording.FORCE_ACT_CLOSE + recording.THINK_CLOSE
    closed = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "deep thought\n\n" + recording.FORCE_ACT_CLOSE,
        "tool_calls": [tool_call],
    }
    calls = [
        _call([sys_, opener]),
        _call([sys_, opener, {"role": "assistant", "content": prefix}], continue_final_message=True),
        _call(
            [
                sys_,
                opener,
                closed,
                {"role": "tool", "tool_call_id": "t1", "content": "ok"},
                {"role": "user", "content": "You yielded control on the turn time budget. Go on."},
            ]
        ),
    ]
    reply, finish = recording.recorded_reply(calls, 0)
    assert finish == "length" and reply["reasoning_content"] == "deep thought" and reply["content"] == ""
    reply, finish = recording.recorded_reply(calls, 1)
    assert finish == "tool_calls" and reply["tool_calls"] == [tool_call]
    assert recording.yielded_after(calls, 0) is False and recording.yielded_after(calls, 1) is True

    # a level summary: the text later openers show under the level's heading; a history compaction: the next request
    summary_prompt = {"role": "user", "content": "Level 1 was completed at action 9. Summarize what would have helped"}
    shown = "Level summary — lessons through completed level 1:\n- push the block" + recording.LEVEL_SUMMARY_END
    compact_prompt = {"role": "user", "content": recording.COMPACTION_PROMPT + ", the oldest part ..."}
    calls = [
        _call(
            [sys_, opener, summary_prompt, {"role": "assistant", "content": recording.THINK_CLOSE}],
            continue_final_message=True,
        ),
        _call([sys_, {"role": "user", "content": "next level" + shown}]),
        _call(
            [sys_, compact_prompt, {"role": "assistant", "content": recording.THINK_CLOSE}],
            continue_final_message=True,
        ),
        _call([sys_, {"role": "user", "content": "[Summary of earlier history (steps 1-5), by you.]\nRULES: x"}]),
    ]
    assert recording.recorded_reply(calls, 0) == ({"role": "assistant", "content": "- push the block"}, "stop")
    assert recording.recorded_reply(calls, 2) == ({"role": "assistant", "content": "RULES: x"}, "stop")
    calls[1]["payload"]["messages"][1]["content"] = "Summary update after level 1 failed. No previous summary."
    assert recording.recorded_reply(calls, 0) == ({"role": "assistant", "content": ""}, "stop")


def test_recording_patterns():
    text = '{"run_elapsed_seconds": 385.18, "time_remaining_seconds": 6160.27}'
    assert recording.clock_pairs(text) == [("385.18", "6160.27")]
    assert recording.without_clock(text) != text
    limit = recording.memory_limit_bytes(['"memory_limit": "Python virtual-memory limit: 2048.0 MiB."'])
    assert limit == 2048 << 20
    assert recording.mask_identities("<f at 0x7fff5543dbc0> 140734799804432") == "<f at 0x#> #id#"
    got = [{"role": "tool", "content": "a"}, {"role": "user", "content": "x"}]
    assert recording.first_difference(got, got) is None
    assert recording.first_difference(got, got[:1])["message"] == 1


# ------------------------------------------------------------------ the oracle
def test_oracle_rolls_the_policy_until_the_level_completes():
    # a counter game: the level completes at 3; the policy steps +1; RESET (0) goes back to 0
    module: dict[str, Any] = {"RESET_ACTION": 0}
    exec(  # noqa: S102 -- a game module defines policy() next to its functions
        "def transition(h, z, a):\n    return h + 1\n"
        "def predict(h):\n    return None\n"
        "def check_level_complete(h, z, level):\n    return h >= 3\n"
        "def get_initial_state(level):\n    return (0, None)\n"
        "def policy(h, z, level):\n    return 1\n",
        module,
    )
    names = ("transition", "predict", "check_level_complete", "get_initial_state")
    game = types.SimpleNamespace(FUNCTIONS={k: module[k] for k in names}, h_t=1, z_t=None)
    assert oracle.distance(game, 0) == 2
    module["policy"] = lambda h, z, level: 0  # always RESET: revisits the start state
    assert oracle.distance(game, 0) is None
    with pytest.raises(LookupError):
        oracle.oracle_for(types.SimpleNamespace(FUNCTIONS={}))


# ------------------------------------------------------------------ the worker's pieces (no harness needed)
def worker_module(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    import worker as module  # the worker imports its siblings by name, as a script

    return module


def test_effort_fields():
    plain = {"messages": [], "chat_template_kwargs": {"enable_thinking": True}}
    assert recording.effort_of(plain) == {} and recording.thinks(plain)
    assert not recording.thinks({"chat_template_kwargs": {"enable_thinking": False}})
    other = recording.with_effort(plain, {"thinking_effort": "low"})
    assert other["chat_template_kwargs"] == {"enable_thinking": True, "thinking_effort": "low"} and plain != other
    assert recording.effort_of(other) == {"thinking_effort": "low"}
    top = recording.with_effort(plain, {"@reasoning_effort": "high"})
    assert top["reasoning_effort"] == "high" and recording.effort_of(top) == {"@reasoning_effort": "high"}
    # a request with an effort of its own (the harness's effort ladder) keeps it
    assert recording.with_effort(other, {"reasoning_effort": "max"}) is other


def test_window_thinking_effort(monkeypatch, tmp_path):
    module = worker_module(monkeypatch)
    snap = POOL / "snap-raw" / "stub-cn04-2fe56bfb.snap.jsonl"
    assert module.Window({"recording": str(snap), "upto": 5}).live_effort == {}  # recorded without: the default
    # a recording whose requests carry an effort the harness itself does not send (set outside the harness)
    lines = []
    for line in snap.read_text().splitlines():
        record = json.loads(line)
        if record.get("t") == "call" and record.get("params") is not None:
            kwargs = record["params"].setdefault("chat_template_kwargs", {})
            if kwargs.get("enable_thinking") is not False:
                kwargs["thinking_effort"] = "high"
        lines.append(json.dumps(record))
    (tmp_path / "effort.snap.jsonl").write_text("\n".join(lines) + "\n")
    harness = [c["payload"] for c in recording.parse(snap.read_bytes())["calls"]]  # what the harness sends

    def window(**spec):
        w = module.Window({"recording": str(tmp_path / "effort.snap.jsonl"), "upto": 5, **spec})
        checked = []
        monkeypatch.setattr(w, "_check", lambda k, payload: checked.append(payload))
        monkeypatch.setattr(w, "_recorded_reply", lambda k: ({"role": "assistant"}, "stop", None))
        w.tempfile, w.clock = types.SimpleNamespace(source=None), types.SimpleNamespace(advance=lambda n: None)
        for k in range(6):
            if k == 5:
                w.to_game.put(("reply", {"role": "assistant"}, "stop", None))
            w.model_call(copy.deepcopy(harness[k]))
        return w, checked, w.to_main.get()[1]

    w, checked, sent = window()
    assert w.live_effort == {"thinking_effort": "high"}
    # every replayed request (and the pivot's) is checked with its recorded effort; the live request carries it
    assert all(p["chat_template_kwargs"]["thinking_effort"] == "high" for p in checked) and len(checked) == 6
    assert sent["chat_template_kwargs"]["thinking_effort"] == "high"
    w, checked, sent = window(reasoning_effort="low")  # the live effort replaces the recorded one, same kwarg
    assert checked[5]["chat_template_kwargs"]["thinking_effort"] == "high"
    assert sent["chat_template_kwargs"]["thinking_effort"] == "low"
    no_effort = module.Window({"recording": str(snap), "upto": 5, "reasoning_effort": "max"})
    assert no_effort.live_effort == {"reasoning_effort": "max"}
    # an effort the collector set past the harness's request (the pool's thinking_effort column): the live default
    pool = module.Window({"recording": str(snap), "upto": 5, "recorded_effort": "low"})
    assert pool.live_effort == {"reasoning_effort": "low"}
    pool = module.Window({"recording": str(snap), "upto": 5, "recorded_effort": "low", "reasoning_effort": "high"})
    assert pool.live_effort == {"reasoning_effort": "high"}


def test_sandbox_seed_goes_right_after_the_import_hook(monkeypatch):
    module = worker_module(monkeypatch)
    bootstrap = "x = 1\n    " + module.SEED_ANCHOR + "\n    y = 2\n"
    seeded = module.seeded_bootstrap(bootstrap, 0)
    assert seeded.index("_gym_fixed_hash_seed(0)") < seeded.index("x = 1")
    lines = seeded.splitlines()
    anchor = lines.index("    " + module.SEED_ANCHOR)
    assert next(line for line in lines[anchor + 1 :] if line.strip()) == "    def _gym_seed_rngs(seed=0):"
    with pytest.raises(RuntimeError, match="seed anchor"):
        module.seeded_bootstrap("no anchor", 0)


def test_virtual_clock(monkeypatch):
    module = worker_module(monkeypatch)
    clock = module.VirtualClock(10.0, 0.5, None, {})
    assert clock.payload() == {"run_elapsed_seconds": 0.0, "time_remaining_seconds": None}
    clock.set(100.0, 900.0)  # a recorded pair: the budget is their sum
    assert clock.payload() == {"run_elapsed_seconds": 100.0, "time_remaining_seconds": 900.0}
    clock.advance(20)  # a live call: 10 s + 20 tokens x 0.5 s
    assert clock.payload() == {"run_elapsed_seconds": 120.0, "time_remaining_seconds": 880.0}
    pinned = module.VirtualClock(10.0, 0.5, 1000.0, {1: (5.0, 995.0)})
    pinned.payload()
    assert pinned.payload()["run_elapsed_seconds"] == 5.0  # read 1 returns the recorded pair


# ------------------------------------------------------------------ setup and data
def test_worker_venv_requirements_are_the_harness_runtime_and_flask():
    requirements = setup_worker.worker_requirements()
    assert "arc-agi==0.9.8" in requirements and "arcengine==0.9.3" in requirements
    assert "flask==3.1.3" in requirements
    assert not any("nemo-gym" in r or r.startswith("#") for r in requirements)


def test_materialize_game_keeps_every_path_in_the_cache(tmp_path):
    import pyarrow.parquet as pq

    entry = pq.read_table(POOL / "games.parquet").to_pylist()[0]  # cn04-2fe56bfb
    cache = tmp_path / "cache"
    root = setup_worker.materialize_game(entry, cache)
    assert (root / "cn04" / "2fe56bfb" / "cn04.py").is_file()
    bad = [
        {"source_sha256": "../../escape"},
        {"source_sha256": "A" * 64},
        {"environment_path": "environments/../../../../escape/2fe56bfb/cn04.py"},
        {"environment_path": "/tmp/cn04/2fe56bfb/cn04.py"},
        {"environment_path": "environments/cn04/2fe56bfb/../../../x.py"},
        {"environment_path": "environments/cn04/2fe56bfb/other.py"},
        {"environment_path": "environments/ls20/9607627b/ls20.py"},  # another game than the entry's
        {"game_id": "ls20-9607627b"},
    ]
    for change in bad:
        with pytest.raises(ValueError):
            setup_worker.materialize_game(dict(entry, **change), cache)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache"]
    assert [p.name for p in cache.iterdir()] == [root.name]  # no staging left behind


def test_harness_root_from_the_kaggle_kit(tmp_path, monkeypatch):
    calls = []

    def fake_ensure_harness(source, dest):  # the arc_agi_3 agent's installer: verified harness/ + config/ at dest
        calls.append((source, dest))
        return dest

    monkeypatch.setattr(setup_worker, "ensure_harness", fake_ensure_harness)
    assert setup_worker.ensure_harness_root(tmp_path, source="kaggle:o/p") == tmp_path / "harness"
    assert calls == [("kaggle:o/p", tmp_path / "harness")]
    with pytest.raises(FileNotFoundError, match="harness_source"):
        setup_worker.ensure_harness_root(tmp_path / "empty")


def test_create_dataset_keeps_restorable_rows(tmp_path):
    def row(pid, game, mode="snapshot", ok=True, **extra):
        return {
            "pivot_id": pid,
            "game_id": game,
            "restore_mode": mode,
            "restore_ok": ok,
            "pivot_json": "{}",
            "harness": "30d2cc8b",
            **extra,
        }

    rows = [
        row("a", "g1", level=1, po_d0=4),
        row("b", "g1", mode="kit_ckpt"),
        row("c", "g1"),
        row("d", "g2", ok=False),
    ]
    kept = create_dataset.select(rows, limit=None, per_game=1, ids=None)
    assert [r["pivot_id"] for r in kept] == ["a"]
    gym_row = create_dataset.gym_row(kept[0])
    assert gym_row["verifier_metadata"]["pivot"] == {"pivot_json": "{}", "harness": "30d2cc8b"}
    assert gym_row["verifier_metadata"]["po_d0"] == 4 and "reasoning_effort" not in gym_row["verifier_metadata"]
    assert create_dataset.gym_row(kept[0], "low")["verifier_metadata"]["reasoning_effort"] == "low"
    assert (
        create_dataset.gym_row(dict(kept[0], thinking_effort="high"))["verifier_metadata"]["pivot"]["thinking_effort"]
        == "high"
    )


def test_committed_example_is_the_example_pool():
    """data/example.jsonl: the 5 pivots (2 games) of data/example_pool, every file a row names present with its pinned
    sha256."""
    import pyarrow.parquet as pq

    rows = example_rows()
    split = create_dataset.select(create_dataset.iter_rows(str(POOL / "example.parquet")), None, None, None)
    assert rows == [create_dataset.gym_row(r) for r in split]
    assert Counter(r["verifier_metadata"]["game_id"] for r in rows) == {"cn04-2fe56bfb": 2, "ls20-9607627b": 3}
    assert len({r["verifier_metadata"]["pivot_id"] for r in rows}) == 5

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    games = {g["game_id"]: g["source_sha256"] for g in pq.read_table(POOL / "games.parquet").to_pylist()}
    for r in rows:
        pivot = r["verifier_metadata"]["pivot"]
        recorded = json.loads(pivot["pivot_json"])
        assert games[recorded["game_id"]] == recorded["game_source_sha256"]
        assert sha256(POOL / pivot["harness_env"]) == pivot["harness_env_sha256"]
        assert sha256(POOL / recorded["snapshot"]["snap_file"]) == recorded["snapshot"]["snap_sha256"]


def test_record_mode_writes_a_recording_that_parses_back(monkeypatch, tmp_path):
    """worker.py with ``record``: every request is live and written with the state it was sent from; the file parses
    back to the same requests, with each answered request's usage and finish reason."""
    module = worker_module(monkeypatch)
    out = tmp_path / "new.snap.jsonl"
    w = module.Window({"record": str(out), "game_id": "g-1", "sandbox_seed": 0, "attempt": "a1"})
    assert w.replaying is False and w.calls == []
    history: list[Any] = []
    w.game = types.SimpleNamespace(
        game_run=types.SimpleNamespace(history=history),
        current_state=types.SimpleNamespace(levels_completed=0, frame=types.SimpleNamespace(data=[[0, 1], [2, 3]])),
    )
    w.engine_game = types.SimpleNamespace(level_index=0)
    w.clock = types.SimpleNamespace(advance=lambda n: None)
    tools = [{"type": "function", "function": {"name": "python"}}]
    sys_, user = {"role": "system", "content": "s"}, {"role": "user", "content": "board"}
    reply = {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]}
    payloads = [
        {"model": "m", "messages": [sys_, user], "tools": tools},
        {"model": "m", "messages": [sys_, user, reply, {"role": "tool", "tool_call_id": "t1", "content": "ok"}]},
    ]
    payloads[1]["tools"] = tools
    usage = {"prompt_tokens": 10, "completion_tokens": 7}
    w.to_game.put(("reply", reply, "tool_calls", usage))
    assert w.model_call(payloads[0]) == (reply, "tool_calls", usage)
    w.steps.append({"name": "ACTION1", "data": {}})  # the reply's tool call acted
    history.append("ACTION1")
    w.to_game.put(("stop",))
    with pytest.raises(module.Stop):
        w.model_call(payloads[1])
    w.close()  # the unanswered request is written at close

    parsed = recording.parse(out.read_bytes())
    assert parsed["header"]["attempt"] == "a1" and parsed["actions"] == [{"id": "ACTION1", "data": {}}]
    calls = parsed["calls"]
    assert [c["payload"] for c in calls] == payloads and [c["step"] for c in calls] == [0, 1]
    assert [c["request_sha256"] for c in calls] == [recording.request_sha256(p) for p in payloads]
    assert calls[0]["usage"] == usage and calls[0]["finish_reason"] == "tool_calls" and "usage" not in calls[1]
    assert calls[1]["messages_keep"] == 2 and calls[0]["kind"] == "policy" and calls[0]["level"] == 1
    assert recording.reply_of(calls, 0) == reply
