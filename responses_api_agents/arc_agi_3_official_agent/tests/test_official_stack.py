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
"""The official harness end to end on CPU: the pinned upstream worker plays the two committed example games (both
harness types) through this agent.

    upstream worker (its own venv) --HTTP--> agent route --> real Gym vLLM model server (TestClient) --> stub vLLM
    upstream worker --HTTP (ONLINE Arcade)--> the real official game server --> scorecard --> real verify

Checks that, with the provider harness, every request reaching "vLLM" resends the reasoning of every earlier reply
(none stripped), also after a compaction; with the standard harness, that earlier replies go back as text only and the
transcript rolls; and that each game verifies against its scorecard (the worker's report matches; RHAE@T from the
server's own times). Skipped when the game files or the harness install are
missing (setup_official_harness.py installs it).
"""

import asyncio
import json
import json as json_module
import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi.testclient import TestClient

from nemo_gym import PARENT_DIR
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from resources_servers.arc_agi_3.app import ArcAgi3ResourcesServer, ArcAgi3ResourcesServerConfig, ArcAgi3VerifyRequest
from responses_api_agents.arc_agi_3_official_agent.app import (
    ArcAgi3OfficialAgent,
    ArcAgi3OfficialAgentConfig,
)
from responses_api_agents.arc_agi_3_official_agent.setup_official_harness import DEFAULT_DEST, venv_python
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig

from .official_stub import StubVLLM


GAMES = Path(
    os.environ.get("ARC3_ENVIRONMENTS_DIR", str(PARENT_DIR / "resources_servers/arc_agi_3/data/environment_files"))
).resolve()
pytestmark = [
    pytest.mark.skipif(not (GAMES / "ls20").is_dir(), reason=f"ARC-AGI-3 game files not found at {GAMES}"),
    pytest.mark.skipif(
        not venv_python(DEFAULT_DEST).is_file(), reason=f"official harness not installed at {DEFAULT_DEST}"
    ),
]
# The committed example games (cn04, ls20) and their rows.
ROWS = [
    json.loads(line)
    for line in (PARENT_DIR / "resources_servers/arc_agi_3/data/example.jsonl").read_text().splitlines()
]


class FakeHttpResponse:
    def __init__(self, text: str, status: int = 200):
        self._text = text
        self.status = status
        self.ok = status < 400
        self.cookies = {}

    async def read(self) -> bytes:
        return self._text.encode()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def row(game: str, index: int) -> dict:
    data = next(r for r in ROWS if r["verifier_metadata"]["game_id"] == game)
    return {**data, "verifier_metadata": {**data["verifier_metadata"], "dispatch_index": index}}


def plain_aiohttp_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gym's request() takes its global aiohttp client from the global config (the CLI of a Gym server); here one
    plain client per event loop stands in for it."""
    import aiohttp

    from nemo_gym.client import http_client

    clients = {}

    def client(*args, **kwargs):
        loop = asyncio.get_running_loop()
        if loop not in clients:
            clients[loop] = aiohttp.ClientSession()
        return clients[loop]

    monkeypatch.setattr(http_client, "get_global_aiohttp_client", client)


def start_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    play_seconds: float,
    benchmark_config: dict,
    harness_type: str = "provider",
):
    plain_aiohttp_clients(monkeypatch)
    stub = StubVLLM()
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="policy_model",
            base_url="http://stub.invalid/v1",
            api_key="unused",
            model="stub-model",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            propagate_context_overflow_errors=True,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )

    async def create_chat_completion(self, **body):
        return stub.reply(body)

    monkeypatch.setattr(model._clients[0].__class__, "create_chat_completion", create_chat_completion)
    model_http = TestClient(model.setup_webserver())
    model_lock = threading.Lock()
    resources = ArcAgi3ResourcesServer(
        config=ArcAgi3ResourcesServerConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="arc_agi_3",
            environments_dir=str(GAMES),
            game_server_dir=str(tmp_path / "game_server"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    resources.game_server.start(len(list(GAMES.glob("*/*/metadata.json"))))

    def call_model(url_path: str, body: dict):
        with model_lock:  # one TestClient, used from worker threads
            return model_http.post(url_path, json=body)

    async def post(server_name, url_path, json=None, cookies=None):
        if server_name == "policy_model":
            response = await asyncio.to_thread(call_model, url_path, json)
            return FakeHttpResponse(response.text, response.status_code)
        if server_name == "arc_agi_3_resources_server" and url_path == "/verify":
            result = await resources.verify(ArcAgi3VerifyRequest.model_validate(json))
            return FakeHttpResponse(result.model_dump_json())
        raise AssertionError(f"unexpected call {server_name} {url_path}")

    port = free_port()
    config = ArcAgi3OfficialAgentConfig(
        host="127.0.0.1",
        port=port,
        entrypoint="app.py",
        name="arc_agi_3_official_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="arc_agi_3_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        environments_dir=str(GAMES),
        output_dir=str(tmp_path),
        run_name="stack",
        play_seconds=play_seconds,
        window_grace_s=20,
        save_every_s=0,
        save_request_logs=True,
        harness_type=harness_type,
        benchmark_configs={harness_type: benchmark_config},
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = post

    async def get(server_name, url_path, **kwargs):
        assert (server_name, url_path) == ("arc_agi_3_resources_server", "/game_server")
        return FakeHttpResponse(json_module.dumps(await resources.game_server_info()))

    server_client.get = get
    agent = ArcAgi3OfficialAgent(config=config, server_client=server_client)
    monkeypatch.setattr(
        ArcAgi3OfficialAgent,
        "model_server_config",
        lambda self: {
            "uses_reasoning_parser": True,
            "uses_interleaved_reasoning": True,
            "propagate_context_overflow_errors": True,
        },
    )
    server = uvicorn.Server(uvicorn.Config(agent.setup_webserver(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        assert time.time() < deadline, "the agent server did not start"
        time.sleep(0.05)
    return stub, agent, server, thread, port, resources


def run_games(port: int, rows: list) -> list:
    import requests

    results = [None] * len(rows)

    def one(i):
        results[i] = requests.post(f"http://127.0.0.1:{port}/run", json=rows[i], timeout=600)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(rows))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def assistant_turns(body: dict) -> list:
    return [m for m in body.get("messages") or [] if m.get("role") == "assistant"]


PROVIDER_CONFIG = {
    "agent": {"MAX_ACTIONS_BASELINE_MULTIPLIER": 5, "MAX_CONTEXT_LENGTH": 262144, "MAX_RETRIES": 2},
    "runtime": {
        "reasoning_replay": "reasoning_content",
        "compaction": {"strategy": "harness_summary", "trigger_tokens": 40000, "summary_input_headroom_tokens": 1024},
    },
    "request": {"max_tokens": 256, "extra_body": {"chat_template_kwargs": {"preserve_thinking": True}}},
}
STANDARD_CONFIG = {
    "agent": {"MAX_ACTIONS_BASELINE_MULTIPLIER": 5, "MAX_CONTEXT_LENGTH": 60000, "MAX_RETRIES": 2},
    "runtime": {"state": "manual_rolling"},
    "request": {"max_tokens": 256, "extra_body": {"chat_template_kwargs": {"preserve_thinking": True}}},
}


def play_two_games(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, benchmark_config: dict, harness_type: str):
    # A long window: both games stop on their action budget first (the window only bounds a slow machine).
    stub, agent, server, thread, port, resources = start_stack(
        monkeypatch, tmp_path, play_seconds=120, benchmark_config=benchmark_config, harness_type=harness_type
    )
    try:
        results = run_games(port, [row("ls20", 0), row("cn04", 1)])
    finally:
        server.should_exit = True
        thread.join(timeout=30)
        resources.game_server.stop()
    for response in results:
        assert response.status_code == 200, response.text[:2000]
        result = response.json()
        assert result["harness"] == "official" and result["harness_type"] == harness_type
        assert result["agent_state"] in ("gave_up", "won"), result
        assert result["result_valid"] is True and result["timing_valid"] is True, result
        assert result["scorecard_runs"] == 1 and result["clock_offset_seconds"] >= 0
        assert result["actions"] > 3 and result["agent_error"] is None
        assert result["reward"] == pytest.approx(result["rhae"] / 100)
        # The call log: every call tagged by the worker with the game's step and level when it was sent.
        log = result["agent_call_log"]
        assert log and all(c["status"] == 200 and c["step"] >= 1 and c["level"] >= 1 for c in log), log[:3]
        assert [c["step"] for c in log] == sorted(c["step"] for c in log)
        assert max(c["step"] for c in log) <= result["actions"] + 1
        gameplay = [c["response_id"] for c in log if c["role"] == "gameplay"]
        (invocation,) = [r for r in result["ng_agent_observations"]["records"] if r["kind"] == "agent_invocation"]
        assert [c["response_id"] for c in invocation["model_calls"]] == gameplay
        assert result["harness_revision"]["commit"].startswith("27f3d03d")
    return stub, resources, results


def test_standard_harness_rolls_its_history_without_reasoning_and_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub, resources, results = play_two_games(monkeypatch, tmp_path, STANDARD_CONFIG, "standard")
    # manual_rolling resends earlier replies as text only, never summarizes, and drops the oldest turns.
    assert stub.requests and not any(
        str(b["messages"][0].get("content", "")).startswith("You create") for b in stub.requests
    )
    for body in stub.requests:
        for message in assistant_turns(body):
            assert not message.get("reasoning_content") and not message.get("reasoning"), message
        assert "<think>" not in json.dumps(body["messages"])
    # Each game plays 100+ turns; a rolled transcript keeps only the latest few.
    assert len(stub.requests) > 200
    assert max(len(assistant_turns(b)) for b in stub.requests) < 50
    assert all(c["role"] == "gameplay" for r in results for c in r.json()["agent_call_log"])
    config = json.loads((tmp_path / "stack" / "run_config.json").read_text())
    assert config["agent"]["harness_type"] == "standard"
    assert config["benchmark_config"]["runtime"]["state"] == "manual_rolling"


def test_two_games_keep_all_earlier_reasoning_and_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stub, resources, results = play_two_games(monkeypatch, tmp_path, PROVIDER_CONFIG, "provider")
    # vLLM side: every assistant turn of every request carries the reasoning of that reply, in both fields.
    game_requests = [b for b in stub.requests if not str(b["messages"][0].get("content", "")).startswith("You create")]
    summary_requests = [b for b in stub.requests if b not in game_requests]
    resent = 0
    for body in stub.requests:
        for message in assistant_turns(body):
            reasoning = stub.reasoning_of[message["content"]]
            assert message.get("reasoning_content") == reasoning and message.get("reasoning") == reasoning, message
            resent += 1
        assert "<think>" not in json.dumps(body["messages"])
        assert body["chat_template_kwargs"] == {"preserve_thinking": True}
    assert resent > 20
    # The history grows turn by turn until a compaction: the turn count only drops right after a summary.
    assert max(len(assistant_turns(b)) for b in game_requests) >= 5
    assert summary_requests, "the low trigger should have compacted at least once"
    for body in summary_requests:
        assert assistant_turns(body), "the summary request sees the history with its reasoning"
    # Upstream's summary calls are the call logs' helper calls, and each starts a compaction record.
    helpers = [c for r in results for c in r.json()["agent_call_log"] if c["role"] == "helper"]
    assert len(helpers) == len(summary_requests) and {c["history_id"] for c in helpers} == {"harness-summary"}
    compactions = [
        record
        for r in results
        for record in r.json()["ng_agent_observations"]["records"]
        if record["kind"] == "context_compaction"
    ]
    assert compactions and all(c["model_calls"] for c in compactions)

    run_dir = tmp_path / "stack"
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["harness"]["provenance"]["commit"].startswith("27f3d03d")
    assert config["game_server"] == resources.game_server.url
    games = sorted((run_dir / "games").iterdir(), key=lambda d: int(d.name.split("-", 1)[0]))
    assert len(games) == 2
    for game_dir in games:
        report = json.loads((game_dir / "result.json").read_text())
        # Upstream's own stop: a level's budget is ceil(baseline x MAX_ACTIONS_BASELINE_MULTIPLIER) actions.
        assert report["exit_reason"] == "ACTION_BUDGET" or report.get("stopped"), report
        assert (game_dir / "requests.jsonl").is_file() and list((game_dir / "recordings").glob("*"))
    cards = list((resources.game_server.out_dir / "cards").glob("*.json"))
    assert len(cards) == 2  # closed by the verifier, kept by the server's hook
    stops = 0
    for response, game_dir in zip(results, games):
        result, report = response.json(), json.loads((game_dir / "result.json").read_text())
        assert result["stop_reason"] == (
            "level_action_budget" if report["exit_reason"] == "ACTION_BUDGET" else "play_window"
        )
        if report["exit_reason"] == "ACTION_BUDGET":
            # The level's budget, from the human baselines that the game server publishes: ceil(5 x baseline) actions
            # on that level (without the baselines upstream falls back to 10 actions per game).
            level = result["levels_completed"]
            baseline = next(r for r in ROWS if r["verifier_metadata"]["game_id"] == result["game_id"])
            budget = -(-baseline["verifier_metadata"]["baseline_actions"][level] * 5 // 1)
            assert result["actions"] >= budget > 11, (result["actions"], budget)
            stops += 1
    assert stops == 2, "both games should stop on upstream's per-level budget"
