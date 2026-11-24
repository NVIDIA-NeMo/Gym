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
"""CPU smoke: the pinned harness plays 2 real games against a scripted model, and the verifier scores them.

Needs Linux (the harness sandbox uses prlimit), the harness install (``setup_harness.py``) and the game files
(``ARC3_ENVIRONMENTS_DIR`` or ``resources_servers/arc_agi_3/data/environment_files``); skipped otherwise. The model
server is emulated in-process: the same request/response transformation the Gym vLLM model server applies with
``uses_reasoning_parser: true``, in front of ``stub_model.stub_reply``.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.arc_agi_3.app import ArcAgi3ResourcesServer, ArcAgi3ResourcesServerConfig, ArcAgi3VerifyRequest
from responses_api_agents.arc_agi_3_agent.call_log import call_log_entries
from responses_api_agents.arc_agi_3_agent.gym_transport import GymChatTransport, RequestsProxy
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    HarnessRuntime,
    compose_harness_env,
    load_harness_env_toml,
)
from responses_api_agents.arc_agi_3_agent.setup_harness import DEFAULT_DEST


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stub_model  # noqa: E402


REPO = HERE.parents[2]
GAMES = Path(
    os.environ.get(
        "ARC3_ENVIRONMENTS_DIR", str(REPO / "resources_servers" / "arc_agi_3" / "data" / "environment_files")
    )
).resolve()
HARNESS = Path(os.environ.get("ARC3_HARNESS_DIR", str(DEFAULT_DEST))).resolve()
PLAY_SECONDS = float(os.environ.get("ARC3_SMOKE_PLAY_SECONDS", "45"))
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the harness sandbox needs Linux"),
    pytest.mark.skipif(not (HARNESS / "harness").is_dir(), reason=f"harness not installed at {HARNESS}"),
    pytest.mark.skipif(not (GAMES / "ls20").is_dir(), reason=f"game files not found at {GAMES}"),
]


ENGINE_400 = (
    "This model's maximum context length is 262144 tokens. However, you requested 270000 tokens in the messages, "
    "Please reduce the length of the messages."
)


class EmulatedGymModel:
    """responses_api_models/vllm_model chat_completions (uses_reasoning_parser, propagate_context_overflow_errors) over
    the scripted model. ``reject_over_messages``: the first request with more messages is rejected once with the
    engine's context-length 400, as vLLM does when the harness's token estimate is too low."""

    def __init__(self, reject_over_messages: int = 0) -> None:
        self.vllm_requests = []
        self.sessions = []  # the game (session cookie) of each request
        self.reject_over_messages = reject_over_messages
        self.rejected = None
        self._next_session = 0

    async def post(self, body, cookies, game_key=None):
        # Like Gym's session middleware: a request without a session cookie opens a new session. The transport
        # keeps one cookie per game, so the cookie identifies the game across history trims.
        session = (cookies or {}).get("session")
        if session is None:
            self._next_session += 1
            session = f"game{self._next_session}"
        vllm = {k: v for k, v in body.items() if k != "metadata"}
        metadata = body.get("metadata") or {}
        vllm = json.loads(metadata.get("extra_body", "{}")) | vllm
        vllm["chat_template_kwargs"] = json.loads(metadata.get("chat_template_kwargs", "{}"))
        for message in vllm["messages"]:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                found = THINK.findall(message["content"])
                if found:
                    message["content"] = THINK.sub("", message["content"])
                    message["reasoning_content"] = found[0]
        self.vllm_requests.append(vllm)
        self.sessions.append(session)
        if (
            self.rejected is None
            and self.reject_over_messages
            and len(vllm["messages"]) > self.reject_over_messages
            and not vllm.get("continue_final_message")
        ):
            self.rejected = (session, len(vllm["messages"]), len(self.vllm_requests) - 1)
            return 400, json.dumps({"detail": ENGINE_400}), {"session": session}
        reply = stub_model.stub_reply(vllm)
        message = reply["choices"][0]["message"]
        reasoning = message.pop("reasoning_content", None)
        if reasoning:
            message["content"] = f"<think>{reasoning}</think>" + (message.get("content") or "")
        return 200, json.dumps(reply), {"session": session}


def test_two_games_play_and_score(tmp_path: Path) -> None:
    rows = [
        json.loads(line) for line in (REPO / "resources_servers/arc_agi_3/data/example.jsonl").read_text().splitlines()
    ]
    rows = rows[:2]
    model = EmulatedGymModel(reject_over_messages=int(os.environ.get("ARC3_SMOKE_REJECT_OVER_MESSAGES", "20")))
    # The authority: the resources server's official game server, which the harness plays through.
    server = ArcAgi3ResourcesServer(
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
    server.game_server.start(len(list(GAMES.glob("*/*/metadata.json"))))  # the 2 committed games, or all 25

    async def main():
        env = compose_harness_env(
            load_harness_env_toml(HARNESS),
            base_url="http://policy/v1",
            model_id="policy",
            hyperparameters={"request_seed": 1},
            overrides={},
        )
        runtime = HarnessRuntime(
            install_root=HARNESS,
            environments_dir=GAMES,
            run_dir=tmp_path / "run",
            env=env,
            play_seconds=PLAY_SECONDS,
            concurrency=len(rows),
            num_attempts=len(rows),
            save_request_logs=True,
            mem_lean=True,
            telemetry=True,
            seed=1,
            save_every_s=10,
            game_server_url=server.game_server.url,
            api_key="smoke-key",
        )
        transport = GymChatTransport(
            asyncio.get_running_loop(), model.post, chat_template_kwargs={"truncate_history_thinking": False}
        )
        await asyncio.to_thread(runtime.start, RequestsProxy(transport), {"smoke": True})
        plays = [
            runtime.play_game(r["verifier_metadata"], r["verifier_metadata"]["dispatch_index"], transport)
            for r in rows
        ]
        attempts = await asyncio.gather(*plays)
        await runtime.shutdown(transport)
        return runtime, attempts, [call_log_entries(transport.take_call_log(a.game_key)) for a in attempts]

    try:
        runtime, attempts, logs = asyncio.run(main())
        scored_rows = [verify_attempt(server, runtime, row, attempt) for row, attempt in zip(rows, attempts)]
    finally:
        server.game_server.stop()
    assert len(scored_rows) == 2
    check_requests_and_outputs(model, runtime)
    # Every call of a game carries the game's step and level when it was sent.
    for attempt, log in zip(attempts, logs):
        assert log and all(c["level"] >= 1 and 1 <= c["step"] <= len(attempt.record["history"]) + 1 for c in log)


def verify_attempt(server, runtime, row, attempt):
    record = attempt.record
    assert attempt.play_error is None and attempt.persistence_error is None
    assert record["state"] in ("gave_up", "won"), record.get("solver_note")
    assert len(record["history"]) > 0, "the scripted model's actions did not reach the game"
    verify = ArcAgi3VerifyRequest(
        responses_create_params=row["responses_create_params"],
        response={
            "id": "r",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "output": [],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
        },
        verifier_metadata=row["verifier_metadata"],
        game_record=record,
        scorecard_id=attempt.scorecard_id,
        game_guid=attempt.guid,
        scorecard_api_key=runtime.api_key,
        benchmark_start_time=runtime.start_time_utc,
    )
    scored = asyncio.run(server.verify(verify))
    # The scorecard agrees with the harness's record (a mismatch raises), and RHAE@T has the server's times.
    assert scored.timing_valid and scored.result_valid and scored.clock_offset_seconds >= 0
    assert scored.actions == len(record["history"]) and scored.scorecard_runs >= 1
    assert scored.levels_completed == record["levels_completed"]
    assert scored.rhae == pytest.approx(record["final_score"])
    assert 0.0 <= scored.reward <= 1.0
    return scored


def check_requests_and_outputs(model, runtime):
    # Requests reached the "vLLM" side in the harness's format.
    first = model.vllm_requests[0]
    assert first["top_k"] == 20 and first["temperature"] == 0.7 and first["seed"] == 1
    assert first["chat_template_kwargs"]["enable_thinking"] is True
    assert first["chat_template_kwargs"]["truncate_history_thinking"] is False
    assert any(
        part.get("type") == "image_url"
        for m in first["messages"]
        if isinstance(m.get("content"), list)
        for part in m["content"]
    )
    assert any(
        m.get("reasoning_content") for r in model.vllm_requests for m in r["messages"] if m["role"] == "assistant"
    )

    # F3: the rejected request was retried smaller (the harness's trim path), not turned into an empty turn.
    if model.reject_over_messages:
        assert model.rejected is not None, "no request grew past the rejection threshold"
        key, size, index = model.rejected
        later = [r for r, s in zip(model.vllm_requests[index + 1 :], model.sessions[index + 1 :]) if s == key]
        assert later and len(later[0]["messages"]) < size
        transcripts = "".join(p.read_text() for p in (runtime.run_dir / "transcripts").glob("*.txt"))
        assert "context_overflow_recovered" in transcripts

    bench = json.loads((runtime.run_dir / "benchmark.json").read_text())
    assert len(bench["game_runs"]) == 2 and bench["start_time"] == runtime.start_time.isoformat()
    assert bench["run_config_sha256"] == runtime.run_config_sha256
    assert json.loads((runtime.run_dir / "harness_env.json").read_text())["ARC3_GDN_OWNERSHIP"] == "0"
    assert runtime.closed and runtime.save_failures == 0
