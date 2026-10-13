# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""OpenAir Congestion client demo -- one scripted episode over HTTP.

Drives one full episode against the served environment over the real HTTP
surface (POST /reset, then a POST /step loop) using a scripted
congestion-relief heuristic in place of an LLM policy. The heuristic selects
one condition-appropriate setpoint and then coasts while it persists. Per-step tool
calls, guardrail verdicts, and rewards are printed, then the episode return.

If a gym is running (`gym env start` with this server's config), the demo connects
to the served `openair_congestion` instance. Otherwise it boots a local
in-process server on the default replay backend, so it runs fully offline --
no 5G lab, no GPU, no model server.

Usage:
    python resources_servers/openair_congestion/client.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import ServerClient
from resources_servers.openair_congestion.app import (
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
)


SERVER_NAME = "openair_congestion"
_EXAMPLE_JSONL = Path(__file__).parent / "data" / "example.jsonl"

# render.to_user_text emits one block per cell; the heuristic reads only those
# model-visible metrics, exactly like the LLM policy it stands in for.
_CELL_BLOCK = re.compile(
    r"- Cell (?P<cell_id>\d+):(?P<body>.*?)(?=\n- Cell |\nLast action:|\nChoose one tool call|\Z)",
    re.DOTALL,
)
_P99 = re.compile(r"p99=(\d+)%")
_FAIRNESS = re.compile(r"Jain fairness ([0-9.]+)")
_SINR = re.compile(r"SINR (-?[0-9.]+) dB")
_BLER = re.compile(r"BLER ([0-9.]+)%")

# One request-body poster; the transport (gym-connected vs. local) is bound in main().
PostFn = Callable[[str, dict], Awaitable[dict]]
_LOGGER = logging.getLogger(__name__)


def _load_task_row() -> dict:
    # First row of the shipped dataset: the same task the gymnasium_agent
    # would forward on /reset (agent_ref is agent-side only).
    with open(_EXAMPLE_JSONL) as f:
        row = json.loads(f.readline())
    row.pop("agent_ref", None)
    return row


def choose_action(observation: str, step_idx: int) -> dict[str, Any]:
    """Choose one condition-appropriate deterministic relief setpoint.

    The policy intervenes once, then coasts while the replay setpoint persists,
    using only rendered KPIs: aggressive handover for cell-edge pressure,
    round-robin for a fairness deficit, and higher UL power only when PRB
    pressure is high and link quality is healthy.
    """
    if step_idx > 0:
        return {"name": "noop", "arguments": {}}

    cells: list[dict[str, float | int]] = []
    for match in _CELL_BLOCK.finditer(observation or ""):
        body = match.group("body")
        p99_match = _P99.search(body)
        fairness_match = _FAIRNESS.search(body)
        sinrs = [float(value) for value in _SINR.findall(body)]
        blers = [float(value) / 100.0 for value in _BLER.findall(body)]
        cells.append(
            {
                "cell_id": int(match.group("cell_id")),
                "p99": float(p99_match.group(1)) / 100.0 if p99_match else 0.0,
                "fairness": float(fairness_match.group(1)) if fairness_match else 1.0,
                "mean_sinr": statistics.fmean(sinrs) if sinrs else 20.0,
                "mean_bler": statistics.fmean(blers) if blers else 0.0,
            }
        )
    if not cells:
        return {"name": "noop", "arguments": {}}

    cell_edge = min(cells, key=lambda cell: (cell["mean_sinr"], -cell["mean_bler"]))
    if cell_edge["mean_sinr"] < 5.0 or cell_edge["mean_bler"] > 0.20:
        return {
            "name": "set_handover_trigger",
            "arguments": {
                "cell_id": cell_edge["cell_id"],
                "a3_offset_db": -24.0,
                "ttt_ms": 0,
            },
        }

    unfair = min(cells, key=lambda cell: cell["fairness"])
    if unfair["fairness"] < 0.95:
        return {
            "name": "set_scheduler_policy",
            "arguments": {"cell_id": unfair["cell_id"], "policy": "RR"},
        }

    congested = max(cells, key=lambda cell: cell["p99"])
    if congested["p99"] >= 0.80:
        return {
            "name": "set_ul_power_control",
            "arguments": {
                "cell_id": congested["cell_id"],
                "p0_dbm": 10.0,
                "alpha": 1.0,
            },
        }
    return {"name": "noop", "arguments": {}}


def _tool_response(name: str, arguments: dict, step_idx: int) -> dict:
    # The response shape the model server would produce for one tool call.
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id=f"call_{step_idx}",
                name=name,
                type="function_call",
                id=f"fc_{step_idx}",
                status="completed",
            )
        ],
        id="r",
        created_at=0.0,
        model="scripted-relief",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ).model_dump()


async def drive_episode(post: PostFn) -> float:
    row = _load_task_row()
    reset = await post("/reset", row)
    try:
        info = reset["info"]
        print(f"reset: episode_id={info['episode_id']} seed={info['seed']} scenario_id={info['scenario_id']}")

        observation = reset["observation"]
        episode_return = 0.0
        step_idx = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = choose_action(observation, step_idx)
            step = await post(
                "/step",
                {
                    "responses_create_params": row["responses_create_params"],
                    "response": _tool_response(action["name"], action["arguments"], step_idx),
                },
            )
            reward = float(step["reward"])
            episode_return += reward
            verdict = (
                "accepted"
                if step["info"].get("guardrail_accepted", True)
                else f"REJECTED ({step['info'].get('rejection_reason')})"
            )
            print(
                f"step {step_idx:2d}: {action['name']}({json.dumps(action['arguments'])}) "
                f"reward={reward:+.4f} {verdict}"
            )
            observation = step["observation"]
            terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
            step_idx += 1
    except BaseException:
        try:
            await post("/close", {})
        except Exception:
            _LOGGER.exception("Failed to close OpenAir congestion episode after client error")
        raise

    try:
        await post("/close", {})
    except Exception:
        # The rollout is already complete and its return is valid. Surface the
        # cleanup failure without discarding the successful result.
        _LOGGER.exception("Failed to close completed OpenAir congestion episode")
    print(f"episode over: terminated={terminated} truncated={truncated} steps={step_idx} return={episode_return:+.4f}")
    return episode_return


async def _run_against_gym(server_client: ServerClient) -> float:
    cookies: dict = {}

    async def post(url_path: str, payload: dict) -> dict:
        response = await server_client.post(server_name=SERVER_NAME, url_path=url_path, json=payload, cookies=cookies)
        response.raise_for_status()
        if response.cookies:
            cookies.update(response.cookies)
        return await response.json()

    return await drive_episode(post)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_local_server() -> str:
    """Boot the env in-process on the replay backend; returns its base URL."""
    config = OpenAirCongestionResourcesServerConfig(
        host="127.0.0.1",
        port=_free_port(),
        entrypoint="app.py",
        name=SERVER_NAME,
    )
    # The env never calls out through server_client; an empty one satisfies
    # the constructor without a head server.
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="", port=0), global_config_dict=OmegaConf.create({})
    )
    env = OpenAirCongestionEnv(config=config, server_client=server_client)
    server = uvicorn.Server(
        uvicorn.Config(env.setup_webserver(), host=config.host, port=config.port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError(f"local server failed to start on {config.host}:{config.port}")
        time.sleep(0.05)
    return f"http://{config.host}:{config.port}"


async def _run_against_local(base_url: str) -> float:
    # unsafe=True: the default cookie jar drops the session cookie for bare-IP
    # hosts like 127.0.0.1.
    async with aiohttp.ClientSession(base_url=base_url, cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:

        async def post(url_path: str, payload: dict) -> dict:
            async with session.post(url_path, json=payload) as response:
                response.raise_for_status()
                return await response.json()

        return await drive_episode(post)


def main() -> None:
    try:
        server_client = ServerClient.load_from_global_config()
    except Exception as e:  # no head server: run standalone
        print(f"No running gym found ({e}); booting a local in-process server (backend: replay).")
        asyncio.run(_run_against_local(_start_local_server()))
    else:
        asyncio.run(_run_against_gym(server_client))


if __name__ == "__main__":
    main()
