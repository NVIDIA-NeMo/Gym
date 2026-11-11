# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym model transport for the author's original VS-Bench gameplay runner.

External source: https://github.com/zelaix/VS-Bench, MIT, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0. The runner and game code are
executed from a separately verified source export. The adapter reports raw
returns and approximate normalized scores using rounded paper anchors.
"""

import asyncio
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import ClientError
from fastapi import Body, Request, Response
from pydantic import ConfigDict, PrivateAttr

from benchmarks.vs_bench.normalize_gameplay import normalized_gameplay_approx, normalized_pong_approx
from benchmarks.vs_bench.paper_gameplay_schedule import paper_competitive_episodes
from benchmarks.vs_bench.prepare import AUTHOR_REVISION, GAMES
from benchmarks.vs_bench.reference_identity import validate_author_checkout
from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, accumulate_response_usage
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.vs_bench_reference.app import author_chat_to_gym, read_frame, write_frame


class GameplayConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    author_root: str
    author_python: str
    episode_root: str
    reasoning_model: bool = True
    temperature: float = 1.0
    max_response_tokens: int
    max_reasoning_tokens: int
    max_author_processes: int = 4
    frame_timeout_seconds: float = 3600


class GameplayRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class GameplayResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]
    terminal_returns: list[list[float]]
    model_seats: list[int]
    episode_steps: list[int]
    author_model_calls: int
    source_results_path: str
    reward_definition: str
    response_scope: str = "suite_summary"


class VSBenchGameplayAgent(SimpleResponsesAPIAgent):
    config: GameplayConfig
    _author_processes: asyncio.Semaphore = PrivateAttr()

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            if len(task) != 1:
                raise ValueError("Gameplay expects one result per task")
            result = task[0]
            game = (result.get("verifier_metadata") or {}).get("game")
            if game not in GAMES:
                raise ValueError("Invalid gameplay game")
            grouped.setdefault(game, []).append(result)
        complete = 0
        paper_complete: set[str] = set()
        normalized_scores: dict[str, float] = {}
        for game, results in grouped.items():
            paper = [r for r in results if (r.get("verifier_metadata") or {}).get("paper_episode") is not None]
            if paper:
                if len(paper) != len(results) or game not in ("kuhn_poker", "breakthrough"):
                    raise ValueError("Mixed gameplay protocols")
                expected = {tuple(asdict(item).values()) for item in paper_competitive_episodes(game)}
                keys = ("game", "block", "index", "environment_seed", "model_seat")
                actual = []
                for row in paper:
                    meta = row["verifier_metadata"]
                    ep = meta["paper_episode"]
                    if meta.get("protocol") != "paper_competitive_v1" or set(ep) != set(keys):
                        raise ValueError("Invalid paper episode metadata")
                    identity = tuple(ep[k] for k in keys)
                    if identity not in expected or row["model_seats"] != [ep["model_seat"]]:
                        raise ValueError("Incorrect paper episode identity or seat")
                    actual.append(identity)
                if len(actual) != len(set(actual)):
                    raise ValueError("Duplicate paper episodes")
                metrics[f"vs_bench/gameplay/{game}/episodes"] = len(actual)
                if set(actual) != expected:
                    continue
                paper_complete.add(game)
                reward = sum(float(r["reward"]) for r in paper) / len(paper)
                if game == "kuhn_poker":
                    for block in range(10):
                        values = [
                            float(r["reward"])
                            for r in paper
                            if r["verifier_metadata"]["paper_episode"]["block"] == block
                        ]
                        metrics[f"vs_bench/gameplay/kuhn_poker/block_{block}/raw_mean"] = sum(values) / len(values)
            else:
                if len(results) != 1:
                    raise ValueError("Duplicate gameplay suites")
                result = results[0]
                returns = result["terminal_returns"]
                seats = result["model_seats"]
                reward = float(result["reward"])
                metrics[f"vs_bench/gameplay/{game}/episodes"] = len(returns)
                if (result.get("verifier_metadata") or {}).get("protocol") == "paper_competitive_v1":
                    if game in ("kuhn_poker", "breakthrough"):
                        raise ValueError("Paper competitive games require per-episode identities")
                    if len(returns) != 10:
                        continue
                    paper_complete.add(game)
                if game in ("coin_dilemma", "monster_hunt", "battle_of_the_colors"):
                    if sorted(seats) != [0, 1] or not returns or any(len(row) <= 1 for row in returns):
                        raise ValueError("Mixed-motive scores require both model seats")
                    means = [sum(float(row[seat]) for row in returns) / len(returns) for seat in (0, 1)]
                    for seat, value in enumerate(means):
                        metrics[f"vs_bench/gameplay/{game}/raw_seat_{seat}_return"] = value
                    reward = sum(means) / 2
                    metrics[f"vs_bench/gameplay/{game}/raw_two_seat_mean"] = reward
            complete += 1
            if paper:
                metrics[f"vs_bench/gameplay/{game}/raw_model_seat_return"] = reward
            elif game not in ("coin_dilemma", "monster_hunt", "battle_of_the_colors"):
                metrics[f"vs_bench/gameplay/{game}/raw_first_model_seat_return"] = reward
            if game == "atari_pong":
                returns, steps, seats = result["terminal_returns"], result["episode_steps"], result["model_seats"]
                if not returns or len(returns) != len(steps) or len(seats) != 1:
                    raise ValueError("Pong requires paired episode returns and steps for one model seat")
                seat = seats[0]
                points = sum(float(row[seat]) for row in returns) / len(returns)
                survival = sum(float(step) for step in steps) / len(steps)
                metrics["vs_bench/gameplay/atari_pong/model_points"] = points
                metrics["vs_bench/gameplay/atari_pong/survival_steps"] = survival
                score = normalized_pong_approx(points, survival)
            else:
                score = normalized_gameplay_approx(game, reward)
            metrics[f"vs_bench/gameplay/{game}/normalized_approx"] = score
            normalized_scores[game] = score
        metrics["vs_bench/gameplay/games_completed"] = complete
        if paper_complete == set(GAMES) and set(normalized_scores) == set(GAMES):
            metrics["vs_bench/gameplay/normalized_equal_game_mean_approx"] = sum(normalized_scores.values()) / len(
                GAMES
            )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        # This headline is emitted only after complete paper-scheduled coverage.
        key = "vs_bench/gameplay/normalized_equal_game_mean_approx"
        return {key: agent_metrics[key]} if key in agent_metrics else {}

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        validate_author_checkout(self.config.author_root, self.config.author_python)
        root = Path(self.config.episode_root)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("Provide an existing, explicitly placed gameplay artifact directory")
        if (
            min(self.config.max_response_tokens, self.config.max_reasoning_tokens, self.config.max_author_processes)
            < 1
        ):
            raise ValueError("Token and process limits must be positive")
        self._author_processes = asyncio.Semaphore(self.config.max_author_processes)

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("The original gameplay runner owns its model calls")

    async def run(self, request: Request, body: GameplayRequest) -> GameplayResponse:
        meta = body.verifier_metadata
        game = meta.get("game")
        if meta.get("track") != "gameplay" or meta.get("author_revision") != AUTHOR_REVISION or game not in GAMES:
            raise ValueError("Invalid author gameplay identity")
        episode = meta.get("paper_episode")
        if episode is not None:
            if meta.get("protocol") != "paper_competitive_v1" or game not in ("kuhn_poker", "breakthrough"):
                raise ValueError("Invalid paper episode protocol")
            allowed = {tuple(asdict(item).values()) for item in paper_competitive_episodes(game)}
            keys = ("game", "block", "index", "environment_seed", "model_seat")
            if set(episode) != set(keys) or tuple(episode[k] for k in keys) not in allowed:
                raise ValueError("Invalid paper episode identity")
        elif meta.get("protocol") == "paper_competitive_v1" and game in ("kuhn_poker", "breakthrough"):
            raise ValueError("Missing paper episode identity")
        episode_dir = str(Path(self.config.episode_root) / f"{game}-{uuid4().hex}")
        task = {
            "mode": "episode" if episode is not None else "suite",
            "paper_episode": episode,
            "game": game,
            "episode_id": 0,
            "episode_dir": episode_dir,
            "reasoning_model": self.config.reasoning_model,
            "temperature": self.config.temperature,
            "max_response_tokens": self.config.max_response_tokens,
            "max_reasoning_tokens": self.config.max_reasoning_tokens,
        }
        payload = body.responses_create_params.model_dump(mode="json", exclude_none=True)
        worker = Path(__file__).resolve().parents[2] / "benchmarks/vs_bench/reference_gameplay_agent.py"
        env = {
            "PATH": os.defpath,
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        async with self._author_processes:
            process = await asyncio.create_subprocess_exec(
                self.config.author_python,
                str(worker),
                self.config.author_root,
                cwd=self.config.author_root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            writer_lock = asyncio.Lock()
            completed_calls = 0
            total_usage = None

            async def route(frame):
                nonlocal completed_calls, total_usage
                author_call = frame["request"]
                params = dict(payload)
                params["input"] = author_chat_to_gym(author_call["messages"])
                params["temperature"] = author_call["temperature"]
                params["max_output_tokens"] = author_call["max_completion_tokens"]
                try:
                    reply = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=self.url_path_for_run("/v1/responses", body),
                        json=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
                        cookies=request.cookies,
                    )
                    await raise_for_status(reply)
                    response = NeMoGymResponse.model_validate(await get_response_json(reply))
                    if (
                        response.status == "incomplete"
                        and response.incomplete_details
                        and response.incomplete_details.reason == "max_output_tokens"
                    ):
                        finish = "length"
                    elif response.status in (None, "completed"):
                        finish = "stop"
                    else:
                        raise ValueError("Unexpected policy response status")
                    completed_calls += 1
                    usage = response.usage
                    total_usage = accumulate_response_usage(total_usage, usage)
                    result = {
                        "type": "model_response",
                        "request_id": frame["request_id"],
                        "finish_reason": finish,
                        "content": response.output_text,
                        "prompt_tokens": usage.input_tokens if usage else None,
                        "completion_tokens": usage.output_tokens if usage else None,
                    }
                except (ClientError, asyncio.TimeoutError, ValueError):
                    result = {"type": "model_error", "request_id": frame["request_id"]}
                async with writer_lock:
                    await write_frame(process.stdin, result)

            try:
                await write_frame(process.stdin, task)
                async with asyncio.TaskGroup() as calls:
                    while True:
                        frame = await asyncio.wait_for(read_frame(process.stdout), self.config.frame_timeout_seconds)
                        if frame["type"] == "complete":
                            break
                        if frame["type"] != "model_request":
                            raise RuntimeError(
                                f"Original gameplay runner failed: {frame.get('error_type')}: {frame.get('message')}"
                            )
                        calls.create_task(route(frame))
                if await process.wait() != 0 or completed_calls == 0 or frame.get("transport_fallbacks", 0):
                    raise RuntimeError("Gameplay run encountered a policy transport failure")
                returns, seats, steps = frame["returns"], frame["model_seats"], frame["steps"]
                if episode is not None:
                    returns, steps = [returns], [steps]
                if not returns or not seats or len(returns) != len(steps):
                    raise ValueError("Original runner did not return complete episode results")
                seat = seats[0]
                reward = sum(float(row[seat]) for row in returns) / len(returns)
                return GameplayResponse(
                    **body.model_dump(),
                    response=NeMoGymResponse.model_validate(
                        {
                            "id": f"vsbench-{uuid4().hex}",
                            "object": "response",
                            "created_at": time.time(),
                            "model": self.config.model_server.name,
                            "status": "completed",
                            "output": [
                                {
                                    "id": "suite-summary",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "annotations": [],
                                            "text": f"Completed {len(returns)} {game} episode(s); results are in terminal_returns.",
                                        }
                                    ],
                                }
                            ],
                            "parallel_tool_calls": False,
                            "tool_choice": "auto",
                            "tools": [],
                            "usage": total_usage.model_dump() if total_usage is not None else None,
                        }
                    ),
                    reward=reward,
                    terminal_returns=returns,
                    model_seats=seats,
                    episode_steps=steps,
                    author_model_calls=frame["model_calls"],
                    source_results_path=frame["source_results_path"],
                    reward_definition="mean return of first model-controlled seat; raw, not normalized",
                )
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()


if __name__ == "__main__":
    VSBenchGameplayAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VSBenchGameplayAgent.run_webserver()
