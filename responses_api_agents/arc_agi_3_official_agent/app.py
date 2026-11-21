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
"""ARC-AGI-3 agent with the official ARC Prize harness (arc-agi-3-benchmarking, pinned; MIT): one game per agent session.

An option next to ``arc_agi_3_agent`` (the Kaggle kit's harness, the default), with the same agent sessions
(``ArcAgi3GameSessions``: the seed binds the row's pinned game, the one activation plays it, the close stops a game
still in play and returns its observations). The same ``arc_agi_3`` resources server scores the games: same rows and
pins, the official game server's scorecard as the authority (reward = score / 100) and RHAE@T from the server's own
action times.

- The harness runs unmodified in its own venv (its openai / anthropic pins conflict with Gym's), one worker process
  per game (``worker.py``). It plays as upstream's own ``Swarm`` does: an ONLINE ``arc_agi.Arcade`` on the official
  game server, on a scorecard that this agent opens for the rollout.
- The harness's model calls go to this server's route ``/official/<game>/v1/chat/completions``, which sends them to the
  Gym model server on the rollout's own path (``url_path_for_run``, so Gym's capture attributes every call to the
  rollout) and maps the formats (``gym_transport``): the reasoning of earlier replies travels as a ``<think>`` content
  prefix that Gym's vLLM model server splits back into ``reasoning_content``, and the reply's reasoning comes back in
  ``reasoning_content``.
- The worker tags every model call with the game's step and level when it was sent and its role (``gameplay``, or
  ``helper`` / ``harness-summary`` for upstream's compaction summary), in request headers (``worker.CALL_HEADERS``).
  The route records each call in the game's call log (``arc_agi_3_agent.call_log``), returned as the session close's
  observations (the accepted gameplay calls in order, one record per compaction) and, in the activation's report,
  ``agent_call_log`` (every call sent: role, history id, step, level, error).
- The worker reports the game (guid, levels, final state, the actions the server answered); the verifier checks that
  report against the scorecard and fails loudly on a mismatch. The rollout's ``stop_reason`` says why the game ended
  (``STOP_REASONS``: e.g. ``level_action_budget`` or ``total_action_budget`` versus ``play_window``, which share the
  agent state ``gave_up``).
- The play clock (``run_clock``): ``shared`` (default, benchmarking) is one run clock, every game's window ends at the
  first game's start + ``play_seconds``; ``per_game`` gives each game ``play_seconds`` from its own start.

What the harness shows the model about its progress: every observation (a user message) starts with
``State: <NOT_FINISHED|GAME_OVER|WIN>`` and ``Levels completed: N`` (the level on the board is N + 1), with ``New
Level:`` before the first frame of a new level; it shows no step number. The call log's ``step`` / ``level`` are the
game's own (actions answered + 1, levels completed + 1 when the call was sent).

Two harness types (``harness_type``), upstream's two harnesses for chat-completions models:

- ``provider`` (default): the Provider Adapter harness, upstream's continuous conversation with the open-source Chat
  Completions adapter. The reasoning of earlier replies is resent (``reasoning_replay``) and the history is compacted by
  a harness summary. The agent refuses to start unless the model server keeps that reasoning
  (``uses_reasoning_parser`` and ``uses_interleaved_reasoning``).
- ``standard``: the Standard harness, ``manual_rolling``. The oldest turns are dropped while the transcript exceeds
  ``agent.MAX_CONTEXT_LENGTH`` characters, earlier replies are resent without their reasoning, and there are no
  summaries.

``benchmark_configs`` holds one upstream ``model_configs.yaml`` entry per type; the defaults in the config are set
for a model served with a 262,144-token window.
"""

import asyncio
import collections
import contextlib
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.agents import BaseResponsesAPIAgentConfig
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import (
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.arc_agi_3.app import ATTEMPT_REPORT_KEY
from resources_servers.arc_agi_3.setup_arc_agi import ensure_flask
from responses_api_agents.arc_agi_3_agent.app import ArcAgi3GameSessions, ArcAgi3Session, redact
from responses_api_agents.arc_agi_3_agent.call_log import (
    GAMEPLAY_ROLE,
    GameCallLog,
    call_log_entries,
    model_unavailable,
    observations,
)
from responses_api_agents.arc_agi_3_agent.gym_transport import from_gym_chat_response, to_gym_chat_request
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    DEFAULT_RESET_TIMEOUT_S,
    GameStartError,
    game_files_sha256,
)
from responses_api_agents.arc_agi_3_official_agent.official_game import OfficialRun
from responses_api_agents.arc_agi_3_official_agent.setup_official_harness import (
    DEFAULT_DEST,
    ensure_official_harness,
    load_manifest,
    venv_python,
)
from responses_api_agents.arc_agi_3_official_agent.worker import CALL_HEADERS


LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
WORKER = HERE / "worker.py"
API_KEY_ENV = "ARC3_OFFICIAL_API_KEY"
CONTINUOUS_CONVERSATION = "continuous_conversation"
MANUAL_ROLLING = "manual_rolling"
OPEN_SOURCE_ADAPTER = "open_source.chat_completions.v1"
HARNESS_TYPES = ("provider", "standard")
# What each harness type fixes in the upstream runtime config; the rest of the entry is the config's.
FIXED_RUNTIME = {
    "provider": {
        "sdk": "openai-python",
        "api": "chat_completions",
        "adapter_id": OPEN_SOURCE_ADAPTER,
        "state": CONTINUOUS_CONVERSATION,
    },
    "standard": {"sdk": "openai-python", "api": "chat_completions", "state": MANUAL_ROLLING},
}
REASONING_REPLAY_MODES = ("reasoning_content", "reasoning", "reasoning_aliases")
# Why a game ended (the record's and the rollout's ``stop_reason``): won, level_action_budget (the level the game was on
# used its own action allowance, upstream's ceil(baseline x MAX_ACTIONS_BASELINE_MULTIPLIER)), total_action_budget (the
# whole game's action cap: upstream's MAX_ACTIONS_HARD_CAP or its total MAX_ACTIONS; not a verdict on the last level),
# play_window (the play window or upstream's MAX_RUNTIME_SECONDS, capped at it), crashed (a harness error: a model call
# that failed, e.g. replies cut at the output cap, or the game server), cancelled (the agent server shut down),
# harness_end (upstream stopped for another reason). From upstream's ExitReason where it gives one; its ACTION_BUDGET is
# split by the worker's report of which budget fired (``action_budget``: "level" or "total"; total when unknown).
STOP_REASONS = {"GAME_WIN": "won", "TIME_BUDGET": "play_window"}
ACTION_BUDGET_STOP_REASONS = {"level": "level_action_budget", "total": "total_action_budget"}
# Environment the worker inherits; nothing else (no proxies for the local route, no provider keys, no harness caps).
WORKER_ENV_NAMES = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")
# How long the agent waits for the game server to close the scorecard of an attempt that aborted before verification.
CARD_CLOSE_TIMEOUT_S = 30.0


def repo_path(path: str) -> Path:
    """A config path; relative paths are relative to the Gym repository root (servers run in their own dir)."""
    return Path(path) if Path(path).is_absolute() else PARENT_DIR / path


def validate_benchmark_config(
    entry: dict[str, Any], model_name: str, harness_type: str = "provider"
) -> dict[str, Any]:
    """A complete upstream ``model_configs.yaml`` entry for this agent, or ``ValueError``.

    Fixed: the harness type's runtime (``FIXED_RUNTIME``: ``provider`` = the continuous conversation with the
    open-source Chat Completions adapter, the only upstream mode that resends the model's earlier reasoning;
    ``standard`` = ``manual_rolling``), non-streaming requests, and the client (this agent's route; set per game). The
    rest is the config's. Upstream's own loader validates the entry again in the worker.
    """
    if harness_type not in HARNESS_TYPES:
        raise ValueError(f"harness_type must be one of {HARNESS_TYPES}, not {harness_type!r}")
    entry = copy.deepcopy(dict(entry or {}))
    entry.setdefault("id", "gym-policy")
    runtime = entry.setdefault("runtime", {})
    for key, want in FIXED_RUNTIME[harness_type].items():
        have = runtime.setdefault(key, want)
        if have != want:
            raise ValueError(f"benchmark_config.runtime.{key} must be {want!r} for harness_type {harness_type}")
    agent = entry.setdefault("agent", {})
    if harness_type == "provider":
        replay = runtime.setdefault("reasoning_replay", "reasoning_content")
        if replay not in REASONING_REPLAY_MODES:
            raise ValueError(f"benchmark_config.runtime.reasoning_replay must be one of {REASONING_REPLAY_MODES}")
        compaction = runtime.get("compaction")
        if compaction is not None and (
            not isinstance(compaction, dict) or compaction.get("strategy") != "harness_summary"
        ):
            raise ValueError("benchmark_config.runtime.compaction must be null or {strategy: harness_summary, ...}")
        if agent.get("analysis_mode"):
            raise ValueError("benchmark_config.agent.analysis_mode applies to manual_rolling only; remove it")
    elif runtime.get("reasoning_replay") or runtime.get("compaction"):
        raise ValueError("harness_type standard (manual_rolling) has no reasoning_replay and no compaction")
    request = entry.setdefault("request", {})
    if request.get("stream"):
        raise ValueError("benchmark_config.request.stream must be false: the Gym model server is non-streaming")
    request["stream"] = False
    request.setdefault("model", model_name)
    client = entry.setdefault("client", {})
    if client.get("base_url"):
        raise ValueError("benchmark_config.client.base_url is set by the agent (its own route); remove it")
    if client.get("api_key_env") not in (None, API_KEY_ENV):
        raise ValueError(f"benchmark_config.client.api_key_env is set by the agent ({API_KEY_ENV}); remove it")
    client["api_key_env"] = API_KEY_ENV
    entry.setdefault("pricing", {})
    return entry


def check_model_server(
    model_config: dict[str, Any], name: str, require_context_errors: bool, harness_type: str = "provider"
) -> None:
    """Raise unless the Gym model server passes context-length errors and, for the provider harness (which resends
    earlier reasoning), keeps the reasoning of earlier turns."""
    if harness_type == "provider" and model_config.get("uses_reasoning_parser") is not True:
        raise RuntimeError(
            f"model server {name!r} must set uses_reasoning_parser: true: the official harness resends earlier "
            "reasoning, which the Gym vLLM model server passes to vLLM only with its reasoning parser on"
        )
    if harness_type == "provider" and model_config.get("uses_interleaved_reasoning", True) is not True:
        raise RuntimeError(
            f"model server {name!r} has uses_interleaved_reasoning: false, which drops the reasoning of earlier "
            "turns; the official harness keeps it (set it true)"
        )
    if require_context_errors and model_config.get("propagate_context_overflow_errors") is not True:
        raise RuntimeError(
            f"model server {name!r} must set propagate_context_overflow_errors: true so a context-length rejection "
            "reaches the harness (it compacts and retries); or set require_context_error_passthrough: false"
        )


class ArcAgi3OfficialAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # The ARC-AGI-3 environment files (<game>/<version>/...), the same directory as the resources server's (pins).
    environments_dir: str
    # Harness install (default: .arc3_official_harness next to this file), installed on first start from
    # harness_source: "git" (clone the pinned commit), a directory or a .tar.gz of the repository at that commit.
    harness_dir: str | None = None
    harness_source: str | None = "git"
    # Run outputs: <output_dir>/<run_name>/{run_config.json, games/<game>/...}.
    output_dir: str = "results/arc_agi_3_official"
    run_name: str | None = None
    # Play budget in seconds. run_clock shared: one run clock, as upstream's benchmark (every game's play window ends at
    # the first game's start + play_seconds; RHAE@T from that origin; benchmarking). per_game: each game has
    # play_seconds from its own start, whenever it starts (data generation, RL; RHAE@T from the game's start).
    play_seconds: float = 7200.0
    run_clock: Literal["shared", "per_game"] = "shared"
    # A worker still running this long after the window's end is stopped.
    window_grace_s: float = 30.0
    # Cancellation and shutdown: a worker gets this long to write its report and exit after SIGTERM.
    cancel_drain_s: float = 30.0
    # How long the arc_agi client waits for a game's RESET (the engine imports the game on its first RESET; the
    # client's own timeout is 10 s, which a slow game module outlasts).
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S
    # Write every model request and reply of the route to <run>/games/<game>/requests.jsonl.
    save_request_logs: bool = False
    # The policy model server must return the engine's context-length 400 (vllm_model
    # propagate_context_overflow_errors: true) so the harness can compact and retry; checked at start.
    require_context_error_passthrough: bool = True
    # The address at which the workers reach this server (default 127.0.0.1 and its port).
    route_host: str = "127.0.0.1"
    # Upstream's harness: "provider" (Provider Adapter: continuous conversation, earlier reasoning resent, summary
    # compaction) or "standard" (manual_rolling: oldest turns dropped, no reasoning resent, no summaries).
    harness_type: Literal["provider", "standard"] = "provider"
    # One upstream model_configs.yaml entry (agent, runtime, request, pricing) per harness type; the selected type's
    # entry is used. See validate_benchmark_config.
    benchmark_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # The thinking effort (e.g. low / high / max), sent as the chat-template kwarg reasoning_effort_kwarg on every
    # request that asks for thinking and sets no effort (an upstream request.reasoning_effort is sent as it is).
    # null = the chat template's default. reasoning_effort_kwarg is the kwarg the model's template reads
    # (reasoning_effort, or e.g. thinking_effort).
    reasoning_effort: str | None = None
    reasoning_effort_kwarg: str = "reasoning_effort"


@dataclass
class Attempt:
    """A game in play: its route key, model-server session cookies and model calls in flight, the model server's URL
    path for its calls (its rollout's path) and its call log."""

    key: str
    playing: bool = True
    cancelled: bool = False
    cookies: dict[str, str] = field(default_factory=dict)
    model_calls: set[asyncio.Task] = field(default_factory=set)
    log_path: Path | None = None
    url_path: str = "/v1/chat/completions"
    calls: GameCallLog = field(default_factory=GameCallLog)
    # The end of the game's play window (time.monotonic()).
    deadline: float = float("inf")

    def seconds_left(self) -> float:
        return self.deadline - time.monotonic()


def header_int(headers: Any, name: str) -> int | None:
    try:
        return int(headers.get(name))
    except (TypeError, ValueError):
        return None


@dataclass
class WorkerOutcome:
    # The worker's result.json: its report of the game.
    report: dict[str, Any] | None = None
    returncode: int | None = None
    window_stopped: bool = False


class ArcAgi3OfficialAgent(ArcAgi3GameSessions):
    config: ArcAgi3OfficialAgentConfig

    def model_post_init(self, context: Any) -> None:
        ensure_flask()
        if self.config.num_workers not in (None, 1):
            raise ValueError("the official agent keeps its games in one process; num_workers must be 1")
        self._benchmark_config = validate_benchmark_config(
            self.config.benchmark_configs.get(self.config.harness_type) or {},
            self.config.model_server.name,
            self.config.harness_type,
        )
        source = self.config.harness_source
        if source and source != "git":
            source = str(repo_path(source))
        self._install_root = ensure_official_harness(
            source, repo_path(self.config.harness_dir) if self.config.harness_dir else DEFAULT_DEST
        )
        self._manifest = load_manifest()
        self._run: OfficialRun | None = None
        self._run_lock = asyncio.Lock()
        self._attempts: dict[str, Attempt] = {}
        self._workers: set[asyncio.subprocess.Process] = set()
        self._game_server_url: str | None = None
        # This agent's key on the game server (its scorecards and games); the verifier gets it to close a card.
        self._game_api_key = f"arc3-official-{uuid4().hex}"
        self._closing = False
        # Games between their registration and their end (the scorecard handed to the verifier or closed); shutdown
        # waits for them, so no worker launch or open scorecard outlives it.
        self._in_play: set[asyncio.Future] = set()
        # Attempts started per route key (repeats of one row share its dispatch index).
        self._passes: collections.Counter = collections.Counter()
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/official/{game_key}/v1/chat/completions")(self.route_chat_completions)
        inner_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app_: FastAPI):
            async with inner_lifespan(app_) as state:
                yield state
            await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    # ---- configuration -------------------------------------------------------------------------------------------

    def run_dir(self) -> Path:
        name = self.config.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return repo_path(self.config.output_dir) / name

    def model_server_config(self) -> dict[str, Any]:
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
        return dict(config)

    def route_base_url(self, key: str) -> str:
        return f"http://{self.config.route_host}:{self.config.port}/official/{key}/v1"

    def harness_revision(self) -> dict[str, Any]:
        """The pinned harness's source (official_harness_manifest.json provenance)."""
        provenance = self._manifest["provenance"]
        return {"name": "arc-agi-3-benchmarking", **{k: provenance.get(k) for k in ("repository", "commit")}}

    def sampling(self) -> dict[str, Any]:
        """The upstream request settings of the selected harness type and the thinking effort the route adds (an
        upstream ``request.reasoning_effort`` or effort chat-template kwarg wins)."""
        request = {k: v for k, v in self._benchmark_config.get("request", {}).items() if k not in ("model", "stream")}
        return redact(
            {
                **request,
                "reasoning_effort": self.config.reasoning_effort,
                "reasoning_effort_kwarg": self.config.reasoning_effort_kwarg,
            }
        )

    def run_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        environments_dir = repo_path(self.config.environments_dir)
        games = {
            f"{meta.parent.parent.name}-{meta.parent.name}": {
                "version": meta.parent.name,
                "files_sha256": game_files_sha256(meta.parent),
            }
            for meta in sorted(environments_dir.glob("*/*/metadata.json"))
        }
        return redact(
            {
                "agent": self.config.model_dump(mode="json"),
                "benchmark_config": self._benchmark_config,
                "harness": {
                    "install_root": str(self._install_root),
                    "provenance": self._manifest["provenance"],
                },
                "game_server": self._game_server_url,
                "model_server": {"name": self.config.model_server.name, "config": model_config},
                "environments": {"dir": str(environments_dir), "games": games},
                "written_at": dt.datetime.now().isoformat(),
            }
        )

    async def _ensure_run(self) -> OfficialRun:
        async with self._run_lock:
            if self._closing:
                raise RuntimeError("the agent server is shutting down; no new games")
            if self._run is None:
                model_config = self.model_server_config()
                check_model_server(
                    model_config,
                    self.config.model_server.name,
                    self.config.require_context_error_passthrough,
                    self.config.harness_type,
                )
                response = await self.server_client.get(
                    server_name=self.config.resources_server.name, url_path="/game_server"
                )
                await raise_for_status(response)
                self._game_server_url = (await get_response_json(response))["url"]
                run = OfficialRun(
                    repo_path(self.config.environments_dir).resolve(),
                    self.run_dir().resolve(),
                    self.config.play_seconds,
                )
                run.run_dir.mkdir(parents=True, exist_ok=True)
                text = json.dumps(self.run_config(model_config), indent=1, sort_keys=True, default=str) + "\n"
                (run.run_dir / "run_config.json").write_text(text)
                run.run_config_sha256 = hashlib.sha256(text.encode()).hexdigest()
                self._run = run
            return self._run

    # ---- the model route -----------------------------------------------------------------------------------------

    async def route_chat_completions(self, game_key: str, request: Request) -> Response:
        """The harness's Chat Completions call for one game, sent to the Gym model server."""
        attempt = self._attempts.get(game_key)
        if attempt is None or not attempt.playing:
            return JSONResponse({"error": {"message": f"no game in play for {game_key!r}"}}, status_code=410)
        payload = await request.json()
        try:
            body = to_gym_chat_request(
                payload,
                reasoning_effort=self.config.reasoning_effort,
                reasoning_effort_kwarg=self.config.reasoning_effort_kwarg,
            )
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        headers = getattr(request, "headers", None) or {}
        call = dict(
            role=headers.get(CALL_HEADERS["role"]) or GAMEPLAY_ROLE,
            history_id=headers.get(CALL_HEADERS["history_id"]) or "main",
            messages=payload.get("messages") or [],
            started_at=time.time(),
            step=header_int(headers, CALL_HEADERS["step"]),
            level=header_int(headers, CALL_HEADERS["level"]),
        )
        task = asyncio.current_task()
        attempt.model_calls.add(task)
        try:
            response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=attempt.url_path,
                json=body,
                cookies=attempt.cookies or None,
            )
            text = (await response.read()).decode("utf-8", errors="replace")
            status = response.status
            attempt.cookies.update({k: m.value for k, m in response.cookies.items()})
        except BaseException as exc:  # no answer (connection error, cancelled): logged, then raised as before
            error = "cancelled" if isinstance(exc, asyncio.CancelledError) else f"{type(exc).__name__}: {exc}"
            attempt.calls.add(status=0, response=None, error=error, **call)
            raise
        finally:
            attempt.model_calls.discard(task)
        if status >= 400:
            attempt.calls.add(status=status, response=None, **call)
            self._log_request(attempt, payload, status, text)
            return Response(content=text, status_code=status, media_type="application/json")
        parsed = json.loads(text)
        attempt.calls.add(status=status, response=parsed, **call)
        reply = from_gym_chat_response(parsed)
        self._log_request(attempt, payload, status, reply)
        return JSONResponse(reply)

    def _log_request(self, attempt: Attempt, payload: dict[str, Any], status: int, reply: Any) -> None:
        if attempt.log_path is None:
            return
        entry = {"t": time.time(), "request": payload, "status": status, "reply": reply}
        with open(attempt.log_path, "a") as log:
            log.write(json.dumps(entry) + "\n")

    # ---- one game ------------------------------------------------------------------------------------------------

    async def _open_card(self, key: str) -> str:
        """A scorecard of this rollout on the official game server."""
        response = await http_request(
            "POST",
            f"{self._game_server_url}/api/scorecard/open",
            json={"tags": ["agent", "arc3-official"], "opaque": {"rollout": key}},
            headers={"X-API-Key": self._game_api_key},
        )
        await raise_for_status(response)
        return (await get_response_json(response))["card_id"]

    async def _close_card(self, card_id: str) -> None:
        """Close a scorecard of an attempt that ended before verification (cancelled, the worker failed to launch or
        sent no report), so the game server does not keep it until its idle timeout. Bounded; a failure is logged, not
        raised over the attempt's own error."""
        try:
            response = await asyncio.wait_for(
                http_request(
                    "POST",
                    f"{self._game_server_url}/api/scorecard/close",
                    json={"card_id": card_id},
                    headers={"X-API-Key": self._game_api_key},
                ),
                timeout=CARD_CLOSE_TIMEOUT_S,
            )
            if response.status != 404:  # 404: the card is already closed
                await raise_for_status(response)
        except Exception as exc:  # the attempt's own error is what the caller sees
            LOG.warning("could not close scorecard %s of an aborted attempt: %s", card_id, exc)

    async def _stop_worker(self, process: asyncio.subprocess.Process) -> None:
        """SIGTERM (the worker writes its report and exits), then SIGKILL after cancel_drain_s."""
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.config.cancel_drain_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()

    def _worker_settings(self, attempt: Attempt, run: OfficialRun, game_id: str, card_id: str) -> dict[str, Any]:
        entry = copy.deepcopy(self._benchmark_config)
        entry["client"]["base_url"] = self.route_base_url(attempt.key)
        return {
            "benchmark_config": entry,
            # No secrets: the route needs no key ("gym-local" satisfies the harness's client); the game key is this
            # agent's local key on the local game server.
            "model_api_key": "gym-local",
            "game_server_url": self._game_server_url,
            "game_api_key": self._game_api_key,
            "card_id": card_id,
            "game_id": game_id,
            "deadline_epoch": time.time() + max(0.0, attempt.seconds_left()),
            "reset_timeout_s": self.config.reset_timeout_s,
            "harness_src": str(self._install_root / "src"),
        }

    async def _play_attempt(
        self, run: OfficialRun, attempt: Attempt, game_dir: Path, game_id: str, card_id: str
    ) -> WorkerOutcome:
        settings_path = game_dir / "settings.json"
        settings_path.write_text(json.dumps(self._worker_settings(attempt, run, game_id, card_id)))
        env = {k: os.environ[k] for k in WORKER_ENV_NAMES if k in os.environ}
        env.update(
            PYTHONUNBUFFERED="1",
            ARC_HARNESS_COMMIT_SHA=self._manifest["provenance"]["commit"],
            NO_PROXY="127.0.0.1,localhost",
        )
        outcome = WorkerOutcome()
        with open(game_dir / "worker.log", "ab") as log:
            process = await asyncio.create_subprocess_exec(
                str(venv_python(self._install_root)),
                "-I",
                str(WORKER),
                "--settings",
                str(settings_path),
                cwd=str(game_dir),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        self._workers.add(process)
        try:
            if attempt.cancelled:  # shutdown began while the worker was launching: it ends at once, as cancelled
                await self._stop_worker(process)
            timeout = max(0.0, attempt.seconds_left()) + self.config.window_grace_s
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=timeout)
            except asyncio.TimeoutError:
                outcome.window_stopped = True
                LOG.warning(
                    "%s: worker still running %.0f s after the window's end; stopping it",
                    attempt.key,
                    self.config.window_grace_s,
                )
                await self._stop_worker(process)
        except BaseException:
            await asyncio.shield(self._stop_worker(process))
            raise
        finally:
            self._workers.discard(process)
            for task in list(attempt.model_calls):
                task.cancel()
        outcome.returncode = process.returncode
        result = game_dir / "result.json"
        outcome.report = json.loads(result.read_text()) if result.is_file() else None
        return outcome

    @staticmethod
    def _end_state(attempt: Attempt, outcome: WorkerOutcome) -> tuple:
        """(state, note, stop reason) of a finished attempt from the worker's report; ``STOP_REASONS``."""
        report = outcome.report or {}
        if report.get("state") == "WIN":
            return "won", None, "won"
        if attempt.cancelled:
            return "cancelled", "the agent server shut down", "cancelled"
        if report.get("error"):
            return "crashed", str(report["error"]), "crashed"
        if outcome.window_stopped:
            return "gave_up", "the worker was stopped after the play window's end", "play_window"
        exit_reason = report.get("exit_reason")
        if exit_reason == "ACTION_BUDGET":
            stop_reason = ACTION_BUDGET_STOP_REASONS.get(report.get("action_budget"), "total_action_budget")
        else:
            stop_reason = STOP_REASONS.get(exit_reason, "harness_end")
        return "gave_up", f"harness exit: {exit_reason}", stop_reason

    def _attempt_response(
        self, state: ArcAgi3Session, record: dict[str, Any] | None, report: dict[str, Any]
    ) -> NeMoGymResponse:
        summary = {
            "game_id": state.game.get("game_id"),
            "state": (record or {}).get("state", "not_started"),
            "levels_completed": (record or {}).get("levels_completed", 0),
            "actions": (record or {}).get("actions", 0),
            "run_dir": str(self._run.run_dir) if self._run else None,
            "harness": f"arc-agi-3-benchmarking@{self._manifest['provenance']['commit'][:8]}",
            "harness_type": self.config.harness_type,
        }
        params = state.activation
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time.time()),
            model=self.config.model_server.name,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=json.dumps(summary), annotations=[])],
                )
            ],
            tool_choice=params.tool_choice if params.tool_choice is not None else "auto",
            tools=params.tools or [],
            parallel_tool_calls=params.parallel_tool_calls if params.parallel_tool_calls is not None else True,
            **{ATTEMPT_REPORT_KEY: report},
        )

    async def play_game(
        self,
        verifier_metadata: dict[str, Any],
        index: int,
        url_path: str = "/v1/chat/completions",
        call_log: GameCallLog | None = None,
    ) -> dict[str, Any]:
        """Play one attempt, its model calls on the model server's ``url_path`` and recorded in ``call_log``; return
        the worker's report as a record, the scorecard id, the game's guid and the call log. Raises GameStartError when it cannot start, and
        RuntimeError when the worker died without a report (the scorecard then cannot be checked) or the server began
        shutting down before the worker started.

        The attempt owns its scorecard until it returns it for verification: on every earlier exit (cancellation, a
        worker that failed to launch or sent no report) its worker is stopped first and then the card is closed."""
        run = await self._ensure_run()
        full_game_id = run.check_game_pins(verifier_metadata)
        game_dir_files = run.environments_dir / str(verifier_metadata["game_id"]) / str(verifier_metadata["version"])
        baselines = [int(x) for x in json.loads((game_dir_files / "metadata.json").read_text())["baseline_actions"]]
        if self.config.run_clock == "shared":
            run.start_clock()
            seconds_left, clock_origin = run.seconds_left(), run.start_time_utc
        else:  # the game's own window, from now
            seconds_left = self.config.play_seconds
            clock_origin = dt.datetime.now(dt.timezone.utc).isoformat()
        deadline = time.monotonic() + seconds_left
        if seconds_left <= 0:
            raise GameStartError(f"{full_game_id}: the run's play window has ended (run_clock shared)")
        key = f"{index}-{full_game_id}"
        self._passes[key] += 1
        if self._passes[key] > 1:  # a repeat of the row: its own route, game directory and scorecard
            key = f"{key}-p{self._passes[key] - 1}"
        # Registered before the first suspension, so a shutdown from here on cancels it and waits for its end.
        attempt = Attempt(key, url_path=url_path, deadline=deadline, calls=call_log or GameCallLog())
        self._attempts[key] = attempt
        in_play = asyncio.get_running_loop().create_future()
        self._in_play.add(in_play)
        card_id: str | None = None
        verified = False  # the card goes to the verifier
        try:
            try:
                card_id = await self._open_card(key)
            except Exception as exc:
                raise GameStartError(f"{full_game_id}: the game server did not open a scorecard: {exc}") from exc
            if attempt.cancelled:
                raise RuntimeError("the agent server is shutting down; no new games")
            game_dir = run.run_dir / "games" / key
            game_dir.mkdir(parents=True, exist_ok=True)
            attempt.log_path = (game_dir / "requests.jsonl") if self.config.save_request_logs else None
            started_at = dt.datetime.now()
            outcome = await self._play_attempt(run, attempt, game_dir, full_game_id, card_id)
            report = outcome.report
            if not report or not report.get("guid"):
                tail = ""
                with contextlib.suppress(OSError):
                    tail = (game_dir / "worker.log").read_text(errors="replace")[-1500:]
                raise RuntimeError(
                    f"{key}: the worker exited ({outcome.returncode}) without a game report; scorecard {card_id} "
                    f"cannot be checked: {(report or {}).get('error') or tail}"
                )
            state, note, stop_reason = self._end_state(attempt, outcome)
            verified = True
        finally:
            attempt.playing = False
            self._attempts.pop(key, None)
            if card_id is not None and not verified:
                close = asyncio.ensure_future(self._close_card(card_id))
                self._in_play.add(close)  # shutdown also waits for the close
                close.add_done_callback(self._in_play.discard)
            self._in_play.discard(in_play)
            in_play.set_result(None)
            if card_id is not None and not verified:
                await asyncio.shield(close)
        record = {
            "game_id": full_game_id,
            "number_of_levels": len(baselines),
            "base_actions_per_level": baselines,
            "state": state,
            "levels_completed": int(report["levels_completed"]),
            "actions": int(report["actions"]),
            "history": [],
            "started_at": started_at.isoformat(),
            "solver_note": note,
            "harness_exit": report.get("exit_reason"),
            "stop_reason": stop_reason,
        }
        return {
            "record": record,
            "scorecard_id": card_id,
            "guid": report["guid"],
            "solver_note": note if state == "crashed" else None,
            "key": key,
            "call_log": attempt.calls,
            "clock_origin": clock_origin,
        }

    async def _play_session(self, state: ArcAgi3Session, url_path: str) -> NeMoGymResponse:
        game = state.game
        record, agent_error, solver_note, scorecard_id, guid, clock_origin = None, None, None, None, None, None
        index = int(game.get("dispatch_index", 0))
        invocation_id, call_log = f"{index}-{game.get('full_game_id') or game.get('game_id')}", GameCallLog()
        model_error, ended = None, False
        try:
            try:
                result = await self.play_game(game, index, url_path=url_path, call_log=call_log)
                record, solver_note = result["record"], result["solver_note"]
                scorecard_id, guid = result["scorecard_id"], result["guid"]
                invocation_id, clock_origin = result["key"], result["clock_origin"]
            except GameStartError as exc:
                LOG.error("ARC-AGI-3 game %s could not start: %s", game.get("game_id"), exc)
                agent_error = f"GameStartError: {exc}"
            ended = True
        finally:  # also when the close stops the game: its observations go with the close
            model_error = model_unavailable(call_log)
            failed = not ended or agent_error or model_error
            state.observations = observations(
                call_log,
                invocation_id=invocation_id,
                model_ref=ModelServerRef(type="responses_api_models", name=self.config.model_server.name),
                status="failed" if failed else ("completed" if record is not None else "unknown"),
                stop_reason=(record or {}).get("state"),
                source="arc_agi_3_official_agent",
            )
        run = self._run
        report = {
            "game_record": record,
            "scorecard_id": scorecard_id,
            "game_guid": guid,
            "scorecard_api_key": self._game_api_key,
            "agent_error": agent_error,
            "model_error": model_error,
            "solver_note": solver_note,
            "stop_reason": (record or {}).get("stop_reason"),
            # RHAE@T's origin: the run clock's (shared) or the game's own start (per_game)
            "benchmark_start_time": clock_origin,
            "run_dir": str(run.run_dir) if run else None,
            "run_config_sha256": run.run_config_sha256 if run else None,
            "harness": "official",
            "harness_type": self.config.harness_type,
            "harness_revision": self.harness_revision(),
            "sampling": self.sampling(),
            "agent_call_log": call_log_entries(call_log),
        }
        return self._attempt_response(state, record, report)

    async def shutdown(self) -> None:
        """Stop every worker (each writes its report; games in play end as cancelled), then wait until every game has
        ended: a game still opening its scorecard is refused (and its card closed), a worker still launching is
        stopped as soon as it exists, and the cards of aborted attempts are closed."""
        self._closing = True
        if self._run is not None:
            self._run.closed = True
        for attempt in list(self._attempts.values()):
            attempt.cancelled = True
        await asyncio.gather(*(self._stop_worker(p) for p in list(self._workers)), return_exceptions=True)
        while self._in_play:
            await asyncio.wait(list(self._in_play))


if __name__ == "__main__":
    ArcAgi3OfficialAgent.run_webserver()
