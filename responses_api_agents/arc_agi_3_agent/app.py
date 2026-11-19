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
"""ARC-AGI-3 agent: plays one game per agent session with the ARC-AGI-3 Kaggle kit's harness.

The harness is the pinned, unmodified harness of the Kaggle kit's public datasets (see ``harness_manifest.json`` and
``setup_harness.py``). Its model calls go to the Gym model server (``gym_transport.py``), on the activation's rollout
path (``url_path_for_request``), so token capture applies to every call.

An environment server drives each game (``ArcAgi3GameSessions``): the seed binds the session to the row's pinned game
(its ``agent_context``), the one activation plays that game for up to ``play_seconds``, and the close stops a game still
in play and returns its model-call observations. The activation's reply carries the agent's report of the game
(``ATTEMPT_REPORT_KEY``: the taaf game record and its scorecard), which the ``arc_agi_3`` resources server verifies
(RHAE) and keeps in the rollout with ``agent_call_log``: every model call sent, with its role, history id, the game's
step and level when it was sent, and its error if it failed; ``harness_log``: the warnings the harness logged while it
played the game (e.g. a summary it rejected), with the game's level then; ``harness_revision`` (the pinned harness's
Kaggle datasets and source commit) and ``sampling`` (the effective sampling and thinking effort). The close's
observations (``call_log.py``) list the game's accepted gameplay calls in order and one compaction record per history
drain.

All harness settings are process-wide (the harness reads them once from the environment), so one agent server is one
hyperparameter configuration. Run the rows of one evaluation concurrently (concurrency >= rows) so that every game
gets the same play window, as in the kit.
"""

import asyncio
import datetime as dt
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.agents import (
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionConflictError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status, request
from nemo_gym.config import (
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    ResourcesServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.trajectory import AgentObservationBundle
from resources_servers.arc_agi_3.app import ATTEMPT_REPORT_KEY
from resources_servers.arc_agi_3.setup_arc_agi import ensure_flask
from responses_api_agents.arc_agi_3_agent.call_log import (
    GameCallLog,
    call_log_entries,
    harness_log_entries,
    model_unavailable,
    observations,
)
from responses_api_agents.arc_agi_3_agent.gym_transport import GymChatTransport, RequestsProxy
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    DEFAULT_RESET_TIMEOUT_S,
    HYPERPARAMETER_ENV,
    GameStartError,
    HarnessRuntime,
    compose_harness_env,
    game_files_sha256,
    load_harness_env_toml,
)
from responses_api_agents.arc_agi_3_agent.setup_harness import DEFAULT_DEST, ensure_harness, load_manifest


LOG = logging.getLogger(__name__)
# The rows' "harness" value (the official option's rows say "official").
HARNESS_NAME = "arc3-kaggle-kit"


def repo_path(path: str) -> Path:
    """A config path; relative paths are relative to the Gym repository root (servers run in their own dir)."""
    return Path(path) if Path(path).is_absolute() else PARENT_DIR / path


# Whole keys that hold credentials (``api_key``, ``hf_token``, ``Authorization``, ...), not settings that merely
# mention tokens (``max_tokens``, ``thinking_token_budget``, ``context_keep_tokens``).
SECRET_KEY_PATTERN = re.compile(
    r"^(?:.*[_-])?(?:api_?key|access_?token|auth_?token|refresh_?token|token|secret(?:_?key)?|password|passwd|"
    r"authorization|cookies?|credentials?|bearer)$",
    re.I,
)


def env_value(value: str | None) -> Any:
    """A harness environment value as the JSON value it spells (``"0.7"`` -> 0.7, ``"true"`` -> True), else as is."""
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return value


def redact(value: Any) -> Any:
    """A JSON-able copy without credentials: secret-looking keys, URL user info and secret query parameters."""
    if isinstance(value, dict):
        return {
            str(k): ("REDACTED" if SECRET_KEY_PATTERN.search(str(k)) and v not in (None, "") else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str) and "://" in value:
        value = re.sub(r"://[^/@\s]*@", "://REDACTED@", value)
        return re.sub(
            r"([?&](?:[^=&]*[_-])?(?:key|token|secret|sig|signature)=)[^&]*", r"\1REDACTED", value, flags=re.I
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# Row-level generation controls (the CLI's --temperature / --top-p / --max-output-tokens) that the harnesses do not
# apply per attempt: their sampling is process-wide agent configuration.
UNSUPPORTED_ROW_CONTROLS = ("temperature", "top_p", "max_output_tokens")


def reject_row_controls(params: NeMoGymResponseCreateParamsNonStreaming) -> None:
    """Raise a 422 when the row sets a generation control the harness would ignore (a different experiment)."""
    unsupported = [name for name in UNSUPPORTED_ROW_CONTROLS if getattr(params, name, None) is not None]
    if unsupported:
        raise HTTPException(
            422,
            f"Row-level controls {unsupported} are unsupported by the ARC-AGI-3 harnesses; configure the agent "
            "server instead (its sampling settings are process-wide)",
        )


class ArcAgi3AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # Harness install: a verified directory (default: .arc3_harness next to this file), installed on first start
    # from harness_source (directory, .tar.gz, or kaggle:<owner>/<prefix>).
    harness_dir: str | None = None
    harness_source: str | None = None
    # The ARC-AGI-3 environment files (<game>/<version>/...), the same directory as the resources server's.
    environments_dir: str
    # Run outputs: <output_dir>/<run_name>/{benchmark.json, transcripts/, artifacts/, telemetry/, harness_env.json}.
    output_dir: str = "results/arc_agi_3"
    run_name: str | None = None

    # Play budget per game (seconds from the game's start; the kit's public25 mode plays 7200 s).
    play_seconds: float = 7200.0
    # shared: one run clock, as the kit (every game's play ends at the first game's start + play_seconds; RHAE@T from
    # that origin). per_game: each game has play_seconds from its own start (RL rollouts, sent whenever the trainer
    # sends them; RHAE@T from the game's start).
    run_clock: Literal["shared", "per_game"] = "shared"
    # Action budget per game (None: no limit, as the kit).
    max_actions_per_game: int | None = None
    # How long the arc_agi client waits for a game's RESET (the engine imports the game on its first RESET; the
    # client's own timeout is 10 s, which a slow game module outlasts).
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S
    # Games played at once (harness worker threads) and attempts sharing the sandbox memory pool.
    concurrency: int = 25
    num_attempts: int = 25

    # Hyperparameters. None keeps the deployed harness_env.toml value (the kit's):
    # temperature 0.7, top_p 0.95, top_k 20, thinking on, max output 12288, context window 131072, keep 24576.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # Seed of the harness's Python ``random``. Seed the engine with the model server's own seed (vLLM --seed).
    seed: int | None = None
    # Sampling seed sent with EVERY request (LOCAL_ANALYZER_SEED). Off as deployed: the harness re-sends a rolled-back
    # turn's request after a failure, and a fixed per-request seed returns the same reply to it.
    request_seed: int | None = None
    enable_thinking: bool | None = None
    # Output cap per request. A reply cut off inside its reasoning is closed and continued to a tool call by the
    # harness (ARC3_FORCE_ACT), so this is also the reasoning-token cap.
    max_output_tokens: int | None = None
    context_window_tokens: int | None = None
    context_keep_tokens: int | None = None
    # Any other harness switch (name -> value), applied last.
    harness_env: dict[str, Any] = Field(default_factory=dict)
    # Extra chat-template kwargs sent with every request (the harness's own, e.g. enable_thinking, win). Model-specific
    # defaults belong in the model server config (e.g. Nemotron's truncate_history_thinking: false).
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Extra vLLM request fields sent with every request.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # The thinking effort (e.g. low / high / max), sent as the chat-template kwarg reasoning_effort_kwarg on every
    # thinking request that sets none of its own (the harness's effort ladder, ARC3_REASONING_EFFORT_LADDER, wins).
    # null = the chat template's default. reasoning_effort_kwarg is the kwarg the model's template reads
    # (reasoning_effort, or e.g. thinking_effort).
    reasoning_effort: str | None = None
    reasoning_effort_kwarg: str = "reasoning_effort"

    # The policy model server must return the engine's context-length 400 (vllm_model
    # propagate_context_overflow_errors: true) so the harness can trim its history and retry; checked at start.
    # Set False only for a model server without that option, accepting that a rejected request becomes an empty turn.
    require_context_error_passthrough: bool = True

    save_request_logs: bool = False
    # Kit step 23 installs: bounded harness RAM (needs glibc), gate telemetry events.
    mem_lean: bool = True
    telemetry: bool = True
    # benchmark.json is rewritten this often during play (the kit's lean diagnostics) and at every game end.
    save_every_s: float = 600.0


class ArcAgi3AgentContext(BaseModel):
    """An ARC-AGI-3 agent session's context: the row's pinned game (``create_dataset.py`` ``AGENT_GAME_KEYS``:
    game_id, full_game_id, version, files_sha256, dispatch_index), checked against the files before it is played."""

    model_config = ConfigDict(extra="forbid")

    game: dict[str, Any]


@dataclass
class ArcAgi3Session(AgentSessionState):
    """One game: its pins, its one activation (the whole game), and the observations the activation leaves for the
    close."""

    game: dict[str, Any]
    activation: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: "asyncio.Task[NeMoGymResponse] | None" = None
    observations: AgentObservationBundle | None = None


class ArcAgi3GameSessions(SimpleResponsesAPIAgent):
    """Agent sessions that each play one ARC-AGI-3 game (both ARC-AGI-3 agents).

    The seed binds the row's pinned game (``agent_context``). The first activation starts the game; a retry of the same
    activation waits for and returns the same game, and a seed repeated for the session starts nothing. The close
    stops a game still in play (its harness stops and its scorecard is released) and returns its observations.
    """

    agent_context_model = ArcAgi3AgentContext

    async def _play_session(self, state: ArcAgi3Session, url_path: str) -> NeMoGymResponse:
        """Play the session's game, its model calls on the model server's ``url_path``; the reply carries the report
        (``ATTEMPT_REPORT_KEY``) and the play sets ``state.observations``."""
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> ArcAgi3Session:
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("An ARC-AGI-3 game is played on the game server; no sandbox is used")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise AgentSeedRejectedError("The ARC-AGI-3 harnesses use no external tools")
        # The base refuses a seed whose context lacks the game (422).
        return ArcAgi3Session(request=body.model_copy(deep=True), game=dict(self.parse_agent_context(body).game))

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """The session's one activation: play its game."""
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise AgentSessionConflictError("An ARC-AGI-3 game is played in a seeded agent session")
        state = self._require_agent_session(session_id)
        rollout_id = request.path_params.get("rollout_id")
        if rollout_id is not None and rollout_id != state.request.episode_id.capture_key:
            raise AgentSessionConflictError("The activation's rollout differs from the seeded session's")
        if state.task is None:
            reject_row_controls(body)
            state.activation = body.model_copy(deep=True)
            state.task = asyncio.ensure_future(
                self._play_session(state, self.url_path_for_request("/v1/chat/completions", request))
            )
        elif state.activation != body:
            raise AgentSessionConflictError("An ARC-AGI-3 session plays one game; retry the same activation")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, ArcAgi3Session)
        if state.task is not None:
            if not state.task.done():
                state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=state.observations,
            cleanup_confirmed=True,
        )


class ArcAgi3Agent(ArcAgi3GameSessions):
    config: ArcAgi3AgentConfig

    def model_post_init(self, context: Any) -> None:
        ensure_flask()
        if self.config.num_workers not in (None, 1):
            raise ValueError("the agent keeps its harness runtime and games in one process; num_workers must be 1")
        source = self.config.harness_source
        if source and not source.startswith("kaggle:"):
            source = str(repo_path(source))
        self._install_root = ensure_harness(
            source, repo_path(self.config.harness_dir) if self.config.harness_dir else DEFAULT_DEST
        )
        self._runtime: HarnessRuntime | None = None
        self._transport: GymChatTransport | None = None
        self._runtime_lock = asyncio.Lock()
        # The runtime's start, owned by the agent (not by the request that triggered it), and the shutdown flag.
        self._runtime_start: asyncio.Future | None = None
        self._closing = False
        # Each game's model-call URL path (its run's rollout path), by transport game key.
        self._game_url_paths: dict[str, str] = {}
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        inner_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app_: FastAPI):
            async with inner_lifespan(app_) as state:
                yield state
            await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        """Stop and finalize every game (records end as cancelled), last snapshot, close the solver. A runtime that is
        still starting is waited for first (its start runs in a thread and cannot be interrupted), then closed."""
        self._closing = True
        start = self._runtime_start
        if start is not None and not start.done():
            await asyncio.wait([start])
        if self._runtime is not None and self._transport is not None:
            await self._runtime.shutdown(self._transport)

    def hyperparameters(self) -> dict[str, Any]:
        c = self.config
        return {
            "temperature": c.temperature,
            "top_p": c.top_p,
            "top_k": c.top_k,
            "request_seed": c.request_seed,
            "enable_thinking": c.enable_thinking,
            "max_output_tokens": c.max_output_tokens,
            "context_window_tokens": c.context_window_tokens,
            "context_keep_tokens": c.context_keep_tokens,
        }

    def run_dir(self) -> Path:
        name = self.config.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return repo_path(self.config.output_dir) / name

    @staticmethod
    def harness_revision() -> dict[str, Any]:
        """The pinned harness's source (harness_manifest.json provenance)."""
        provenance = load_manifest()["provenance"]
        keys = ("kaggle_harness_dataset", "kaggle_config_dataset", "source_commit")
        return {"name": "arc3-kaggle-kit", **{k: provenance.get(k) for k in keys}}

    def sampling(self, env: dict[str, str]) -> dict[str, Any]:
        """The effective sampling of the harness's requests (its environment, after the config's overrides) and the
        thinking effort the transport adds (a request's own effort, e.g. the harness's effort ladder, wins)."""
        out = {field: env_value(env.get(name)) for field, name in HYPERPARAMETER_ENV.items()}
        out.update(
            reasoning_effort=self.config.reasoning_effort,
            reasoning_effort_kwarg=self.config.reasoning_effort_kwarg,
            reasoning_effort_ladder=env.get("ARC3_REASONING_EFFORT_LADDER"),
            chat_template_kwargs=self.config.chat_template_kwargs,
            extra_body=self.config.extra_body,
        )
        return out

    async def _post_model(
        self, body: dict[str, Any], cookies: dict[str, str] | None, game_key: str | None = None
    ) -> tuple[int, str, dict]:
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self._game_url_paths.get(game_key or "", "/v1/chat/completions"),
            json=body,
            cookies=cookies,
        )
        text = (await response.read()).decode("utf-8", errors="replace")
        return response.status, text, {k: m.value for k, m in response.cookies.items()}

    def model_server_config(self) -> dict[str, Any]:
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
        return dict(config)

    async def observed_engine(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """What the engine behind the model server reports (``/v1/models``: served name, max_model_len)."""
        urls = model_config.get("base_url")
        url = (urls[0] if isinstance(urls, (list, tuple)) else urls) or ""
        if not url:
            return {"error": "the model server config has no base_url"}
        try:
            response = await request("GET", url.rstrip("/") + "/models")
            models = (await get_response_json(response)).get("data") or []
            return {"models": [{k: m.get(k) for k in ("id", "root", "max_model_len")} for m in models]}
        except Exception as exc:  # recorded, not fatal: the run config says the engine could not be read
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def run_config(self, env: dict[str, str], model_config: dict[str, Any]) -> dict[str, Any]:
        manifest = load_manifest()
        environments_dir = repo_path(self.config.environments_dir)
        games = {}
        for meta_path in sorted(environments_dir.glob("*/*/metadata.json")):
            games[f"{meta_path.parent.parent.name}-{meta_path.parent.name}"] = {
                "version": meta_path.parent.name,
                "files_sha256": game_files_sha256(meta_path.parent),
            }
        return redact(
            {
                "agent": self.config.model_dump(mode="json"),
                "seed_semantics": "seed = harness Python random; request_seed = LOCAL_ANALYZER_SEED on every request "
                "(None = unset); engine seed = the model server's own (vllm --seed)",
                "transport": {
                    "chat_template_kwargs": self.config.chat_template_kwargs,
                    "extra_body": self.config.extra_body,
                    "reasoning_effort": self.config.reasoning_effort,
                    "reasoning_effort_kwarg": self.config.reasoning_effort_kwarg,
                },
                "harness_env": env,
                "harness": {"install_root": str(self._install_root), "provenance": manifest["provenance"]},
                "model_server": {"name": self.config.model_server.name, "config": model_config},
                "engine_observed": await self.observed_engine(model_config),
                "environments": {"dir": str(environments_dir), "games": games},
                "written_at": dt.datetime.now().isoformat(),
            }
        )

    async def _ensure_runtime(self) -> HarnessRuntime:
        """The server's one runtime. Its start is a task of the agent: a request cancelled during the start leaves it
        running, the next request waits for the same start, and shutdown waits for it and closes the runtime."""
        async with self._runtime_lock:
            if self._closing:
                raise RuntimeError("the agent server is shutting down; no new games")
            if self._runtime is not None:
                return self._runtime
            start = self._runtime_start
            if start is None or start.done():  # the first request, or the previous start failed
                start = self._runtime_start = asyncio.ensure_future(self._start_runtime())
        return await asyncio.shield(start)

    async def _start_runtime(self) -> HarnessRuntime:
        model_config = self.model_server_config()
        if (
            self.config.require_context_error_passthrough
            and model_config.get("propagate_context_overflow_errors") is not True
        ):
            raise RuntimeError(
                f"model server {self.config.model_server.name!r} must set propagate_context_overflow_errors: true "
                "(vllm_model) so a context-length rejection reaches the harness's trim-and-retry; or set "
                "require_context_error_passthrough: false on the agent"
            )
        loop = asyncio.get_running_loop()
        transport = GymChatTransport(
            loop,
            self._post_model,
            chat_template_kwargs=self.config.chat_template_kwargs,
            extra_body=self.config.extra_body,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_effort_kwarg=self.config.reasoning_effort_kwarg,
        )
        base_url = self.resolve_model_base_url(self.config.model_server.name)
        game_server = await self.server_client.get(
            server_name=self.config.resources_server.name, url_path="/game_server"
        )
        await raise_for_status(game_server)
        game_server_url = (await get_response_json(game_server))["url"]
        env = compose_harness_env(
            load_harness_env_toml(self._install_root),
            base_url=base_url,
            model_id=self.config.model_server.name,
            hyperparameters=self.hyperparameters(),
            overrides=self.config.harness_env,
        )
        runtime = HarnessRuntime(
            install_root=self._install_root,
            environments_dir=repo_path(self.config.environments_dir).resolve(),
            run_dir=self.run_dir().resolve(),
            env=env,
            play_seconds=self.config.play_seconds,
            concurrency=self.config.concurrency,
            num_attempts=self.config.num_attempts,
            save_request_logs=self.config.save_request_logs,
            mem_lean=self.config.mem_lean,
            telemetry=self.config.telemetry,
            seed=self.config.seed,
            save_every_s=self.config.save_every_s,
            game_server_url=game_server_url,
            api_key=f"arc3-{uuid4().hex}",
            shared_clock=self.config.run_clock == "shared",
            max_actions_per_game=self.config.max_actions_per_game,
            reset_timeout_s=self.config.reset_timeout_s,
        )
        run_config = await self.run_config(env, model_config) | {"game_server": game_server_url}
        try:
            await asyncio.to_thread(runtime.start, RequestsProxy(transport), run_config)
        except Exception:
            await asyncio.to_thread(runtime.abort_start)  # tear down what the failed start already built
            raise
        LOG.info("ARC-AGI-3 harness started: run dir %s", runtime.run_dir)
        self._transport, self._runtime = transport, runtime
        return runtime

    def _attempt_response(
        self, state: ArcAgi3Session, record: dict[str, Any] | None, report: dict[str, Any]
    ) -> NeMoGymResponse:
        """A one-message Responses API record of the attempt (the full transcript is in the run directory), with the
        agent's report of it for the verifier."""
        summary = {
            "game_id": state.game.get("game_id"),
            "state": (record or {}).get("state", "not_started"),
            "levels_completed": (record or {}).get("levels_completed", 0),
            "actions": len((record or {}).get("history") or []),
            "run_dir": str(self._runtime.run_dir) if self._runtime else None,
        }
        params = state.activation
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
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

    async def _play_session(self, state: ArcAgi3Session, url_path: str) -> NeMoGymResponse:
        runtime = await self._ensure_runtime()
        game = state.game
        record, agent_error, persistence_error, solver_note = None, None, None, None
        scorecard_id, guid, started_at, model_error = None, None, None, None
        index = int(game.get("dispatch_index", 0))
        game_key = f"{index}:{game.get('full_game_id') or game.get('game_id')}:{uuid4().hex[:12]}"
        self._game_url_paths[game_key] = url_path
        ended = False
        try:
            try:
                attempt = await runtime.play_game(game, index, self._transport, game_key=game_key)
                record, agent_error = attempt.record, attempt.play_error
                persistence_error, solver_note = attempt.persistence_error, attempt.solver_note
                scorecard_id, guid, started_at = attempt.scorecard_id, attempt.guid, attempt.started_at_utc
            except GameStartError as exc:  # never started: no record, scored as not started
                LOG.error("ARC-AGI-3 game %s could not start: %s", game.get("game_id"), exc)
                agent_error = f"GameStartError: {exc}"
            ended = True
        finally:  # also when the close stops the game: its observations go with the close
            self._game_url_paths.pop(game_key, None)
            call_log = self._transport.take_call_log(game_key) if self._transport is not None else GameCallLog()
            model_error = model_unavailable(call_log)
            failed = not ended or agent_error or model_error
            state.observations = observations(
                call_log,
                invocation_id=game_key,
                model_ref=ModelServerRef(type="responses_api_models", name=self.config.model_server.name),
                status="failed" if failed else ("completed" if record is not None else "unknown"),
                stop_reason=(record or {}).get("state"),
            )
        report = {
            "game_record": record,
            "scorecard_id": scorecard_id,
            "game_guid": guid,
            "scorecard_api_key": runtime.api_key,
            "agent_error": agent_error,
            "model_error": model_error,
            "persistence_error": persistence_error,
            "solver_note": solver_note,
            "benchmark_start_time": runtime.start_time_utc if self.config.run_clock == "shared" else started_at,
            "run_dir": str(runtime.run_dir),
            "run_config_sha256": runtime.run_config_sha256,
            "harness": HARNESS_NAME,
            "harness_revision": self.harness_revision(),
            "sampling": redact(self.sampling(getattr(runtime, "env", None) or {})),
            "agent_call_log": call_log_entries(call_log),
            "harness_log": harness_log_entries(call_log),
        }
        return self._attempt_response(state, record, report)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """The resources server's RHAE metrics (overall, at 1 h / 2 h, counts of invalid results)."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/aggregate_metrics", json=body
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    ArcAgi3Agent.run_webserver()
