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
"""Runs the pinned ARC-AGI-3 harness inside the Gym agent, as the Kaggle kit runs it on Kaggle.

What the kit's notebook steps do, and where it happens here:

- step 04 (environment): :func:`compose_harness_env` exports ``harness_env.toml`` ([notebook], [setup_env] and, with
  ``use_priority_scheduling``, [priority]), then the engine adaptation (:data:`GYM_ENGINE_ENV`), then the agent
  config's hyperparameters and ``harness_env`` overrides;
- step 11 (sources on ``sys.path``): :func:`harness_sys_path`;
- steps 18/20 (benchmark and solver settings): :meth:`HarnessRuntime.start` builds ``HarnessSolver`` with the
  deployed settings (analyzer timeout 900 s, request logs, drain 60 s) and the play budget;
- step 23 (game list, installs): offline ``taaf.game_api.GameAPI`` games and the ``fast_state`` installs;
- step 24 (play): :meth:`HarnessRuntime.play_game` runs ``HarnessSolver._play_one`` for one game in the solver's
  worker pool, and ``benchmark.json`` is saved every ``save_every_s`` and at each game end.

The harness reads its settings from the environment once, at import, so one agent process is one configuration.
"""

import asyncio
import collections
import contextlib
import datetime as dt
import functools
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger(__name__)

# Engine-specific switches of the deployed harness_env.toml that have no meaning on the Gym model server:
# - ARC3_GDN_OWNERSHIP / ARC3_GDN_CPU_RETENTION: request metadata and side calls of the kit's SGLang engine's GDN
#   state protocol;
# - ARC3_WARMUP_ACTION_GAMES: a Kaggle idle-timeout keepalive (one RESET while the engine loads), which would add one
#   action to one game.
GYM_ENGINE_ENV: dict[str, str] = {
    "ARC3_GDN_OWNERSHIP": "0",
    "ARC3_GDN_CPU_RETENTION": "0",
    "ARC3_WARMUP_ACTION_GAMES": "0",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
}

# Agent config field -> harness environment variable.
HYPERPARAMETER_ENV: dict[str, str] = {
    "temperature": "LOCAL_ANALYZER_TEMPERATURE",
    "top_p": "LOCAL_ANALYZER_TOP_P",
    "top_k": "LOCAL_ANALYZER_TOP_K",
    "request_seed": "LOCAL_ANALYZER_SEED",
    "enable_thinking": "LOCAL_ANALYZER_ENABLE_THINKING",
    "max_output_tokens": "LOCAL_ANALYZER_MAX_OUTPUT",
    "context_window_tokens": "LOCAL_ANALYZER_CONTEXT_WINDOW",
    "context_keep_tokens": "ARC3_CONTEXT_KEEP_TOKENS",
}

# Harness solver settings of the deployed kit (pickled benchmark + steps 20/23/24, public25 mode).
DEPLOYED_ANALYZER_TIMEOUT_S = 900.0
DEPLOYED_CANCEL_DRAIN_S = 60.0
# Step 04 of a non-submission (public) run.
KIT_PROCESS_ENV: dict[str, str] = {
    "MPLBACKEND": "Agg",
    "TAAF_RUN_AS_SUBMISSION": "0",
    "TAAF_MINIMAL_DIAGNOSTICS": "0",
    "ONLY_RESET_LEVELS": "true",
}

# Variables the harness reads (its settings). An inherited value that the agent config does not set would change the
# run without a trace (e.g. a shell's LOCAL_ANALYZER_SEED), so :meth:`HarnessRuntime.export_env` removes it.
HARNESS_ENV_PREFIXES = ("ARC3_", "LOCAL_ANALYZER_", "TAAF_", "MULTIMODAL_", "EXPOSE_")
HARNESS_ENV_NAMES = frozenset(
    {"ONLY_RESET_LEVELS", "INFERENCE_ANALYZER_MODEL", "OPENAI_PROVIDER", "CHAT_PROVIDER", "GSQ_NVFP4_F16"}
)
# Gym's own settings with a harness-like name (tests and launch scripts), not read by the harness.
GYM_OWN_ENV_NAMES = frozenset({"ARC3_ENVIRONMENTS_DIR", "ARC3_HARNESS_DIR"})
GYM_OWN_ENV_PREFIXES = ("ARC3_SMOKE_",)


def inherited_harness_env(environ: dict[str, str], configured: dict[str, str]) -> list[str]:
    """Names of harness settings present in ``environ`` that the configuration does not set."""
    return sorted(
        name
        for name in environ
        if name not in configured
        and (name in HARNESS_ENV_NAMES or name.startswith(HARNESS_ENV_PREFIXES))
        and name not in GYM_OWN_ENV_NAMES
        and not name.startswith(GYM_OWN_ENV_PREFIXES)
    )


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def compose_harness_env(
    harness_env_toml: dict[str, Any],
    *,
    base_url: str,
    model_id: str,
    hyperparameters: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, str]:
    """The environment the harness runs under, in the kit's order; later entries win."""
    env: dict[str, str] = dict(KIT_PROCESS_ENV)
    env.update({k: _env_value(v) for k, v in harness_env_toml["notebook"].items()})
    env.update(
        {
            "LOCAL_ANALYZER_BASE_URL": base_url,
            "OPENAI_BASE_URL": base_url,
            "LOCAL_ANALYZER_MODEL_ID": model_id,
            "INFERENCE_ANALYZER_MODEL": model_id,
        }
    )
    env.update({k: _env_value(v) for k, v in harness_env_toml["setup_env"].items()})
    if harness_env_toml.get("use_priority_scheduling"):
        env.update({k: _env_value(v) for k, v in harness_env_toml["priority"].items()})
    env.update(GYM_ENGINE_ENV)
    for field, name in HYPERPARAMETER_ENV.items():
        value = hyperparameters.get(field)
        if value is not None:
            env[name] = _env_value(value)
    env.update({k: _env_value(v) for k, v in overrides.items()})
    return env


def harness_sys_path(install_root: Path) -> list[str]:
    """Import roots in the kit's order (step 11: each bundled repo's src/ or itself, reverse-sorted; step 04: public110)."""
    harness = install_root / "harness"
    entries = []
    for repo in sorted((harness / "taaf" / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(str(candidate))
                break
    entries.append(str(harness / "public110"))
    return entries


def load_harness_env_toml(install_root: Path) -> dict[str, Any]:
    return tomllib.loads((install_root / "config" / "harness_env.toml").read_text())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``path`` through a unique temporary file in the same directory, then rename it into place."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def game_files_sha256(game_dir: Path) -> dict[str, str]:
    """sha256 of a game version's files except metadata.json (create_dataset.py ``files_sha256``)."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(game_dir.iterdir())
        if p.is_file() and p.name != "metadata.json"
    }


def install_call_roles(tool_agent_cls: Any, transport: Any) -> None:
    """Mark every harness model call with its role (``gdn_role``: gameplay or helper, and ``gdn_history_id``) on the
    transport, which records it in the game's call log. The harness files are unchanged; the method is wrapped once."""
    original = tool_agent_cls._chat_completion
    if getattr(original, "_gym_call_roles", False):
        return

    @functools.wraps(original)
    def chat_completion(self: Any, *args: Any, **kwargs: Any) -> Any:
        role = kwargs.get("gdn_role", "gameplay")
        with transport.call_role(str(role), str(kwargs.get("gdn_history_id", "main"))):
            return original(self, *args, **kwargs)

    chat_completion._gym_call_roles = True
    tool_agent_cls._chat_completion = chat_completion


# The arc_agi client's first RESET of a game (the engine imports the game module then; some synthetic games take over
# a minute) waits this long by default; the client's own timeout is 10 s.
DEFAULT_RESET_TIMEOUT_S = 120.0


def make_exact_version(arcade_cls: Any) -> None:
    """Make ``arc_agi.Arcade.make("<game>-<version>")`` in ONLINE mode play exactly that version.

    arc_agi's ONLINE make asks the server for the metadata of the game name only (``sg0005``) and plays the version the
    server returns for it. A games directory can hold several versions of one name (the synthetic sets do), so the
    metadata is asked for the full id instead, which the game server answers exactly. Idempotent. The official agent's
    ``worker.py`` has a copy (it runs in the harness's own venv).
    """
    original = arcade_cls._create_remote_wrapper
    if getattr(original, "exact_version", False):
        return

    def create_remote_wrapper(self: Any, game_id: str, version: str | None, *args: Any, **kwargs: Any) -> Any:
        if version:
            game_id, version = f"{game_id}-{version}", None
        return original(self, game_id, version, *args, **kwargs)

    create_remote_wrapper.exact_version = True
    arcade_cls._create_remote_wrapper = create_remote_wrapper


def set_reset_timeout(wrapper_cls: Any, seconds: float) -> None:
    """Make arc_agi's ``RemoteEnvironmentWrapper.reset`` wait ``seconds`` for the server (its own timeout is 10 s).

    The engine imports a game on its first RESET; a slow game module (some synthetic games take 10-70 s) outlasts the
    10 s and the client gives up on a game that is still loading. Only RESET waits longer; every other call keeps its
    timeout. Idempotent (the last ``seconds`` wins). The official agent's ``worker.py`` has a copy.
    """
    original = getattr(wrapper_cls.reset, "original_reset", wrapper_cls.reset)

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        session = self._session
        post = session.post

        def post_with_reset_timeout(*post_args: Any, **post_kwargs: Any) -> Any:
            post_kwargs["timeout"] = seconds
            return post(*post_args, **post_kwargs)

        session.post = post_with_reset_timeout
        try:
            return original(self, *args, **kwargs)
        finally:
            del session.post

    reset.original_reset = original
    wrapper_cls.reset = reset


def game_progress(game: Any) -> tuple[int, int]:
    """``(step, level)`` of a taaf game now: actions taken + 1 (the harness's "Current state: step S") and levels
    completed + 1 (the level on the board)."""
    return len(game.game_run.history) + 1, int(game.current_state.levels_completed) + 1


class GameStartError(RuntimeError):
    """The attempt could not start (pin mismatch, missing game, engine refused); it has no record."""


@dataclass
class AttemptResult:
    """One played attempt. ``record`` is always the game's record once it started."""

    record: dict[str, Any]
    # The harness's note when the game crashed (it keeps its score).
    solver_note: str | None = None
    # An exception outside the harness's own handling while the game was being played.
    play_error: str | None = None
    # The final run snapshot (benchmark.json) could not be written; the record is still valid.
    persistence_error: str | None = None
    # The attempt's scorecard on the official game server and its game run's guid (what the verifier checks).
    scorecard_id: str | None = None
    guid: str | None = None
    # The game's own start (UTC ISO) and its transport key (the game's call log).
    started_at_utc: str | None = None
    game_key: str | None = None


def _abandon_started_game(start: "asyncio.Future[Any]") -> None:
    if start.cancelled() or start.exception() is not None:
        return
    game = start.result()
    if game.game_run is not None and game.game_run.state == "playing":
        game.game_run.state = "cancelled"
        game.finish_game()


class HarnessRuntime:
    """One harness configuration in this process: environment, solver, games, benchmark.json."""

    def __init__(
        self,
        *,
        install_root: Path,
        environments_dir: Path,
        run_dir: Path,
        env: dict[str, str],
        play_seconds: float,
        concurrency: int,
        num_attempts: int,
        save_request_logs: bool,
        mem_lean: bool,
        telemetry: bool,
        seed: int | None,
        save_every_s: float,
        game_server_url: str,
        api_key: str,
        cancel_drain_s: float = DEPLOYED_CANCEL_DRAIN_S,
        shared_clock: bool = True,
        max_actions_per_game: int | None = None,
        reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S,
    ):
        self.install_root = install_root
        # The official arc_agi game server (the scorecard authority) and this run's key on it.
        self.game_server_url = game_server_url
        self.api_key = api_key
        self.environments_dir = environments_dir
        self.run_dir = run_dir
        self.env = env
        self.play_seconds = play_seconds
        self.concurrency = concurrency
        self.num_attempts = num_attempts
        self.save_request_logs = save_request_logs
        self.mem_lean = mem_lean
        self.telemetry = telemetry
        self.seed = seed
        self.save_every_s = save_every_s
        self.cancel_drain_s = cancel_drain_s
        # shared_clock (the kit): every game's play ends at the run origin + play_seconds. Otherwise each game has its
        # own play_seconds from its own start (RL rollouts that start whenever the trainer sends them).
        self.shared_clock = shared_clock
        self.max_actions_per_game = max_actions_per_game
        self.reset_timeout_s = reset_timeout_s
        self._attempts = collections.Counter()
        self._attempt_serial = 0
        self.solver = None
        self.tool_agent = None
        self.run_config_sha256: str | None = None
        self._session = None
        self._games: list[Any] = []
        self._inflight: dict[str, "asyncio.Future[None]"] = {}
        self._games_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._start_time: dt.datetime | None = None
        self._saver: asyncio.Task | None = None
        self.save_failures = 0
        self.last_save_error: str | None = None
        self.closed = False
        self.installs: list[str] = []

    @property
    def start_time(self) -> dt.datetime | None:
        """The run clock's origin (benchmark.json ``start_time``, local time): taken just before the first game's
        start, so every game's first event on the game server is after it."""
        return self._start_time

    @property
    def start_time_utc(self) -> str | None:
        """The run clock's origin as a UTC ISO time (what the verifier measures the server's action times from)."""
        return self._start_time.astimezone(dt.timezone.utc).isoformat() if self._start_time else None

    def start(self, requests_proxy: Any, run_config: dict[str, Any] | None = None) -> None:
        """Export the environment, import the harness, route its model calls, build the solver (once per process)."""
        cleared = self.export_env()
        # The harness's arc_agi client reads its key from the environment (taaf's ArcadeSpec does not carry it).
        os.environ["ARC_API_KEY"] = self.api_key
        if run_config is not None:
            run_config = dict(run_config, harness_env_cleared=cleared)
        for entry in reversed(harness_sys_path(self.install_root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        if self.seed is not None:
            random.seed(self.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        import inference.agent.tool_agent as tool_agent
        import taaf.game
        from inference.framework.solver import HarnessSolver
        from inference.utils.sandbox_memory import SandboxMemoryPool

        tool_agent.requests = requests_proxy
        transport = getattr(requests_proxy, "_transport", None)
        if transport is not None:
            install_call_roles(tool_agent.ToolAgent, transport)
            transport.capture_logs()  # the harness's warnings per game (the rollout's harness_log)
        self.tool_agent = tool_agent

        import fast_state

        self.installs.append(fast_state.install())
        self.installs.append(fast_state.install_incremental())
        if self.mem_lean:
            self.installs.append(fast_state.install_mem_lean(600.0))
        self.installs.append(fast_state.install_lean_viewer())
        if self.telemetry:
            import gate_telemetry

            self.installs.append(str(gate_telemetry.install(self.run_dir / "telemetry", time.time(), sample_s=2.0)))

        solver = HarnessSolver(
            label="arc3-kit-harness",
            model="local",
            analyzer_timeout=DEPLOYED_ANALYZER_TIMEOUT_S,
            max_actions_per_game=self.max_actions_per_game,
            max_runtime_s_per_game=float(self.play_seconds),
            concurrency=int(self.concurrency),
            save_request_logs=bool(self.save_request_logs),
            start_local_server=False,
            cancel_drain_timeout_s=self.cancel_drain_s,
        )
        self.solver = solver  # from here on, a failed start is torn down by abort_start
        solver.job_dir = self.run_dir
        solver.minimal_diagnostics = True
        solver._setup()
        solver._sandbox_memory_pool = SandboxMemoryPool.from_env(int(self.num_attempts))
        solver._stop_event.clear()
        self._session = taaf.game.RunSession(record_intermediate_states=False)
        (self.run_dir / "harness_env.json").write_text(json.dumps(self.env, indent=1, sort_keys=True) + "\n")
        (self.run_dir / "installs.json").write_text(json.dumps(self.installs, indent=1) + "\n")
        if run_config is not None:
            self.write_run_config(run_config)

    def export_env(self) -> list[str]:
        """Make the process environment's harness settings exactly the configured ones.

        Removes inherited harness settings that the configuration does not set (logged by name, values are not
        logged), then exports the configuration. Returns the removed names (recorded in ``run_config.json``).
        """
        cleared = inherited_harness_env(os.environ, self.env)
        for name in cleared:
            del os.environ[name]
        if cleared:
            LOG.warning("removed inherited harness settings not in the agent config: %s", ", ".join(cleared))
        os.environ.update(self.env)
        return cleared

    def write_run_config(self, run_config: dict[str, Any]) -> str:
        """Persist the (already redacted) effective configuration; every output carries its sha256."""
        text = json.dumps(run_config, indent=1, sort_keys=True, default=str) + "\n"
        self.run_config_sha256 = hashlib.sha256(text.encode()).hexdigest()
        (self.run_dir / "run_config.json").write_text(text)
        return self.run_config_sha256

    def abort_start(self) -> None:
        """Undo a :meth:`start` that raised: tear down the solver it already built. The runtime is closed after."""
        solver, self.solver = self.solver, None
        self.closed = True
        if solver is not None:
            try:
                solver._teardown()
            except Exception:
                LOG.exception("solver teardown after a failed start")

    def check_game_pins(self, verifier_metadata: dict[str, Any]) -> str:
        """The row's full game id, after checking its version and file sha256 against the files to be played."""
        game = verifier_metadata.get("game_id")
        version = verifier_metadata.get("version")
        pinned = verifier_metadata.get("files_sha256")
        full_game_id = verifier_metadata.get("full_game_id")
        if not (game and version and pinned and full_game_id):
            raise GameStartError(f"{game}: the dataset row lacks full_game_id/version/files_sha256 pins")
        game_dir = self.environments_dir / str(game) / str(version)
        if not game_dir.is_dir():
            raise GameStartError(f"{game}: pinned version {version} is not under {self.environments_dir}")
        actual = game_files_sha256(game_dir)
        if actual != pinned:
            raise GameStartError(f"{game}: files under {game_dir} differ from the dataset's sha256 pins")
        if full_game_id != f"{game}-{version}":
            raise GameStartError(f"{game}: full_game_id {full_game_id!r} does not match version {version!r}")
        return str(full_game_id)

    def _start_game(self, full_game_id: str) -> Any:
        if self.closed:
            raise GameStartError("the agent is shutting down")
        import arc_agi
        import arc_agi.remote_wrapper
        import taaf.game_api

        make_exact_version(arc_agi.Arcade)
        set_reset_timeout(arc_agi.remote_wrapper.RemoteEnvironmentWrapper, self.reset_timeout_s)
        # Play through the official game server (the authority): taaf's own ONLINE path, one scorecard per game.
        # environments_dir is explicit: taaf's default resolves the re-arc-3 snapshot (not shipped) even when ONLINE.
        spec = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.ONLINE,
            arc_base_url=self.game_server_url,
            environments_dir=str(self.environments_dir),
        )
        game = taaf.game_api.GameAPI(env_name=full_game_id, arcade_spec=spec)
        with self._start_lock:
            if self.closed:
                raise GameStartError("the agent is shutting down")
            origin = dt.datetime.now() if self._start_time is None else None
            game._gym_started_at = dt.datetime.now().astimezone()
            game.start_game(self._session)
            if game.game_id != full_game_id:
                raise GameStartError(f"the engine started {game.game_id!r}, not the pinned {full_game_id!r}")
            with self._games_lock:
                if self._start_time is None:
                    # One run clock (the kit's): every game's play window ends at the origin + play_seconds, and
                    # RHAE@T is measured from it. The origin is taken before the first game's server RESET.
                    self._start_time = origin
                    if self.shared_clock:
                        self.solver.soft_end_time = self._start_time + dt.timedelta(seconds=float(self.play_seconds))
                self._games.append(game)
        return game

    def _play(self, game: Any, key: str, index: int, transport: Any, pass_index: int = 0) -> None:
        transport.bind_game(key, progress=functools.partial(game_progress, game))
        try:
            self.solver._play_one(game, index, pass_index, None)
        finally:
            transport.bind_game(None)
            run = game.game_run
            if run is not None and run.final_score is None:
                game.finish_game()

    def cancel_attempt(self, game: Any, key: str, transport: Any) -> None:
        """Stop one attempt only: its session sees state != playing at its next check, and its pending model wait
        is abandoned. The shared solver stop event (all games) is left alone."""
        run = game.game_run
        if run is not None and run.state == "playing":
            run.state = "cancelled"
        transport.cancel(key)

    async def play_game(
        self, verifier_metadata: dict[str, Any], index: int, transport: Any, game_key: str | None = None
    ) -> AttemptResult:
        """Play one attempt. Raises :class:`GameStartError` when it cannot start; once started, the record is kept
        whatever happens after (play exception, failed snapshot save)."""
        full_game_id = self.check_game_pins(verifier_metadata)
        start = asyncio.ensure_future(asyncio.to_thread(self._start_game, full_game_id))
        try:
            game = await asyncio.shield(start)
        except asyncio.CancelledError:
            start.add_done_callback(_abandon_started_game)  # a game that still starts ends as cancelled at once
            raise
        except GameStartError:
            raise
        except Exception as exc:
            raise GameStartError(f"{full_game_id}: {type(exc).__name__}: {exc}") from exc
        self._ensure_saver()
        # Repeats of one game (RL groups) get their own pass index (their files: <game>_p<pass>) and transport key.
        with self._games_lock:
            pass_index = self._attempts[full_game_id]
            self._attempts[full_game_id] += 1
            self._attempt_serial += 1
            key = game_key or f"{index}:{full_game_id}:{self._attempt_serial}"
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self.solver._worker_pool, functools.partial(self._play, game, key, index, transport, pass_index)
        )
        self._inflight[key] = future
        play_error = None
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            self.cancel_attempt(game, key, transport)
            await asyncio.wait({future}, timeout=self.cancel_drain_s)
            if not future.done():
                LOG.warning("%s: still running %.0f s after cancellation", key, self.cancel_drain_s)
            await asyncio.to_thread(self._save_quietly)
            raise
        except Exception as exc:
            play_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("%s: play failed outside the harness's own handling", key)
        finally:
            if future.done():
                self._inflight.pop(key, None)
        run = game.game_run
        if run.final_score is None:
            game.finish_game()
        record = run.to_json_dict()
        persistence_error = None
        try:
            await asyncio.to_thread(self.save_benchmark_json)
        except Exception as exc:
            persistence_error = f"{type(exc).__name__}: {exc}"
            LOG.error("%s: run snapshot not saved: %s", key, persistence_error)
        return AttemptResult(
            record=record,
            solver_note=record.get("solver_note") if record.get("state") == "crashed" else None,
            play_error=play_error,
            persistence_error=persistence_error,
            scorecard_id=getattr(game, "_scorecard_id", None),
            guid=game.current_state.raw.guid,
            started_at_utc=(
                game._gym_started_at.astimezone(dt.timezone.utc).isoformat()
                if getattr(game, "_gym_started_at", None) is not None
                else None
            ),
            game_key=key,
        )

    def benchmark_json(self) -> dict[str, Any]:
        """taaf ``Benchmark.to_json_dict`` fields that the kit's rhae.py reads, plus the run's config digest."""
        with self._games_lock:
            games = list(self._games)
        runs = []
        for game in games:
            for attempt in range(5):
                try:
                    runs.append(game.game_run.to_json_dict())
                    break
                except RuntimeError:  # the game thread appended to its history during the copy: copy again
                    if attempt == 4:
                        raise
        return {
            "label": "arc3-kit-harness-gym",
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "saved_at": dt.datetime.now().isoformat(),
            "play_seconds": self.play_seconds,
            "run_config_sha256": self.run_config_sha256,
            "game_runs": runs,
        }

    def save_benchmark_json(self) -> None:
        """Snapshot and publish under one lock (saves never interleave)."""
        with self._save_lock:
            atomic_write_json(self.run_dir / "benchmark.json", self.benchmark_json())

    def _save_quietly(self) -> None:
        try:
            self.save_benchmark_json()
        except Exception as exc:
            self.save_failures += 1
            self.last_save_error = f"{type(exc).__name__}: {exc}"
            LOG.error("benchmark.json save failed: %s", self.last_save_error)

    def _ensure_saver(self) -> None:
        if self._saver is None and self.save_every_s > 0:
            self._saver = asyncio.ensure_future(self._save_loop())

    async def _save_loop(self) -> None:
        """Periodic snapshots; a failed save is logged and counted, and the loop keeps going."""
        while True:
            await asyncio.sleep(self.save_every_s)
            await asyncio.to_thread(self._save_quietly)

    async def shutdown(self, transport: Any) -> None:
        """Server shutdown: stop every game, drain within ``cancel_drain_s``, finish the records as cancelled, write
        the last snapshot, stop the saver and close the solver."""
        if self.closed:
            return
        self.closed = True
        with self._games_lock:
            games = list(self._games)
        for game in games:  # as the harness's _finish_remaining: a game stopped mid-play ends as cancelled
            run = game.game_run
            if run is not None and run.state == "playing":
                run.state = "cancelled"
        if self.solver is not None:
            self.solver._stop_event.set()
        transport.cancel_all()
        pending = [f for f in self._inflight.values() if not f.done()]
        if pending:
            await asyncio.wait(pending, timeout=self.cancel_drain_s)
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                with contextlib.suppress(Exception):
                    game.finish_game()
        if self._saver is not None:
            self._saver.cancel()
            await asyncio.gather(self._saver, return_exceptions=True)
        await asyncio.to_thread(self._save_quietly)
        if self.solver is not None:
            self.solver._teardown()
