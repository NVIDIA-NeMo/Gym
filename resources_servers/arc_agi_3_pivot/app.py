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
"""ARC-AGI-3 pivot windows: RL from recorded game states (PivotRL).

A pivot is one recorded model call of an ARC-AGI-3 game played with the Kaggle kit's harness. A window restores the game at
that call and lets the policy continue it for a few calls; the reward is how much closer to completing the level it got.

Each window runs the recorded harness itself in a worker process (``worker.py``). The harness plays the game unmodified;
its model is a pipe. Up to the pivot, the worker answers every request with the recorded reply, after checking that the
request, the board and the action count equal the recorded ones, so the harness re-runs the recorded cells and actions
and the restored state is its own. From the pivot on, each request the harness sends is the policy's turn:

- ``/seed_session``: restore the row's pivot. A seed by an environment server (``resources_session_id``) returns the
  session; a cookie seed returns the pivot request, as ``/window`` does.
- ``/window``: the window's request (messages, tools, chat template kwargs). A pivot that does not restore exactly
  returns ``restore_error`` (the rollout is masked).
- ``/step``: the policy's reply (an assistant message) -> the harness runs its tool calls and sends its next request;
  the response holds that request's messages, ``done``, and the call's record.
- ``/verify``: reward = ``d0 - d_end``, the oracle distance to completing the level at the pivot minus at the window's
  end (``d0`` when the level completes). A window without both distances (the oracle found none) is masked. Verify
  releases the window; a retry of the same verify returns the same score.
- ``/close_session``: releases the window (its worker and work directory) on every path, also when the episode never
  reached verify.

The window ends, at a call boundary, at the first of: the level completed, ``max_actions`` actions, ``max_calls`` calls,
``dollar_cap``, or the game's end. A next request that does not extend the previous one plus its reply (the harness
compacting or trimming its context, a helper request, a continuation of a cut-off reply) also ends it, and the window is
masked, not scored: a window is one chain of calls.

Cost: every call is priced from its tokens, ``(uncached * in + cached * cached + decoded * out) / 1e6`` USD. ``cached`` is
the prompt prefix the previous call of the window already holds (its prompt + reply); the rest is ``uncached``.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import tomllib
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Annotated, Any, ClassVar

from fastapi import FastAPI, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr

from nemo_gym import PARENT_DIR
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import ENVIRONMENT_SERVER_FAILED, MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR, VERIFIER_ERROR
from nemo_gym.core.failure_kinds import ENVIRONMENT_PREPARE_FAILED
from nemo_gym.server_runtime import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.arc_agi_3.setup_arc_agi import ensure_flask
from resources_servers.arc_agi_3_pivot import recording
from resources_servers.arc_agi_3_pivot.setup_worker import (
    DEFAULT_DEST,
    HARNESS_COMMIT,
    ensure_harness_root,
    ensure_worker_python,
    materialize_game,
)
from resources_servers.arc_agi_3_pivot.task_data import PivotRow
from responses_api_agents.arc_agi_3_agent.harness_runtime import (
    GYM_ENGINE_ENV,
    HARNESS_ENV_NAMES,
    HARNESS_ENV_PREFIXES,
    harness_sys_path,
)


LOG = logging.getLogger(__name__)
WORKER = Path(__file__).with_name("worker.py")
# USD per 1M tokens (Alibaba's Qwen3.8-Flash-Next list prices).
DEFAULT_PRICES = {"in": 0.20, "cached": 0.016, "out": 0.47}
# Recording-side capture switches of a harness env file, never applied at play.
CAPTURE_ONLY_ENV = {"ARC3_SNAPSHOT_DIR"}
MODEL_URL = "http://model.invalid/v1"  # the harness's model endpoint: never contacted, its requests come to the window
# Seconds a worker gets to exit after close (then its process group is killed).
WORKER_CLOSE_S = 30.0
# The only variables of this server's environment the worker inherits (no credentials: game code runs in the worker).
WORKER_ENV_NAMES = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ")
# What a recorded harness env file may set: the harness's own settings, and the kit's library import switches.
RECORDED_ENV_NAMES = HARNESS_ENV_NAMES | {"TRANSFORMERS_NO_TF", "TRANSFORMERS_NO_TORCHVISION", "USE_TF"}


class RestoreError(RuntimeError):
    """The pivot could not be restored exactly (infrastructure: the window is masked)."""


def repo_path(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else PARENT_DIR / path


def cached_prompt_tokens(prompt: int, reported: int | None, previous: int | None) -> int:
    """Prompt tokens of a call already held by the window: ``reported`` when the agent measured it (the token prefix
    shared with the previous call's prompt + reply), else the previous call's prompt + reply (a window is one chain:
    every request extends the previous one). The window's first call counts as all cached."""
    if reported is not None:
        return min(max(int(reported), 0), prompt)
    return prompt if previous is None else min(prompt, previous)


def dollars(uncached: float, cached: float, decoded: float, prices: dict[str, float]) -> float:
    return (uncached * prices["in"] + cached * prices["cached"] + decoded * prices["out"]) / 1e6


def pool_row(verifier_metadata: dict[str, Any]) -> dict[str, Any]:
    """The task's pool row (``verifier_metadata.pivot``), parsed with the task-data schema's ``PivotRow``."""
    return PivotRow.model_validate(verifier_metadata.get("pivot")).model_dump(exclude_none=True)


def harness_env_tables(path: Path) -> dict[str, str]:
    """The environment a recording exported from its harness env file: [notebook], [setup_env], and [priority] when
    ``use_priority_scheduling`` is true; harness settings only (``RECORDED_ENV_NAMES``, ``HARNESS_ENV_PREFIXES``), any
    other name (a credential, a cache path) is dropped."""
    table = tomllib.loads(path.read_text())
    env: dict[str, Any] = {}
    for name in ("notebook", "setup_env"):
        env.update(table.get(name) or {})
    if table.get("use_priority_scheduling"):
        env.update(table.get("priority") or {})
    return {
        str(k): str(v)
        for k, v in env.items()
        if k not in CAPTURE_ONLY_ENV and (k in RECORDED_ENV_NAMES or str(k).startswith(HARNESS_ENV_PREFIXES))
    }


def extends(previous: list[dict[str, Any]], following: list[dict[str, Any]]) -> bool:
    """Whether ``following`` is ``previous`` plus the reply (an assistant message) and what came after it."""
    return (
        len(following) > len(previous)
        and following[: len(previous)] == previous
        and following[len(previous)].get("role") == "assistant"
    )


def _reply_view(message: dict[str, Any]) -> tuple:
    calls = tuple(
        (tc.get("id"), (tc.get("function") or {}).get("name"), (tc.get("function") or {}).get("arguments"))
        for tc in message.get("tool_calls") or []
    )
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return message.get("content") or "", reasoning, calls


def same_reply(history: dict[str, Any], reply: dict[str, Any]) -> bool:
    """Whether the harness's history holds the policy's reply as generated: content, reasoning and tool calls (id,
    name, arguments). Otherwise the next request does not continue what the policy generated (the chain breaks)."""
    return _reply_view(history) == _reply_view(reply)


def tool_error(messages: list[dict[str, Any]], ids: list[str]) -> str | None:
    """The first error the harness reported in its tool results for these tool call ids."""
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_call_id") not in ids:
            continue
        content = message.get("content")
        if isinstance(content, str) and '"error"' in content:
            try:
                error = json.loads(content).get("error")
            except (ValueError, AttributeError):
                error = None
            if error:
                return str(error)
    return None


class HarnessWorker:
    """One window's harness in a child process (``worker.py``) under the worker interpreter, in its own process group
    (the harness's sandbox children included). Calls are synchronous and bounded by ``timeout_s``; an error in the
    worker is raised here as a RestoreError (its ``info`` is the worker's report). A worker that does not answer in
    time is killed. ``close`` never waits on a pending call: it interrupts it."""

    def __init__(self, python: str, env: dict[str, str], cwd: str, timeout_s: float = 3600.0):
        ours, theirs = socket.socketpair()
        self._proc = subprocess.Popen(
            [python, str(WORKER), str(theirs.fileno())],
            pass_fds=(theirs.fileno(),),
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        theirs.close()
        # The socket object stays ours for shutdown(): a close alone does not end a recv blocked in another thread.
        self._sock, self._conn = ours, Connection(os.dup(ours.fileno()))
        self._lock = threading.Lock()
        self._timeout_s = timeout_s
        self._closed = False

    def call(self, op: str, *args: Any) -> Any:
        with self._lock:
            try:
                self._conn.send((op, args))
                if not self._conn.poll(self._timeout_s):
                    self._kill()
                    raise RestoreError(f"the harness worker did not answer {op} within {self._timeout_s:g} s")
                ok, value = self._conn.recv()
            except (EOFError, OSError) as exc:
                raise RestoreError(f"the harness worker exited (code {self._proc.poll()}): {exc}") from None
        if ok:
            return value
        error = RestoreError(value.get("reason") or "worker error")
        error.info = value
        raise error

    def _kill(self) -> None:
        """SIGKILL the worker's process group: the worker and the sandbox processes it started."""
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acknowledged = False
        if self._lock.acquire(blocking=False):  # no call pending: ask the worker to exit
            try:
                self._conn.send(("close", ()))
                if self._conn.poll(WORKER_CLOSE_S):
                    self._conn.recv()
                    acknowledged = True
            except (EOFError, OSError):
                pass
            finally:
                self._lock.release()
        try:  # ends a call pending in another thread (its recv sees the end of the stream)
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        if acknowledged:  # the worker is exiting: give it the time to stop its game thread
            try:
                self._proc.wait(timeout=WORKER_CLOSE_S)
            except subprocess.TimeoutExpired:
                pass
        self._kill()  # a worker that did not exit (or is stuck in a call), and the sandbox processes a worker leaves
        self._proc.wait()  # reaped
        if self._lock.acquire(timeout=WORKER_CLOSE_S):  # the interrupted call has returned: its connection can close
            self._conn.close()
            self._lock.release()


class ArcAgi3PivotConfig(BaseResourcesServerConfig):
    # The pivot pool directory: games.parquet, env/ (recorded harness env files), snap-raw/ (the recordings). Rows name
    # their files relative to it.
    pool_dir: str
    # The harness install root (<root>/harness/taaf/src, <root>/config); null = the one setup_worker.py installs under
    # .runtime/ from harness_source (the Kaggle kit's harness: kaggle:<owner>/<prefix>, a directory or a tarball).
    harness_root: str | None = None
    harness_source: str | None = None
    # The recorded harness revision the installed harness reproduces; rows recorded under another one are refused.
    harness_revision: str = HARNESS_COMMIT[:8]
    # The worker interpreter: a version (e.g. "3.12": a uv venv setup_worker.py builds under .runtime/), a path to an
    # interpreter with the harness runtime, or null = this server's. Pivots recorded under Python 3.12 restore exactly
    # only under 3.12 when the harness's retry guard fired before the pivot (its code ids hash ast.dump, whose output
    # changed in 3.13).
    worker_python: str | None = "3.12"
    # The interpreter of the harness's Python sandbox; null = the worker's (the harness's own default).
    sandbox_python: str | None = None
    # Window rules: at most max_calls model calls and max_actions game actions (checked at call boundaries).
    max_calls: int = Field(24, ge=1)
    max_actions: int = Field(12, ge=1)
    # USD; end the window after the call whose cumulative cost reaches it. null = no cap.
    dollar_cap: float | None = Field(None, gt=0)
    # USD per 1M tokens: in (uncached prompt), cached (prompt prefix the window already holds), out (generated).
    prices: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PRICES))
    # The run clock the harness shows the model: each live call advances it by call_seconds + token_seconds per
    # generated token (a replay takes the recorded values).
    clock_call_seconds: float = 7.71
    clock_token_seconds: float = 0.0302
    # Replays again with the recorded clock pinned where a cell printed a clock value an earlier call read.
    clock_repasses: int = Field(4, ge=0)
    # The sandbox's memory limit until a recorded tool result reports another one (MiB).
    sandbox_memory_mib: float = 1024
    # The sandbox timeout for replayed cells that finished in the recording (a recorded timeout keeps the harness's).
    replay_timeout_s: int = 600
    # The thinking effort of the window's live requests (e.g. low / high / max), sent as a chat-template kwarg on every
    # thinking request; a row's verifier_metadata.reasoning_effort wins. null = the pivot's recorded effort (none
    # recorded: the chat template's default). The replay before the pivot always takes the recorded effort.
    reasoning_effort: str | None = None
    # Seconds the server waits for one answer of the window's worker (the replay up to the pivot, a reply's tool calls,
    # an oracle distance); a worker that does not answer in time is killed and the window masked.
    worker_timeout_s: float = Field(3600.0, gt=0)


class PivotSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class PivotSeedResponse(BaseSeedSessionResponse):
    """The window's request: a cookie seed's response, and ``/window``'s."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # A pivot that could not be restored (infrastructure): no window, the rollout is masked.
    restore_error: str | None = None


class PivotStepRequest(BaseModel):
    # The model's reply: an assistant chat message (content, reasoning_content, tool_calls).
    message: dict[str, Any]
    finish_reason: str = "stop"
    gen_tokens: int | None = None
    prompt_tokens: int | None = None
    # Prompt tokens shared with the previous call's prompt + reply, when the agent has the token ids to measure it.
    cached_tokens: int | None = None


class PivotStepResponse(BaseModel):
    # The harness's next request (none when done): its messages, and the part of them after the reply.
    request_messages: list[dict[str, Any]] | None = None
    messages: list[dict[str, Any]]
    done: bool
    call: dict[str, Any]


class PivotVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    # Set by the agent when the window could not be played to its end (a failed model call); an agent session's
    # activation reports it in its response's metadata.agent_error.
    agent_error: str | None = None


class PivotVerifyResponse(BaseMultiRewardVerifyResponse):
    model_config = ConfigDict(extra="allow")
    progress: float = 0.0
    level_completed: float = 0.0
    actions: float = 0.0
    calls: float = 0.0
    tool_errors: float = 0.0
    gen_tokens: float = 0.0
    uncached_tokens: float = 0.0
    cached_tokens: float = 0.0
    dollars: float = 0.0
    dollar_cap_reached: float = 0.0
    d0: int | None = None
    d_end: int | None = None
    # the thinking effort the window's live requests carried (null: the chat template's default)
    reasoning_effort: str | None = None


def _set_event() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


@dataclasses.dataclass
class _Window:
    # Owned by the window while published here; read and replaced under ``guard`` (the restore runs in a thread).
    worker: HarnessWorker | None = None
    workdir: str | None = None
    request: dict[str, Any] = dataclasses.field(default_factory=dict)
    level_index: int = 0
    levels_completed: int = 0
    d0: int | None = None
    d: int | None = None
    restore_error: str | None = None
    # The worker failed or did not answer during the window (infrastructure: the window is masked).
    worker_error: str | None = None
    restore: dict[str, Any] = dataclasses.field(default_factory=dict)
    done: bool = False
    discard: str | None = None
    actions: int = 0
    calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    gen_tokens: int = 0
    uncached_tokens: int = 0
    cached_tokens: int = 0
    dollars: float = 0.0
    dollar_cap_reached: bool = False
    # The previous call's prompt + generated tokens (what the engine's prefix cache holds for the next call).
    held_tokens: int | None = None
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    guard: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    # Set by release: the restore starts no worker after it, and closes one it started before.
    released: bool = False
    # Clear while the restore thread runs.
    restore_done: threading.Event = dataclasses.field(default_factory=_set_event)

    def adopt(self, worker: HarnessWorker) -> bool:
        """Publish the restore's new worker; False when the window was released (the caller closes the worker)."""
        with self.guard:
            if self.released:
                return False
            self.worker = worker
            return True

    def take_worker(self, worker: HarnessWorker | None = None) -> HarnessWorker | None:
        """Detach the window's worker (only when it is ``worker``, if given); the caller closes what it gets."""
        with self.guard:
            if self.worker is None or (worker is not None and self.worker is not worker):
                return None
            taken, self.worker = self.worker, None
            return taken

    def release(self) -> HarnessWorker | None:
        """Mark the window released and detach its worker (the caller closes it)."""
        with self.guard:
            self.released = True
            taken, self.worker = self.worker, None
            return taken

    def cost(self) -> dict[str, float]:
        return {
            "gen_tokens": float(self.gen_tokens),
            "uncached_tokens": float(self.uncached_tokens),
            "cached_tokens": float(self.cached_tokens),
            "dollars": self.dollars,
            "dollar_cap_reached": float(self.dollar_cap_reached),
        }


class ArcAgi3PivotResourcesServer(SimpleResourcesServer):
    config: ArcAgi3PivotConfig
    seed_request_model: ClassVar[type[PivotSeedRequest]] = PivotSeedRequest
    _windows: dict[str, _Window] = PrivateAttr(default_factory=dict)
    _games: dict[tuple, dict[str, Any]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        ensure_flask()  # arc_agi imports flask (when the worker runs on this server's interpreter)
        if self.config.num_workers not in (None, 1):
            raise ValueError("windows live in this process (and their workers); num_workers must be 1")
        c = self.config
        if c.harness_root:
            self._harness_root = repo_path(c.harness_root)
        else:
            self._harness_root = ensure_harness_root(DEFAULT_DEST, source=c.harness_source)
        self._pool = repo_path(c.pool_dir)
        games = self._pool / "games.parquet"
        if not games.is_file():
            raise FileNotFoundError(f"{games} missing (pool_dir must hold the pool's games.parquet)")
        import pyarrow.parquet as pq

        columns = ["game_id", "source_sha256", "environment_code", "environment_path", "sdk_metadata_json"]
        for entry in pq.read_table(games, columns=columns).to_pylist():
            self._games[(entry["game_id"], entry["source_sha256"])] = entry
        python = c.worker_python
        if not python:
            python = sys.executable
        elif not Path(python).is_file():
            python = str(ensure_worker_python(python, DEFAULT_DEST))
        self._python = python
        self._game_cache = DEFAULT_DEST / "games"
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/window")(self.window)
        app.post("/step")(self.step)
        return app

    # ------------------------------------------------------------------ restore
    def _pool_file(self, rel: str, pid: Any) -> Path:
        """A file a row names, relative to the pool; one that resolves outside the pool is refused."""
        path = (self._pool / rel).resolve()
        if not path.is_relative_to(self._pool.resolve()):
            raise RestoreError(f"pivot {pid}: {rel} is outside the pool {self._pool}")
        return path

    def _spec(self, row: dict[str, Any], workdir: str, overrides: dict[int, tuple]) -> dict[str, Any]:
        pivot = json.loads(row["pivot_json"])
        pid = pivot.get("pivot_id")
        harness = str(row.get("harness") or pivot.get("env_harness") or "")
        if harness[:8] != self.config.harness_revision[:8]:
            raise RestoreError(
                f"pivot {pid} was recorded under harness {harness}, this server runs {self.config.harness_revision}"
            )
        ref = pivot.get("snapshot") or {}
        if not ref.get("snap_file"):
            raise RestoreError(f"pivot {pid} has no recording (snapshot.snap_file)")
        snap = self._pool_file(ref["snap_file"], pid)
        if not snap.is_file():
            raise RestoreError(f"pivot {pid}: recording {snap} missing")
        raw = snap.read_bytes()
        if ref.get("snap_sha256") and hashlib.sha256(raw).hexdigest() != ref["snap_sha256"]:
            raise RestoreError(f"pivot {pid}: recording sha256 differs from the row")
        parsed = recording.parse(raw)
        call = int((pivot.get("provenance") or {}).get("call", pivot.get("call_index", -1)))
        upto = next((i for i, c in enumerate(parsed["calls"]) if int(c["n"]) == call), None)
        if upto is None:
            raise RestoreError(f"pivot {pid}: call {call} is not in its recording")
        env_rel = row.get("harness_env") or pivot.get("harness_env")
        if not env_rel:
            raise RestoreError(f"pivot {pid} names no harness env file")
        env_file = self._pool_file(env_rel, pid)
        if not env_file.is_file():
            raise RestoreError(f"pivot {pid}: harness env {env_file} missing")
        want = row.get("harness_env_sha256") or pivot.get("harness_env_sha256")
        if want and hashlib.sha256(env_file.read_bytes()).hexdigest() != want:
            raise RestoreError(f"pivot {pid}: harness env {env_file} differs from the row's sha256")
        game_id = str(pivot.get("game_id") or parsed["header"].get("game_id"))
        source_sha256 = pivot.get("game_source_sha256")
        entry = self._games.get((game_id, source_sha256))
        if entry is None and source_sha256:  # the pinned revision only: another one can diverge after the pivot
            raise RestoreError(f"pivot {pid}: game {game_id} at source sha256 {source_sha256} is not in games.parquet")
        if entry is None:  # a legacy row without a pin: the pool's only revision of the game
            matches = [e for (gid, _), e in self._games.items() if gid == game_id]
            if len(matches) != 1:
                raise RestoreError(f"pivot {pid}: game {game_id} is not (uniquely) in games.parquet")
            entry = matches[0]
        if int(pivot.get("seed") or 0) != 0:
            raise RestoreError(f"pivot {pid}: game seed {pivot.get('seed')} (only seed 0 is played)")
        model = parsed["calls"][0]["payload"].get("model") or "policy"
        env = harness_env_tables(env_file)
        env.update(GYM_ENGINE_ENV)  # no engine-specific protocols (GDN state ownership, CPU retention) under Gym
        env.update(
            {
                "LOCAL_ANALYZER_BASE_URL": MODEL_URL,
                "OPENAI_BASE_URL": MODEL_URL,
                "LOCAL_ANALYZER_MODEL_ID": str(model),
                "INFERENCE_ANALYZER_MODEL": str(model),
            }
        )
        c = self.config
        return {
            "env": env,
            "harness_paths": harness_sys_path(self._harness_root),
            "environments_dir": str(materialize_game(entry, self._game_cache)),
            "game_id": json.loads(entry["sdk_metadata_json"])["game_id"],
            "recording": str(snap),
            "upto": upto,
            "sandbox_seed": int(parsed["header"].get("sandbox_seed") or 0),
            "sandbox_python": c.sandbox_python,
            "clock": {"call_s": c.clock_call_seconds, "token_s": c.clock_token_seconds, "budget": None},
            "clock_overrides": {str(k): list(v) for k, v in overrides.items()},
            "memory_mib": c.sandbox_memory_mib,
            "replay_timeout_s": c.replay_timeout_s,
            "reasoning_effort": row.get("reasoning_effort") or c.reasoning_effort,
            "recorded_effort": row.get("thinking_effort"),
            "workdir": workdir,
        }

    def _worker_env(self, spec_env: dict[str, str]) -> dict[str, str]:
        """A minimal environment: ``WORKER_ENV_NAMES`` of this server's, then the window's harness settings."""
        env = {k: os.environ[k] for k in WORKER_ENV_NAMES if k in os.environ}
        env.update(spec_env)
        return env

    def _restore(self, row: dict[str, Any], window: _Window) -> None:
        """Start the window's worker and replay up to the pivot (again with the recorded clock pinned where needed).

        Runs in a thread that a cancelled seed does not stop: once the window is released, it starts no worker and
        closes the one it started; ``restore_done`` is set when it returns."""
        try:
            self._replay(row, window)
        finally:
            window.restore_done.set()

    def _replay(self, row: dict[str, Any], window: _Window) -> None:
        pivot = json.loads(row["pivot_json"])
        overrides: dict[int, tuple] = {}
        workdir = window.workdir = tempfile.mkdtemp(prefix="arc3-pivot-")
        released = RestoreError(f"pivot {pivot.get('pivot_id')}: the session was released during the restore")
        for attempt in range(1 + self.config.clock_repasses):
            spec = self._spec(row, workdir, overrides)
            if window.released:
                raise released
            worker = HarnessWorker(self._python, self._worker_env(spec["env"]), workdir, self.config.worker_timeout_s)
            if not window.adopt(worker):
                worker.close()
                raise released
            try:  # a release meanwhile closes the worker, which interrupts the replay
                start = worker.call("start", spec)
                break
            except RestoreError as exc:
                if window.take_worker(worker) is not None:
                    worker.close()
                more = {
                    int(k): tuple(v)
                    for k, v in ((getattr(exc, "info", {}) or {}).get("clock_overrides") or {}).items()
                }
                more = {k: v for k, v in more.items() if overrides.get(k) != v}
                if not more or attempt == self.config.clock_repasses:
                    info = getattr(exc, "info", None) or {"reason": str(exc)}
                    detail = json.dumps({k: v for k, v in info.items() if k not in ("traceback",)}, default=str)
                    raise RestoreError(f"pivot {pivot.get('pivot_id')}: {detail[:1500]}") from None
                overrides.update(more)
        try:
            if start["request"] is None or start["role"] != "gameplay" or start["continuation"]:
                raise RestoreError(f"pivot {pivot.get('pivot_id')}: the pivot request is not a gameplay request")
            window.request = start["request"]
            window.level_index = int(start["level_index"])
            window.levels_completed = int(start["levels_completed"])
            window.d0 = window.d = worker.call("distance", window.level_index)
            if pivot.get("po_d0") is not None and window.d0 != int(pivot["po_d0"]):
                raise RestoreError(
                    f"pivot {pivot.get('pivot_id')}: oracle distance {window.d0} != recorded {pivot['po_d0']}"
                )
            window.restore = {
                k: start[k]
                for k in ("masked", "engine_fields", "replayed", "action_count", "reasoning_effort")
                if k in start
            }
            window.restore["clock_repasses"] = attempt
        except BaseException:
            if window.take_worker(worker) is not None:
                worker.close()
            raise

    # Session state hooks (the base seeds through them and releases on close / retirement; verify releases too).
    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._windows

    async def initialize_session_state(self, session_id: str, body: PivotSeedRequest) -> None:
        window = self._windows[session_id] = _Window()
        meta = body.verifier_metadata
        try:
            row = pool_row(meta)
            row["reasoning_effort"] = meta.get("reasoning_effort")
            window.restore_done.clear()
            await asyncio.to_thread(self._restore, row, window)
        except Exception as exc:  # a pivot that does not restore exactly, a bad row: never a policy outcome
            window.restore_error = f"{type(exc).__name__}: {exc}"
            LOG.warning("pivot %s not restored: %s", meta.get("pivot_id"), window.restore_error)

    async def release_session_state(self, session_id: str) -> None:
        window = self._windows.get(session_id)
        if window is None:
            return
        worker = window.release()
        if worker is not None:
            await asyncio.to_thread(worker.close)  # also interrupts a restore's pending call
        # A restore still running (its seed was cancelled) stops at its next step and closes what it started; the
        # window stays here, owned and retryable, until it has.
        while not window.restore_done.is_set():
            await asyncio.sleep(0.05)
        if window.workdir:
            await asyncio.to_thread(shutil.rmtree, window.workdir, True)
        self._windows.pop(session_id, None)

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest | Annotated[PivotSeedRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | PivotSeedResponse:
        seeded = await super().seed_session(body, request=request)
        if isinstance(body, ResourcesSeedSessionRequest):
            return seeded  # the agent reads the window's request from /window
        return self._request_view(self._windows[request.session[SESSION_ID_KEY]])

    async def window(self, request: Request) -> PivotSeedResponse:
        """The window's current request (the pivot request until the first step)."""
        window = self._windows.get(request.session[SESSION_ID_KEY])
        if window is None:
            raise ResourcesSessionLostError("no window for this session (not seeded, verified or closed)")
        return self._request_view(window)

    @staticmethod
    def _request_view(window: _Window) -> PivotSeedResponse:
        if window.restore_error is not None:
            return PivotSeedResponse(restore_error=window.restore_error)
        return PivotSeedResponse(
            messages=window.request["messages"],
            tools=window.request.get("tools") or [],
            chat_template_kwargs=window.request.get("chat_template_kwargs") or {},
        )

    # ------------------------------------------------------------------ play
    @staticmethod
    def harness_reply(message: dict[str, Any]) -> dict[str, Any]:
        """The policy's reply as a chat completion message for the harness."""
        out: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            out["reasoning_content"] = reasoning
        if message.get("tool_calls"):
            out["tool_calls"] = message["tool_calls"]
        return out

    async def step(self, request: Request, body: PivotStepRequest) -> PivotStepResponse:
        window = self._windows.get(request.session[SESSION_ID_KEY])
        if window is None:
            raise ResourcesSessionLostError("no window for this session (not seeded, verified or closed)")
        if window.worker is None:
            raise ValueError("step on a session without a restored window")
        c = self.config
        async with window.lock:
            worker = window.worker
            if window.done or worker is None:
                raise ValueError("step after the window ended")
            usage = {"completion_tokens": int(body.gen_tokens or 0), "prompt_tokens": int(body.prompt_tokens or 0)}
            try:
                out = await asyncio.to_thread(
                    worker.call, "reply", self.harness_reply(body.message), body.finish_reason, usage
                )
            except RestoreError as exc:  # the worker failed or did not answer: the window ends, masked
                return await self._worker_failed(window, exc)
            # actions count up to the one that completed the pivot's level
            actions = out["actions"]
            completed_at = next(
                (i for i, a in enumerate(actions) if a["levels_completed"] > window.levels_completed), None
            )
            if completed_at is not None:
                actions = actions[: completed_at + 1]
            level_completed = completed_at is not None
            ids = [str(tc.get("id")) for tc in body.message.get("tool_calls") or []]
            following = (out.get("request") or {}).get("messages") or []
            call = {
                "actions": [a["name"] for a in actions],
                "level_completed": level_completed,
                "game_over": any(a["state"] == "GAME_OVER" for a in actions),
                "tool_error": tool_error(following, ids),
                "d_before": window.d,
            }
            try:
                window.d = (
                    0 if level_completed else await asyncio.to_thread(worker.call, "distance", window.level_index)
                )
            except RestoreError as exc:
                return await self._worker_failed(window, exc)
            call["d_after"] = window.d
            window.actions += len(actions)
            window.calls.append(call)

            prompt, gen = int(body.prompt_tokens or 0), int(body.gen_tokens or 0)
            cached = cached_prompt_tokens(prompt, body.cached_tokens, window.held_tokens)
            window.held_tokens = prompt + gen
            window.gen_tokens += gen
            window.cached_tokens += cached
            window.uncached_tokens += prompt - cached
            window.dollars += dollars(prompt - cached, cached, gen, c.prices)
            call["dollars"] = window.dollars

            reason = None
            if level_completed:
                reason = "level_completed"
            elif window.actions >= c.max_actions:
                reason = "action_budget"
            elif len(window.calls) >= c.max_calls:
                reason = "max_calls"
            elif body.finish_reason == "abort":
                reason = "abort"
            elif c.dollar_cap is not None and window.dollars >= c.dollar_cap:
                reason, window.dollar_cap_reached = "dollar_cap", True
            elif out.get("request") is None:
                reason = f"game_ended:{out.get('ended')}"
            elif out["role"] != "gameplay" or out["continuation"]:
                reason, window.discard = (
                    "helper_request",
                    f"the harness sent a {out['role']} request" + (" (a continuation)" if out["continuation"] else ""),
                )
            elif not extends(window.request["messages"], following):
                reason, window.discard = "context_rewrite", "the harness's next request does not extend the window"
            elif not same_reply(following[len(window.request["messages"])], body.message):
                reason, window.discard = "reply_rewritten", "the harness's history holds the reply changed"
            call["done_reason"] = reason
            window.done = reason is not None
            if not window.done:
                new = following[len(window.request["messages"]) + 1 :]
                window.request = out["request"]
            else:  # no next turn: the reply's tool results only
                new = [m for m in following if m.get("role") == "tool" and m.get("tool_call_id") in ids]
            if window.done and (worker := window.take_worker()) is not None:
                await asyncio.to_thread(worker.close)
        return PivotStepResponse(
            request_messages=None if window.done else window.request["messages"],
            messages=new,
            done=window.done,
            call=json.loads(json.dumps(call, default=str)),
        )

    async def _worker_failed(self, window: "_Window", exc: RestoreError) -> PivotStepResponse:
        window.done, window.worker_error = True, str(exc)
        if (worker := window.take_worker()) is not None:
            await asyncio.to_thread(worker.close)
        return PivotStepResponse(
            request_messages=None, messages=[], done=True, call={"done_reason": "worker_error", "error": str(exc)}
        )

    # ------------------------------------------------------------------ score
    async def verify(self, request: Request, body: PivotVerifyRequest) -> PivotVerifyResponse:
        """Score the window and release it (its worker exits now, not at session close). The score is retained: a
        retry of the verify (its first response lost) returns it, until the session is closed."""
        session_id = request.session[SESSION_ID_KEY]
        payload = body.model_dump()
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                return PivotVerifyResponse(
                    **(payload | lost_session_verification("no window for this session (not seeded or closed)")),
                    reward_components={"progress": 0.0},
                )
            if verification.is_live:
                result = self._score(self._windows[session_id], body)
                verification.retain({k: v for k, v in result.model_dump().items() if k not in payload})
                return result
            return PivotVerifyResponse(**(payload | verification.retained))

    def _score(self, window: _Window, body: PivotVerifyRequest) -> PivotVerifyResponse:
        payload = body.model_dump()
        effort = next(iter((window.restore.get("reasoning_effort") or {}).values()), None)
        agent_error = body.agent_error or (body.response.metadata or {}).get("agent_error")
        masked = None
        if window.restore_error is not None:
            masked = (ENVIRONMENT_PREPARE_FAILED, window.restore_error)
        elif window.worker_error is not None:
            masked = (ENVIRONMENT_SERVER_FAILED, f"the window's harness worker failed: {window.worker_error}")
        elif window.discard is not None:
            masked = (MODEL_CONTEXT_LIMIT, f"{window.discard} (discarded, not scored)")
        elif agent_error or not window.done:
            masked = (MODEL_REQUEST_ERROR, agent_error or "the window ended before its termination rule")
        if masked is not None:
            return PivotVerifyResponse(
                **payload,
                reward=0.0,
                reward_components={"progress": 0.0},
                mask_sample=True,
                failure_kind=masked[0],
                failure_reason=masked[1],
                d0=window.d0,
                calls=float(len(window.calls)),
                reasoning_effort=effort,
                **window.cost(),
            )
        level_completed = any(c.get("level_completed") for c in window.calls)
        d_end = 0 if level_completed else window.d
        # Progress needs both distances; distance zero is the observed completion only (the oracle's None, its search
        # cap or a loop, or a zero the game did not confirm, is no measurement).
        unmeasured = window.d0 is None or d_end is None or (d_end == 0 and not level_completed)
        if unmeasured:
            return PivotVerifyResponse(
                **payload,
                reward=0.0,
                reward_components={"progress": 0.0},
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=f"no oracle distance to score (d0 {window.d0}, d_end {window.d})",
                level_completed=float(level_completed),
                actions=float(window.actions),
                calls=float(len(window.calls)),
                d0=window.d0,
                d_end=window.d,
                reasoning_effort=effort,
                **window.cost(),
            )
        progress = float(window.d0 - d_end)
        tool_errors = float(sum(1 for c in window.calls if c.get("tool_error")))
        return PivotVerifyResponse(
            **payload,
            reward=progress,
            # the reward's only term (a trainer checks reward == sum of components); the window's other measures are
            # the fields below
            reward_components={"progress": progress},
            progress=progress,
            level_completed=float(level_completed),
            actions=float(window.actions),
            calls=float(len(window.calls)),
            tool_errors=tool_errors,
            d0=window.d0,
            **window.cost(),
            d_end=d_end,
            reasoning_effort=effort,
        )


if __name__ == "__main__":
    ArcAgi3PivotResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArcAgi3PivotResourcesServer.run_webserver()  # noqa: F401
