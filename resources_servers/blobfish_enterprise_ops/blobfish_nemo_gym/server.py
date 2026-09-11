# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""NeMo Gym resources server that hosts one Blobfish suite in-process.

Contract with NeMo Gym's ``simple_agent``:

* ``POST /seed_session``  the dataset row (``task_id`` and friends) arrives as the body; the
  session cookie NeMo Gym attaches identifies the rollout. A fresh, isolated world is built.
* ``POST /<tool>``         every function call the policy emits is forwarded here with the parsed
  arguments as the JSON body. The tool name is the Responses-API-safe public name.
* ``POST /verify``         the row plus the completed response arrive; the package's own
  deterministic verifier scores the final world state and trace.

Rewards: ``reward`` is the criterion fraction by default (dense signal for RL on tasks frontier
models mostly fail), the strict pass when ``reward_mode: strict``, or the criterion fraction times
the earned share of the suite's outcome families when ``reward_mode: gated`` (recommended for
training: a wrong final write then costs most of the reward instead of a tenth of it; see
``suite.OUTCOME_FAMILIES``). ``strict_pass``, ``criterion_fraction`` and ``outcome_fraction`` are
returned as top-level floats so NeMo Gym's aggregate metrics report them, and
``reward_components`` carries them for multi-objective trainers.

``SuiteHost`` holds the suite, the episode registry and the scoring; the HTTP server and
``BlobfishVerifier`` (what NeMo Gym's onboarding checks build for each verifier-fixture case) share
it, so ``gym env test`` exercises exactly the code ``/verify`` runs.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from starlette.concurrency import run_in_threadpool

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


try:  # NeMo Gym main: the multi-objective base that owns ``reward_components``.
    from nemo_gym.base_resources_server import BaseMultiRewardVerifyResponse as _VerifyResponseBase
except ImportError:  # NeMo Gym 0.4.x wheels
    _VerifyResponseBase = BaseVerifyResponse

from .dataset import build_row
from .suite import Episode, Suite, VerifierError, outcome_fraction


__all__ = [
    "BlobfishSeedSessionRequest",
    "BlobfishSeedSessionResponse",
    "BlobfishSuiteResourcesServer",
    "BlobfishSuiteResourcesServerConfig",
    "BlobfishVerifier",
    "BlobfishVerifyRequest",
    "BlobfishVerifyResponse",
    "HostSettings",
    "ScriptedEpisode",
    "ToolCall",
    "resolve_repo_path",
]

RESERVED_ROUTES = frozenset({"seed_session", "verify", "aggregate_metrics", "mcp", "docs", "openapi.json"})


class BlobfishSuiteResourcesServerConfig(BaseResourcesServerConfig):
    suite_dir: str
    # JSON {task_dir_name: sha256} pinning every task package (emitted next to the environment).
    # When set, the server refuses to start on packages that differ from the reviewed ones.
    suite_manifest: str | None = None
    reward_mode: str = "fraction"  # "fraction" (criterion fraction), "strict" (all criteria met) or "gated"
    # Override the measured outcome families for ``gated`` (defaults to suite.OUTCOME_FAMILIES by suite name).
    outcome_families: list[str] | None = None
    max_open_episodes: int = 4096
    episode_ttl_seconds: float = 7200.0
    max_concurrent_verifies: int = 8


class BlobfishSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str


class BlobfishSeedSessionResponse(BaseSeedSessionResponse):
    task_id: str
    suite: str
    tool_count: int


class BlobfishVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str


class BlobfishVerifyResponse(_VerifyResponseBase):  # type: ignore[misc,valid-type]
    task_id: str
    strict_pass: float
    criterion_fraction: float
    tool_calls: float
    failed_tool_calls: float
    outcome_fraction: float | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    verifier_profile: str = ""
    note: str = ""


class ToolCall(BaseModel):
    """One call in a scripted episode; ``tool`` is the public (``a__b``) or package (``a.b``) name."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ScriptedEpisode(BaseModel):
    """A policy's tool calls for one task, replayed on a fresh world and scored without HTTP."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    calls: list[ToolCall] = Field(default_factory=list)


@dataclass(frozen=True)
class HostSettings:
    reward_mode: str = "fraction"
    outcome_families: tuple[str, ...] | None = None
    max_open_episodes: int = 4096
    episode_ttl_seconds: float = 7200.0
    max_concurrent_verifies: int = 8


class SuiteHost:
    """Suite loading, the episode registry and scoring, shared by the server and the verifier."""

    def _init_host(self, suite_dir: str | Path, suite_manifest: str | Path | None, settings: HostSettings) -> None:
        expected = None
        if suite_manifest:
            manifest = json.loads(resolve_repo_path(suite_manifest).read_text(encoding="utf-8"))
            expected = manifest.get("packages", manifest)
        self._suite = Suite.load(resolve_repo_path(suite_dir), expected_digests=expected)
        self._episodes = {}
        self._lock = threading.Lock()
        self._verify_slots = threading.BoundedSemaphore(max(1, settings.max_concurrent_verifies))
        self._settings = settings

    @property
    def suite(self) -> Suite:
        return self._suite

    def open_episodes(self) -> int:
        with self._lock:
            return sum(1 for episode in self._episodes.values() if episode is not None)

    def reset(self) -> None:
        """Close every open episode; the next one is rebuilt from the package seed."""
        with self._lock:
            stale = list(self._episodes.values())
            self._episodes.clear()
        for episode in stale:
            if episode is not None:
                episode.close()

    async def exercise(
        self, task_id: str, calls: Iterable[ToolCall | tuple[str, dict[str, Any]]]
    ) -> BlobfishVerifyResponse:
        """Seed, call and score one scripted episode: the path a rollout takes, minus HTTP."""
        try:
            task = self._suite.get(task_id)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from exc
        row = build_row(task, self._suite.name)
        episode = await run_in_threadpool(Episode, task)
        try:
            for call in calls:
                tool, arguments = (call.tool, call.arguments) if isinstance(call, ToolCall) else call
                await run_in_threadpool(episode.call, tool, dict(arguments or {}))
        except BaseException:
            episode.close()
            raise
        return await self._score(
            episode,
            task_id=task_id,
            responses_create_params=row["responses_create_params"],
            response=_empty_response(),
        )

    async def _score(
        self, episode: Episode, *, task_id: str, responses_create_params: Any, response: Any
    ) -> BlobfishVerifyResponse:
        """Run the package verifier on a finished episode and close it; verifier errors score zero."""
        try:
            with self._verify_slots:
                verdict = await run_in_threadpool(episode.score)
            calls, failed = episode.stats()
        except VerifierError as exc:
            calls, failed = episode.stats()
            return BlobfishVerifyResponse(
                responses_create_params=responses_create_params,
                response=response,
                reward=0.0,
                task_id=task_id,
                strict_pass=0.0,
                criterion_fraction=0.0,
                tool_calls=float(calls),
                failed_tool_calls=float(failed),
                reward_components={"strict_pass": 0.0, "criterion_fraction": 0.0},
                note=f"verifier error: {exc}",
                **_failure_reason(f"verifier error, reward is not a measure of policy quality: {exc}"),
            )
        finally:
            episode.close()
        strict = float(verdict.strict_pass)
        fraction = float(verdict.reward_fraction)
        outcome = outcome_fraction(verdict, self._suite.name, self._settings.outcome_families)
        mode = self._settings.reward_mode
        if mode == "strict":
            reward = strict
        elif mode == "gated" and outcome is not None:
            reward = fraction * outcome
        else:
            reward = fraction
        components = {"strict_pass": strict, "criterion_fraction": fraction}
        if outcome is not None:
            components["outcome_fraction"] = outcome
        return BlobfishVerifyResponse(
            responses_create_params=responses_create_params,
            response=response,
            reward=reward,
            task_id=task_id,
            strict_pass=strict,
            criterion_fraction=fraction,
            tool_calls=float(calls),
            failed_tool_calls=float(failed),
            outcome_fraction=outcome,
            reward_components=components,
            verifier_profile=verdict.profile,
            note=""
            if mode != "gated" or outcome is not None
            else "gated mode: no outcome families for this suite, reward is the plain fraction",
        )

    def _sweep_expired(self) -> None:
        cutoff = time.time() - self._settings.episode_ttl_seconds
        with self._lock:
            expired = [
                sid for sid, episode in self._episodes.items() if episode is not None and episode.started_at < cutoff
            ]
            stale = [self._episodes.pop(sid) for sid in expired]
        for episode in stale:
            if episode is not None:
                episode.close()


class BlobfishVerifier(SuiteHost):
    """The scorer without the web server: NeMo Gym's onboarding checks build one per fixture case."""

    def __init__(
        self,
        suite_dir: str | Path,
        *,
        suite_manifest: str | Path | None = None,
        reward_mode: str = "fraction",
        outcome_families: Iterable[str] | None = None,
    ):
        families = tuple(outcome_families) if outcome_families is not None else None
        self._init_host(suite_dir, suite_manifest, HostSettings(reward_mode=reward_mode, outcome_families=families))

    async def verify(self, body: ScriptedEpisode) -> BlobfishVerifyResponse:
        return await self.exercise(body.task_id, body.calls)


class BlobfishSuiteResourcesServer(SuiteHost, SimpleResourcesServer):
    config: BlobfishSuiteResourcesServerConfig

    _suite: Suite = PrivateAttr()
    _episodes: dict[str, Episode | None] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _verify_slots: threading.BoundedSemaphore = PrivateAttr()
    _settings: HostSettings = PrivateAttr(default_factory=HostSettings)

    def model_post_init(self, __context: Any) -> None:
        self._init_host(
            self.config.suite_dir,
            self.config.suite_manifest,
            HostSettings(
                reward_mode=self.config.reward_mode,
                outcome_families=tuple(self.config.outcome_families) if self.config.outcome_families else None,
                max_open_episodes=self.config.max_open_episodes,
                episode_ttl_seconds=self.config.episode_ttl_seconds,
                max_concurrent_verifies=self.config.max_concurrent_verifies,
            ),
        )

    # ------------------------------------------------------------------ routes
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Registered after the base routes so /seed_session, /verify and /aggregate_metrics win.
        app.post("/{tool_name}")(self.call_tool)
        return app

    async def seed_session(self, request: Request, body: BlobfishSeedSessionRequest) -> BlobfishSeedSessionResponse:  # type: ignore[override]
        session_id = _session_id(request)
        try:
            task = self._suite.get(body.task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        self._sweep_expired()
        with self._lock:
            if len(self._episodes) >= self._settings.max_open_episodes and session_id not in self._episodes:
                raise HTTPException(status_code=503, detail="too many open episodes; retry later")
            self._episodes[session_id] = None  # reserve the slot while the world is built
        try:
            episode = await run_in_threadpool(Episode, task)
        except Exception:
            with self._lock:
                self._episodes.pop(session_id, None)
            raise
        with self._lock:
            previous = self._episodes.get(session_id)
            self._episodes[session_id] = episode
        if previous is not None:
            previous.close()
        return BlobfishSeedSessionResponse(task_id=task.task_id, suite=self._suite.name, tool_count=len(task.tools))

    async def call_tool(
        self,
        tool_name: str,
        request: Request,
        body: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        if tool_name in RESERVED_ROUTES:
            raise HTTPException(status_code=404, detail=f"{tool_name} is not a tool")
        episode = self._episode_for(request)
        if episode is None:
            raise HTTPException(status_code=409, detail="no open episode for this session; call /seed_session first")
        if episode.resolve_tool(tool_name) is None:
            # Returned as a tool result, not an HTTP error, so the policy sees its mistake.
            return {"error": f"unknown tool: {tool_name}"}
        return await run_in_threadpool(episode.call, tool_name, body or {})

    async def verify(self, request: Request, body: BlobfishVerifyRequest) -> BlobfishVerifyResponse:  # type: ignore[override]
        session_id = _session_id(request)
        with self._lock:
            episode = self._episodes.pop(session_id, None)
        if episode is None:
            return BlobfishVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=0.0,
                task_id=body.task_id,
                strict_pass=0.0,
                criterion_fraction=0.0,
                tool_calls=0.0,
                failed_tool_calls=0.0,
                reward_components={"strict_pass": 0.0, "criterion_fraction": 0.0},
                note="no open episode for this session; the rollout never seeded a world",
                **_failure_reason("rollout never seeded a world; reward is not a measure of policy quality"),
            )
        self._sweep_expired()
        return await self._score(
            episode, task_id=body.task_id, responses_create_params=body.responses_create_params, response=body.response
        )

    # --------------------------------------------------------------- helpers
    def _episode_for(self, request: Request) -> Episode | None:
        session_id = _session_id(request)
        with self._lock:
            return self._episodes.get(session_id)


def _search_roots() -> list[Path]:
    """Where a repository-relative config path may live: the working directory, NeMo Gym's checkout root
    (servers are launched with the server directory as working directory), and the checkout this file is
    vendored into (resources_servers/<server>/blobfish_nemo_gym/server.py)."""
    roots = [Path.cwd()]
    try:
        from nemo_gym import PARENT_DIR

        roots.append(Path(PARENT_DIR))
    except ImportError:  # pragma: no cover - nemo_gym is a dependency of this module
        pass
    roots.append(Path(__file__).resolve().parents[3])
    return roots


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a config path as given, or relative to the roots above; error names every location tried."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    tried = []
    for root in _search_roots():
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
        tried.append(str(candidate))
    raise FileNotFoundError(f"{value!s} not found; looked in: {', '.join(dict.fromkeys(tried))}")


def _empty_response() -> dict[str, Any]:
    """A response with no model output: scripted episodes carry their policy in the trace."""
    return {
        "id": "resp_scripted",
        "created_at": 0,
        "model": "scripted",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _failure_reason(text: str) -> dict[str, str]:
    """``failure_reason`` exists on NeMo Gym main's verify response; older wheels reject unknown fields."""
    return {"failure_reason": text} if "failure_reason" in BlobfishVerifyResponse.model_fields else {}


def _session_id(request: Request) -> str:
    session_id = request.session.get(SESSION_ID_KEY)
    if not session_id:
        raise HTTPException(status_code=400, detail="missing NeMo Gym session")
    return str(session_id)


if __name__ == "__main__":  # pragma: no cover
    BlobfishSuiteResourcesServer.run_webserver()
