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
"""Interactive-browser resources server for NeMo-Gym.

Stateful environment: each rollout (`session_id`) owns one isolated live browser
context. The policy drives it via tool calls (navigate/click/type/observe/finish);
`verify()` scores task completion against the live browser state. Where that
browser runs is a config choice — a local Chromium (`local_playwright`, the
default) or one supplied over CDP by a session provider (`remote_cdp`) — and
nothing else in the environment changes with it. See `browser/`.

Session lifetime: a browser is released when the rollout is scored (`verify`),
when the same `session_id` is re-seeded, or when an Environment Server closes the
session through `/close_session`. An Environment Server assigns the session id and
registers that close before seeding, so a rollout abandoned before `verify` still
has its browser released. An Agent's `/run` does not close, so there a rollout
abandoned without `verify` or a re-seed keeps its browser until the process exits
(local) or the provider reclaims it (remote).
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Tuple, Union

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY


logger = logging.getLogger(__name__)


try:  # package import (gym loads the resources server as a module)
    from .browser import BrowserBackend, create_backend
except ImportError:  # script/standalone import (python app.py, local tests)
    from browser import BrowserBackend, create_backend


# Sparse outcome reward; extend with new spec keys as tasks need them.
_SCORING_KEYS = ("final_url", "url_contains", "dom_contains", "answer_equals")

# Where the browser comes from when the config says nothing: a local Chromium,
# which needs no account, no network and no quota.
_DEFAULT_BACKEND: Dict[str, Any] = {"local_playwright": {"headless": True}}


class InteractiveBrowserConfig(BaseResourcesServerConfig):
    # Single-key mapping: {backend_name: {backend kwargs}} — see browser/registry.py.
    backend: Dict[str, Any] = _DEFAULT_BACKEND
    max_elements: int = 50  # elements collected + shown per observation


class BrowserSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    initial_url: str = "about:blank"
    # Grading spec, e.g. {"final_url": "..."} or {"dom_contains": "Success"}.
    verifier_metadata: Optional[Dict[str, Any]] = None


class NavigateRequest(BaseModel):
    url: str


class ElementRequest(BaseModel):
    element_id: int


class TypeRequest(BaseModel):
    element_id: int
    text: str


class FinishRequest(BaseModel):
    answer: str = ""


class ToolResponse(BaseModel):
    observation: str
    done: bool = False
    error: Optional[str] = None


# Fields ``BaseVerifyResponse`` owns. A verify request may carry them as extras, so they
# are dropped from the spread rather than passed twice.
_RESPONSE_OWNED_FIELDS = frozenset({"reward", "failure_reason", "mask_sample"})


class _ScoringUnavailable(RuntimeError):
    """The browser could not be read, so no measurement of the policy exists."""


class BrowserVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class _SessionState:
    __slots__ = ("backend", "answer", "gt", "finished_url", "finished_text")

    def __init__(self, backend: BrowserBackend, gt: Dict[str, Any]):
        self.backend = backend
        self.answer: Optional[str] = None
        self.gt = gt
        # The page as it was the first time the model said it was done. `done` is a
        # hint the agent loop does not enforce, so an episode can keep navigating
        # after finishing; grading the live page would then score whatever it
        # wandered onto. None until the model finishes, and never overwritten.
        self.finished_url: Optional[str] = None
        self.finished_text: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.finished_url is not None


class InteractiveBrowserResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: InteractiveBrowserConfig
    # Per-rollout session state. A private attr (leading underscore) so pydantic
    # does not try to build a schema for the non-pydantic _SessionState.
    _session_id_to_state: Dict[str, _SessionState] = {}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Session state lives in this process, so a second worker would serve tool calls
        # for browsers it does not hold.
        if self.config.num_workers not in (None, 1):
            raise ValueError("interactive_browser keeps browsers in-process and requires num_workers=1")
        self._session_id_to_state = {}
        # Sessions an Environment Server seeded under its own resources_session_id.
        self._typed_identity: Dict[str, Tuple[EpisodeId, TaskId]] = {}
        self._typed_locks: Dict[str, asyncio.Lock] = {}
        # Closed ids stay known: a seed that lands after its close would otherwise open a
        # browser nobody will ever close.
        self._typed_closed: Dict[str, EpisodeId] = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/browser_navigate")(self.browser_navigate)
        app.post("/browser_click")(self.browser_click)
        app.post("/browser_type")(self.browser_type)
        app.post("/browser_observe")(self.browser_observe)
        app.post("/browser_finish")(self.browser_finish)
        return app

    # ----- lifecycle ----------------------------------------------------- #
    async def seed_session(
        self,
        request: Request,
        # Typed first and left to right: the legacy model allows extra fields, so it would
        # otherwise swallow a typed body and answer it with an empty response.
        body: Annotated[
            Union[ResourcesSeedSessionRequest, BrowserSeedSessionRequest],
            Field(union_mode="left_to_right"),
        ],
    ) -> Union[ResourcesSeedSessionResponse, BaseSeedSessionResponse]:
        if not isinstance(body, ResourcesSeedSessionRequest):
            await self._open_session(request.session[SESSION_ID_KEY], body)
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._typed_locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._typed_closed:
                raise ValueError(f"resources session is already closed: {session_id}")
            identity = self._typed_identity.get(session_id)
            if identity is not None:
                # A retried seed for the same episode gets the browser it already has.
                if identity != (body.episode_id, body.task_id):
                    raise ValueError("resources_session_id is already bound to another episode or task")
                return ResourcesSeedSessionResponse(resources_session_id=session_id)
            spec = BrowserSeedSessionRequest.model_validate(body.task_data)
            await self._open_session(session_id, spec, episode_id=body.episode_id)
            self._typed_identity[session_id] = (body.episode_id, body.task_id)
            return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(
        self,
        request: Request,
        body: Annotated[Optional[Dict[str, Any]], Body()] = None,
    ) -> Union[ResourcesCloseSessionResponse, Dict[str, bool]]:
        # An empty body closes the cookie session, for callers that seeded through /run.
        if not body:
            session_id = request.session.get(SESSION_ID_KEY)
            if session_id is not None:
                await self._release(session_id)
            return {"closed": True}

        try:
            typed = ResourcesCloseSessionRequest.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error
        session_id = typed.resources_session_id
        async with self._typed_locks.setdefault(session_id, asyncio.Lock()):
            closed_episode = self._typed_closed.get(session_id)
            if closed_episode is not None:
                if typed.episode_id != closed_episode:
                    raise ValueError("episode_id does not match the closed resources session")
                request.session.pop(SESSION_ID_KEY, None)
                return ResourcesCloseSessionResponse(resources_session_id=session_id)
            identity = self._typed_identity.get(session_id)
            if identity is not None and typed.episode_id != identity[0]:
                raise ValueError("episode_id does not match the seeded resources session")
            # Usually `verify` has already released the browser; this is the path for an
            # episode that ended before it, and a no-op otherwise.
            await self._release(session_id)
            self._typed_identity.pop(session_id, None)
            self._typed_closed[session_id] = typed.episode_id
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _release(self, session_id: str) -> None:
        st = self._session_id_to_state.pop(session_id, None)
        if st is None:
            return
        try:
            await st.backend.close()
        except Exception:
            # A browser we could not close is still held by this run; say so rather than
            # let the close report success silently.
            logger.warning("could not close the browser for session %s", session_id, exc_info=True)

    async def _open_session(
        self, session_id: str, body: BrowserSeedSessionRequest, episode_id: Optional[EpisodeId] = None
    ) -> None:
        # Resolve a repo-relative initial_url (e.g. "site/index.html") to an
        # absolute file:// URI, so example tasks don't hard-code machine paths.
        initial_url = body.initial_url
        if initial_url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", initial_url):
            initial_url = (Path(__file__).parent / initial_url).as_uri()

        # If this session_id is re-seeded (e.g. a retried rollout), release the
        # old browser first so we don't leak a session/process.
        await self._release(session_id)

        # Identifiers travel with the session so a remote provider can tag (and later
        # account for) the browser it hands out. The episode is what the training side
        # records; the session id alone is internal to whoever seeded it.
        session_metadata = {"rollout_session_id": session_id}
        if episode_id is not None:
            session_metadata["rollout_id"] = episode_id.rollout_id
            session_metadata["attempt"] = str(episode_id.attempt)
        backend = create_backend(self.config.backend, session_metadata=session_metadata)
        # `open()` unwinds its own partial state — including any provider
        # session it acquired — before it raises.
        await backend.open(initial_url)
        self._session_id_to_state[session_id] = _SessionState(backend=backend, gt=(body.verifier_metadata or {}))

    def _state(self, request: Request) -> Optional[_SessionState]:
        return self._session_id_to_state.get(request.session[SESSION_ID_KEY])

    @staticmethod
    def _no_session() -> "ToolResponse":
        return ToolResponse(observation="", error="no active session; seed_session must be called first")

    async def _render(self, st: _SessionState) -> str:
        obs = await st.backend.observe(max_elements=self.config.max_elements)
        return obs.render(max_elements=self.config.max_elements)

    # ----- tools (errors returned to model, never raised) ----------------- #
    async def browser_navigate(self, request: Request, body: NavigateRequest) -> ToolResponse:
        st = self._state(request)
        if st is None:
            return self._no_session()
        try:
            await st.backend.goto(body.url)
            return ToolResponse(observation=await self._render(st))
        except Exception as e:
            return ToolResponse(observation="", error=f"navigate failed: {e}")

    async def browser_click(self, request: Request, body: ElementRequest) -> ToolResponse:
        st = self._state(request)
        if st is None:
            return self._no_session()
        try:
            await st.backend.click(body.element_id)
            return ToolResponse(observation=await self._render(st))
        except Exception as e:
            return ToolResponse(observation="", error=f"click failed: {e}")

    async def browser_type(self, request: Request, body: TypeRequest) -> ToolResponse:
        st = self._state(request)
        if st is None:
            return self._no_session()
        try:
            await st.backend.type(body.element_id, body.text)
            return ToolResponse(observation=await self._render(st))
        except Exception as e:
            return ToolResponse(observation="", error=f"type failed: {e}")

    async def browser_observe(self, request: Request) -> ToolResponse:
        st = self._state(request)
        if st is None:
            return self._no_session()
        try:
            return ToolResponse(observation=await self._render(st))
        except Exception as e:
            return ToolResponse(observation="", error=f"observe failed: {e}")

    async def browser_finish(self, request: Request, body: FinishRequest) -> ToolResponse:
        st = self._state(request)
        if st is None:
            return self._no_session()
        if not st.finished:
            # First finish wins: the reward reflects the state the model committed to.
            st.answer = body.answer
            try:
                obs = await st.backend.observe(max_elements=0)
                st.finished_url = await st.backend.current_url()
                st.finished_text = obs.title + " " + await st.backend.text()
            except Exception:
                # A browser that died at the moment of finishing leaves no snapshot;
                # `_score` falls back to the live page and reports if that fails too.
                logger.warning("could not snapshot the page at finish", exc_info=True)
        return ToolResponse(observation="", done=True)

    # ----- reward -------------------------------------------------------- #
    def _verify_response(
        self, body: BrowserVerifyRequest, reward: float, failure_reason: Optional[str] = None
    ) -> BaseVerifyResponse:
        """Build the response without letting a request field collide with a response one.

        ``BrowserVerifyRequest`` allows extra fields, so a caller can put ``reward`` or
        ``failure_reason`` in the body; spreading the dump and also passing them as
        keywords would raise ``TypeError: got multiple values``.
        """
        return BaseVerifyResponse(
            **{k: v for k, v in body.model_dump().items() if k not in _RESPONSE_OWNED_FIELDS},
            reward=reward,
            failure_reason=failure_reason,
        )

    async def verify(self, request: Request, body: BrowserVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        st = self._session_id_to_state.get(session_id)
        if st is None:
            # The session this rollout ran in is gone: it was never seeded, it was already
            # verified, or the server lost it. There is nothing to measure, so say so
            # instead of reporting a zero that reads as a policy that solved nothing.
            return self._verify_response(body, reward=0.0, failure_reason="browser session not found at verify")

        reward, failure_reason = 0.0, None
        try:
            reward = await self._score(st)
        except _ScoringUnavailable as exc:
            # The browser died before it could be read. The rollout itself may have been
            # fine; what failed is the measurement. Raising here would abort the whole
            # collection run, because only JudgeError is converted to a routed row.
            failure_reason = str(exc)
        finally:
            await self._release(session_id)
        return self._verify_response(body, reward=reward, failure_reason=failure_reason)

    async def _score(self, st: _SessionState) -> float:
        gt = st.gt or {}
        if "answer_equals" in gt:
            # Answered from state the episode already produced; no browser call needed.
            return float((st.answer or "").strip() == str(gt["answer_equals"]).strip())
        try:
            # Grade what the model committed to, not where the episode drifted after.
            if st.finished:
                url, text = st.finished_url, st.finished_text
            else:
                # The model never finished, so the live page is the only page there is.
                obs = await st.backend.observe(max_elements=0)
                url, text = await st.backend.current_url(), obs.title + " " + await st.backend.text()
            if "final_url" in gt:
                return float(url == gt["final_url"])
            if "url_contains" in gt:
                return float(gt["url_contains"] in url)
            if "dom_contains" in gt:
                # Title + full visible page text, not just interactive elements, so
                # non-interactive DOM text (e.g. a <p>) is matched too.
                return float(str(gt["dom_contains"]).lower() in text.lower())
        except Exception as exc:
            raise _ScoringUnavailable(f"browser unreachable while scoring: {type(exc).__name__}: {exc}") from exc
        # Fail loudly rather than scoring 0: a dataset whose verifier_metadata
        # carries an unsupported (or misspelled) key would otherwise give every
        # rollout reward 0, which is indistinguishable from a policy that never
        # solves the task.
        raise ValueError(
            f"verifier_metadata has no supported scoring key (got {sorted(gt)}; expected one of {list(_SCORING_KEYS)})"
        )


if __name__ == "__main__":
    InteractiveBrowserResourcesServer.run_webserver()
