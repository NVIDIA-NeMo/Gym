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

Session lifetime: a browser is released when the rollout is scored (`verify`) or
when the same `session_id` is re-seeded. There is no independent episode TTL, so
a rollout abandoned without either leaves its browser open until the process
exits (local) or the provider reclaims it (remote).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
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
    config: InteractiveBrowserConfig
    # Per-rollout session state. A private attr (leading underscore) so pydantic
    # does not try to build a schema for the non-pydantic _SessionState.
    _session_id_to_state: Dict[str, _SessionState] = {}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._session_id_to_state = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/browser_navigate")(self.browser_navigate)
        app.post("/browser_click")(self.browser_click)
        app.post("/browser_type")(self.browser_type)
        app.post("/browser_observe")(self.browser_observe)
        app.post("/browser_finish")(self.browser_finish)
        return app

    # ----- lifecycle ----------------------------------------------------- #
    async def seed_session(self, request: Request, body: BrowserSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Resolve a repo-relative initial_url (e.g. "site/index.html") to an
        # absolute file:// URI, so example tasks don't hard-code machine paths.
        initial_url = body.initial_url
        if initial_url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", initial_url):
            initial_url = (Path(__file__).parent / initial_url).as_uri()

        # If this session_id is re-seeded (e.g. a retried rollout), release the
        # old browser first so we don't leak a session/process.
        old = self._session_id_to_state.pop(session_id, None)
        if old is not None:
            try:
                await old.backend.close()
            except Exception:
                # Reporting beats hiding: a browser we could not close is a resource this
                # run is still holding, and the next seed proceeds either way.
                logger.warning("could not close the previous browser for session %s", session_id, exc_info=True)

        # The rollout id travels with the session so a remote provider can tag
        # (and later account for) the browser it hands out.
        backend = create_backend(self.config.backend, session_metadata={"rollout_session_id": session_id})
        # `open()` unwinds its own partial state — including any provider
        # session it acquired — before it raises.
        await backend.open(initial_url)
        self._session_id_to_state[session_id] = _SessionState(backend=backend, gt=(body.verifier_metadata or {}))
        return BaseSeedSessionResponse()

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
            try:
                await st.backend.close()
            except Exception:
                logger.warning("could not close the browser for session %s", session_id, exc_info=True)
            finally:
                self._session_id_to_state.pop(session_id, None)
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
