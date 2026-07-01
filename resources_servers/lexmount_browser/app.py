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
"""Lexmount interactive-browser resources server for NeMo-Gym.

Stateful environment: each rollout (`session_id`) owns one isolated live browser
context. The policy drives it via tool calls (navigate/click/type/scroll/observe/
finish); `verify()` scores task completion against the live browser state. The
browser itself is pluggable (`backend: playwright | lexmount`) — see `backend.py`.
"""

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

try:  # package import (gym loads the resources server as a module)
    from .backend import BrowserBackend, make_backend
except ImportError:  # script/standalone import (python app.py, local tests)
    from backend import BrowserBackend, make_backend


class LexmountBrowserConfig(BaseResourcesServerConfig):
    backend: str = "playwright"        # "playwright" (reference) | "lexmount"
    headless: bool = True
    endpoint: Optional[str] = None     # optional LEXMOUNT_BASE_URL override (backend: lexmount)
    browser_mode: str = "normal"       # Lexmount cloud browser mode (backend: lexmount)
    max_elements: int = 50             # elements shown per observation


class LexmountSeedSessionRequest(BaseSeedSessionRequest):
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


class LexmountVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class _SessionState:
    __slots__ = ("backend", "answer", "gt")

    def __init__(self, backend: BrowserBackend, gt: Dict[str, Any]):
        self.backend = backend
        self.answer: Optional[str] = None
        self.gt = gt


class LexmountBrowserResourcesServer(SimpleResourcesServer):
    config: LexmountBrowserConfig
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
    async def seed_session(
        self, request: Request, body: LexmountSeedSessionRequest
    ) -> BaseSeedSessionResponse:
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
                pass

        backend = make_backend(
            self.config.backend,
            headless=self.config.headless,
            endpoint=self.config.endpoint,
            browser_mode=getattr(self.config, "browser_mode", "normal"),
        )
        try:
            await backend.open(initial_url)
        except Exception:
            try:
                await backend.close()
            finally:
                raise
        self._session_id_to_state[session_id] = _SessionState(
            backend=backend, gt=(body.verifier_metadata or {})
        )
        return BaseSeedSessionResponse()

    def _state(self, request: Request) -> Optional[_SessionState]:
        return self._session_id_to_state.get(request.session[SESSION_ID_KEY])

    @staticmethod
    def _no_session() -> "ToolResponse":
        return ToolResponse(observation="", error="no active session; seed_session must be called first")

    async def _render(self, st: _SessionState) -> str:
        obs = await st.backend.observe()
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
        st.answer = body.answer
        return ToolResponse(observation="", done=True)

    # ----- reward -------------------------------------------------------- #
    async def verify(self, request: Request, body: LexmountVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        st = self._session_id_to_state.get(session_id)
        reward = 0.0
        try:
            if st is not None:
                reward = await self._score(st)
        finally:
            if st is not None:
                try:
                    await st.backend.close()
                finally:
                    self._session_id_to_state.pop(session_id, None)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    async def _score(self, st: _SessionState) -> float:
        gt = st.gt or {}
        # Sparse outcome reward; extend with new spec keys as tasks need them.
        if "final_url" in gt:
            return float(await st.backend.current_url() == gt["final_url"])
        if "url_contains" in gt:
            return float(gt["url_contains"] in await st.backend.current_url())
        if "dom_contains" in gt:
            # Check title + full visible page text (not just interactive elements),
            # so non-interactive DOM text (e.g. a <p>) is matched too.
            obs = await st.backend.observe()
            haystack = (obs.title + " " + await st.backend.text()).lower()
            return float(str(gt["dom_contains"]).lower() in haystack)
        if "answer_equals" in gt:
            return float((st.answer or "").strip() == str(gt["answer_equals"]).strip())
        return 0.0


if __name__ == "__main__":
    LexmountBrowserResourcesServer.run_webserver()
