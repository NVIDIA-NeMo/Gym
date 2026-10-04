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
"""TimeWarp resources server: browse one UI era of the Wiki, News and Shop sites, then answer.

Each rollout gets an isolated browser context opened on the task's start site, restricted to
the sites that serve the row's ``ui_version``. The policy drives it with tool calls and ends
the episode with a final assistant message; ``verify`` scores that message with TimeWarp's
verifiers (``scoring.py``) and releases the browser.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import SESSION_LOST
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.reward_profile import compute_subset_metrics, highest_k_metrics
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.timewarp.browser import (
    BrowserSession,
    SessionLimits,
    SharedBrowser,
    ToolInputError,
    describe_error,
    origin_of,
)
from resources_servers.timewarp.scoring import (
    DETERMINISTIC_SCORERS,
    build_judge_messages,
    judge_references,
    parse_judge_verdict,
    strip_thinking,
    validate_eval_types,
)
from resources_servers.timewarp.setup_chromium import ensure_chromium


logger = logging.getLogger(__name__)

UI_VERSIONS = range(1, 7)
# What the policy calls each site, mapped to TimeWarp's site keys.
SITE_NAMES: Dict[str, str] = {"wiki": "wiki", "news": "news", "shop": "webshop"}
JUDGE_NOT_CONFIGURED = "timewarp:judge_not_configured"


class SiteUrls(BaseModel):
    """Base URLs of the three sites serving one UI version (upstream ``TW_WIKI``/``TW_NEWS``/``TW_WEBSHOP``)."""

    wiki: str
    news: str
    webshop: str


class TimewarpResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # UI version (1-6) -> where that version's sites are served. Rows for an unlisted version fail at seed.
    site_urls: Dict[int, SiteUrls] = Field(default_factory=dict)
    headless: bool = True
    # Observation budget; longer pages are split into parts the policy pages through with `observe`.
    max_observation_chars: int = 12000
    action_timeout_ms: int = 5000
    navigation_timeout_ms: int = 30000
    # Only the llm_judge tasks (one per split) need a judge. Without one they are masked, not scored 0.
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class TimewarpSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    ui_version: int
    start_site: Literal["wiki", "news", "webshop"]


class TimewarpTaskSpec(BaseModel):
    """A task's TimeWarp ``eval`` block: which verifiers run (AND-ed) and their references."""

    model_config = ConfigDict(extra="allow")

    eval_types: List[str]
    reference_answers: Dict[str, Any]


class TimewarpVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: TimewarpTaskSpec
    # The task question; the LLM judge sees it.
    intent: str
    ui_version: Optional[int] = None
    sites: List[str] = Field(default_factory=list)


class TimewarpVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: str
    # Subset keys for the per-version and per-site breakdowns in aggregate metrics.
    timewarp_version: Optional[str] = None
    timewarp_site: Optional[str] = None


# Fields the verify response owns. The request allows extras, so a caller may send them too; they are
# dropped from the spread instead of being passed twice.
_RESPONSE_OWNED_FIELDS = frozenset(TimewarpVerifyResponse.model_fields) - frozenset(BaseVerifyRequest.model_fields)


class JudgeNotConfiguredError(RuntimeError):
    """An llm_judge task was scored without a judge model server."""


def extract_final_answer(response: NeMoGymResponse) -> str:
    """The assistant message that ended the episode, without reasoning.

    Upstream, the agent answers through ``send_msg_to_user``; here the answer is the reply the
    policy gives instead of calling another tool. An episode whose last step was a tool call
    (for instance one stopped by ``max_steps``) has no answer.
    """
    texts: List[str] = []
    for item in reversed(response.output):
        if item.type == "reasoning":
            continue
        if item.type != "message" or getattr(item, "role", None) != "assistant":
            break
        texts.append("".join(part.text for part in item.content if part.type == "output_text"))
    return strip_thinking("\n".join(reversed(texts)))


def _site_label(sites: List[str]) -> Optional[str]:
    if not sites:
        return None
    return sites[0] if len(sites) == 1 else "multi"


class TimewarpVerifier:
    """Scores the final answer against the task's TimeWarp spec. Needs no browser."""

    async def verify(self, body: TimewarpVerifyRequest) -> TimewarpVerifyResponse:
        spec = body.verifier_metadata
        validate_eval_types(spec.eval_types)
        answer = extract_final_answer(body.response)
        fields = {k: v for k, v in body.model_dump().items() if k not in _RESPONSE_OWNED_FIELDS} | {
            "extracted_answer": answer,
            "timewarp_version": f"v{body.ui_version}" if body.ui_version is not None else None,
            "timewarp_site": _site_label(body.sites),
        }
        # Upstream scores only an answer the agent actually sent.
        reward = 0.0
        if answer:
            reward = 1.0
            for eval_type in spec.eval_types:
                if eval_type == "llm_judge":
                    try:
                        score = await self._llm_judge(question=body.intent, spec=spec, answer=answer)
                    except JudgeNotConfiguredError as error:
                        return TimewarpVerifyResponse(
                            **fields,
                            reward=0.0,
                            mask_sample=True,
                            failure_kind=JUDGE_NOT_CONFIGURED,
                            failure_reason=str(error),
                        )
                else:
                    score = DETERMINISTIC_SCORERS[eval_type](answer, spec.reference_answers)
                reward *= score
        return TimewarpVerifyResponse(**fields, reward=reward)

    async def _llm_judge(self, *, question: str, spec: TimewarpTaskSpec, answer: str) -> float:
        """1.0 if the judge accepts the answer against any ``fuzzy_match`` gold."""
        for reference in judge_references(spec.reference_answers):
            if reference == "N/A":
                if answer.strip().upper() == "N/A":
                    return 1.0
                continue
            verdict = await self._ask_judge(question=question, reference=reference, answer=answer)
            if parse_judge_verdict(verdict) > 0:
                return 1.0
        return 0.0

    async def _ask_judge(self, *, question: str, reference: str, answer: str) -> str:
        raise JudgeNotConfiguredError(
            "this task is scored by an LLM judge and no judge_model_server is configured; "
            "set judge_model_server on the timewarp resources server"
        )


@dataclass
class _Episode:
    session: BrowserSession
    sites: SiteUrls
    # Set when the browser died mid-episode: the policy could no longer act, so its reward is no measurement.
    lost: bool = False


class ObserveRequest(BaseModel):
    part: int = 1


class OpenSiteRequest(BaseModel):
    site: Literal["wiki", "news", "shop"]


class GotoRequest(BaseModel):
    url: str


class RefRequest(BaseModel):
    ref: str


class FillRequest(BaseModel):
    ref: str
    text: str


class PressRequest(BaseModel):
    ref: str
    key: str


class SelectOptionRequest(BaseModel):
    ref: str
    option: str


class TimewarpResourcesServer(TimewarpVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: TimewarpResourcesServerConfig

    _episodes: Dict[str, _Episode] = {}

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Browser contexts live in this process; another worker would receive tool calls for
        # contexts it does not hold.
        if self.config.num_workers not in (None, 1):
            raise ValueError("timewarp keeps browser sessions in-process and requires num_workers=1")
        ensure_chromium()
        self._episodes = {}
        self._browser = SharedBrowser(headless=self.config.headless)
        self._limits = SessionLimits(
            max_observation_chars=self.config.max_observation_chars,
            action_timeout_ms=self.config.action_timeout_ms,
            navigation_timeout_ms=self.config.navigation_timeout_ms,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for tool in (
            "observe",
            "open_site",
            "goto",
            "click",
            "fill",
            "press",
            "select_option",
            "go_back",
            "go_forward",
        ):
            # Plain text, so the policy reads the snapshot's lines rather than an escaped JSON string.
            app.post(f"/{tool}", response_class=PlainTextResponse)(getattr(self, tool))

        main_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            async with main_lifespan(app) as maybe_state:
                try:
                    yield maybe_state
                finally:
                    await self._browser.close()

        app.router.lifespan_context = lifespan
        return app

    # ----- lifecycle --------------------------------------------------------------------- #
    async def seed_session(self, request: Request, body: TimewarpSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        # A retried rollout re-seeds the same cookie session; never leak the old context.
        await self._release(session_id)
        sites = self.config.site_urls.get(body.ui_version)
        if sites is None:
            raise ValueError(
                f"no site_urls configured for ui_version {body.ui_version}; "
                f"configured versions: {sorted(self.config.site_urls)}"
            )
        session = await BrowserSession.open(
            await self._browser.get(),
            start_url=getattr(sites, body.start_site),
            allowed_origins=frozenset(origin_of(url) for url in (sites.wiki, sites.news, sites.webshop)),
            limits=self._limits,
        )
        self._episodes[session_id] = _Episode(session=session, sites=sites)
        return BaseSeedSessionResponse()

    async def verify(self, request: Request, body: TimewarpVerifyRequest) -> TimewarpVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        episode = self._episodes.get(session_id)
        try:
            response = await super().verify(body)
        finally:
            await self._release(session_id)
        if episode is not None and episode.lost:
            return response.model_copy(
                update={
                    "mask_sample": True,
                    "failure_kind": SESSION_LOST,
                    "failure_reason": "the browser closed during the episode, so the policy could not keep browsing",
                }
            )
        return response

    async def _release(self, session_id: str) -> None:
        episode = self._episodes.pop(session_id, None)
        if episode is None:
            return
        try:
            await episode.session.close()
        except Exception:
            logger.warning("could not close the browser context for session %s", session_id, exc_info=True)

    async def _ask_judge(self, *, question: str, reference: str, answer: str) -> str:
        if self.config.judge_model_server is None:
            return await super()._ask_judge(question=question, reference=reference, answer=answer)
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(**message)
            for message in build_judge_messages(question=question, reference=reference, answer=answer)
        ]
        response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        return strip_thinking(response.output_text or "")

    # ----- tools: errors go back to the policy as text, never raised --------------------- #
    async def _run_tool(self, request: Request, action: Callable[[_Episode], Awaitable[str]]) -> str:
        episode = self._episodes.get(request.session[SESSION_ID_KEY])
        if episode is None:
            return "Error: this rollout has no browser session; seed_session must be called first."
        try:
            return await action(episode)
        except ToolInputError as error:
            return f"Error: {error}"
        except Exception as error:
            if not episode.session.is_alive():
                logger.warning("browser session lost mid-episode", exc_info=True)
                episode.lost = True
                return "Error: the browser session was lost; this episode cannot continue."
            logger.debug("tool call failed", exc_info=True)
            return f"Error: {describe_error(error)}"

    async def observe(self, request: Request, body: ObserveRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.observe(body.part))

    async def open_site(self, request: Request, body: OpenSiteRequest) -> str:
        return await self._run_tool(
            request, lambda episode: episode.session.goto(getattr(episode.sites, SITE_NAMES[body.site]))
        )

    async def goto(self, request: Request, body: GotoRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.goto(body.url))

    async def click(self, request: Request, body: RefRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.click(body.ref))

    async def fill(self, request: Request, body: FillRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.fill(body.ref, body.text))

    async def press(self, request: Request, body: PressRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.press(body.ref, body.key))

    async def select_option(self, request: Request, body: SelectOptionRequest) -> str:
        return await self._run_tool(request, lambda episode: episode.session.select_option(body.ref, body.option))

    async def go_back(self, request: Request) -> str:
        return await self._run_tool(request, lambda episode: episode.session.go_back())

    async def go_forward(self, request: Request) -> str:
        return await self._run_tool(request, lambda episode: episode.session.go_forward())

    # ----- metrics ----------------------------------------------------------------------- #
    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Success rates per UI version (``v1``..``v6``) and per site (``wiki``, ``news``, ``webshop``, ``multi``)."""
        metrics: Dict[str, Any] = {}
        for subset_key in ("timewarp_version", "timewarp_site"):
            metrics.update(compute_subset_metrics(tasks, subset_key))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key_metrics = super().get_key_metrics(agent_metrics)
        for version in UI_VERSIONS:
            key_metrics.update(
                highest_k_metrics(agent_metrics, f"v{version}/pass@1[avg-of-{{k}}]", score_names=["accuracy"])
            )
        return key_metrics


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=TimewarpVerifier,
    request_model=TimewarpVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    TimewarpResourcesServer.run_webserver()
