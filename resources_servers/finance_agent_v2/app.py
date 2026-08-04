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
"""
Finance Agent v2 (FABv2) Resource Server.

Tools-only reuse of Vals's official finance-agent-v2 benchmark: this server
imports the upstream ``finance_agent.tools.*`` ``Tool`` classes directly (no
reimplementation) and exposes each as an HTTP endpoint, so the existing
nemo-gym ``finance_agent`` agent loop can drive them. The public FABv2 release
ships no official grader, so scoring uses **our own** approximation: the
``/verify`` endpoint reuses the v1 ``[[0]]/[[1]]/[[2]]`` judge from
``resources_servers/finance_sec_search``. (The Vals private rubric grader is
deliberately not reproduced here — it derives from privately licensed material.)

Upstream tool surface (``finance_agent.tools``):
  - web_search (TavilyWebSearch)        — needs Tavily API key
  - edgar_search (EDGARSearch)          — needs sec-api.io key
  - parse_html_page (ParseHtmlPage)     — writes to per-session data storage
  - retrieve_information (RetrieveInformation) — LLM over stored docs
  - calculator (Calculator)             — no key (simpleeval)
  - price_history (PriceHistory)        — needs Tiingo pricing key
  - submit_final_result (SubmitFinalResult)

Each upstream tool implements ``async execute(args, state, logger) -> ToolOutput``
and shares a per-session ``state`` dict (parse_html_page writes, retrieve_information
reads), which this server scopes by HTTP session cookie.
"""

import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml
from fastapi import Body, FastAPI

# Upstream Vals finance-agent-v2 tool classes (installed via requirements.txt).
from finance_agent.tools import (
    Calculator,
    EDGARSearch,
    ParseHtmlPage,
    PriceHistory,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
)
from pydantic import BaseModel, Field
from starlette.requests import Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json


# Local cache layer. Support both package import (tests:
# resources_servers.finance_agent_v2.app) and flat script execution (the nemo-gym
# entrypoint runs app.py directly, so relative imports would fail).
try:
    from .cache import ToolCache
    from .cached_tools import (
        CachedEDGARSearch,
        CachedParseHtmlPage,
        CachedPriceHistory,
        SecFilingSearch,
    )
except ImportError:  # pragma: no cover - exercised only under flat entrypoint execution
    from cached_tools import (
        CachedEDGARSearch,
        CachedParseHtmlPage,
        CachedPriceHistory,
        SecFilingSearch,
    )

    from cache import ToolCache

logger = logging.getLogger(__name__)


class FinanceAgentV2ResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Finance Agent v2 resource server."""

    # --- Tool API keys (external services the upstream tools call) -----------
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for the web_search tool.")
    sec_api_key: Optional[str] = Field(default=None, description="sec-api.io API key for the edgar_search tool.")
    pricing_data_api_key: Optional[str] = Field(default=None, description="Tiingo API key for the price_history tool.")

    # --- Retrieval model (powers retrieve_information) -----------------------
    retrieval_model_server: Optional[ModelServerRef] = Field(
        default=None, description="Model server for retrieve_information LLM calls."
    )
    retrieval_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None, description="Parameters for retrieval model requests (temperature, top_p, etc.)."
    )
    retrieval_system_prompt: Optional[str] = Field(
        default=None,
        description="Inline retrieval system prompt. Takes priority over retrieval_system_prompt_fpath.",
    )
    retrieval_system_prompt_fpath: str = Field(
        default="prompt_templates/finance_agent_v2_retrieval.yaml",
        description="Fallback file path for retrieval system prompt.",
    )
    retrieval_max_output_tokens: Optional[int] = Field(
        default=None,
        description="Max output tokens for retrieve_information LLM calls. None leaves it unset "
        "so the call inherits the full generation budget (eval); set an int to cap it (training).",
    )

    # --- Judge model (powers /verify scoring, path A) ------------------------
    judge_model_server: Optional[ModelServerRef] = Field(
        default=None, description="Reference to the judge model server."
    )
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = Field(
        default=None, description="Parameters for judge model requests."
    )
    rubric_judge_prompt_template: Optional[str] = Field(
        default=None,
        description="Inline rubric judge prompt template. Takes priority over "
        "rubric_judge_prompt_template_fpath. Supports {question}, {generated_answer}, "
        "{criterion} placeholders and grades one criterion per call.",
    )
    rubric_judge_prompt_template_fpath: str = Field(
        default="prompt_templates/finance_agent_v2_rubric_judge.yaml",
        description="Fallback file path for the rubric judge prompt template.",
    )
    judge_call_timeout: Optional[float] = Field(
        default=60.0,
        description="Per-call timeout in seconds for judge LLM requests. None disables.",
    )

    # --- Scoring behavior ----------------------------------------------------
    # Each criterion is judged repeatedly and decided by majority vote. An attempt
    # only counts as a success when the reply parses and yields an integer 0/1;
    # API errors, timeouts, empty output, and unparseable replies are all retried
    # against the same budget.
    judge_required_successes: int = Field(
        default=3,
        description="Successful (parsed) judge verdicts needed per criterion before "
        "voting. Keep odd so the majority cannot tie.",
    )
    judge_max_attempts: int = Field(
        default=10,
        description="Hard cap on judge calls per criterion. Whichever comes first, "
        "judge_required_successes or this, ends the criterion. Falling short leaves "
        "the criterion unresolved (score null).",
    )
    judge_max_concurrency: int = Field(
        default=4,
        description="Max concurrent judge calls per rollout, across criteria. Bounds "
        "wall time (a question has ~9 criteria x 3-10 calls) without stampeding the "
        "judge endpoint. 1 makes judging fully sequential.",
    )

    # --- Tool response caching -----------------------------------------------
    # Disk cache for pricing (Tiingo), edgar_search (sec-api), and sec.gov filing
    # documents. Caches the raw upstream response and reuses the upstream
    # serializer, so a hit is byte-identical to a live call.
    use_cache: bool = Field(
        default=True,
        description="Enable the on-disk tool response cache. True: serve hits and "
        "persist misses (read+write); False: run tools fully live.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Root dir for the tool response cache (used when use_cache is True). "
        "None falls back to ~/.cache/nemo_gym/finance_agent_v2. Relative paths resolve from cwd.",
    )

    # --- SEC tool surface ----------------------------------------------------
    enabled_sec_tools: List[str] = Field(
        default_factory=lambda: ["edgar_search"],
        description="Which SEC tools to expose to the agent. 'edgar_search' "
        "(upstream sec-api.io full-text search, byte-parity with Vals — use for eval); "
        "'sec_filing_search' (data.sec.gov ticker->CIK listing, cheaper/no key, NOT "
        "byte-parity — use for training/SDG). Adding sec_filing_search changes the tool "
        "surface vs Vals.",
    )
    user_agent: str = Field(
        default="Gym-SEC-Search/1.0 (research@nvidia.com)",
        description="User-Agent header for data.sec.gov requests made by sec_filing_search.",
    )
    max_filing_results: int = Field(
        default=200,
        description="Maximum number of filing metadata entries returned by sec_filing_search.",
    )

    # --- Rollout controls ----------------------------------------------------
    max_rollout_time_seconds: Optional[float] = Field(
        default=None,
        description="Per-rollout wall-clock budget in seconds. When exceeded, tool calls return an "
        "error asking the model to submit immediately. None disables.",
    )
    max_end_date: Optional[str] = Field(
        default="2026-03-01",
        description="Informational only. The upstream finance_agent tools self-clamp dates to their "
        "own MAX_END_DATE (2026-03-01); this server does not re-clamp.",
    )


# ============================================================================
# Request / Response models
# ============================================================================


class FinanceAgentV2RunRequest(BaseRunRequest):
    """Run request with question and (optional) expected answer.

    ``expected_answer`` / ``rubric`` are optional to support an unlabeled
    dry-run that exercises the agent + tools path before labels are available.
    """

    question: str = ""
    expected_answer: Optional[str] = None


class FinanceAgentV2VerifyRequest(FinanceAgentV2RunRequest, BaseVerifyRequest):
    """Verify request for Finance Agent v2 tasks."""

    # The scoring input: every criterion is judged separately (see verify()).
    rubric: Optional[str] = Field(
        default=None,
        description="JSON string of the dataset's rubric criteria — the scoring input. "
        "Each entry has 'criteria' (the claim to grade) and 'operator'. Absent/empty "
        "means an unlabeled dry run, which cannot be scored.",
    )


class RubricJudgement(BaseModel):
    """Per-criterion judging record: the verdict plus everything needed to audit it.

    ``score`` is None when the criterion never produced ``judge_required_successes``
    parsable verdicts within ``judge_max_attempts``. That is a judge failure, not a
    "not met" — the enclosing question's reward drops to 0.0 (``BaseVerifyResponse.reward``
    is a non-optional float) but ``judge_error`` is set so those rows stay filterable.
    """

    criteria: str
    operator: Optional[str] = None
    score: Optional[int] = None
    # Successful verdicts in call order. Length == judge_required_successes unless
    # the criterion is unresolved.
    votes: List[int] = Field(default_factory=list)
    votes_for: int = 0
    votes_against: int = 0
    # False when the judge contradicted itself across votes. Aggregated into a
    # disagreement rate so judge flakiness is trackable over time.
    unanimous: Optional[bool] = None
    attempts_used: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    # One entry per successful vote, in the same order as ``votes``.
    evidence: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class FinanceAgentV2VerifyResponse(BaseVerifyResponse):
    """Verify response for Finance Agent v2 tasks."""

    expected_answer: Optional[str] = None
    # Full per-criterion record — the debugging trail and the disagreement signal.
    rubric_judgements: Optional[List[RubricJudgement]] = None
    rubric_total: Optional[int] = None
    rubric_passed: Optional[int] = None
    rubric_unresolved: Optional[int] = None
    # Partial credit, reported alongside the all-or-nothing reward so a 0.0 that
    # missed one criterion is distinguishable from one that missed everything.
    rubric_fraction: Optional[float] = None
    rubric_all_pass: Optional[bool] = None
    # Set when scoring could not be completed (no rubric, or any criterion left
    # unresolved). Distinguishes a *judge failure* — where reward 0.0 is not a
    # meaningful verdict and should be filtered — from a genuine miss.
    judge_error: Optional[str] = None


# ============================================================================
# Retrieval LLM shim
# ============================================================================


class _NemoGymRetrievalLLM:
    """Duck-typed ``model_library.base.LLM`` substitute for RetrieveInformation.

    The upstream ``RetrieveInformation`` tool only calls ``await llm.query(prompt)``
    and reads ``.output_text_str`` / ``.metadata`` off the result, so we avoid
    pulling in model_library's registry/LLM machinery and instead route the call
    through nemo-gym's configured retrieval model server.
    """

    def __init__(self, server: "FinanceAgentV2ResourcesServer"):
        self._server = server

    async def query(self, prompt: str) -> SimpleNamespace:
        return await self._server._run_retrieval(prompt)


# ============================================================================
# Resource server
# ============================================================================


class FinanceAgentV2ResourcesServer(SimpleResourcesServer):
    """Exposes the upstream Vals finance-agent-v2 tools as HTTP endpoints."""

    config: FinanceAgentV2ResourcesServerConfig

    # Tool name -> upstream Tool instance (None when the tool is unavailable,
    # e.g. a required API key was not configured).
    _tools: Dict[str, Any]

    def model_post_init(self, context):
        # session_id -> {key -> stored text}; shared `state` dict the upstream
        # parse_html_page / retrieve_information tools read and write.
        self._data_storage: Dict[str, Dict[str, str]] = {}
        self._session_start_times: Dict[str, float] = {}

        # Shared disk cache for pricing / edgar / SEC docs (disabled when use_cache is False).
        self._cache = ToolCache(self.config.cache_dir, use_cache=self.config.use_cache)
        if self._cache.enabled:
            logger.info("Tool response cache enabled at %s", self._cache.root)

        # Retrieval system prompt (inline takes priority over file).
        if self.config.retrieval_system_prompt:
            self._retrieval_system_prompt = self.config.retrieval_system_prompt.strip()
        else:
            with open(self.config.retrieval_system_prompt_fpath, "r") as f:
                self._retrieval_system_prompt = yaml.safe_load(f)["retrieval_system_prompt"].strip()

        # Rubric judge prompt: rendered once per criterion by verify().
        if self.config.rubric_judge_prompt_template:
            self._rubric_judge_prompt_template = self.config.rubric_judge_prompt_template.strip()
        else:
            with open(self.config.rubric_judge_prompt_template_fpath, "r") as f:
                self._rubric_judge_prompt_template = yaml.safe_load(f)["rubric_judge_prompt_template"].strip()

        self._tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------
    def _build_tools(self) -> Dict[str, Any]:
        """Instantiate upstream Vals tools, skipping any whose key is missing.

        Tools requiring an unavailable key (or model server) are registered as
        ``None`` so their endpoint returns a helpful "unavailable" error rather
        than failing to start the server.
        """
        tools: Dict[str, Any] = {}
        cache = self._cache

        # No-key tools: always available. parse_html_page is cached (sec.gov docs
        # only) when the cache is on; behavior/output is otherwise identical.
        tools["calculator"] = Calculator()
        tools["parse_html_page"] = CachedParseHtmlPage(cache) if cache.enabled else ParseHtmlPage()
        tools["submit_final_result"] = SubmitFinalResult()

        # web_search (Tavily). Gate on the configured key (like edgar_search /
        # price_history below and the V1 finance_sec_search server) so availability is
        # deterministic. Upstream TavilyWebSearch falls back to os.getenv("TAVILY_API_KEY"),
        # which would otherwise make this env-dependent; the key still flows from the
        # shell via the config's ${oc.env:TAVILY_API_KEY} resolver, so behavior is
        # functionally identical to Vals when a key is present.
        if self.config.tavily_api_key:
            tools["web_search"] = self._try_build("web_search", lambda: TavilyWebSearch(self.config.tavily_api_key))
        else:
            logger.info("No tavily_api_key configured — web_search will be unavailable")
            tools["web_search"] = None

        # edgar_search (sec-api.io) — only exposed when selected (default: eval).
        if "edgar_search" in self.config.enabled_sec_tools:
            if self.config.sec_api_key:
                tools["edgar_search"] = self._try_build(
                    "edgar_search",
                    lambda: (
                        CachedEDGARSearch(sec_api_key=self.config.sec_api_key, cache=cache)
                        if cache.enabled
                        else EDGARSearch(sec_api_key=self.config.sec_api_key)
                    ),
                )
            else:
                logger.info("No sec_api_key configured — edgar_search will be unavailable")
                tools["edgar_search"] = None

        # sec_filing_search (data.sec.gov) — training/SDG; no key required. Not
        # byte-parity with Vals, so it is only exposed when explicitly selected.
        if "sec_filing_search" in self.config.enabled_sec_tools:
            tools["sec_filing_search"] = self._try_build(
                "sec_filing_search",
                lambda: SecFilingSearch(
                    cache=cache,
                    user_agent=self.config.user_agent,
                    max_filing_results=self.config.max_filing_results,
                ),
            )

        # price_history (Tiingo).
        if self.config.pricing_data_api_key:
            tools["price_history"] = self._try_build(
                "price_history",
                lambda: (
                    CachedPriceHistory(self.config.pricing_data_api_key, cache)
                    if cache.enabled
                    else PriceHistory(self.config.pricing_data_api_key)
                ),
            )
        else:
            logger.info("No pricing_data_api_key configured — price_history will be unavailable")
            tools["price_history"] = None

        # retrieve_information (LLM over stored docs).
        if self.config.retrieval_model_server:
            tools["retrieve_information"] = RetrieveInformation(llm=_NemoGymRetrievalLLM(self))
        else:
            logger.info("No retrieval_model_server configured — retrieve_information will be unavailable")
            tools["retrieve_information"] = None

        available = sorted(name for name, tool in tools.items() if tool is not None)
        logger.info("Finance Agent v2 tools available: %s", ", ".join(available))
        return tools

    @staticmethod
    def _try_build(name: str, factory) -> Any:
        try:
            return factory()
        except Exception as e:  # noqa: BLE001 — missing key / init failure -> tool unavailable
            logger.warning("Tool '%s' unavailable: %s: %s", name, type(e).__name__, e)
            return None

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def _get_session_storage(self, session_id: str) -> Dict[str, str]:
        if session_id not in self._data_storage:
            self._data_storage[session_id] = {}
        return self._data_storage[session_id]

    def _check_time_budget(self, session_id: str) -> Optional[str]:
        """Return an error message if the rollout exceeded its time budget, else None."""
        if not self.config.max_rollout_time_seconds:
            return None
        start = self._session_start_times.get(session_id)
        if start is None:
            return None
        elapsed = time.monotonic() - start
        if elapsed > self.config.max_rollout_time_seconds:
            logger.warning(
                "Session %s exceeded time budget (%.0fs > %.0fs)",
                session_id,
                elapsed,
                self.config.max_rollout_time_seconds,
            )
            return json.dumps(
                {
                    "error": f"Time budget exhausted ({elapsed:.0f}s / {self.config.max_rollout_time_seconds:.0f}s). "
                    "No further tool calls will be executed. Call submit_final_result immediately with your best answer."
                }
            )
        return None

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        """Reset per-question data storage for this session."""
        session_id = request.session[SESSION_ID_KEY]
        self._data_storage[session_id] = {}
        self._session_start_times[session_id] = time.monotonic()
        logger.debug("seed_session: reset data storage for session %s", session_id)
        if len(self._data_storage) > 128:
            logger.warning(
                "data_storage has %d active sessions — possible leak (verify cleanup failing?)",
                len(self._data_storage),
            )
        return await super().seed_session(body)

    # ------------------------------------------------------------------
    # Webserver wiring
    # ------------------------------------------------------------------
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        for tool_name in self._tools:
            app.post(f"/{tool_name}")(self._make_tool_handler(tool_name))

        available = ", ".join(sorted(self._tools))

        @app.post("/{tool_name}")
        async def handle_unknown_tool(tool_name: str):
            return {
                "results": json.dumps({"error": f"Tool '{tool_name}' does not exist. Available tools: {available}"})
            }

        return app

    def _make_tool_handler(self, tool_name: str):
        async def handler(request: Request, body: dict = Body(default={})):
            return await self._dispatch_tool(tool_name, request, body)

        return handler

    async def _dispatch_tool(self, tool_name: str, request: Request, args: dict) -> Dict[str, str]:
        session_id = request.session.get(SESSION_ID_KEY, "")

        if timeout_msg := self._check_time_budget(session_id):
            return {"results": timeout_msg}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "results": json.dumps(
                    {
                        "error": f"Tool '{tool_name}' is not available (required API key or model server not configured)."
                    }
                )
            }

        state = self._get_session_storage(session_id)
        if not isinstance(args, dict):
            args = {}

        try:
            output = await tool.execute(args, state, logger)
        except Exception as e:  # noqa: BLE001 — surface as a tool error, never 500 the agent
            logger.warning("Tool '%s' raised %s: %s", tool_name, type(e).__name__, e)
            return {"results": json.dumps({"error": f"{type(e).__name__}: {e}"})}

        return {"results": output.output}

    # ------------------------------------------------------------------
    # Retrieval LLM call (used by RetrieveInformation via the shim)
    # ------------------------------------------------------------------
    async def _run_retrieval(self, prompt: str) -> SimpleNamespace:
        """Send an already-substituted retrieval prompt to the retrieval model server.

        The upstream RetrieveInformation tool performs the {{key}} substitution
        itself before calling this, so ``prompt`` is the final user content.
        """
        if not self.config.retrieval_model_server:
            raise RuntimeError("retrieve_information is not configured (retrieval_model_server is unset).")

        retrieval_params = (
            self.config.retrieval_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        retrieval_params.input = [
            NeMoGymEasyInputMessage(role="system", content=self._retrieval_system_prompt),
            NeMoGymEasyInputMessage(role="user", content=prompt),
        ]
        if retrieval_params.max_output_tokens is None:
            retrieval_params.max_output_tokens = self.config.retrieval_max_output_tokens

        llm_response = await self.server_client.post(
            server_name=self.config.retrieval_model_server.name,
            url_path="/v1/responses",
            json=retrieval_params,
        )
        if not llm_response.ok:
            body_text = (await llm_response.text())[:500]
            raise RuntimeError(f"Retrieval LLM HTTP {llm_response.status}: {body_text}")

        llm_response_obj = NeMoGymResponse.model_validate(await get_response_json(llm_response))

        result_text = ""
        for output_item in llm_response_obj.output:
            if getattr(output_item, "type", None) == "message":
                for content_item in getattr(output_item, "content", []):
                    if getattr(content_item, "type", None) == "output_text":
                        result_text += getattr(content_item, "text", "")

        if not result_text:
            diagnostic_parts: List[str] = []
            incomplete_details = getattr(llm_response_obj, "incomplete_details", None)
            if incomplete_details is not None and getattr(incomplete_details, "reason", None):
                diagnostic_parts.append(f"incomplete_details.reason={incomplete_details.reason}")
            status = getattr(llm_response_obj, "status", None)
            if status:
                diagnostic_parts.append(f"status={status}")
            diagnostic = (" (" + ", ".join(diagnostic_parts) + ")") if diagnostic_parts else ""
            raise RuntimeError(f"Retrieval LLM returned no output.{diagnostic}")

        return SimpleNamespace(output_text_str=result_text, metadata={})

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_rubric(rubric: Optional[str]) -> List[Dict[str, Any]]:
        """Parse the dataset's rubric JSON string into criteria dicts.

        Returns [] for anything unusable (absent, malformed, not a list, or no
        entry carrying a non-empty ``criteria``), which verify() treats as an
        unscorable dry run rather than a zero.
        """
        if not rubric:
            return []
        try:
            parsed = json.loads(rubric)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Rubric is not valid JSON — cannot score")
            return []
        if not isinstance(parsed, list):
            logger.warning("Rubric JSON is %s, expected a list — cannot score", type(parsed).__name__)
            return []

        criteria: List[Dict[str, Any]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            text = entry.get("criteria")
            if isinstance(text, str) and text.strip():
                criteria.append({"criteria": text.strip(), "operator": entry.get("operator")})
        return criteria

    @staticmethod
    def _extract_score(judge_text: str) -> Optional[int]:
        """Pull the binary ``score`` out of a judge reply, or None if unusable.

        Accepts a bare JSON object or one fenced in markdown, and scans candidate
        objects from the end so a worked example quoted mid-reasoning cannot
        outrank the judge's actual verdict. Only integer 0/1 counts — a string
        ``"pass"`` or a float is a parse failure, because silently coercing those
        would let a confused judge vote.
        """
        if not judge_text or not judge_text.strip():
            return None

        candidates = re.findall(r"\{.*?\}", judge_text, re.DOTALL)
        for blob in reversed(candidates):
            try:
                payload = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict) or "score" not in payload:
                continue
            score = payload["score"]
            # bool is an int subclass in Python; True/False is not a verdict.
            if isinstance(score, int) and not isinstance(score, bool) and score in (0, 1):
                return score
            return None
        return None

    @staticmethod
    def _judge_reply_text(judge_response: NeMoGymResponse) -> str:
        """Concatenate the visible text of a judge response."""
        text = ""
        for item in judge_response.output:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                chunk = getattr(part, "text", None)
                if isinstance(chunk, str):
                    text += chunk
        return text

    async def _judge_attempt(self, prompt: str, attempt: int) -> tuple[Optional[int], Optional[str], str]:
        """Make one judge call. Returns (score, error, raw_reply_text).

        ``score`` is None when the call failed or the reply could not be parsed;
        ``error`` describes which. Callers retry either case.
        """
        params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]

        # Escalate after a failed attempt. Reasoning judge models can spend the
        # entire max_output_tokens budget on hidden reasoning and emit no visible
        # text, so retrying with identical params reproduces the empty output.
        # Give later attempts a real chance: raise the output budget and lower the
        # reasoning effort so the model actually emits the JSON verdict.
        if attempt > 0:
            if getattr(params, "max_output_tokens", None):
                params.max_output_tokens = min(max(params.max_output_tokens, 4096) * 2, 32768)
            if getattr(params, "reasoning", None) is not None:
                try:
                    params.reasoning.effort = "low"
                except Exception:  # noqa: BLE001
                    pass

        try:
            response = await asyncio.wait_for(
                self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                ),
                timeout=self.config.judge_call_timeout,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except Exception as e:  # noqa: BLE001
            return None, f"judge call failed: {type(e).__name__}: {e}", ""

        reply = self._judge_reply_text(judge_response)
        score = self._extract_score(reply)
        if score is None:
            error = (
                "judge returned empty output (likely token budget exhausted by reasoning)"
                if not reply.strip()
                else "judge reply had no integer 0/1 score"
            )
            return None, error, reply
        return score, None, reply

    async def _judge_criterion(
        self, question: str, generated_answer: str, criterion: Dict[str, Any]
    ) -> RubricJudgement:
        """Judge one criterion, voting over repeated calls.

        Calls the judge until ``judge_required_successes`` replies parse to a 0/1
        or ``judge_max_attempts`` is spent, whichever comes first. API errors,
        timeouts, and unparseable replies all consume an attempt and are retried.
        Falling short of the required successes leaves ``score`` None (unresolved),
        which is a judge failure rather than a "not met".
        """
        prompt = (
            self._rubric_judge_prompt_template.replace("{question}", question)
            .replace("{generated_answer}", generated_answer)
            .replace("{criterion}", criterion["criteria"])
        )

        record = RubricJudgement(criteria=criterion["criteria"], operator=criterion.get("operator"))
        needed = max(1, self.config.judge_required_successes)
        last_error: Optional[str] = None

        max_attempts = max(1, self.config.judge_max_attempts)
        for attempt in range(max_attempts):
            if len(record.votes) >= needed:
                break
            record.attempts_used += 1
            score, error, reply = await self._judge_attempt(prompt, attempt)

            if score is None:
                last_error = error
                if reply:
                    record.parse_failures += 1
                else:
                    record.api_failures += 1
                logger.warning(
                    "Rubric judge attempt %d failed (%s) for criterion: %.80s",
                    record.attempts_used,
                    error,
                    criterion["criteria"],
                )
                # Back off between failures only, capped so a flaky judge cannot
                # stretch one criterion past the rollout budget, and skipped on the
                # last attempt where there is nothing left to wait for.
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                continue

            record.votes.append(score)
            payload = self._extract_json_payload(reply)
            record.evidence.append(str(payload.get("extracted_evidence", ""))[:1000])
            record.reasons.append(str(payload.get("reason", ""))[:1000])

        record.votes_for = sum(record.votes)
        record.votes_against = len(record.votes) - record.votes_for

        if len(record.votes) < needed:
            record.error = last_error or f"only {len(record.votes)}/{needed} successful judge verdicts"
            logger.error(
                "Criterion unresolved after %d attempts (%s): %.80s",
                record.attempts_used,
                record.error,
                criterion["criteria"],
            )
            return record

        record.score = 1 if record.votes_for * 2 > len(record.votes) else 0
        record.unanimous = record.votes_for == 0 or record.votes_against == 0
        if not record.unanimous:
            logger.info(
                "Judge disagreed %d-%d on criterion: %.80s",
                record.votes_for,
                record.votes_against,
                criterion["criteria"],
            )
        return record

    @staticmethod
    def _extract_json_payload(judge_text: str) -> Dict[str, Any]:
        """Return the scored JSON object from a reply, or {} if not found.

        Mirrors _extract_score's last-object-wins scan so the recorded evidence
        and reason come from the same object the verdict came from.
        """
        for blob in reversed(re.findall(r"\{.*?\}", judge_text or "", re.DOTALL)):
            try:
                payload = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and "score" in payload:
                return payload
        return {}

    async def verify(self, request: Request, body: FinanceAgentV2VerifyRequest) -> FinanceAgentV2VerifyResponse:
        """Score the agent's answer against the dataset's rubric criteria.

        Each criterion is judged independently and decided by majority vote (see
        ``_judge_criterion``). Reward is all-or-nothing: 1.0 only when every
        criterion passes, else 0.0. ``rubric_fraction`` carries partial credit so a
        near miss is distinguishable from a total one, and any unresolved criterion
        sets ``judge_error`` so judge failures never masquerade as low scores.
        """
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id:
            self._data_storage.pop(session_id, None)
            self._session_start_times.pop(session_id, None)

        question = ""
        for msg in body.responses_create_params.input or []:
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    question = content

        generated_answer = ""
        for output_item in reversed(body.response.output):
            if getattr(output_item, "type", None) == "function_call":
                if getattr(output_item, "name", None) == "submit_final_result":
                    try:
                        args = json.loads(getattr(output_item, "arguments", "{}"))
                        generated_answer = args.get("final_result", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

        # No submission is a genuine failure, not a judge failure: the agent had its
        # chances (the loop nudges it until max_steps). Return before spending on the
        # judge.
        if not generated_answer:
            return FinanceAgentV2VerifyResponse(**body.model_dump(), reward=0.0)

        criteria = self._parse_rubric(body.rubric)
        if not criteria:
            return FinanceAgentV2VerifyResponse(
                **body.model_dump(),
                reward=0.0,
                judge_error="no usable rubric criteria — unlabeled dry run, not scorable",
            )

        if not self.config.judge_model_server:
            return FinanceAgentV2VerifyResponse(
                **body.model_dump(),
                reward=0.0,
                rubric_total=len(criteria),
                judge_error="no judge_model_server configured — cannot score the rubric",
            )

        # Criteria are independent, so judge them concurrently under a semaphore:
        # a question carries ~9 criteria x 3-10 calls, which is slow serially but
        # would stampede the judge endpoint unbounded.
        semaphore = asyncio.Semaphore(max(1, self.config.judge_max_concurrency))

        async def judge_one(criterion: Dict[str, Any]) -> RubricJudgement:
            async with semaphore:
                return await self._judge_criterion(question, generated_answer, criterion)

        judgements = await asyncio.gather(*(judge_one(c) for c in criteria))

        total = len(judgements)
        passed = sum(1 for j in judgements if j.score == 1)
        unresolved = sum(1 for j in judgements if j.score is None)
        all_pass = unresolved == 0 and passed == total

        judge_error = None
        if unresolved:
            first = next(j for j in judgements if j.score is None)
            judge_error = f"{unresolved}/{total} criteria unresolved (e.g. {first.error})"

        logger.info(
            "Rubric verdict: %d/%d passed, %d unresolved, reward=%.1f",
            passed,
            total,
            unresolved,
            1.0 if all_pass else 0.0,
        )

        return FinanceAgentV2VerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_pass else 0.0,
            rubric_judgements=list(judgements),
            rubric_total=total,
            rubric_passed=passed,
            rubric_unresolved=unresolved,
            rubric_fraction=(passed / total) if total else None,
            rubric_all_pass=all_pass,
            judge_error=judge_error,
        )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Rubric-level metrics on top of the RewardProfiler baseline.

        The headline reward is all-or-nothing, which hides how close a failing
        answer was and cannot distinguish a genuine miss from a judge that never
        returned a verdict. These fill both gaps:

        - ``mean/rubric_fraction`` is the partial-credit view of the same runs.
        - ``mean/criterion_pass_rate`` pools every criterion so the denominator is
          criteria, not questions, and is not skewed by rubric length.
        - ``mean/judge_disagreement_rate`` is the share of resolved criteria where
          the judge contradicted itself across votes. Track it over time: a rise
          means the judge is getting less decisive on this data, independent of
          model quality.
        - the ``rubric/*`` counters keep judge failures and retry cost visible
          instead of letting them hide inside the score.

        Verify responses arrive as dicts, so ``rubric_judgements`` is read as
        serialized dicts rather than ``RubricJudgement`` instances.
        """
        rollouts = [r for task in tasks for r in task]
        if not rollouts:
            return {}

        fractions = [r["rubric_fraction"] for r in rollouts if r.get("rubric_fraction") is not None]
        all_pass_flags = [bool(r["rubric_all_pass"]) for r in rollouts if r.get("rubric_all_pass") is not None]

        criteria: List[Dict[str, Any]] = []
        for r in rollouts:
            for j in r.get("rubric_judgements") or []:
                if isinstance(j, dict):
                    criteria.append(j)

        resolved = [j for j in criteria if j.get("score") is not None]
        unresolved = len(criteria) - len(resolved)
        # Only resolved criteria carry a meaningful unanimity flag.
        contested = sum(1 for j in resolved if j.get("unanimous") is False)
        judge_error_rollouts = sum(1 for r in rollouts if r.get("judge_error"))
        no_submission = sum(1 for r in rollouts if not r.get("rubric_judgements") and not r.get("judge_error"))

        metrics: Dict[str, Any] = {
            "rubric/rollouts": len(rollouts),
            "rubric/criteria_total": len(criteria),
            "rubric/criteria_resolved": len(resolved),
            "rubric/criteria_unresolved": unresolved,
            "rubric/rollouts_with_judge_error": judge_error_rollouts,
            "rubric/rollouts_without_submission": no_submission,
            "rubric/parse_failures": sum(int(j.get("parse_failures") or 0) for j in criteria),
            "rubric/api_failures": sum(int(j.get("api_failures") or 0) for j in criteria),
        }
        if fractions:
            metrics["mean/rubric_fraction"] = sum(fractions) / len(fractions)
        if all_pass_flags:
            metrics["mean/rubric_all_pass"] = sum(all_pass_flags) / len(all_pass_flags)
        if resolved:
            metrics["mean/criterion_pass_rate"] = sum(1 for j in resolved if j.get("score") == 1) / len(resolved)
            metrics["mean/judge_disagreement_rate"] = contested / len(resolved)
        if criteria:
            metrics["rubric/mean_attempts_per_criterion"] = sum(
                int(j.get("attempts_used") or 0) for j in criteria
            ) / len(criteria)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline metrics: the usual mean/* plus judge-health counters.

        The counters are promoted deliberately — a run whose judge failed on a
        third of its criteria should not be read as a low score, and that is only
        obvious if the failure count sits next to the score.
        """
        key = {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}
        for k in (
            "rubric/criteria_unresolved",
            "rubric/rollouts_with_judge_error",
            "rubric/rollouts_without_submission",
        ):
            if k in agent_metrics:
                key[k] = agent_metrics[k]
        return key


if __name__ == "__main__":
    FinanceAgentV2ResourcesServer.run_webserver()
