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

logger = logging.getLogger(__name__)


class FinanceAgentV2ResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the Finance Agent v2 resource server."""

    # --- Tool API keys (external services the upstream tools call) -----------
    tavily_api_key: Optional[str] = Field(
        default=None, description="Tavily API key for the web_search tool."
    )
    sec_api_key: Optional[str] = Field(
        default=None, description="sec-api.io API key for the edgar_search tool."
    )
    pricing_data_api_key: Optional[str] = Field(
        default=None, description="Tiingo API key for the price_history tool."
    )

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
    judge_prompt_template: Optional[str] = Field(
        default=None,
        description="Inline judge prompt template. Takes priority over judge_prompt_template_fpath. "
        "Supports {question}, {expected_answer}, {generated_answer} placeholders.",
    )
    judge_prompt_template_fpath: str = Field(
        default="prompt_templates/finance_agent_v2_judge.yaml",
        description="Fallback file path for the legacy judge prompt template.",
    )
    judge_call_timeout: Optional[float] = Field(
        default=60.0,
        description="Per-call timeout in seconds for judge LLM requests. None disables.",
    )

    # --- Scoring behavior ----------------------------------------------------
    reward_mode: str = Field(
        default="binary",
        description="How the [[N]] judge rating maps to reward. "
        "'binary': only [[2]] -> 1.0, else 0.0. "
        "'scaled': [[0]] -> 0.0, [[1]] -> 0.5, [[2]] -> 1.0.",
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

    # Carried through from the dataset for completeness/reference only. The public
    # FABv2 release has no official grader, so this is NOT used for reward here —
    # scoring uses our own [[N]] judge (see verify()).
    rubric: Optional[str] = Field(
        default=None,
        description="Reference-only: JSON string of the dataset's rubric criteria. "
        "Not used for scoring.",
    )


class FinanceAgentV2VerifyResponse(BaseVerifyResponse):
    """Verify response for Finance Agent v2 tasks."""

    expected_answer: Optional[str] = None
    judge_rating: Optional[int] = None
    judge_text: Optional[str] = None
    # Set when the judge failed to produce a usable [[N]] verdict (call error, or
    # no rating after all retries). Distinguishes a *judge failure* (reward 0.0 is
    # not meaningful — filter/ignore these) from a genuine [[0]] "incorrect".
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

        # Retrieval system prompt (inline takes priority over file).
        if self.config.retrieval_system_prompt:
            self._retrieval_system_prompt = self.config.retrieval_system_prompt.strip()
        else:
            with open(self.config.retrieval_system_prompt_fpath, "r") as f:
                self._retrieval_system_prompt = yaml.safe_load(f)["retrieval_system_prompt"].strip()

        # Judge prompt (legacy [[0]]/[[1]]/[[2]] mode).
        if self.config.judge_prompt_template:
            self._judge_prompt_template = self.config.judge_prompt_template.strip()
        else:
            with open(self.config.judge_prompt_template_fpath, "r") as f:
                self._judge_prompt_template = yaml.safe_load(f)["judge_prompt_template"].strip()

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

        # No-key tools: always available.
        tools["calculator"] = Calculator()
        tools["parse_html_page"] = ParseHtmlPage()
        tools["submit_final_result"] = SubmitFinalResult()

        # web_search (Tavily). Gate on the configured key (like edgar_search /
        # price_history below and the V1 finance_sec_search server) so availability is
        # deterministic. Upstream TavilyWebSearch falls back to os.getenv("TAVILY_API_KEY"),
        # which would otherwise make this env-dependent; the key still flows from the
        # shell via the config's ${oc.env:TAVILY_API_KEY} resolver, so behavior is
        # functionally identical to Vals when a key is present.
        if self.config.tavily_api_key:
            tools["web_search"] = self._try_build(
                "web_search", lambda: TavilyWebSearch(self.config.tavily_api_key)
            )
        else:
            logger.info("No tavily_api_key configured — web_search will be unavailable")
            tools["web_search"] = None

        # edgar_search (sec-api.io).
        if self.config.sec_api_key:
            tools["edgar_search"] = self._try_build(
                "edgar_search", lambda: EDGARSearch(sec_api_key=self.config.sec_api_key)
            )
        else:
            logger.info("No sec_api_key configured — edgar_search will be unavailable")
            tools["edgar_search"] = None

        # price_history (Tiingo).
        if self.config.pricing_data_api_key:
            tools["price_history"] = self._try_build(
                "price_history", lambda: PriceHistory(self.config.pricing_data_api_key)
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
                session_id, elapsed, self.config.max_rollout_time_seconds,
            )
            return json.dumps({
                "error": f"Time budget exhausted ({elapsed:.0f}s / {self.config.max_rollout_time_seconds:.0f}s). "
                "No further tool calls will be executed. Call submit_final_result immediately with your best answer."
            })
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
                "results": json.dumps(
                    {"error": f"Tool '{tool_name}' does not exist. Available tools: {available}"}
                )
            }

        return app

    def _make_tool_handler(self, tool_name: str):
        async def handler(request: Request, body: dict = Body(default={})):
            return await self._dispatch_tool(tool_name, request, body)

        return handler

    async def _dispatch_tool(self, tool_name: str, request: Request, args: dict) -> Dict[str, str]:
        session_id = request.session.get(SESSION_ID_KEY, "")

        if (timeout_msg := self._check_time_budget(session_id)):
            return {"results": timeout_msg}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "results": json.dumps(
                    {"error": f"Tool '{tool_name}' is not available (required API key or model server not configured)."}
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
    async def verify(self, request: Request, body: FinanceAgentV2VerifyRequest) -> FinanceAgentV2VerifyResponse:
        """Verify the agent's answer (path A: own judge).

        Rating scale (reward depends on config.reward_mode):
            [[2]] = fully correct  -> binary: 1.0 | scaled: 1.0
            [[1]] = partial        -> binary: 0.0 | scaled: 0.5
            [[0]] = incorrect      -> binary: 0.0 | scaled: 0.0
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

        if not generated_answer:
            return FinanceAgentV2VerifyResponse(**body.model_dump(), reward=0.0)

        # Unlabeled dry-run: no expected_answer and no judge -> reward 0.
        if not self.config.judge_model_server:
            if body.expected_answer is None:
                return FinanceAgentV2VerifyResponse(**body.model_dump(), reward=0.0)
            reward = 1.0 if body.expected_answer.lower() in generated_answer.lower() else 0.0
            return FinanceAgentV2VerifyResponse(**body.model_dump(), reward=reward)

        # Legacy [[0]]/[[1]]/[[2]] judge.
        judge_user_prompt = self._judge_prompt_template
        judge_user_prompt = judge_user_prompt.replace("{question}", question)
        judge_user_prompt = judge_user_prompt.replace("{expected_answer}", body.expected_answer or "")
        judge_user_prompt = judge_user_prompt.replace("{generated_answer}", generated_answer)

        judge_params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        judge_params.input = [NeMoGymEasyInputMessage(role="user", content=judge_user_prompt)]

        max_judge_retries = 3
        judge_text = ""
        rating = None
        judge_error = None
        for attempt in range(max_judge_retries):
            # Escalate after a no-verdict attempt. Reasoning judge models can spend
            # the entire max_output_tokens budget on hidden reasoning and emit no
            # visible text, so retrying with identical params reproduces the empty
            # output. Give later attempts a real chance: raise the output budget and
            # lower the reasoning effort so the model actually emits the [[N]] tag.
            attempt_params = judge_params.model_copy(deep=True)
            if attempt > 0:
                if getattr(attempt_params, "max_output_tokens", None):
                    attempt_params.max_output_tokens = min(max(attempt_params.max_output_tokens, 4096) * 2, 32768)
                if getattr(attempt_params, "reasoning", None) is not None:
                    try:
                        attempt_params.reasoning.effort = "low"
                    except Exception:  # noqa: BLE001
                        pass

            try:
                response = await asyncio.wait_for(
                    self.server_client.post(
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/responses",
                        json=attempt_params,
                    ),
                    timeout=self.config.judge_call_timeout,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
            except Exception as e:  # noqa: BLE001
                judge_error = f"judge call failed: {type(e).__name__}: {e}"
                logger.warning("Judge call attempt %d/%d failed: %s: %s", attempt + 1, max_judge_retries, type(e).__name__, e)
                if attempt < max_judge_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.error("Judge model call failed after %d attempts", max_judge_retries)
                # Judge failure: reward 0.0 is not a meaningful verdict — flag it.
                return FinanceAgentV2VerifyResponse(
                    **body.model_dump(), reward=0.0, judge_rating=None,
                    judge_text=judge_text, judge_error=judge_error,
                )

            judge_text = ""
            try:
                last_output = judge_response.output[-1]
                if getattr(last_output, "type", None) == "message":
                    judge_text = getattr(last_output.content[-1], "text", "")
            except Exception:  # noqa: BLE001
                pass

            rating_match = re.search(r"\[\[(\d+)\]\]", judge_text)
            rating = int(rating_match.group(1)) if rating_match else None
            if rating is not None:
                judge_error = None
                break

            judge_error = (
                "judge returned empty output (likely token budget exhausted by reasoning)"
                if not judge_text.strip()
                else "judge returned no [[N]] verdict"
            )
            logger.warning(
                "Judge returned no [[N]] rating (attempt %d/%d). Output: %s",
                attempt + 1, max_judge_retries, judge_text[:200],
            )
            if attempt < max_judge_retries - 1:
                await asyncio.sleep(2**attempt)

        # No usable verdict after all retries: surface as a judge failure rather
        # than silently scoring a (misleading) 0.0.
        if rating is None:
            logger.error("Judge produced no verdict after %d attempts: %s", max_judge_retries, judge_error)
            return FinanceAgentV2VerifyResponse(
                **body.model_dump(), reward=0.0, judge_rating=None,
                judge_text=judge_text, judge_error=judge_error,
            )

        if self.config.reward_mode == "scaled":
            reward = {0: 0.0, 1: 0.5, 2: 1.0}.get(rating, 0.0)
        else:
            reward = 1.0 if rating == 2 else 0.0

        return FinanceAgentV2VerifyResponse(
            **body.model_dump(), reward=reward, judge_rating=rating, judge_text=judge_text
        )


if __name__ == "__main__":
    FinanceAgentV2ResourcesServer.run_webserver()
