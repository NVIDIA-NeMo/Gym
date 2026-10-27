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
"""FrontierFinance resources server.

Runs FrontierFinance queries on the Finance Agent v2 tools, with every dated tool
call capped at the query's ``as_of_date``, and grades answers with the official
FrontierFinance grader (``frontier_finance`` package): its prompt, batching, reply
parsing, judge-panel majority vote and metrics. Only the judge transport differs:
judges are Gym model servers instead of provider SDKs.
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from frontier_finance.grading import Grader
from frontier_finance.judges import Judge
from frontier_finance.metrics import MetricsReport
from frontier_finance.models import EvalItem, ItemResult, Rubric
from pydantic import BaseModel, Field
from starlette.requests import Request

from nemo_gym.base_resources_server import BaseSeedSessionRequest, BaseSeedSessionResponse, BaseVerifyResponse
from nemo_gym.config_types import ModelServerRef
from nemo_gym.failure_kinds import JUDGE_FAILED, VERIFIER_ERROR
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json
from resources_servers.finance_agent_v2.app import (
    FinanceAgentV2ResourcesServer,
    FinanceAgentV2ResourcesServerConfig,
    FinanceAgentV2RunRequest,
    FinanceAgentV2VerifyRequest,
)


logger = logging.getLogger(__name__)

DATE_CAPPED_TOOLS = frozenset({"web_search", "edgar_search", "price_history"})
SUBMIT_TOOL = "submit_final_result"
METRIC_PREFIX = "frontier_finance/"
_MAX_RECORDED_REPLY_CHARS = 20_000

HEADLINE_METRICS = (
    "macro_avg_qualification_rate_on_all_queries",
    "micro_avg_qualification_rate_on_all_queries",
    "macro_avg_qualification_rate_must_have_on_all_queries",
    "micro_avg_qualification_rate_must_have_on_all_queries",
    "macro_avg_qualification_rate_on_success_queries",
    "micro_avg_qualification_rate_on_success_queries",
    "macro_avg_qualification_rate_must_have_on_success_queries",
    "micro_avg_qualification_rate_must_have_on_success_queries",
)


class FrontierFinanceJudgeConfig(BaseModel):
    name: str = Field(description="Label for this judge in rollouts and metrics, e.g. 'claude-sonnet-4-6'.")
    model_server: ModelServerRef
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature; null leaves it to the provider default."
    )


class FrontierFinanceResourcesServerConfig(FinanceAgentV2ResourcesServerConfig):
    judge_panel: List[FrontierFinanceJudgeConfig] = Field(
        description="Ordered judge panel. Each rubric is decided by majority vote of the judges that "
        "succeeded on the query; on a tie the earliest of them wins.",
    )
    max_rubrics_per_call: int = Field(default=30, description="Rubrics graded per judge call.")
    max_json_parse_retries: int = Field(
        default=1, description="Re-prompts with a format reminder after an unparseable judge reply."
    )
    judge_max_output_tokens: int = Field(default=32000, description="Output budget per judge call.")
    judge_max_retries: int = Field(default=2, description="Retries of a judge call that errored.")
    judge_thread_pool_size: int = Field(
        default=256, description="Threads that run the blocking official grader across concurrent verifies."
    )
    max_tool_calls: Optional[int] = Field(
        default=None,
        description="Tool calls allowed per rollout, or null for no limit. Past it, every tool except "
        "submit_final_result returns an error asking the model to submit.",
    )


class FrontierFinanceRubric(BaseModel):
    rubric_id: int
    rubric_text: str
    must_have: bool = False
    rubric_type: Optional[str] = None
    rubric_subtype: Optional[str] = None
    data_source_type: Optional[str] = None
    data_source_subtype: Optional[str] = None
    data_source_detailed: List[str] = Field(default_factory=list)


class FrontierFinanceSeedSessionRequest(BaseSeedSessionRequest):
    as_of_date: Optional[str] = None


class FrontierFinanceRunRequest(FinanceAgentV2RunRequest):
    query_id: str = ""
    as_of_date: Optional[str] = None
    use_cases: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    rubrics: List[FrontierFinanceRubric] = Field(default_factory=list)


class FrontierFinanceVerifyRequest(FrontierFinanceRunRequest, FinanceAgentV2VerifyRequest):
    pass


class FrontierFinanceVerifyResponse(FrontierFinanceVerifyRequest, BaseVerifyResponse):
    """``reward`` is the query's rubric qualification rate (qualified / total rubrics)."""

    rubric_labels: Optional[List[bool]] = None
    judge_labels: Dict[str, Optional[List[bool]]] = Field(default_factory=dict)
    judge_replies: Dict[str, List[str]] = Field(default_factory=dict)
    failed_checks_by_judge: Dict[str, int] = Field(default_factory=dict)
    num_rubrics: int = 0
    num_qualified: int = 0
    num_must_have: int = 0
    num_must_have_qualified: int = 0
    # The official grader's failure_reason: "no_response" or "judge_error".
    grading_failure: Optional[str] = None


def cap_tool_dates(args: Dict[str, Any], as_of_date: str) -> Dict[str, Any]:
    """Return ``args`` with ``end_date`` and ``start_date`` no later than ``as_of_date`` (YYYY-MM-DD)."""
    capped = dict(args)
    end_date = capped.get("end_date")
    if not isinstance(end_date, str) or not end_date or end_date > as_of_date:
        capped["end_date"] = as_of_date
    start_date = capped.get("start_date")
    if isinstance(start_date, str) and start_date > as_of_date:
        capped["start_date"] = as_of_date
    return capped


def extract_final_answer(response: NeMoGymResponse) -> str:
    for output_item in reversed(response.output):
        if getattr(output_item, "type", None) == "function_call" and getattr(output_item, "name", None) == SUBMIT_TOOL:
            try:
                args = json.loads(getattr(output_item, "arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                return ""
            answer = args.get("final_result") if isinstance(args, dict) else None
            return answer if isinstance(answer, str) else ""
    return ""


def item_result_from_rollout(row: Dict[str, Any]) -> ItemResult:
    """Rebuild the official grader's per-query result from a verify response."""
    rubrics = [Rubric.from_dict(r) for r in row.get("rubrics") or []]
    failure = row.get("grading_failure")
    return ItemResult(
        query_id=str(row.get("query_id") or ""),
        rubrics=rubrics,
        labels=[] if failure else list(row.get("rubric_labels") or []),
        failed=bool(failure),
        failure_reason=failure,
        failed_checks_by_judge=dict(row.get("failed_checks_by_judge") or {}),
    )


def _use_case_breakdown(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    totals: Dict[str, int] = {}
    qualified: Dict[str, int] = {}
    for row in rows:
        if row.get("grading_failure"):
            continue
        for use_case in row.get("use_cases") or ["(unspecified)"]:
            totals[use_case] = totals.get(use_case, 0) + int(row.get("num_rubrics") or 0)
            qualified[use_case] = qualified.get(use_case, 0) + int(row.get("num_qualified") or 0)
    return {
        use_case: {
            "num_rubrics": totals[use_case],
            "num_qualified": qualified[use_case],
            "qualification_rate": qualified[use_case] / totals[use_case] if totals[use_case] else 0.0,
        }
        for use_case in sorted(totals)
    }


def frontier_finance_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Official FrontierFinance metrics over verify responses, flattened under ``frontier_finance/``."""
    report = MetricsReport([item_result_from_rollout(r) for r in rows]).compute()
    report["breakdown_by_use_case"] = _use_case_breakdown(rows)

    metrics: Dict[str, Any] = {}
    for name, value in report.items():
        if name.startswith("breakdown_by_"):
            group = name.removeprefix("breakdown_by_")
            for key, stats in value.items():
                metrics[f"{METRIC_PREFIX}{group}/{key}"] = stats["qualification_rate"]
        elif name == "failed_criteria_checks_by_judge":
            for judge, count in value.items():
                metrics[f"{METRIC_PREFIX}failed_checks/{judge}"] = count
        else:
            metrics[f"{METRIC_PREFIX}{name}"] = value
    return metrics


class _GymJudge(Judge):
    """Adapter that lets the official (blocking) grader call a Gym judge model server.

    ``complete`` runs on a worker thread and blocks on a coroutine scheduled on the
    server's event loop, so it must never be called from that loop's own thread.
    """

    def __init__(
        self,
        server: "FrontierFinanceResourcesServer",
        judge: FrontierFinanceJudgeConfig,
        loop: asyncio.AbstractEventLoop,
    ):
        config = server.config
        super().__init__(
            judge.name,
            max_tokens=config.judge_max_output_tokens,
            timeout=config.judge_call_timeout or 0,
            max_retries=config.judge_max_retries,
        )
        self._server = server
        self._judge = judge
        self._loop = loop
        self.replies: List[str] = []

    def complete(self, system: str, user: str) -> str:
        future = asyncio.run_coroutine_threadsafe(self._server._judge_complete(self._judge, system, user), self._loop)
        reply = future.result()
        self.replies.append(reply[:_MAX_RECORDED_REPLY_CHARS])
        return reply


class FrontierFinanceResourcesServer(FinanceAgentV2ResourcesServer):
    config: FrontierFinanceResourcesServerConfig

    def model_post_init(self, context):
        super().model_post_init(context)
        if not self.config.judge_panel:
            raise ValueError("judge_panel must list at least one judge.")
        names = [j.name for j in self.config.judge_panel]
        if len(set(names)) != len(names):
            raise ValueError(f"judge_panel names must be unique, got {names}")
        self._as_of_dates: Dict[str, str] = {}
        self._tool_call_counts: Dict[str, int] = {}
        self._judge_executor = ThreadPoolExecutor(
            max_workers=self.config.judge_thread_pool_size, thread_name_prefix="frontier_finance_judge"
        )

    async def seed_session(self, request: Request, body: FrontierFinanceSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        if body.as_of_date:
            date.fromisoformat(body.as_of_date)
            self._as_of_dates[session_id] = body.as_of_date
        else:
            self._as_of_dates.pop(session_id, None)
        self._tool_call_counts[session_id] = 0
        return await super().seed_session(request, body)

    async def _dispatch_tool(self, tool_name: str, request: Request, args: dict) -> Dict[str, str]:
        session_id = request.session.get(SESSION_ID_KEY, "")
        if tool_name != SUBMIT_TOOL and self.config.max_tool_calls is not None:
            used = self._tool_call_counts.get(session_id, 0)
            if used >= self.config.max_tool_calls:
                return {
                    "results": json.dumps(
                        {
                            "error": f"Tool call limit reached ({used} / {self.config.max_tool_calls}). "
                            "No further tool calls will be executed. Call submit_final_result immediately "
                            "with your best answer."
                        }
                    )
                }
            self._tool_call_counts[session_id] = used + 1
        as_of_date = self._as_of_dates.get(session_id)
        if as_of_date and tool_name in DATE_CAPPED_TOOLS and isinstance(args, dict):
            args = cap_tool_dates(args, as_of_date)
        return await super()._dispatch_tool(tool_name, request, args)

    async def _judge_complete(self, judge: FrontierFinanceJudgeConfig, system: str, user: str) -> str:
        params: Dict[str, Any] = {
            "input": [
                NeMoGymEasyInputMessage(role="system", content=system),
                NeMoGymEasyInputMessage(role="user", content=user),
            ],
            "max_output_tokens": self.config.judge_max_output_tokens,
        }
        if judge.temperature is not None:
            params["temperature"] = judge.temperature
        create_params = NeMoGymResponseCreateParamsNonStreaming(**params)

        for attempt in range(max(0, self.config.judge_max_retries)):
            try:
                return await self._judge_call(judge, create_params)
            except Exception as e:  # noqa: BLE001
                logger.warning("Judge %s call failed (%s: %s); retrying", judge.name, type(e).__name__, e)
                await asyncio.sleep(min(2**attempt, 8))
        return await self._judge_call(judge, create_params)

    async def _judge_call(
        self, judge: FrontierFinanceJudgeConfig, create_params: NeMoGymResponseCreateParamsNonStreaming
    ) -> str:
        response = await asyncio.wait_for(
            self.server_client.post(server_name=judge.model_server.name, url_path="/v1/responses", json=create_params),
            timeout=self.config.judge_call_timeout,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
        return self._judge_reply_text(judge_response)

    async def _grade_with_judge(
        self, judge: FrontierFinanceJudgeConfig, item: EvalItem
    ) -> tuple[ItemResult, List[str]]:
        adapter = _GymJudge(self, judge, asyncio.get_running_loop())
        grader = Grader(
            [adapter],
            max_rubrics_per_call=self.config.max_rubrics_per_call,
            max_json_parse_retries=self.config.max_json_parse_retries,
        )
        result = await asyncio.get_running_loop().run_in_executor(self._judge_executor, grader.grade, item)
        return result, adapter.replies

    async def verify(self, request: Request, body: FrontierFinanceVerifyRequest) -> FrontierFinanceVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id:
            self._data_storage.pop(session_id, None)
            self._session_start_times.pop(session_id, None)
            self._as_of_dates.pop(session_id, None)
            self._tool_call_counts.pop(session_id, None)

        rubrics = [Rubric.from_dict(r.model_dump()) for r in body.rubrics]
        num_must_have = sum(1 for r in rubrics if r.must_have)
        base = {**body.model_dump(), "num_rubrics": len(rubrics), "num_must_have": num_must_have}

        if not rubrics:
            return FrontierFinanceVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason="sample has no rubrics",
            )

        answer = extract_final_answer(body.response)
        if not answer:
            return FrontierFinanceVerifyResponse(**base, reward=0.0, grading_failure="no_response")

        item = EvalItem(
            query_id=body.query_id,
            query=body.question,
            query_date=body.as_of_date or "",
            rubrics=rubrics,
            system_response=answer,
        )
        graded = await asyncio.gather(*(self._grade_with_judge(j, item) for j in self.config.judge_panel))

        judge_labels: Dict[str, Optional[List[bool]]] = {}
        judge_replies: Dict[str, List[str]] = {}
        failed_checks: Dict[str, int] = {}
        per_model: List[List[bool]] = []
        for judge, (result, replies) in zip(self.config.judge_panel, graded):
            judge_replies[judge.name] = replies
            for name, count in result.failed_checks_by_judge.items():
                failed_checks[name] = failed_checks.get(name, 0) + count
            if result.failed:
                judge_labels[judge.name] = None
            else:
                judge_labels[judge.name] = result.labels
                per_model.append(result.labels)

        if not per_model:
            return FrontierFinanceVerifyResponse(
                **base,
                reward=0.0,
                judge_labels=judge_labels,
                judge_replies=judge_replies,
                failed_checks_by_judge=failed_checks,
                grading_failure="judge_error",
                mask_sample=True,
                failure_kind=JUDGE_FAILED,
                failure_reason=f"all {len(self.config.judge_panel)} judges failed",
            )

        labels = Grader.majority_vote(per_model)
        qualified = sum(labels)
        return FrontierFinanceVerifyResponse(
            **base,
            reward=qualified / len(rubrics),
            rubric_labels=labels,
            judge_labels=judge_labels,
            judge_replies=judge_replies,
            failed_checks_by_judge=failed_checks,
            num_qualified=qualified,
            num_must_have_qualified=sum(1 for r, label in zip(rubrics, labels) if r.must_have and label),
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task in tasks for r in task]
        return frontier_finance_metrics(rows) if rows else {}

    def compute_repeat_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = self.compute_metrics(tasks)
        names = [f"{METRIC_PREFIX}{name}" for name in HEADLINE_METRICS]
        return {name: metrics[name] for name in names if name in metrics}

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key = {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}
        for name in (*HEADLINE_METRICS, "num_records", "num_failed_queries", "num_judge_errors"):
            if f"{METRIC_PREFIX}{name}" in agent_metrics:
                key[f"{METRIC_PREFIX}{name}"] = agent_metrics[f"{METRIC_PREFIX}{name}"]
        return key


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    for _logger_name in (__name__, "resources_servers"):
        logging.getLogger(_logger_name).setLevel(os.environ.get("NEMO_GYM_LOG_LEVEL", "INFO").upper())
    FrontierFinanceResourcesServer.run_webserver()
