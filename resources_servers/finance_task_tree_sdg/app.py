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
"""Task-tree finance question SDG: one resources server, configured per stage.

Each question stage is one rollout followed by an in-verify judge. The judge
returns only pass or fail; its full output is written to ``audit_dir`` and never
appears in the verify response, so no judge token reaches a later stage's model.

The ``eligibility`` stage runs once per task-tree leaf before sampling: it maps
the leaf to the SIC industries whose companies can support it, with no judge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional

from big_finance_harness.tools import PythonExecTool, ToolError
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field
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
from nemo_gym.judge import JudgeError, reraise_judge_errors
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.big_finance.app import (
    _parse_json_object,
    _responses_output_text,
    _tool_call_error_statuses,
    _tool_call_is_error,
    extract_final_answer,
    format_trace,
)
from resources_servers.big_finance.sec_cache import SecHttpCache
from resources_servers.finance_task_tree_sdg.sec_tools import AsOfEdgarSearchTool, AsOfFetchUrlTool, parse_as_of


logger = logging.getLogger(__name__)

Stage = Literal["eligibility", "question_gen", "leak_proposal", "leak_removal", "answer_gen", "adjudicate"]

REJECTION_CODES = ("data_unavailable", "condition_not_met", "company_ineligible", "document_mismatch", "other")
LEAK_ACTIONS = ("delete", "generalize")
ELIGIBILITY_SCOPES = ("any_industry", "restricted", "unsupported")

_PROMPT_DIR = Path(__file__).with_name("prompts")
_JUDGE_PROMPTS: dict[str, str] = {
    "question_gen": "judge_question_gen.txt",
    "leak_proposal": "judge_leak_proposal.txt",
    "leak_removal": "judge_leak_removal.txt",
}
_DONE_TOOLS: dict[str, tuple[str, ...]] = {
    "eligibility": ("submit_eligibility",),
    "question_gen": ("submit_question", "submit_rejection"),
    "leak_proposal": ("submit_leak_proposal",),
    "leak_removal": ("submit_revised_question",),
    "answer_gen": ("final_answer",),
    "adjudicate": ("submit_verdict",),
}
_STAGE_TOOLS: dict[str, tuple[str, ...]] = {
    "eligibility": ("submit_eligibility",),
    "question_gen": ("edgar_search", "fetch_url", "python_exec", "submit_question", "submit_rejection"),
    "leak_proposal": ("submit_leak_proposal",),
    "leak_removal": ("submit_revised_question",),
    "answer_gen": ("edgar_search", "fetch_url", "python_exec", "final_answer"),
    "adjudicate": ("edgar_search", "fetch_url", "python_exec", "submit_verdict"),
}
_SEC_TOOLS = ("edgar_search", "fetch_url")
# failed_criteria and reason are written to the audit file only; verify never returns them.
_VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "failed_criteria", "reason"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "failed_criteria": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
}
_SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9._-]+")


class FinanceTaskTreeSdgConfig(BaseResourcesServerConfig):
    stage: Stage
    sec_edgar_user_agent: Optional[str] = None
    use_cache: bool = False
    cache_dir: Optional[str] = None
    edgar_timeout_s: float = 20.0
    fetch_timeout_s: float = 30.0
    python_exec_timeout_s: float = 5.0
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    judge_call_timeout_s: Optional[float] = 1800.0
    judge_prompt_fpath: Optional[str] = None
    audit_dir: Optional[str] = None
    min_question_chars: int = 40
    max_question_chars: int = 4000


class SdgRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    sdg: dict[str, Any] = Field(default_factory=dict)


class SdgVerifyRequest(SdgRunRequest, BaseVerifyRequest):
    pass


class SdgVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    sdg: dict[str, Any] = Field(default_factory=dict)
    sdg_result: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class _Session:
    as_of: Optional[date]
    sdg: dict[str, Any] = field(default_factory=dict)


def _tool_error(message: str) -> PlainTextResponse:
    return PlainTextResponse(message, headers={"x-nemo-gym-tool-error": "true"})


def _tool_ok(message: str) -> PlainTextResponse:
    return PlainTextResponse(message, headers={"x-nemo-gym-tool-error": "false"})


def _non_empty_str(args: dict[str, Any], key: str) -> Optional[str]:
    value = args.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def validate_eligibility(args: dict[str, Any], sic_codes: list[str]) -> dict[str, Any]:
    scope = args.get("scope")
    if scope not in ELIGIBILITY_SCOPES:
        raise ToolError(f"scope must be one of {list(ELIGIBILITY_SCOPES)}")
    codes = args.get("sic_codes")
    if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
        raise ToolError("sic_codes must be a list of strings (use an empty list unless scope is 'restricted')")
    codes = sorted({code.strip() for code in codes})
    if scope == "restricted":
        if not codes:
            raise ToolError("sic_codes must not be empty when scope is 'restricted'")
        unknown = [code for code in codes if code not in set(sic_codes)]
        if unknown:
            raise ToolError(f"these SIC codes are not in the provided list: {unknown}")
    elif codes:
        raise ToolError(f"sic_codes must be empty when scope is {scope!r}")
    reason = _non_empty_str(args, "reason")
    if reason is None:
        raise ToolError("reason is required")
    return {"scope": scope, "sic_codes": codes, "reason": reason}


def validate_question(args: dict[str, Any], config: FinanceTaskTreeSdgConfig) -> str:
    question = _non_empty_str(args, "question")
    if question is None:
        raise ToolError("question is required and must be a non-empty string")
    if not config.min_question_chars <= len(question) <= config.max_question_chars:
        raise ToolError(
            f"question must be between {config.min_question_chars} and {config.max_question_chars} characters"
        )
    return question


def validate_rejection(args: dict[str, Any]) -> dict[str, str]:
    code = args.get("reason_code")
    if code not in REJECTION_CODES:
        raise ToolError(f"reason_code must be one of {list(REJECTION_CODES)}")
    explanation = _non_empty_str(args, "explanation")
    if explanation is None:
        raise ToolError("explanation is required")
    return {"reason_code": code, "explanation": explanation}


def validate_leak_proposal(args: dict[str, Any], question: str) -> dict[str, Any]:
    items = args.get("items")
    if not isinstance(items, list):
        raise ToolError("items must be a list (use an empty list when the question has no leak)")
    cleaned: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ToolError(f"items[{index}] must be an object")
        span = item.get("span")
        if not isinstance(span, str) or not span.strip():
            raise ToolError(f"items[{index}].span must be a non-empty string")
        if span not in question:
            raise ToolError(f"items[{index}].span must be copied verbatim from the question: {span!r}")
        action = item.get("action")
        if action not in LEAK_ACTIONS:
            raise ToolError(f"items[{index}].action must be one of {list(LEAK_ACTIONS)}")
        replacement = item.get("replacement")
        if action == "generalize" and not (isinstance(replacement, str) and replacement.strip()):
            raise ToolError(f"items[{index}].replacement is required when action is 'generalize'")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolError(f"items[{index}].reason must be a non-empty string")
        entry = {"span": span, "action": action, "reason": reason.strip()}
        if action == "generalize":
            entry["replacement"] = replacement.strip()
        cleaned.append(entry)
    no_leak_reason = _non_empty_str(args, "no_leak_reason")
    if not cleaned and no_leak_reason is None:
        raise ToolError("no_leak_reason is required when items is empty")
    return {"items": cleaned, "no_leak_reason": no_leak_reason}


def validate_revised_question(
    args: dict[str, Any], original: str, proposal: dict[str, Any], config: FinanceTaskTreeSdgConfig
) -> str:
    question = validate_question(args, config)
    remaining = [item["span"] for item in proposal.get("items", []) if item["span"] in question]
    if remaining:
        raise ToolError(f"these proposed spans are still in the question: {remaining}")
    if proposal.get("items") and question == original:
        raise ToolError("the revised question is identical to the original question")
    return question


def validate_verdict(args: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    if not isinstance(args.get("question_valid"), bool):
        raise ToolError("question_valid must be a boolean")
    selected = args.get("selected")
    allowed = [*labels, "none"]
    if selected not in allowed:
        raise ToolError(f"selected must be one of {allowed}")
    return {"question_valid": args["question_valid"], "selected": selected}


def accepted_submission(response: NeMoGymResponse, names: tuple[str, ...]) -> Optional[tuple[str, dict[str, Any]]]:
    """Return the last successful terminal call that ended the loop."""
    if (response.metadata or {}).get("stop_reason") != "done_tool":
        return None
    statuses = _tool_call_error_statuses(response)
    outputs = {
        getattr(item, "call_id", None): str(getattr(item, "output", "") or "")
        for item in response.output
        if getattr(item, "type", None) == "function_call_output"
    }
    for item in reversed(response.output):
        if getattr(item, "type", None) != "function_call" or getattr(item, "name", None) not in names:
            continue
        call_id = getattr(item, "call_id", None)
        output = outputs.get(call_id)
        if output is None or _tool_call_is_error(statuses, call_id, output):
            continue
        try:
            args = json.loads(getattr(item, "arguments", "") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(args, dict):
            return item.name, args
    return None


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _task_card(sdg: dict[str, Any]) -> str:
    keys = ("task", "companies", "fiscal_year", "anchor_documents", "as_of_date", "reveal_level")
    return _json_block({key: sdg.get(key) for key in keys if key in sdg})


def judge_user_prompt(stage: str, sdg: dict[str, Any], result: dict[str, Any], trace: str) -> str:
    if stage == "question_gen":
        return (
            f"TASK CARD:\n{_task_card(sdg)}\n\n"
            f"QUESTION:\n{result['question']}\n\n"
            f"QUESTION WRITER'S TRACE (tool calls and results):\n{trace}"
        )
    if stage == "leak_proposal":
        return (
            f"TASK CARD:\n{_task_card(sdg)}\n\n"
            f"QUESTION:\n{sdg.get('question', '')}\n\n"
            f"QUESTION WRITER'S TRACE (tool calls and results):\n{sdg.get('question_gen_trace', '')}\n\n"
            f"LEAK-REMOVAL PROPOSAL:\n{_json_block(result['proposal'])}"
        )
    if stage == "leak_removal":
        return (
            f"TASK CARD:\n{_task_card(sdg)}\n\n"
            f"ORIGINAL QUESTION:\n{sdg.get('question', '')}\n\n"
            f"APPROVED LEAK-REMOVAL PROPOSAL:\n{_json_block(sdg.get('leak_proposal', {}))}\n\n"
            f"REVISED QUESTION:\n{result['question']}"
        )
    raise ValueError(f"stage {stage!r} has no judge")


class FinanceTaskTreeSdgServer(SimpleResourcesServer):
    config: FinanceTaskTreeSdgConfig

    def model_post_init(self, context: Any) -> None:
        cache = SecHttpCache(cache_dir=self.config.cache_dir, use_cache=self.config.use_cache)
        self._sessions: dict[str, _Session] = {}
        self._edgar: Optional[AsOfEdgarSearchTool] = None
        self._fetch: Optional[AsOfFetchUrlTool] = None
        if self.config.sec_edgar_user_agent:
            self._edgar = AsOfEdgarSearchTool(
                cache=cache,
                user_agent=self.config.sec_edgar_user_agent,
                timeout_s=self.config.edgar_timeout_s,
            )
            self._fetch = AsOfFetchUrlTool(
                cache=cache,
                history=self._edgar.history,
                timeout_s=self.config.fetch_timeout_s,
                sec_user_agent=self.config.sec_edgar_user_agent,
            )
        self._python = PythonExecTool(timeout_s=self.config.python_exec_timeout_s)
        self._judge_system = ""
        if self.config.stage in _JUDGE_PROMPTS:
            path = Path(self.config.judge_prompt_fpath or _PROMPT_DIR / _JUDGE_PROMPTS[self.config.stage])
            self._judge_system = path.read_text(encoding="utf-8")

    def _session(self, request: Request) -> _Session:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise ToolError("no seeded session for this rollout")
        return session

    async def seed_session(self, request: Request, body: SdgRunRequest) -> BaseSeedSessionResponse:
        sdg = dict(body.sdg or {})
        self._sessions[request.session[SESSION_ID_KEY]] = _Session(as_of=parse_as_of(sdg.get("as_of_date")), sdg=sdg)
        if len(self._sessions) > 4096:
            logger.warning("%d live SDG sessions; verify may not be releasing them", len(self._sessions))
        return await super().seed_session(BaseSeedSessionRequest())

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in _STAGE_TOOLS[self.config.stage]:
            app.post(f"/{name}")(self._handler(name))
        return app

    def _handler(self, name: str):
        async def handler(request: Request, body: dict = Body(default={})):
            args = body if isinstance(body, dict) else {}
            try:
                return _tool_ok(await self._run_tool(name, request, args))
            except ToolError as exc:
                return _tool_error(str(exc))
            except Exception as exc:  # noqa: BLE001 - tool errors are observations, not HTTP 500s
                return _tool_error(f"unexpected tool error: {type(exc).__name__}: {exc}")

        return handler

    async def _run_tool(self, name: str, request: Request, args: dict[str, Any]) -> str:
        if name == "python_exec":
            return await self._python.run(args)
        session = self._session(request)
        if name in _SEC_TOOLS:
            tool = self._edgar if name == "edgar_search" else self._fetch
            if tool is None:
                raise ToolError(f"Tool '{name}' is not configured.")
            return await tool.run(args, as_of=session.as_of)
        if name == "submit_eligibility":
            validate_eligibility(args, list(session.sdg.get("sic_codes", [])))
            return "Eligibility submitted."
        if name == "submit_question":
            validate_question(args, self.config)
            return "Question submitted."
        if name == "submit_rejection":
            validate_rejection(args)
            return "Rejection submitted."
        if name == "submit_leak_proposal":
            validate_leak_proposal(args, str(session.sdg.get("question", "")))
            return "Proposal submitted."
        if name == "submit_revised_question":
            validate_revised_question(
                args, str(session.sdg.get("question", "")), session.sdg.get("leak_proposal") or {}, self.config
            )
            return "Revised question submitted."
        if name == "final_answer":
            answer = _non_empty_str(args, "answer")
            if answer is None:
                raise ToolError("answer is required and must be a non-empty string")
            return answer
        if name == "submit_verdict":
            validate_verdict(args, [item["label"] for item in session.sdg.get("candidates", [])])
            return "Verdict submitted."
        raise ToolError(f"Tool '{name}' is not available in stage {self.config.stage!r}.")

    def _audit_path(self, body: SdgVerifyRequest) -> Optional[Path]:
        if not self.config.audit_dir:
            return None
        sample = _SAFE_PATH_RE.sub("_", str(body.sdg.get("sample_id") or "unknown"))
        rollout = _SAFE_PATH_RE.sub("_", body.capture_rollout_id or uuid.uuid4().hex)
        return Path(self.config.audit_dir) / self.config.stage / sample / f"{rollout}.json"

    def _write_audit(self, path: Optional[Path], record: dict[str, Any]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(path)

    async def _judge(self, body: SdgVerifyRequest, user_prompt: str) -> dict[str, Any]:
        """Return only the verdict fields; the judge's own text goes to the audit file."""
        if self.config.judge_model_server is None:
            raise JudgeError("judge_model_server is not configured")
        params = (
            self.config.judge_responses_create_params or NeMoGymResponseCreateParamsNonStreaming(input=[])
        ).model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(role="system", content=self._judge_system),
            NeMoGymEasyInputMessage(role="user", content=user_prompt),
        ]
        params.text = {
            "format": {"type": "json_schema", "name": "sdg_verdict", "strict": True, "schema": _VERDICT_SCHEMA}
        }

        async def request_judge() -> Any:
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
            )
            await raise_for_status(response)
            return await get_response_json(response)

        raw = await reraise_judge_errors(asyncio.wait_for(request_judge(), timeout=self.config.judge_call_timeout_s))
        verdict: Optional[str] = None
        parsed: dict[str, Any] = {}
        try:
            parsed = _parse_json_object(_responses_output_text(raw))
            value = parsed.get("verdict")
            verdict = value if value in ("pass", "fail") else None
        except (ValueError, RecursionError):
            verdict = None
        model = raw.get("model") if isinstance(raw, dict) else None
        self._write_audit(
            self._audit_path(body),
            {
                "stage": self.config.stage,
                "sample_id": body.sdg.get("sample_id"),
                "rollout_id": body.capture_rollout_id,
                "judge_model": model,
                "verdict": verdict,
                "failed_criteria": parsed.get("failed_criteria"),
                "reason": parsed.get("reason"),
                "judge_input": [message.model_dump() for message in params.input],
                "judge_response": raw,
            },
        )
        return {"verdict": verdict, "model": model, "parsed": verdict is not None}

    async def verify(self, request: Request, body: SdgVerifyRequest) -> SdgVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id:
            self._sessions.pop(session_id, None)
        stage = self.config.stage
        result: dict[str, Any] = {"stage": stage}
        reward = 0.0

        if stage == "answer_gen":
            answer = extract_final_answer(body.response)
            result.update(status="answered" if answer else "no_answer", final_answer=answer)
            return self._respond(body, 1.0, result)

        submission = accepted_submission(body.response, _DONE_TOOLS[stage])
        if submission is None:
            result["status"] = "no_submission"
            return self._respond(body, 1.0 if stage == "adjudicate" else 0.0, result)
        name, args = submission

        if stage == "adjudicate":
            labels = [item["label"] for item in body.sdg.get("candidates", [])]
            result.update(status="adjudicated", **validate_verdict(args, labels))
            return self._respond(body, 1.0, result)
        if stage == "eligibility":
            result.update(status="mapped", **validate_eligibility(args, list(body.sdg.get("sic_codes", []))))
            return self._respond(body, 1.0, result)

        trace = format_trace(body.response)
        if stage == "question_gen":
            if name == "submit_rejection":
                result.update(status="rejected", rejection=validate_rejection(args), trace=trace)
                return self._respond(body, 0.0, result)
            result.update(question=validate_question(args, self.config), trace=trace)
        elif stage == "leak_proposal":
            result["proposal"] = validate_leak_proposal(args, str(body.sdg.get("question", "")))
        elif stage == "leak_removal":
            result["question"] = validate_revised_question(
                args, str(body.sdg.get("question", "")), body.sdg.get("leak_proposal") or {}, self.config
            )

        judge = await self._judge(body, judge_user_prompt(stage, body.sdg, result, trace))
        result["judge"] = judge
        if not judge["parsed"]:
            result["status"] = "judge_unparsed"
        else:
            result["status"] = f"judge_{judge['verdict']}"
            reward = 1.0 if judge["verdict"] == "pass" else 0.0
        return self._respond(body, reward, result)

    @staticmethod
    def _respond(body: SdgVerifyRequest, reward: float, result: dict[str, Any]) -> SdgVerifyResponse:
        return SdgVerifyResponse(**body.model_dump(), reward=reward, sdg_result=result)


if __name__ == "__main__":
    FinanceTaskTreeSdgServer.run_webserver()
