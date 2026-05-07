# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Judge resources server that grades a list of behavioral assertions in one LLM call."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import nullcontext
from typing import Any, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
)
from nemo_gym.server_utils import get_response_json
from resources_servers.skill_judge.schemas import (
    EVIDENCE_MAX_CHARS,
    AssertionGrade,
    AssertionGradeResponse,
    JudgeRequest,
    JudgeResponse,
    JudgeUsage,
    SkillJudgeResourcesServerConfig,
    SkillJudgeVerifyRequest,
    ToolCallLogEntry,
)


logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _extract_json_array(text: str) -> Optional[list[Any]]:
    stripped = _strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_ARRAY_RE.search(stripped)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _format_tool_calls(tool_calls: list[ToolCallLogEntry]) -> str:
    if not tool_calls:
        return "(no tool calls)"
    lines = []
    for i, tc in enumerate(tool_calls, start=1):
        header = f"{i}. {tc.name}({tc.arguments})"
        if tc.exit_code is not None:
            header += f" → exit {tc.exit_code}"
        lines.append(header)
        if tc.stdout_snippet:
            lines.append(f"   stdout: {tc.stdout_snippet}")
        if tc.stderr_snippet:
            lines.append(f"   stderr: {tc.stderr_snippet}")
        if tc.truncated:
            lines.append("   (output truncated)")
    return "\n".join(lines)


def _format_assertions(assertions: list[str]) -> str:
    return "\n".join(f"{i}. {a}" for i, a in enumerate(assertions, start=1))


def _normalize_grades(raw_grades: list[Any], n_assertions: int) -> list[AssertionGrade]:
    """Project raw judge output onto exactly n_assertions grades (ids 1..n)."""
    by_id: dict[int, dict[str, Any]] = {}
    for item in raw_grades:
        if not isinstance(item, dict):
            continue
        try:
            gid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        by_id[gid] = item

    grades: list[AssertionGrade] = []
    for gid in range(1, n_assertions + 1):
        item = by_id.get(gid)
        if item is None:
            grades.append(
                AssertionGrade(id=gid, satisfied=False, evidence="judge did not return a grade for this assertion")
            )
            continue
        satisfied = bool(item.get("satisfied", False))
        evidence = str(item.get("evidence", ""))[:EVIDENCE_MAX_CHARS]
        grades.append(AssertionGrade(id=gid, satisfied=satisfied, evidence=evidence))
    return grades


def _usage_from_response(resp: NeMoGymResponse) -> JudgeUsage:
    if resp.usage is None:
        return JudgeUsage()
    return JudgeUsage(
        input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        total_tokens=getattr(resp.usage, "total_tokens", 0) or 0,
    )


def _last_assistant_text(resp: NeMoGymResponse) -> str:
    for o in reversed(resp.output):
        if getattr(o, "type", None) != "message" or getattr(o, "role", None) != "assistant":
            continue
        content = getattr(o, "content", None)
        if isinstance(content, list):
            parts = []
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    parts.append(t)
            return "\n".join(parts).strip()
        if isinstance(content, str):
            return content.strip()
    return ""


class SkillJudgeResourcesServer(SimpleResourcesServer):
    config: SkillJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_semaphore: Any = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_semaphore = nullcontext()

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/judge")(self.judge)
        return app

    def _build_user_prompt(self, req: JudgeRequest) -> str:
        return self._judge_prompt_template.format(
            prompt=req.prompt,
            expected_output=req.expected_output or "(not provided)",
            response=req.response or "(empty response)",
            tool_calls=_format_tool_calls(req.tool_calls),
            assertions=_format_assertions(req.assertions),
            n=len(req.assertions),
        )

    async def _call_judge(self, req: JudgeRequest) -> JudgeResponse:
        n = len(req.assertions)
        if n == 0:
            return JudgeResponse(grades=[], reward=0.0)

        cfg = self.config
        create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=self._build_user_prompt(req)))
        create_params.input = msgs

        async with self._judge_semaphore:
            raw_response = await self.server_client.post(
                server_name=cfg.judge_model_server.name,
                url_path="/v1/responses",
                json=create_params,
            )
            judge_response = NeMoGymResponse.model_validate(await get_response_json(raw_response))

        judge_text = _last_assistant_text(judge_response)
        usage = _usage_from_response(judge_response)

        raw_grades = _extract_json_array(judge_text)
        if raw_grades is None:
            logger.warning("skill_judge could not parse JSON array from judge output")
            return JudgeResponse(
                grades=[
                    AssertionGrade(id=i, satisfied=False, evidence="judge output was not parseable JSON")
                    for i in range(1, n + 1)
                ],
                reward=0.0,
                judge_usage=usage,
                parse_error="could not extract JSON array from judge output",
            )

        grades = _normalize_grades(raw_grades, n)
        satisfied = sum(1 for g in grades if g.satisfied)
        return JudgeResponse(
            grades=grades,
            reward=satisfied / n,
            judge_usage=usage,
        )

    async def judge(self, body: JudgeRequest) -> JudgeResponse:
        return await self._call_judge(body)

    async def verify(self, body: SkillJudgeVerifyRequest) -> AssertionGradeResponse:
        metadata = body.verifier_metadata or {}

        assertions = metadata.get("assertions") or []
        if not isinstance(assertions, list):
            assertions = []

        tool_call_entries: list[ToolCallLogEntry] = []
        for item in metadata.get("tool_calls") or []:
            if isinstance(item, dict):
                tool_call_entries.append(ToolCallLogEntry.model_validate(item))

        prompt = ""
        for msg in reversed(body.responses_create_params.input or []):
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    prompt = content
                break

        judge_req = JudgeRequest(
            prompt=prompt,
            expected_output=metadata.get("expected_output"),
            response=_last_assistant_text(body.response),
            tool_calls=tool_call_entries,
            assertions=[str(a) for a in assertions],
        )
        judge_resp = await self._call_judge(judge_req)

        return AssertionGradeResponse(
            **body.model_dump(),
            reward=judge_resp.reward,
            grades=judge_resp.grades,
            judge_usage=judge_resp.judge_usage,
            parse_error=judge_resp.parse_error,
        )


if __name__ == "__main__":
    SkillJudgeResourcesServer.run_webserver()
