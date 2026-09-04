# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebVoyager screenshot-and-answer evaluators behind a Gym resource boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from urllib.parse import urlparse

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.judge import call_judge, reraise_judge_errors
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.web.models import WebVerifierResult
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig
from resources_servers.webvoyager_judge.models import (
    WebVoyagerJudgeRequest,
    WebVoyagerJudgeResponse,
    WebVoyagerStandardVerifyRequest,
    WebVoyagerStandardVerifyResponse,
)
from resources_servers.webvoyager_judge.prompts import WEBVOYAGER_GEMINI_JUDGE_PROMPT


LOG = logging.getLogger("nemo_gym.resources_servers.webvoyager_judge")


def _origins(urls: list[str]) -> str:
    origins: list[str] = []
    for value in urls:
        parsed = urlparse(value)
        if not parsed.hostname:
            continue
        port = f":{parsed.port}" if parsed.port else ""
        origin = f"{parsed.scheme or 'unknown'}://{parsed.hostname}{port}"
        if origin not in origins:
            origins.append(origin)
    return ",".join(origins) or "none"


def _extract_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def parse_gemini_verdict(text: str) -> tuple[bool, dict[str, str]] | None:
    """Parse the maintained WebVoyager JSON verdict contract."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    verdict = str(payload.get("verdict", "")).upper()
    if verdict not in {"SUCCESS", "FAILURE"}:
        return None
    normalized = {"thought": str(payload.get("thought", "")), "verdict": verdict}
    return verdict == "SUCCESS", normalized


class WebVoyagerJudgeResourcesServer(SimpleResourcesServer):
    config: WebVoyagerJudgeConfig

    async def _judge_evidence(self, body: WebVoyagerJudgeRequest) -> WebVoyagerJudgeResponse:
        started = time.monotonic()
        LOG.info(
            "event=webvoyager_judge_start benchmark=%s task=%s screenshots_received=%d "
            "urls_received=%d origins=%s final_answer_present=%s",
            body.task.benchmark.value,
            body.task.task_id,
            len(body.screenshots),
            len(body.page_urls),
            _origins(body.page_urls),
            bool(body.final_answer.strip()),
        )
        screenshots = body.screenshots[-self.config.max_screenshots :]
        if self.config.require_screenshot and not screenshots:
            LOG.warning(
                "event=webvoyager_judge_short_circuit task=%s reason=missing_judge_evidence "
                "valid_sample=%s elapsed_seconds=%.3f",
                body.task.task_id,
                True,
                time.monotonic() - started,
            )
            return WebVoyagerJudgeResponse(
                result=WebVerifierResult(
                    valid_sample=True,
                    failure_kind="missing_judge_evidence",
                    verifier_version=self.config.verifier_version,
                )
            )

        task_instruction = body.task.intent
        if body.task.start_urls:
            task_instruction = f"On {body.task.start_urls[0]}: {task_instruction}"
        content = [
            {
                "type": "input_text",
                "text": (
                    f"Task: {task_instruction}\n\n"
                    f"Result Response: {body.final_answer or 'No answer'}\n\n"
                    f"{len(screenshots)} screenshots recorded during the task:"
                ),
            }
        ]
        for index, screenshot in enumerate(screenshots, start=1):
            content.append({"type": "input_text", "text": f"Step {index}"})
            content.append({"type": "input_image", "image_url": screenshot, "detail": "high"})
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.instructions = WEBVOYAGER_GEMINI_JUDGE_PROMPT
        params.input = [NeMoGymEasyInputMessage(role="user", content=content)]
        model_started = time.monotonic()
        LOG.info(
            "event=webvoyager_judge_model_start task=%s model_server=%s screenshots_used=%d",
            body.task.task_id,
            self.config.judge_model_server.name,
            len(screenshots),
        )
        try:
            judge_response = await reraise_judge_errors(
                asyncio.wait_for(
                    call_judge(
                        self.server_client,
                        server_name=self.config.judge_model_server.name,
                        url_path="/v1/responses",
                        json=params,
                        response_model=NeMoGymResponse,
                    ),
                    timeout=self.config.judge_call_timeout_secs,
                )
            )
        except Exception:
            LOG.exception(
                "event=webvoyager_judge_model_failed task=%s model_server=%s elapsed_seconds=%.3f",
                body.task.task_id,
                self.config.judge_model_server.name,
                time.monotonic() - model_started,
            )
            raise
        judge_text = _extract_output_text(judge_response)
        LOG.info(
            "event=webvoyager_judge_model_complete task=%s model_server=%s output_chars=%d "
            "output_sha256=%s elapsed_seconds=%.3f",
            body.task.task_id,
            self.config.judge_model_server.name,
            len(judge_text),
            hashlib.sha256(judge_text.encode("utf-8")).hexdigest()[:12],
            time.monotonic() - model_started,
        )
        parsed_verdict = parse_gemini_verdict(judge_text)
        verdict = parsed_verdict[0] if parsed_verdict is not None else None
        if verdict is None:
            LOG.warning(
                "event=webvoyager_judge_unparseable task=%s output_chars=%d output_sha256=%s",
                body.task.task_id,
                len(judge_text),
                hashlib.sha256(judge_text.encode("utf-8")).hexdigest()[:12],
            )
            result = WebVerifierResult(
                # A response was received successfully; malformed verdict text
                # is therefore a benchmark outcome rather than infrastructure.
                valid_sample=True,
                failure_kind="judge_unparseable",
                verifier_version=self.config.verifier_version,
                metadata={"judge_text": judge_text},
            )
        else:
            score = float(verdict)
            result = WebVerifierResult(
                reward=score,
                raw_score=score,
                task_success=verdict,
                valid_sample=True,
                verifier_version=self.config.verifier_version,
                metadata={
                    "judge_text": judge_text,
                    "judge_profile": "gemini_screenshot_trajectory",
                    **({"parsed": parsed_verdict[1]} if parsed_verdict is not None else {}),
                    "screenshots_used": len(screenshots),
                    "page_urls": body.page_urls[-self.config.max_screenshots :],
                },
            )
        LOG.info(
            "event=webvoyager_judge_complete task=%s valid_sample=%s success=%s reward=%s "
            "failure_kind=%s screenshots_used=%d elapsed_seconds=%.3f",
            body.task.task_id,
            result.valid_sample,
            result.task_success,
            result.reward,
            result.failure_kind or "none",
            len(screenshots),
            time.monotonic() - started,
        )
        return WebVoyagerJudgeResponse(result=result, judge_text=judge_text)

    async def verify(
        self,
        body: WebVoyagerStandardVerifyRequest,
    ) -> WebVoyagerStandardVerifyResponse:
        judged = await self._judge_evidence(
            WebVoyagerJudgeRequest(
                task=body.web_task,
                final_answer=body.final_answer,
                screenshots=body.screenshots,
                page_urls=body.page_urls,
            )
        )
        result = judged.result
        return WebVoyagerStandardVerifyResponse.model_validate(
            {
                "responses_create_params": body.responses_create_params,
                "response": body.response,
                "reward": result.reward if result.valid_sample else 0.0,
                "raw_score": result.raw_score,
                "task_success": result.task_success,
                "mask_sample": not result.valid_sample,
                "failure_kind": result.failure_kind,
                "judge_text": judged.judge_text,
                "verifier_metadata": result.metadata,
                "verifier_version": result.verifier_version,
            }
        )


if __name__ == "__main__":
    WebVoyagerJudgeResourcesServer.run_webserver()
