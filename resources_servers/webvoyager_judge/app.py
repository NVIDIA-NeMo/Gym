# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebVoyager's screenshot-and-answer evaluator behind a Gym resource boundary."""

from __future__ import annotations

from fastapi import FastAPI

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.web.models import WebVerifierResult
from resources_servers.webvoyager_judge.config import WebVoyagerJudgeConfig
from resources_servers.webvoyager_judge.models import (
    WebVoyagerJudgeRequest,
    WebVoyagerJudgeResponse,
    WebVoyagerStandardVerifyRequest,
    WebVoyagerStandardVerifyResponse,
)


SYSTEM_PROMPT = """You are evaluating the result of a web-navigation task. You receive:
1. The original web task instruction.
2. The final screenshots from the browser trajectory.
3. The agent's final textual response.

Do not interact with websites. Judge only the supplied evidence. Check every requirement in multi-part tasks. If the
response contradicts a screenshot, the screenshot takes precedence. If the response contains relevant details that
are not visible in the screenshots and are not contradicted by them, you may accept those details.

Explain the assessment briefly, then end with exactly one definitive verdict: SUCCESS or NOT SUCCESS."""


def _extract_output_text(response: NeMoGymResponse) -> str:
    parts: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "output_text":
                parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def parse_verdict(text: str) -> bool | None:
    """Match upstream semantics while treating a missing verdict as judge failure."""

    upper = text.upper()
    if "NOT SUCCESS" in upper:
        return False
    if "SUCCESS" in upper:
        return True
    return None


class WebVoyagerJudgeResourcesServer(SimpleResourcesServer):
    config: WebVoyagerJudgeConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/verify_webvoyager")(self.verify_webvoyager)
        return app

    async def verify_webvoyager(self, body: WebVoyagerJudgeRequest) -> WebVoyagerJudgeResponse:
        if not body.final_answer.strip():
            return WebVoyagerJudgeResponse(
                result=WebVerifierResult(
                    valid_sample=True,
                    failure_kind="agent_no_final_answer",
                    verifier_version=self.config.verifier_version,
                )
            )

        screenshots = body.screenshots[-self.config.max_screenshots :]
        if self.config.require_screenshot and not screenshots:
            return WebVoyagerJudgeResponse(
                result=WebVerifierResult(
                    valid_sample=False,
                    failure_kind="missing_judge_evidence",
                    verifier_version=self.config.verifier_version,
                )
            )

        content = [
            {
                "type": "input_text",
                "text": (
                    f"TASK: {body.task.intent}\n"
                    f"Result Response: {body.final_answer}\n"
                    f"{len(screenshots)} screenshots from the end of the trajectory follow."
                ),
            }
        ]
        content.extend(
            {"type": "input_image", "image_url": screenshot, "detail": "high"} for screenshot in screenshots
        )
        content.append({"type": "input_text", "text": "Your verdict:"})

        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.instructions = SYSTEM_PROMPT
        params.input = [NeMoGymEasyInputMessage(role="user", content=content)]
        raw_response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
        )
        await raise_for_status(raw_response)
        judge_response = NeMoGymResponse.model_validate(await get_response_json(raw_response))
        judge_text = _extract_output_text(judge_response)
        verdict = parse_verdict(judge_text)
        if verdict is None:
            result = WebVerifierResult(
                valid_sample=False,
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
                    "screenshots_used": len(screenshots),
                    "page_urls": body.page_urls[-self.config.max_screenshots :],
                },
            )
        return WebVoyagerJudgeResponse(result=result, judge_text=judge_text)

    async def verify(
        self,
        body: WebVoyagerStandardVerifyRequest,
    ) -> WebVoyagerStandardVerifyResponse:
        judged = await self.verify_webvoyager(
            WebVoyagerJudgeRequest(
                task=body.web_task,
                final_answer=body.final_answer,
                screenshots=body.screenshots,
                page_urls=body.page_urls,
            )
        )
        result = judged.result
        return WebVoyagerStandardVerifyResponse.model_validate(
            body.model_dump()
            | {
                "reward": result.reward if result.valid_sample else 0.0,
                "raw_score": result.raw_score,
                "task_success": result.task_success,
                "mask_sample": not result.valid_sample,
                "failure_kind": result.failure_kind,
                "judge_text": judged.judge_text,
                "verifier_metadata": result.metadata,
            }
        )


if __name__ == "__main__":
    WebVoyagerJudgeResourcesServer.run_webserver()
