# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_rl.environments.mmpr_filtered_reward import mmpr_filtered_reward


class OmniVerifierResourcesServerConfig(BaseResourcesServerConfig):
    pass


class OmniVerifierRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    ground_truth: Optional[str] = None
    verifier: Optional[str] = None
    expected_answer: Optional[str] = None
    answer: Optional[str] = None
    think_mode: Optional[Literal["think", "nothink"]] = None
    format_weight: float = 0.1
    dynamic_format_reward: bool = True
    asr_reward_min: Optional[float] = -1.0
    metadata: Optional[dict[str, Any]] = None


class OmniVerifierVerifyRequest(OmniVerifierRunRequest, BaseVerifyRequest):
    pass


class OmniVerifierVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    reward: float
    is_correct: bool
    ground_truth: str
    verifier: Optional[str] = None
    expected_answer: Optional[str] = None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_reasoning_text(output_item: Any) -> str:
    summaries = _field(output_item, "summary")
    if not isinstance(summaries, list):
        return ""

    texts: list[str] = []
    for summary in summaries:
        text = _field(summary, "text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "\n".join(texts).strip()


def _wrap_reasoning(text: str) -> str:
    return f"<think>{text}</think>\n" if text else ""


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    pending_reasoning: list[str] = []
    for output_item in body.response.output:
        output_type = _field(output_item, "type")
        if output_type == "reasoning":
            reasoning_text = _extract_reasoning_text(output_item)
            if reasoning_text:
                pending_reasoning.append(reasoning_text)
            continue

        if output_type != "message":
            continue
        if _field(output_item, "role") != "assistant":
            continue

        message_texts: list[str] = []
        content = _field(output_item, "content")
        if isinstance(content, list):
            for content_item in content:
                text = _field(content_item, "text")
                if isinstance(text, str):
                    message_texts.append(text)
        elif isinstance(content, str):
            message_texts.append(content)

        reasoning_prefix = _wrap_reasoning("\n".join(pending_reasoning).strip())
        pending_reasoning.clear()
        if message_texts or reasoning_prefix:
            message_text = "\n".join(message_texts)
            texts.append(f"{reasoning_prefix}{message_text}".strip())
    return "\n".join(texts).strip()


def _build_ground_truth(body: OmniVerifierRunRequest) -> str:
    if body.ground_truth:
        return body.ground_truth

    answer = body.expected_answer if body.expected_answer is not None else body.answer
    if not body.verifier or answer is None:
        raise ValueError(
            "omni_verifier requires either ground_truth or verifier plus expected_answer/answer"
        )

    verifier_answer = f"{body.verifier}:{answer}"
    if body.think_mode:
        return f"{body.think_mode}:{verifier_answer}"
    return verifier_answer


class OmniVerifierResourcesServer(SimpleResourcesServer):
    config: OmniVerifierResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(
        self, body: OmniVerifierVerifyRequest
    ) -> OmniVerifierVerifyResponse:
        text = _extract_last_assistant_text(body)
        ground_truth = _build_ground_truth(body)
        reward, is_correct = mmpr_filtered_reward(
            ground_truth,
            text,
            format_weight=body.format_weight,
            dynamic_format_reward=body.dynamic_format_reward,
            asr_reward_min=body.asr_reward_min,
        )

        return OmniVerifierVerifyResponse(
            **body.model_dump(
                exclude={"ground_truth", "verifier", "expected_answer", "answer"}
            ),
            reward=float(reward),
            is_correct=bool(is_correct),
            ground_truth=ground_truth,
            verifier=body.verifier,
            expected_answer=(
                body.expected_answer if body.expected_answer is not None else body.answer
            ),
        )


if __name__ == "__main__":
    OmniVerifierResourcesServer.run_webserver()
