# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep native ClawEval judging while preserving NVIDIA provider model routes."""

from __future__ import annotations

from urllib.parse import quote, urlsplit

import httpx
from claw_eval.graders.llm_judge import LLMJudge


class NvidiaInferenceGeminiJudge(LLMJudge):
    """Use the full configured model ID in Inference Hub's Gemini endpoint.

    The native ClawEval judge strips provider prefixes from Gemini model IDs.
    NVIDIA Inference Hub routes by the full ID. Evaluation prompts, generation
    settings, parsing, retries and score computation remain the native judge's.
    """

    def __init__(self, *, model_id: str, api_key: str, base_url: str) -> None:
        endpoint = urlsplit(base_url)
        if (
            endpoint.scheme != "https"
            or endpoint.hostname != "inference-api.nvidia.com"
            or endpoint.path.rstrip("/") not in {"", "/v1"}
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("NVIDIA Gemini judge requires an HTTPS Inference Hub root or /v1 base URL")
        self.gemini_endpoint = f"https://{endpoint.netloc}/v1beta/models/{quote(model_id, safe='/')}:generateContent"
        self._hub_api_key = api_key
        super().__init__(model_id=model_id, api_key=api_key, base_url=base_url)

    def _call_gemini(self, system_prompt: str, user_content: str | list[dict]) -> str:
        if isinstance(user_content, str):
            parts = [{"text": user_content}]
        else:
            parts = []
            for item in user_content:
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("data:"):
                        raise ValueError("Native ClawEval visual evidence must be inline image data")
                    metadata, data = url.split(",", 1)
                    parts.append({"inline_data": {"mime_type": metadata.split(":")[1].split(";")[0], "data": data}})
                else:
                    raise ValueError("Unsupported native ClawEval Gemini evidence part")
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
        }
        if system_prompt:
            body["system_instruction"] = {"parts": [{"text": system_prompt}]}
        response = httpx.post(
            self.gemini_endpoint,
            json=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._hub_api_key}"},
            timeout=300.0,
        )
        response.raise_for_status()
        candidate = response.json()["candidates"][0]
        text = "".join(
            part.get("text", "") for part in candidate.get("content", {}).get("parts", []) if not part.get("thought")
        )
        if not text:
            raise ValueError("NVIDIA Gemini judge returned no answer text")
        return text


def create_judge(*, model_id: str, api_key: str, base_url: str) -> LLMJudge:
    """Select the Hub route adapter only for NVIDIA-hosted Gemini judges."""
    if urlsplit(base_url).hostname == "inference-api.nvidia.com" and "gemini" in model_id.lower():
        return NvidiaInferenceGeminiJudge(model_id=model_id, api_key=api_key, base_url=base_url)
    return LLMJudge(model_id=model_id, api_key=api_key, base_url=base_url)
