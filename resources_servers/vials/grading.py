# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run VIALS judging through Gym's asynchronous model-server transport."""

import asyncio

from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.vials.upstream import (
    _JUDGE_SYSTEM,
    _MULTIMODAL_SYSTEM,
    FreeFormJudgeResponse,
    _build_judge_user_message,
    _salvage_json,
)


class VialsJudge:
    """Preserve upstream judge prompts and parsing while using Gym model routing."""

    def __init__(self, server_client: ServerClient, *, server_name: str, max_concurrent: int = 4) -> None:
        self.server_client = server_client
        self.server_name = server_name
        self._slots = asyncio.Semaphore(max_concurrent)

    async def _evaluate(self, system: str, content: str | list[dict]) -> dict:
        schema = FreeFormJudgeResponse.model_json_schema()
        schema["additionalProperties"] = False
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            max_output_tokens=512,
            text={"format": {"type": "json_schema", "name": "vials_verdict", "schema": schema, "strict": True}},
        )
        total_in = total_out = 0
        for attempt in range(3):
            async with self._slots:
                response = await call_judge(
                    self.server_client,
                    server_name=self.server_name,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                )
            if response.usage is not None:
                total_in += response.usage.input_tokens
                total_out += response.usage.output_tokens
            parsed = _salvage_json(response.output_text)
            if parsed is not None:
                return {
                    "correct": parsed.equivalent,
                    "judge_reasoning": parsed.reasoning,
                    "judge_unverifiable": parsed.reasoning.lstrip().upper().startswith("UNVERIFIABLE:"),
                    "judge_input_tokens": total_in,
                    "judge_output_tokens": total_out,
                    "judge_model": response.model,
                }
            if attempt < 2:
                await asyncio.sleep(attempt + 1)
        # Gym distinguishes a received-but-unparseable reply from a failed API call.
        return {
            "correct": False,
            "judge_reasoning": "Could not parse a judge verdict after three attempts.",
            "judge_unparseable": True,
            "judge_input_tokens": total_in,
            "judge_output_tokens": total_out,
            "judge_model": response.model,
        }

    async def grade(self, answer: str, item: dict, model_input: list[dict], *, multimodal_fallback: bool) -> dict:
        """Grade text first, then use every original panel to resolve label ambiguity."""
        text_only = await self._evaluate(_JUDGE_SYSTEM, _build_judge_user_message(item, answer))
        if not (multimodal_fallback and text_only.get("judge_unverifiable")):
            return text_only
        images = [
            part
            for message in model_input
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "input_image"
        ]
        if not images:
            raise ValueError("Image-assisted grading requires images in the policy input")
        text = (
            _build_judge_user_message(item, answer).rsplit(
                "Are the candidate and ground-truth final answers equivalent?", 1
            )[0]
            + "Using the image only to resolve labelling ambiguities, are the candidate and ground-truth final answers equivalent?"
        )
        multimodal = await self._evaluate(_MULTIMODAL_SYSTEM, [*images, {"type": "input_text", "text": text}])
        return {
            **multimodal,
            "judge_multimodal": True,
            "judge_input_tokens": text_only["judge_input_tokens"] + multimodal["judge_input_tokens"],
            "judge_output_tokens": text_only["judge_output_tokens"] + multimodal["judge_output_tokens"],
        }
