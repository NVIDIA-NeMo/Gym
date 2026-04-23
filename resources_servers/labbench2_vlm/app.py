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
"""
labbench2 VLM benchmark resource server.

Supports figqa2-img, figqa2-pdf, tableqa2-img, tableqa2-pdf.

JSONL rows store lightweight media references (``verifier_metadata.media_dir``)
instead of inline base64.  The custom ``labbench2_vlm_agent`` embeds images/PDFs
at rollout time via ``embed_media_into_row`` before sending to the model.
verify() scores the model's free-text answer against the GOLD ideal using an
LLM judge.
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
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
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LabbenchVLMConfig(BaseResourcesServerConfig):
    name: str = "labbench2_vlm"

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_prompt_template_fpath: str = "prompt_templates/judge.txt"
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"

    # Limit concurrent calls to the judge model. Set to None to disable.
    judge_endpoint_max_concurrency: Optional[int] = 64


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LabbenchVLMVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: Optional[str] = None


class LabbenchVLMVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None
    judge_evaluations: list[JudgeEvaluation] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_question(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Return the question text from the last user message.

    Our JSONL format puts a multimodal content list in the user message:
    [...image blocks..., {"type": "input_text", "text": "<question>"}]

    We walk the content blocks in reverse and return the first input_text we find.
    Falls back to the raw string content if the message is not multimodal.
    """
    for msg in reversed(params.input or []):
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in reversed(content):
                # Pydantic model block
                if getattr(block, "type", None) == "input_text":
                    return (getattr(block, "text", "") or "").strip()
                # Plain dict block (e.g. when loaded from JSON)
                if isinstance(block, dict) and block.get("type") == "input_text":
                    return (block.get("text") or "").strip()
        elif isinstance(content, str):
            return content.strip()
    return ""


def _extract_generated_answer(response: NeMoGymResponse) -> str:
    """Return the last assistant message text from the model response."""
    for item in reversed(response.output):
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            content = getattr(item, "content", None)
            if isinstance(content, list):
                texts = [getattr(c, "text", "") for c in content if getattr(c, "type", None) == "output_text"]
                return "\n".join(t for t in texts if t).strip()
            if isinstance(content, str):
                return content.strip()
    return ""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class LabbenchVLMResourcesServer(SimpleResourcesServer):
    config: LabbenchVLMConfig

    def model_post_init(self, __context: Any) -> None:
        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_sem = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_sem = nullcontext()

        with open(self.config.judge_prompt_template_fpath) as f:
            self._judge_prompt = f.read().strip()

    async def verify(self, body: LabbenchVLMVerifyRequest) -> LabbenchVLMVerifyResponse:
        meta = body.verifier_metadata or {}
        ideal = str(meta.get("ideal", ""))
        question = _extract_question(body.responses_create_params)
        generated = _extract_generated_answer(body.response)

        is_equal, evaluation = await self._judge(
            question=question,
            expected_answer=ideal,
            generated_answer=generated,
        )

        reward = 1.0 if is_equal else 0.0
        return LabbenchVLMVerifyResponse(
            **body.model_dump(),
            reward=reward,
            judge_evaluations=[evaluation],
        )

    async def _judge(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        cfg = self.config
        user_prompt = self._judge_prompt.format(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

        params = cfg.judge_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=user_prompt)]

        async with self._judge_sem:
            try:
                raw = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                )
                judge_response = NeMoGymResponse.model_validate(await get_response_json(raw))
            except Exception as e:
                print(f"[labbench2_vlm] judge HTTP error: {type(e).__name__}: {e}", flush=True)
                raise

        eval_record = JudgeEvaluation(
            responses_create_params=params,
            response=judge_response,
            verdict_label=None,
        )

        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return False, eval_record
            text = getattr(last_output.content[-1], "text", "")
        except Exception:
            return False, eval_record

        eq_pos = text.find(cfg.judge_equal_label)
        neq_pos = text.find(cfg.judge_not_equal_label)

        if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
            eval_record.verdict_label = cfg.judge_equal_label
            return True, eval_record

        eval_record.verdict_label = cfg.judge_not_equal_label if neq_pos >= 0 else None
        return False, eval_record

    # -------------------------------------------------------------------------
    # Aggregate metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _score_fn(r: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {"accuracy": float(r.get("reward", 0.0))}
        evaluations = r.get("judge_evaluations") or []
        if evaluations:
            verdict = (evaluations[0] or {}).get("verdict_label")
            scores["judge_no_verdict"] = 1.0 if verdict is None else 0.0
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Overall pass@k/accuracy plus per-tag breakdown (figqa2-img, tableqa2-pdf, …)."""
        overall, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)

        # Per-tag subset metrics — tag lives at verifier_metadata.tag
        subsets: Dict[str, List[List[Dict[str, Any]]]] = {}
        for task_rollouts in tasks:
            tag = ((task_rollouts[0].get("verifier_metadata") or {}).get("tag")) if task_rollouts else None
            if tag:
                subsets.setdefault(tag, []).append(task_rollouts)

        metrics: Dict[str, Any] = {**overall}
        for tag, subset_tasks in subsets.items():
            subset_m, _, _, _ = compute_pass_majority_metrics(subset_tasks, score_fn=self._score_fn)
            for key, value in subset_m.items():
                if key == "per_sample_aggregate":
                    continue
                metrics[f"{tag}/{key}"] = value

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        return key


if __name__ == "__main__":
    LabbenchVLMResourcesServer.run_webserver()
