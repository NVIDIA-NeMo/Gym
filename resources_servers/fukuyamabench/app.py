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

import json
import logging
import re
from enum import Enum
from typing import Any, Optional

from metrics import score_pathway

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_subset_metrics


logger = logging.getLogger(__name__)

DEFAULT_LENIENT = True

# compute_subset_metrics prefixes each key with the case_set value, e.g.
# "B/pass@1/accuracy". Tiers are single upper-case letters upstream.
_TIER_METRIC_RE = re.compile(r"^[A-Z]/")


class FukuyamaBenchStatus(str, Enum):
    SCORED = "scored"
    EMPTY_OUTPUT = "empty_output"
    NO_PATHWAY = "no_pathway"
    BAD_GOLD = "bad_gold"


class FukuyamaBenchRunRequest(BaseRunRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class FukuyamaBenchVerifyRequest(FukuyamaBenchRunRequest, BaseVerifyRequest):
    pass


class FukuyamaBenchVerifyResponse(BaseVerifyResponse):
    status: Optional[str] = None
    case_id: Optional[str] = None
    case_set: Optional[str] = None
    checkpoints_correct: Optional[float] = None
    checkpoints_total: Optional[float] = None
    checkpoint_accuracy: Optional[float] = None
    any_checkpoint_correct: Optional[bool] = None
    pred_length: Optional[float] = None
    gt_steps: Optional[float] = None


class FukuyamaBenchResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Report each difficulty tier separately.

        Published Set B and Set C baselines differ by close to an order of
        magnitude, so a mean pooled over a mixed split describes no benchmark
        anyone reports. The default aggregation emits only that pooled scalar,
        so per-tier metrics are added here and a mixed run always carries them.
        """
        return compute_subset_metrics(tasks, "case_set")

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Promote the per-tier results and drop the pooled reward.

        Emitting tier metrics is not enough on its own: the inherited selection
        keeps every ``mean/*`` key, so ``mean/reward`` stays the headline that
        dashboards read — the exact cross-tier average this server exists to
        avoid. Once tier metrics are present, that pooled reward is removed.
        """
        tiers = {k: v for k, v in agent_metrics.items() if _TIER_METRIC_RE.match(k)}
        if not tiers:
            return super().get_key_metrics(agent_metrics)
        means = {k: v for k, v in agent_metrics.items() if k.startswith("mean/") and k != "mean/reward"}
        return {**means, **tiers}

    async def verify(self, body: FukuyamaBenchVerifyRequest) -> FukuyamaBenchVerifyResponse:
        meta = body.verifier_metadata or {}
        case_id = meta.get("case_id")
        case_set = meta.get("case_set")

        gt_pathway = meta.get("gt_pathway")
        checkpoints = meta.get("checkpoints")
        if not gt_pathway or not checkpoints:
            logger.warning("Missing gold pathway or checkpoints for case %r", case_id)
            return _response(body, FukuyamaBenchStatus.BAD_GOLD, None, case_id, case_set)

        text = _extract_last_assistant_text(body)
        if not text:
            return _response(body, FukuyamaBenchStatus.EMPTY_OUTPUT, None, case_id, case_set)

        pred_pathway = _extract_pathway(text)
        if pred_pathway is None:
            return _response(body, FukuyamaBenchStatus.NO_PATHWAY, None, case_id, case_set)

        scores = score_pathway(
            pred_pathway,
            gt_pathway,
            checkpoints,
            lenient=meta.get("lenient", DEFAULT_LENIENT),
        )
        return _response(body, FukuyamaBenchStatus.SCORED, scores, case_id, case_set)


def _response(
    body: FukuyamaBenchVerifyRequest,
    status: FukuyamaBenchStatus,
    scores: Optional[dict],
    case_id: Optional[str],
    case_set: Optional[str],
) -> FukuyamaBenchVerifyResponse:
    scores = scores or {}
    extra = {
        "status": status.value,
        "case_id": case_id,
        "case_set": case_set,
        # Left None when nothing was scored, so an unscored rollout is never read
        # as a run that matched zero checkpoints.
        "checkpoints_correct": scores.get("checkpoints_correct"),
        "checkpoints_total": scores.get("checkpoints_total"),
        "checkpoint_accuracy": scores.get("checkpoint_accuracy"),
        "any_checkpoint_correct": scores.get("any_checkpoint_correct"),
        "pred_length": scores.get("pred_length"),
        "gt_steps": scores.get("gt_steps"),
    }
    return FukuyamaBenchVerifyResponse(
        **body.model_dump(exclude=set(extra)),
        reward=1.0 if scores.get("exact_match") else 0.0,
        **extra,
    )


_FENCED_ARRAY_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_STEP_OBJECT_RE = re.compile(r'\{[^{}]*"step_id"[^{}]*\}', re.DOTALL)
_RESULT_HEADER_RE = re.compile(r"##\s*Result\s*\n")


def _is_pathway(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and any(isinstance(s, dict) and ("product_smiles" in s or "products" in s) for s in value)
    )


def _extract_pathway(text: str) -> Optional[list[dict]]:
    """Recover the predicted step array from the model's output.

    Tried in upstream's order: fenced JSON arrays (last first, so a model that
    revises its answer is read at its conclusion), then any balanced array after
    the ``## Result`` header, then loose per-step objects.
    """
    for block in reversed(_FENCED_ARRAY_RE.findall(text)):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if _is_pathway(parsed):
            return parsed

    header = _RESULT_HEADER_RE.search(text)
    search_text = text[header.end() :] if header else text
    for start in (m.start() for m in re.finditer(r"\[", search_text)):
        depth = 0
        for i in range(start, len(search_text)):
            if search_text[i] == "[":
                depth += 1
            elif search_text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(search_text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if _is_pathway(parsed):
                        return parsed
                    break

    steps = []
    for match in _STEP_OBJECT_RE.finditer(text):
        try:
            step = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if "product_smiles" in step or "products" in step:
            steps.append(step)
    return steps or None


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


if __name__ == "__main__":
    FukuyamaBenchResourcesServer.run_webserver()
