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
"""SpartQA resources server — spatial reasoning as direct answer generation.

Ported from the nemo-evaluator BYOB benchmark ``spartqa``
(``benchmarks/spartqa/byob_spartqa.py``). The MTEB ``mteb/SpartQA`` retrieval
dataset is joined at prep time (``prepare_spartqa.py``) into one row per query
whose ``target`` is the accepted answer phrase (all accepted phrases in
``all_targets``). The model is shown the query and must return the matching
answer phrase, ending with a ``Final answer: <phrase>`` line.

The per-sample reward is ``1.0`` on an exact-or-answer-containing match against
any accepted answer, else ``0.0`` — so ``compute_metrics``'s mean-of-rewards
equals corpus accuracy. ``exact`` (strict match) and ``parsed`` (a non-empty
answer was extracted) ride on each row for downstream inspection.

The prompt and answer-extraction / scoring logic are ported verbatim from
``byob_spartqa.py`` so the metric is identical.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse


PROMPT = """\
Answer the spatial reasoning query below.
Return one concise final answer phrase. If the query gives answer options,
copy the matching option phrase exactly when possible.

End your response with one line in this exact format:
Final answer: <answer phrase>

Query:
{question}
"""


# ── Answer extraction + normalization (verbatim from byob_spartqa.py) ──────


def _normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    normalized = text.strip().lower().translate(table)
    return " ".join(normalized.split())


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<\|channel\>thought\s*.*?<channel\|>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _clean_candidate(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*•\s]+", "", text)
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    return text.strip().strip("\"'` ")


def _extract_answer(text: str) -> str:
    text = _strip_reasoning(text)
    patterns = [
        r"(?:^|\n)\s*(?:[*_`#>\-\s]*)final\s+answer(?:\s+is)?\s*(?:[*_`\s])*[:\-]\s*(.+)",
        r"(?:^|\n)\s*(?:[*_`#>\-\s]*)selected\s+answer(?:\s+is)?\s*(?:[*_`\s])*[:\-]\s*(.+)",
        r"(?:^|\n)\s*(?:[*_`#>\-\s]*)answer(?:\s+is)?\s*(?:[*_`\s])*[:\-]\s*(.+)",
        r"\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|would\s+be|should\s+be)\s*[:\-]?\s*(.+)",
        r"\bselected\s+(?:option|answer)\s+(?:is|would\s+be)\s*[:\-]?\s*(.+)",
    ]
    extracted = None
    for pattern in patterns:
        matches = list(
            re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        )
        if matches:
            extracted = matches[-1].group(1).strip()
            break
    if extracted is not None:
        text = extracted

    lines = [_clean_candidate(line) for line in text.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if line and _normalize(line) not in {"final answer", "answer"}
    ]
    if not lines:
        return ""

    if extracted is None and len(lines) > 1:
        first = _normalize(lines[0])
        if first.startswith(
            ("thinking process", "analysis", "the user wants", "we need")
        ):
            return lines[-1]
    return lines[0]


def _response_text(response: NeMoGymResponse) -> str:
    """Best-effort extraction of the assistant text from a NeMoGymResponse."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for c in content or []:
            t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


# ── Request / response shapes ─────────────────────────────────────────────


class SpartqaResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "spartqa"


class SpartqaRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # The first accepted answer phrase, and all accepted phrases. ``target``
    # (a scalar) survives the nemo-evaluator ``gym://...protocol=native`` driver,
    # which forwards a row's top-level scalar fields onto ``/verify`` but DROPS
    # list/dict fields (``all_targets`` never arrives that way). The full accepted
    # set therefore also rides in ``verifier_metadata``, which the driver forwards
    # intact; verify() falls back to it so all phrases are always available.
    target: str = ""
    all_targets: List[str] = Field(default_factory=list)
    verifier_metadata: Optional[Dict[str, Any]] = None


class SpartqaVerifyRequest(SpartqaRunRequest, BaseVerifyRequest):
    pass


class SpartqaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    exact: bool = False
    parsed: bool = False
    extracted: str = ""


class SpartqaResourcesServer(SimpleResourcesServer):
    config: SpartqaResourcesServerConfig

    async def verify(self, body: SpartqaVerifyRequest) -> SpartqaVerifyResponse:
        prediction = _extract_answer(_response_text(body.response))
        prediction_norm = _normalize(prediction)
        # Prefer the explicit list; fall back to verifier_metadata (the only path
        # that survives the native driver) and finally the scalar target.
        meta = body.verifier_metadata or {}
        targets = body.all_targets or meta.get("all_targets") or [
            body.target or meta.get("target", "")
        ]

        exact = False
        contains = False
        for target in targets:
            target_norm = _normalize(str(target))
            if not target_norm:
                continue
            if prediction_norm == target_norm:
                exact = True
                contains = True
                break
            if target_norm in prediction_norm:
                contains = True

        return SpartqaVerifyResponse(
            **body.model_dump(),
            reward=1.0 if (exact or contains) else 0.0,
            exact=exact,
            parsed=bool(prediction_norm),
            extracted=prediction[:200],
        )

    # --- aggregation -----------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task_rollouts in tasks for r in task_rollouts]
        if not rows:
            return {}

        metrics: Dict[str, Any] = {}
        rewards = [r["reward"] for r in rows if isinstance(r.get("reward"), (int, float))]
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)
            metrics["count"] = len(rewards)
        metrics["exact_match_rate"] = sum(1 for r in rows if r.get("exact")) / len(rows)
        metrics["parse_rate"] = sum(1 for r in rows if r.get("parsed")) / len(rows)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: agent_metrics[k]
            for k in ("mean_reward", "exact_match_rate")
            if k in agent_metrics
        }


if __name__ == "__main__":
    SpartqaResourcesServer.run_webserver()
