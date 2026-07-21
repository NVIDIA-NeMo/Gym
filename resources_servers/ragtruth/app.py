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
"""RAGTruth resources server — case-level hallucination detection.

Ported from the nemo-evaluator BYOB benchmarks ``ragtruth_qa`` /
``ragtruth_summary`` / ``ragtruth_data2txt``. Each task gives the model a
``(reference context, candidate_response)`` pair (already formatted into the
prompt by ``prepare_ragtruth.py``) and asks it to emit a
``{"hallucination list": [...]}`` JSON object. The per-sample reward is ``1.0``
when the model's binary "any hallucination?" verdict matches the gold label
(``is_halu``), else ``0.0``.

The three task slices (``QA``, ``Summary``, ``Data2txt``) differ only by the
prompt template applied at prep time; the scoring logic here is identical
across them. ``task_type`` rides on each row so ``compute_metrics`` can break
results down per slice.

``compute_metrics`` reports both the BYOB headline metric (mean per-sample
accuracy, also the reward) and the corpus-level precision / recall / F1 that
the original RAGTruth paper reports — reconstructed from the per-row
``is_halu`` (gold) and ``pred_halu`` (predicted) flags.

Build the dataset with ``prepare_ragtruth.py`` (fetches the public
ParticleMedia/RAGTruth GitHub dataset). Upstream:
``benchmarks/ragtruth/ragtruth/baseline/{dataset,prepare_dataset,predict_and_evaluate}.py``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse


LOG = logging.getLogger(__name__)

_FENCE_WHOLE_RE = re.compile(
    r"^```(?:json)?\s*\r?\n?(.*)\r?\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_INNER_RE = re.compile(
    r"```(?:json)?\s*\r?\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# ── Response parsing (adapted from byob_ragtruth.py) ──────────────────────


def _make_think_res(tag: str) -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    """Compile the three regexes needed to strip a thinking-model tag pair."""
    t = re.escape(tag)
    pair = re.compile(rf"<{t}>.*?</{t}>", re.DOTALL | re.IGNORECASE)
    orphan_close = re.compile(rf".*</{t}>\s*", re.DOTALL | re.IGNORECASE)
    orphan_open = re.compile(rf"<{t}>\s*", re.IGNORECASE)
    return pair, orphan_close, orphan_open


_DEFAULT_THINK_REGEXES = _make_think_res("think")


def _strip_think(text: str, tag: str = "think") -> str:
    if not text:
        return ""
    pair, orphan_close, orphan_open = _DEFAULT_THINK_REGEXES if tag == "think" else _make_think_res(tag)
    # Remove paired <tag>…</tag> blocks, then handle stray tags from thinking
    # models: a lone </tag> (reasoning emitted with no opening tag) drops
    # everything up to and including it; a lone <tag> is dropped on its own.
    cleaned = pair.sub("", text)
    cleaned = orphan_close.sub("", cleaned)
    cleaned = orphan_open.sub("", cleaned)
    return cleaned.strip()


def _strip_json_fence(text: str) -> str:
    if not text or "```" not in text:
        return text
    s = text.strip()
    whole = _FENCE_WHOLE_RE.match(s)
    if whole:
        return whole.group(1).strip()
    inner = _FENCE_INNER_RE.search(s)
    if inner:
        candidate = inner.group(1).strip()
        if candidate:
            return candidate
    return s


def _parse_response(raw: str, think_tag: str = "think") -> dict | None:
    cleaned = _strip_json_fence(_strip_think(raw, tag=think_tag))
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _has_hallucination(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    lst = parsed.get("hallucination list")
    return isinstance(lst, list) and len(lst) > 0


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


def _f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Binary precision / recall / F1 for the positive (hallucination) class."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Request / response shapes ─────────────────────────────────────────────


class RagtruthResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "ragtruth"
    # Tag name for stripping thinking-model reasoning blocks (e.g. "think" for
    # <think>…</think>, "reasoning" for <reasoning>…</reasoning>).
    think_tag: str = "think"


class RagtruthRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # Gold "does this case contain a hallucination?" label, precomputed at prep
    # time as ``bool(labels)`` from the upstream RAGTruth annotations.
    is_halu: bool = False
    task_type: str = ""


class RagtruthVerifyRequest(RagtruthRunRequest, BaseVerifyRequest):
    pass


class RagtruthVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_type: str = ""
    # Per-sample flags surfaced so corpus-level F1 can be recomputed downstream.
    is_halu: int = 0
    pred_halu: int = 0
    parse_fail: int = 0
    # The model's extracted hallucination list (truncated), for inspection.
    extracted: str = ""


class RagtruthResourcesServer(SimpleResourcesServer):
    config: RagtruthResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: RagtruthVerifyRequest) -> RagtruthVerifyResponse:
        parsed = _parse_response(_response_text(body.response), think_tag=self.config.think_tag)
        parse_fail = parsed is None
        pred_halu = _has_hallucination(parsed)
        is_halu = bool(body.is_halu)
        correct = pred_halu == is_halu

        if isinstance(parsed, dict) and isinstance(parsed.get("hallucination list"), list):
            extracted = json.dumps(parsed["hallucination list"], ensure_ascii=False)
        else:
            extracted = ""

        data = body.model_dump()
        data.pop("is_halu", None)  # re-emitted below as an int flag
        return RagtruthVerifyResponse(
            **data,
            reward=1.0 if correct else 0.0,
            is_halu=int(is_halu),
            pred_halu=int(pred_halu),
            parse_fail=int(parse_fail),
            extracted=extracted[:500],
        )

    # --- aggregation -----------------------------------------------------

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task_rollouts in tasks for r in task_rollouts]
        metrics: Dict[str, Any] = {}

        rewards = [r["reward"] for r in rows if isinstance(r.get("reward"), (int, float))]
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)
            metrics["count"] = len(rewards)

        parse_fails = [r.get("parse_fail", 0) for r in rows]
        if rows:
            metrics["parse_fail_rate"] = sum(parse_fails) / len(rows)

        # Corpus precision / recall / F1 over the binary halu labels, both
        # overall and broken down by task_type — the original RAGTruth metric.
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_type[r.get("task_type") or "unknown"].append(r)

        def _confusion(group: List[Dict[str, Any]]) -> Dict[str, Any]:
            tp = sum(1 for r in group if r.get("pred_halu") and r.get("is_halu"))
            fp = sum(1 for r in group if r.get("pred_halu") and not r.get("is_halu"))
            fn = sum(1 for r in group if not r.get("pred_halu") and r.get("is_halu"))
            grp_rewards = [r["reward"] for r in group if isinstance(r.get("reward"), (int, float))]
            out = _f1(tp, fp, fn)
            out["accuracy"] = sum(grp_rewards) / len(grp_rewards) if grp_rewards else 0.0
            out["count"] = len(group)
            return out

        if rows:
            overall = _confusion(rows)
            for key in ("precision", "recall", "f1"):
                metrics[key] = overall[key]

            for task_type, group in sorted(by_type.items()):
                stats = _confusion(group)
                for key, val in stats.items():
                    metrics[f"task_type/{task_type}/{key}"] = val

        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("mean_reward", "f1", "precision", "recall"):
            if k in agent_metrics:
                out[k] = agent_metrics[k]
        for k, v in agent_metrics.items():
            if k.endswith("/f1") or k.endswith("/accuracy"):
                out[k] = v
        return out


if __name__ == "__main__":
    RagtruthResourcesServer.run_webserver()
