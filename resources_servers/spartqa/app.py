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
"""SpartQA resources server — the CO (choose-object) question type.

``mteb/SpartQA`` is the MTEB retrieval form of SpartQA (Mirzaee et al., NAACL
2021) and contains CO questions only: a story plus "… <object X> or <object
Y>?". Per the paper (Table 8 fn.), CO is a **four-label single-choice** task —
``X``, ``Y``, ``both of them``, ``none of them`` — and the metric is accuracy
over that label set.

``prepare_spartqa.py`` recovers that label set from the retrieval qrels (which
flatten "both of them" into three relevant documents), renders the four
candidates into the prompt, and stores the single gold label in ``target`` and
the two object labels in ``options``.

The per-sample reward is ``1.0`` iff the model's answer resolves to the gold
label, else ``0.0`` — so ``compute_metrics``'s mean-of-rewards equals CO
accuracy. Resolution is a verbatim match against one candidate, or a
containment match when it is *unambiguous*: a response naming two candidates
(e.g. echoing "X or Y" back) resolves to nothing and scores 0.0. Scoring any of
the three flattened qrels phrases as correct would let that echo score ~95%.
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
Answer the spatial reasoning question below.
Choose exactly one of the candidate answers and copy it verbatim. Answer
"both of them" if both listed objects satisfy the question, and "none of them"
if neither of them does.

End your response with one line in this exact format:
Final answer: <candidate answer>

Story and question:
{question}

Candidate answers:
{candidates}
"""

# The two story-independent CO labels. Aliases let a model phrase them its own
# way; each alias maps to exactly one label, so this adds no leniency across
# labels.
BOTH_LABEL = "both of them"
NONE_LABEL = "none of them"

_BOTH_ALIASES = frozenset({"both", "both of them", "both objects", "both of the objects", "both of these"})
# The paper treats DK / None / [] alike: none of the candidate objects hold.
_NONE_ALIASES = frozenset(
    {
        "none",
        "none of them",
        "none of the objects",
        "neither",
        "neither of them",
        "neither object",
        "no object",
        "dk",
        "do not know",
        "dont know",
    }
)


# ── Answer extraction + normalization ───────────────────────────────────────


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
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        # Drop the instruction-template echo (``Final answer: <candidate
        # answer>``) and empty captures; keep the last remaining real answer.
        candidates = [
            captured
            for m in matches
            if (captured := m.group(1).strip()) and _normalize(captured) != "candidate answer"
        ]
        if candidates:
            extracted = candidates[-1]
            break
    if extracted is not None:
        text = extracted

    lines = [_clean_candidate(line) for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line and _normalize(line) not in {"final answer", "answer"}]
    if not lines:
        return ""

    if extracted is None and len(lines) > 1:
        first = _normalize(lines[0])
        if first.startswith(("thinking process", "analysis", "the user wants", "we need")):
            return lines[-1]
    return lines[0]


def _label_key(text: str) -> str:
    """Normalize a candidate label or a prediction for comparison.

    Article-insensitive, because the qrels phrases and the story's own wording
    disagree on leading articles ("medium yellow square …" vs "the medium
    yellow square …").
    """
    return re.sub(r"^(?:a|an|the)\s+", "", _normalize(text)).strip()


def candidate_labels(options: List[str]) -> List[str]:
    """The four CO labels: the two story objects, then ``both`` / ``none``."""
    labels: List[str] = []
    seen: set[str] = set()
    for label in [*options, BOTH_LABEL, NONE_LABEL]:
        key = _label_key(str(label))
        if key and key not in seen:
            seen.add(key)
            labels.append(str(label).strip())
    return labels


def match_label(prediction: str, labels: List[str]) -> Optional[str]:
    """Resolve a free-text answer to exactly one candidate label, or ``None``.

    Verbatim (article-insensitive) equality wins outright. Otherwise the answer
    must *contain* exactly one label — a response naming two of them is
    ambiguous and resolves to ``None`` rather than being credited for whichever
    one happens to be gold.
    """
    key = _label_key(prediction)
    if not key:
        return None

    by_key = {_label_key(label): label for label in labels}
    if key in by_key:
        return by_key[key]
    if key in _BOTH_ALIASES and BOTH_LABEL in by_key.values():
        return by_key[_label_key(BOTH_LABEL)]
    if key in _NONE_ALIASES and NONE_LABEL in by_key.values():
        return by_key[_label_key(NONE_LABEL)]

    hits = [label_key for label_key in by_key if label_key and label_key in key]
    # One option can nest inside the other ("a triangle in block B" inside "a
    # big blue triangle in block B"); the most specific match is the intended
    # one, so drop labels wholly contained in another hit.
    hits = [h for h in hits if not any(other != h and h in other for other in hits)]
    return by_key[hits[0]] if len(hits) == 1 else None


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

    # ``target`` is the single gold CO label; ``options`` are the two story
    # objects offered by the question. ``target`` (a scalar) survives the
    # nemo-evaluator ``gym://...protocol=native`` driver, which forwards a row's
    # top-level scalar fields onto ``/verify`` but DROPS list/dict fields
    # (``options`` never arrives that way). The options therefore also ride in
    # ``verifier_metadata``, which the driver forwards intact; verify() falls
    # back to it so the full candidate set is always available.
    target: str = ""
    options: List[str] = Field(default_factory=list)
    verifier_metadata: Optional[Dict[str, Any]] = None


class SpartqaVerifyRequest(SpartqaRunRequest, BaseVerifyRequest):
    pass


class SpartqaVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    exact: bool = False
    parsed: bool = False
    extracted: str = ""
    predicted_label: str = ""


class SpartqaResourcesServer(SimpleResourcesServer):
    config: SpartqaResourcesServerConfig

    async def verify(self, body: SpartqaVerifyRequest) -> SpartqaVerifyResponse:
        prediction = _extract_answer(_response_text(body.response))
        # Prefer the explicit fields; fall back to verifier_metadata, the only
        # path that survives the native driver.
        meta = body.verifier_metadata or {}
        gold = body.target or str(meta.get("target", ""))
        options = body.options or list(meta.get("options") or [])

        labels = candidate_labels(options) if options else candidate_labels([gold])
        predicted = match_label(prediction, labels)
        gold_key = _label_key(gold)
        correct = bool(gold_key) and predicted is not None and _label_key(predicted) == gold_key

        return SpartqaVerifyResponse(
            **body.model_dump(),
            reward=1.0 if correct else 0.0,
            # ``exact`` means the gold label was copied verbatim, as instructed,
            # rather than recovered from a longer sentence.
            exact=correct and _label_key(prediction) == gold_key,
            parsed=bool(_normalize(prediction)),
            extracted=prediction[:200],
            predicted_label=predicted or "",
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
        # Fraction of answers that resolved to some candidate label at all; a low
        # value means the model is not following the copy-a-candidate format.
        metrics["label_resolve_rate"] = sum(1 for r in rows if r.get("predicted_label")) / len(rows)
        # Accuracy split by gold label. A model that ignores "both of them" —
        # the gold for 44% of the corpus — shows up here as a near-zero slice
        # even when the headline number looks healthy.
        for label in (BOTH_LABEL, NONE_LABEL):
            slice_rows = [r for r in rows if _label_key(str(r.get("target", ""))) == _label_key(label)]
            if slice_rows:
                key = label.replace(" ", "_")
                metrics[f"accuracy_{key}"] = sum(1 for r in slice_rows if r.get("reward") == 1.0) / len(slice_rows)
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {k: agent_metrics[k] for k in ("mean_reward", "exact_match_rate") if k in agent_metrics}


if __name__ == "__main__":
    SpartqaResourcesServer.run_webserver()
