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
"""Generic VLMEvalKit driver (audio-first) as a NeMo Gym resources server.

A single generic scorer that dispatches on the
VLMEvalKit ``DATASET_TYPE`` (MCQ / QA / Y-N / VQA) instead of a hand-written
``_score_<bench>`` per benchmark. Audio rows carry their clip via
``responses_create_params.metadata.audio_path`` / ``audio_data`` — the
``vllm_model`` server splices it into an ``audio_url`` content block (see
responses_api_models/vllm_model/app.py). Inference is done by Gym's model
server; this server only prepares data (prepare_data.py) and scores here.

Scoring reuses VLMEvalKit's ``can_infer`` for option-letter extraction when the
``vlmeval`` package is importable (the faithful path, used on the cluster /
container). When it is not (offline unit tests, lightweight venv), it falls
back to an equivalent local extractor so ``verify()`` never hard-depends on the
heavy ``vlmeval`` import chain (torch/opencv/...). Hard types (LLM-judge,
IoU-grounding, ASR-WER, circular MMBench) stay bespoke — not covered here.
"""

import re
import string
from typing import Any, Dict, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

# Dataset types (VLMEvalKit ``DATASET_TYPE``) scored generatively here.
_MCQ_TYPES = {"MCQ", "QA", "MCQ_MMMU_PRO"}
_YORN_TYPES = {"Y/N", "YORN", "Y-N"}


def _extract_text(response: Any) -> str:
    """Concatenate assistant text from a Responses-API result.

    Prefers the ``output_text`` convenience accessor (as the image
    ``vlm_eval_kit`` server does); falls back to walking ``output`` items.
    """
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    return "".join(parts)


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` reasoning before parsing the answer.

    Thinking models leak option letters inside the reasoning block; scoring the
    raw text would match those instead of the final answer.
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def infer_option_vlmeval(prediction: str, choices: Dict[str, str]) -> Optional[str]:
    """Extract an option letter via VLMEvalKit's native ``can_infer`` (reuse backend).

    Requires the ``vlmeval`` package to be installed; raises loudly if it is
    not, so a run configured for the vlmeval backend never silently degrades.
    ``can_infer`` returns a letter, ``"Z"`` (no match), or ``False``.
    """
    from vlmeval.utils.matching_util import can_infer

    result = can_infer(prediction, choices if choices else {k: k for k in "ABCD"})
    if result and result != "Z":
        return str(result)
    return None


def infer_option(prediction: str, choices: Dict[str, str]) -> Optional[str]:
    """Extract a single option letter — dependency-free local implementation."""
    valid = set(choices.keys()) if isinstance(choices, dict) and choices else set("ABCD")

    match = re.search(r"(?i)\b(?:answer|option|choice)\b\D{0,12}([A-Z])\b", prediction)
    if match and match.group(1) in valid:
        return match.group(1)

    clean = prediction
    for ch in ".()[],:;!*#{}\"'":
        clean = clean.replace(ch, " ")
    tokens = clean.split()
    found = [t for t in tokens if t in valid]
    if len(found) == 1:
        return found[0]
    for t in tokens:
        if t in valid:
            return t
    return None


class VlmEvalKitAudioResourcesServerConfig(BaseResourcesServerConfig):
    # Option-letter extraction backend: "local" = dependency-free reimplementation
    # (default); "vlmeval" = VLMEvalKit's native can_infer (requires the vlmeval
    # package — fails loudly if missing, never silently falls back).
    scoring_backend: str = "local"


class VlmEvalKitAudioVerifyRequest(BaseVerifyRequest):
    # Many VLMEvalKit benchmarks run through this one server, so allow extra
    # per-benchmark fields to ride on the request.
    model_config = ConfigDict(extra="allow")

    benchmark_name: str = ""
    dataset_type: str = "QA"
    answer: Any = None
    choices: Dict[str, str] = {}


class VlmEvalKitAudioVerifyResponse(VlmEvalKitAudioVerifyRequest, BaseVerifyResponse):
    pass


class VlmEvalKitAudioResourcesServer(SimpleResourcesServer):
    config: VlmEvalKitAudioResourcesServerConfig

    def _score(self, dataset_type: str, prediction: str, answer: Any, choices: Dict[str, str]) -> Dict[str, Any]:
        dtype = (dataset_type or "QA").upper()
        backend = self.config.scoring_backend
        if backend not in ("local", "vlmeval"):
            raise ValueError(f"Unsupported scoring_backend: {backend!r}. Use 'local' or 'vlmeval'.")

        if dtype in _MCQ_TYPES:
            extract = infer_option_vlmeval if backend == "vlmeval" else infer_option
            pred = extract(prediction, choices)
            correct = pred is not None and answer is not None and str(pred).upper() == str(answer).strip().upper()
            return {"reward": 1.0 if correct else 0.0, "extracted": pred, "scoring_backend": backend}

        if dtype in _YORN_TYPES:
            low = prediction.lower()
            pred = "yes" if "yes" in low else ("no" if "no" in low else None)
            correct = pred is not None and answer is not None and pred == str(answer).strip().lower()
            return {"reward": 1.0 if correct else 0.0, "extracted": pred}

        # VQA / default: exact or containment match (naive; bespoke scorers
        # handle anls/judge for the harder VQA benchmarks).
        pred_norm = prediction.strip().lower()
        ans_norm = str(answer).strip().lower() if answer is not None else ""
        correct = bool(ans_norm) and (pred_norm == ans_norm or ans_norm in pred_norm)
        return {"reward": 1.0 if correct else 0.0, "extracted": prediction[:200]}

    async def verify(self, body: VlmEvalKitAudioVerifyRequest) -> VlmEvalKitAudioVerifyResponse:
        prediction = strip_think(_extract_text(body.response))
        score = self._score(body.dataset_type, prediction, body.answer, body.choices)
        return VlmEvalKitAudioVerifyResponse(**body.model_dump(), **score)


if __name__ == "__main__":
    VlmEvalKitAudioResourcesServer.run_webserver()
