# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from functools import partial
from typing import Any, Optional

from metrics import FINGERPRINTS, canonical_smiles, decode_safe, score_fragments, score_safe, score_smiles

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


logger = logging.getLogger(__name__)

# How the gold answer is compared. Task 3 answers are whole molecules, written
# either as SMILES or as SAFE; Tasks 1 and 2 answers are SAFE fragment sets,
# which are not parseable as molecules and are compared as string multisets.
#
# Tasks 1 and 2 share a reward but not a metric set: Appendix D.2 defines a
# fragment-level edit distance for Task 2 that Appendix D.1 does not define for
# Task 1. The two are separate modes so that the scoring mode alone determines
# which metrics are produced, rather than the scorer reaching for a second field.
SCORERS = {
    "smiles": score_smiles,
    "safe": score_safe,
    "fragments_task1": partial(score_fragments, task="task1"),
    "fragments_task2": partial(score_fragments, task="task2"),
}
DEFAULT_SCORING_MODE = "smiles"

# RDKit's canonical ranking recurses per atom, so `Chem.MolToSmiles` on a long
# enough chain overflows the C stack and kills the process — no Python exception,
# no HTTP 500, just SIGSEGV taking every concurrent request with it. Measured on
# an 8 MiB stack: 14,000 characters returns after 5 s, 15,000 segfaults.
#
# The longest gold answer in the 7,904-row test set is 328 characters, so nothing
# near this cap is a real attempt. It also bounds the O(n*m) Levenshtein DP.
MAX_ANSWER_CHARS = 2000


class MolDeToxStatus(str, Enum):
    SCORED = "scored"
    EMPTY_OUTPUT = "empty_output"
    NO_ANSWER = "no_answer"
    ANSWER_TOO_LONG = "answer_too_long"
    UNSUPPORTED_MODE = "unsupported_mode"
    BAD_GOLD = "bad_gold"
    BAD_METADATA = "bad_metadata"


# Statuses where a zero reward is a fault in the task row rather than a judgement
# on the policy. `failure_reason` is populated for exactly these, so a consumer
# can separate "the model got it wrong" from "this row was never scorable".
# The model-behaviour statuses (empty output, no answer, oversized answer) are
# deliberately absent: those zeros are real.
HARNESS_FAULTS = {
    MolDeToxStatus.UNSUPPORTED_MODE: "scoring_mode is not one of " + ", ".join(sorted(SCORERS)),
    MolDeToxStatus.BAD_GOLD: "gold answer is missing or unusable for the scoring mode",
    MolDeToxStatus.BAD_METADATA: "verifier_metadata field has the wrong type",
}


class MolDeToxRunRequest(BaseRunRequest):
    verifier_metadata: Optional[dict[str, Any]] = None


class MolDeToxVerifyRequest(MolDeToxRunRequest, BaseVerifyRequest):
    pass


class MolDeToxVerifyResponse(BaseVerifyResponse):
    status: Optional[str] = None
    extracted_answer: Optional[str] = None
    endpoint: Optional[str] = None
    dataset_name: Optional[str] = None
    levenshtein: Optional[float] = None
    # Molecule modes only (Task 3, smiles and safe).
    validity: Optional[float] = None
    tanimoto_rdk: Optional[float] = None
    tanimoto_maccs: Optional[float] = None
    tanimoto_morgan: Optional[float] = None
    # Fragment mode only (Tasks 1 and 2).
    fragment_f1: Optional[float] = None
    n_fragments_predicted: Optional[float] = None


class MolDeToxResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Promote accuracy and token counts only.

        The default promotes every `mean/*` metric, which would put the
        similarity scores on the headline line. Those are the one number on this
        benchmark that must not be read as performance: echoing the toxic input
        unchanged, the degenerate non-edit, scores a mean Morgan Tanimoto of
        0.711 while scoring zero accuracy. Validity behaves the same way, since
        copying the input is always valid.

        Exact-match accuracy is the reward and the only figure upstream reports
        as headline. The similarity metrics stay in `agent_metrics` for
        diagnosis; they are simply not promoted.
        """
        keep = ("mean/reward", "mean/input_tokens", "mean/output_tokens", "mean/total_tokens")
        return {name: agent_metrics[name] for name in keep if name in agent_metrics}

    async def verify(self, body: MolDeToxVerifyRequest) -> MolDeToxVerifyResponse:
        meta = body.verifier_metadata if isinstance(body.verifier_metadata, dict) else {}
        # `verifier_metadata` is typed `dict[str, Any]`, so nothing upstream
        # guarantees these are strings. They are echoed into a typed response and
        # used as a dict key and a string, each of which raises on the wrong type.
        # A malformed row must come back as a scored-zero with a reason, not as an
        # HTTP 500 that fails the whole eval.
        endpoint = _as_str(meta.get("endpoint"))
        dataset_name = _as_str(meta.get("dataset_name"))

        mode = meta.get("scoring_mode", DEFAULT_SCORING_MODE)
        if not isinstance(mode, str):
            logger.warning("scoring_mode is %s, not a string: %r", type(mode).__name__, mode)
            return _response(body, MolDeToxStatus.BAD_METADATA, None, None, endpoint, dataset_name)

        scorer = SCORERS.get(mode)
        if scorer is None:
            logger.warning("Unknown scoring_mode %r", mode)
            return _response(body, MolDeToxStatus.UNSUPPORTED_MODE, None, None, endpoint, dataset_name)

        gold = meta.get("answer")
        if gold is not None and not isinstance(gold, str):
            logger.warning("answer is %s, not a string: %r", type(gold).__name__, gold)
            return _response(body, MolDeToxStatus.BAD_METADATA, None, None, endpoint, dataset_name)

        if not gold or not _gold_is_usable(gold, mode):
            logger.warning("Gold answer missing or unusable for mode %r: %r", mode, gold)
            return _response(body, MolDeToxStatus.BAD_GOLD, None, None, endpoint, dataset_name)

        text = _extract_last_assistant_text(body)
        if not text:
            return _response(body, MolDeToxStatus.EMPTY_OUTPUT, None, None, endpoint, dataset_name)

        answer = _extract_json_answer(text)
        if answer is None:
            return _response(body, MolDeToxStatus.NO_ANSWER, None, None, endpoint, dataset_name)

        if len(answer) > MAX_ANSWER_CHARS:
            logger.warning("Answer of %d characters exceeds the %d cap", len(answer), MAX_ANSWER_CHARS)
            return _response(
                body, MolDeToxStatus.ANSWER_TOO_LONG, answer[:MAX_ANSWER_CHARS], None, endpoint, dataset_name
            )

        return _response(body, MolDeToxStatus.SCORED, answer, scorer(answer, gold), endpoint, dataset_name)


def _as_str(value: Any) -> Optional[str]:
    """Echo a provenance field only when it is actually a string.

    `endpoint` and `dataset_name` are reporting labels, so a wrong type should
    cost the label, not the row. Returning None keeps the typed response valid
    and leaves the prediction scorable.
    """
    if value is None or isinstance(value, str):
        return value
    logger.warning("Dropping non-string provenance value %r", value)
    return None


def _gold_is_usable(gold: str, mode: str) -> bool:
    if mode == "smiles":
        return canonical_smiles(gold) is not None
    if mode == "safe":
        return decode_safe(gold) is not None
    return bool(gold.strip())


def _response(
    body: MolDeToxVerifyRequest,
    status: MolDeToxStatus,
    extracted_answer: Optional[str],
    scores: Optional[dict],
    endpoint: Optional[str],
    dataset_name: Optional[str],
) -> MolDeToxVerifyResponse:
    scores = scores or {}
    extra = {
        "status": status.value,
        "extracted_answer": extracted_answer,
        "endpoint": endpoint,
        "dataset_name": dataset_name,
        "levenshtein": scores.get("levenshtein"),
    }
    # Only the metrics the scoring mode actually produced are populated; the rest
    # stay None so an absent metric is never read as a zero.
    for key in ("validity", "fragment_f1", "n_fragments_predicted", *(f"tanimoto_{k}" for k in FINGERPRINTS)):
        extra[key] = scores.get(key)

    return MolDeToxVerifyResponse(
        **body.model_dump(exclude=set(extra) | {"failure_reason"}),
        reward=scores.get("exact_match", 0.0),
        failure_reason=HARNESS_FAULTS.get(status),
        **extra,
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _extract_json_answer(text: str) -> Optional[str]:
    """Pull the SMILES out of the `{"answer": ...}` object the prompt demands.

    The whole response is tried first, since a well-behaved reply is bare JSON.
    Otherwise every fenced block and bare object is ordered by **position**,
    rightmost first, so a model that reasons before answering is read at its
    conclusion. Ordering by kind instead would let a fenced first draft outrank a
    later correction written in plain text.
    """
    located = [(m.start(), m.group(1)) for m in _FENCE_RE.finditer(text)]
    located += [(m.start(), m.group(0)) for m in _OBJECT_RE.finditer(text)]

    candidates = [text.strip()]
    candidates.extend(fragment for _, fragment in sorted(located, key=lambda pair: -pair[0]))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            answer = parsed["answer"]
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    return None


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
    MolDeToxResourcesServer.run_webserver()
