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

"""NMRArena verifier: ranked SMILES from 1H/13C NMR peak lists, scored by exact structure match.

The policy answers with a JSON object of up to ten ranked SMILES. Following
upstream (odanchem/NMRArena), the candidates are canonicalised with RDKit with
stereochemistry stripped, de-duplicated and truncated to ten; a molecule is solved
when a candidate equals the canonical truth. ``reward`` is Top-1 (the rank-1
candidate is the truth); ``top10`` and the rank-1 Tanimoto similarity are carried
as additional fields. Everything is deterministic and model-free.
"""

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel
from scoring import DEFAULT_MAX_SMILES_CHARS, canonical, parse_candidates, sanitize_strings, score_candidates

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


logger = logging.getLogger(__name__)


class NMRArenaStatus(str, Enum):
    SCORED = "scored"  # at least one valid candidate was compared with the truth
    EMPTY_OUTPUT = "empty_output"  # no assistant text at all
    FORMAT_FAIL = "format_fail"  # no loadable ``"candidates"`` JSON in the text (upstream's name)
    NO_VALID_CANDIDATE = "no_valid_candidate"  # JSON found, but nothing in it parses as a molecule
    INVALID_CANDIDATE = "invalid_candidate"  # strict mode: an invalid, oversize or duplicate member
    BAD_METADATA = "bad_metadata"  # harness fault: the row's truth is missing or unparseable


# The only status where a zero reward is not the policy's doing.
HARNESS_FAULTS = {NMRArenaStatus.BAD_METADATA: "verifier_metadata lacks a parseable gold smiles"}


class NMRArenaResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.STATELESS

    # Upstream ``NUM_CANDIDATES``: candidates kept per prediction and the k of Top-k.
    num_candidates: int = 10
    # Longest model-controlled string RDKit is allowed to parse; see ``scoring.py``.
    max_smiles_chars: int = DEFAULT_MAX_SMILES_CHARS
    # False reproduces upstream: an invalid, oversize or duplicate entry is a miss at its
    # position and the rest of the list is scored. True disqualifies the whole prediction.
    strict_candidates: bool = False
    # Recover complete candidate objects from a JSON block cut off at the token budget,
    # as upstream's published per-item lists do. False reproduces the notebook parser.
    salvage_truncated_json: bool = True


class NMRArenaRunRequest(BaseRunRequest):
    # ``Any`` on purpose: a row whose metadata is not an object must still reach ``verify``
    # and come back as a labelled harness fault rather than a 422.
    verifier_metadata: Optional[Any] = None


class NMRArenaVerifyRequest(NMRArenaRunRequest, BaseVerifyRequest):
    pass


class NMRArenaVerifyResponse(BaseVerifyResponse):
    status: Optional[str] = None
    compound_id: Optional[str] = None
    primary_class: Optional[str] = None
    n_complex: Optional[float] = None
    truth_canonical: Optional[str] = None
    # Canonical SMILES per position, ``None`` where the entry did not parse.
    candidates: list[Optional[str]] = []
    n_raw_candidates: Optional[int] = None
    n_valid: int = 0
    n_invalid: int = 0
    n_oversize: int = 0
    n_duplicate: int = 0
    salvaged: bool = False
    hit_rank: Optional[int] = None
    top1: float = 0.0
    top10: float = 0.0
    # Upstream's Tanimoto is conditional on the position-1 candidate parsing; ``None``
    # otherwise, which the aggregator skips, so ``mean/tanimoto_top1`` is the conditional mean.
    tanimoto_top1: Optional[float] = None
    answered: float = 0.0
    # True when the model response reports it stopped at ``max_output_tokens``.
    response_incomplete: bool = False
    harness_failure: float = 0.0


class _Task(BaseModel):
    smiles: str
    compound_id: Optional[str] = None
    primary_class: Optional[str] = None
    n_complex: Optional[float] = None


def _parse_task(meta: Any, max_chars: int) -> tuple[_Task, str]:
    """Validate the row. A bad truth is a harness fault; a bad provenance label only costs the label."""
    if not isinstance(meta, dict):
        raise ValueError("verifier_metadata is not an object")
    smiles = meta.get("smiles")
    if not isinstance(smiles, str):
        raise ValueError("gold smiles is missing or not a string")
    truth = canonical(smiles, max_chars=max_chars)
    if truth is None:
        raise ValueError("gold smiles does not parse")
    n_complex = meta.get("n_complex")
    task = _Task(
        smiles=smiles,
        compound_id=meta.get("compound_id") if isinstance(meta.get("compound_id"), str) else None,
        primary_class=meta.get("primary_class") if isinstance(meta.get("primary_class"), str) else None,
        n_complex=float(n_complex)
        if isinstance(n_complex, (int, float)) and not isinstance(n_complex, bool)
        else None,
    )
    return task, truth


class NMRArenaResourcesServer(SimpleResourcesServer):
    config: NMRArenaResourcesServerConfig

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Make the Tanimoto denominator explicit next to the conditional mean the profiler computes."""
        rows = [r for rollouts in tasks for r in rollouts]
        answered = [r for r in rows if isinstance(r.get("tanimoto_top1"), (int, float))]
        out: dict[str, Any] = {"count/rows": len(rows), "count/answered": len(answered)}
        if answered:
            out["tanimoto_top1/answered_only"] = sum(r["tanimoto_top1"] for r in answered) / len(answered)
        return out

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Top-1, Top-10, the answer rate and the harness-fault rate.

        The conditional Tanimoto is deliberately not a headline: upstream reports it
        "conditional on valid outputs", and a model that answers only when confident is
        paid for its silence (upstream's own footnote shows 0.89 on 9 of 315 answers). It
        stays in ``agent_metrics`` as ``mean/tanimoto_top1`` and
        ``tanimoto_top1/answered_only`` next to ``count/answered``.
        """
        keep = {
            "mean/reward",
            "mean/top1",
            "mean/top10",
            "mean/answered",
            "mean/harness_failure",
            "mean/response_incomplete",
            "mean/input_tokens",
            "mean/output_tokens",
            "mean/total_tokens",
        }
        return {k: v for k, v in agent_metrics.items() if k in keep}

    async def verify(self, body: NMRArenaVerifyRequest) -> NMRArenaVerifyResponse:
        try:
            task, truth = _parse_task(body.verifier_metadata, self.config.max_smiles_chars)
        except ValueError as e:
            logger.warning("Unusable verifier_metadata: %s", e)
            return _response(body, NMRArenaStatus.BAD_METADATA)

        text = sanitize_strings(_extract_last_assistant_text(body))
        if not text:
            return _response(body, NMRArenaStatus.EMPTY_OUTPUT, task=task, truth=truth)

        parsed = parse_candidates(
            text,
            n=self.config.num_candidates,
            max_smiles_chars=self.config.max_smiles_chars,
            strict=self.config.strict_candidates,
            salvage=self.config.salvage_truncated_json,
        )
        if parsed.disqualified is not None:
            status = NMRArenaStatus.INVALID_CANDIDATE
        elif not parsed.found_json:
            status = NMRArenaStatus.FORMAT_FAIL
        elif parsed.n_valid == 0:
            status = NMRArenaStatus.NO_VALID_CANDIDATE
        else:
            status = NMRArenaStatus.SCORED
        return _response(body, status, task=task, truth=truth, parsed=parsed)


def _response(
    body: NMRArenaVerifyRequest,
    status: NMRArenaStatus,
    *,
    task: Optional[_Task] = None,
    truth: Optional[str] = None,
    parsed=None,
) -> NMRArenaVerifyResponse:
    candidates = parsed.candidates if parsed is not None else []
    scores = score_candidates(truth, candidates) if truth is not None else None
    extra: dict[str, Any] = {
        "status": status.value,
        "compound_id": task.compound_id if task else None,
        "primary_class": task.primary_class if task else None,
        "n_complex": task.n_complex if task else None,
        "truth_canonical": truth,
        "candidates": candidates,
        "n_raw_candidates": parsed.n_raw if parsed is not None else None,
        "n_valid": parsed.n_valid if parsed is not None else 0,
        "n_invalid": parsed.n_invalid if parsed is not None else 0,
        "n_oversize": parsed.n_oversize if parsed is not None else 0,
        "n_duplicate": parsed.n_duplicate if parsed is not None else 0,
        "salvaged": parsed.salvaged if parsed is not None else False,
        "hit_rank": scores.hit_rank if scores else None,
        "top1": scores.top1 if scores else 0.0,
        "top10": scores.top10 if scores else 0.0,
        "tanimoto_top1": scores.tanimoto_top1 if scores else None,
        "answered": scores.answered if scores else 0.0,
        "response_incomplete": _response_incomplete(body),
        "harness_failure": 1.0 if status in HARNESS_FAULTS else 0.0,
        "failure_reason": HARNESS_FAULTS.get(status),
    }
    fields = set(NMRArenaVerifyResponse.model_fields)
    # The echoed request is model-controlled too: sanitise it before it is re-encoded.
    base = sanitize_strings(body.model_dump(exclude={k for k in fields if k in extra}))
    return NMRArenaVerifyResponse(**base, reward=extra["top1"], **extra)


def _response_incomplete(body: NMRArenaVerifyRequest) -> bool:
    response = body.response
    if getattr(response, "status", None) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) == "max_output_tokens"


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
    NMRArenaResourcesServer.run_webserver()
