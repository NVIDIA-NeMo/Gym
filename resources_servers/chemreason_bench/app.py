# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemReason-Bench resources server.

Single-turn, deterministic, no tools and no code execution: the model returns one
JSON object and a pure-Python scorer compares it to gold. Six task types are
dispatched on ``task_type``; see ``metrics.py`` for the formulas and their
provenance.

Reward vs. reported metric
--------------------------
``reward`` is a per-row signal in [0, 1]. It is NOT the benchmark's metric. Four
of the six published metrics are only defined over a corpus -- ``f1_positive``
needs the full confusion matrix, ``step_completion_score`` applies a
corpus-level format-error penalty -- so the headline numbers are computed in
``compute_metrics`` by reducing per-row contributions, not by averaging rewards.
``mean/reward`` is therefore deliberately kept out of ``get_key_metrics``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import metrics as M
from pydantic import model_validator
from response_parsing import extract_json, to_prediction, to_prediction_lm

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ChemReasonBenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ChemReasonBenchVerifyRequest(BaseVerifyRequest):
    # Prepared benchmark rows are flat, so these arrive at the top level; the
    # committed example.jsonl nests them under `verifier_metadata` instead.
    # Accept both rather than requiring one shape.
    verifier_metadata: Optional[Dict[str, Any]] = None
    task_type: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    benchmark_id: Optional[int] = None
    # Question-side vocabulary upstream's post-processors need. Not gold.
    expected_step_ids: Optional[List[str]] = None
    options: Optional[List[Any]] = None
    # "gen" (JSON reply) or "lm" (bare decision token). Rows without it are gen,
    # so a dataset prepared before the lm protocol existed still scores.
    protocol: str = "gen"

    @model_validator(mode="before")
    @classmethod
    def _lift_verifier_metadata(cls, data: Any) -> Any:
        """Accept the row's fields nested under `verifier_metadata` or at the top level.

        Top level wins, matching the sibling servers. Without this a flat row
        reaches verify() with task_type unset and every instance is charged to
        the harness -- which is exactly how the first smoke run scored 0.00 on
        all six tasks with bad_task_type=50.
        """
        if isinstance(data, dict) and isinstance(data.get("verifier_metadata"), dict):
            return {**data["verifier_metadata"], **data}
        return data


class ChemReasonBenchVerifyResponse(ChemReasonBenchVerifyRequest, BaseVerifyResponse):
    status: Optional[str] = None
    # Per-row contributions; compute_metrics reduces these into the six primaries.
    contributions: Optional[Dict[str, float]] = None
    harness_failure: Optional[bool] = None


def _first_output_logprobs(response: Any) -> Any:
    """Per-token logprobs of the first output text part, or None.

    Only lm rows request them (prepare.py sets logprobs/top_logprobs on those
    rows), so this is None for every gen row and the text path is used.
    """
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            logprobs = getattr(part, "logprobs", None)
            if logprobs:
                return [lp if isinstance(lp, dict) else lp.model_dump() for lp in logprobs]
    return None


def _sanitize(text: Optional[str]) -> Optional[str]:
    """Drop lone surrogates so the response can be encoded for the wire.

    A surrogate reaching the JSON encoder raises while the response is being
    built, which no guard inside the scorer can catch.
    """
    if text is None:
        return None
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


class ChemReasonBenchResourcesServer(SimpleResourcesServer):
    """Scores one ChemReason-Bench instance against its gold record."""

    config: ChemReasonBenchResourcesServerConfig

    async def verify(self, body: ChemReasonBenchVerifyRequest) -> ChemReasonBenchVerifyResponse:
        task_type = body.task_type
        ground_truth = body.ground_truth
        # `task_id` and `task_type` are declared on the request, so they are already
        # in model_dump(); passing them again as keywords is a TypeError. Build the
        # payload once and override.
        payload = body.model_dump()
        payload["task_id"] = _sanitize(payload.get("task_id"))

        # A malformed row is the harness's fault, not the model's: report it as a
        # status so the run continues and the rate stays visible, rather than
        # raising a 500 that would end the whole job.
        if not isinstance(task_type, str) or task_type not in M.TASK_TYPES:
            return ChemReasonBenchVerifyResponse(**payload, reward=0.0, status="bad_task_type", harness_failure=True)
        if not isinstance(ground_truth, dict):
            return ChemReasonBenchVerifyResponse(
                **payload, reward=0.0, status="bad_ground_truth", harness_failure=True
            )

        if body.protocol == "lm":
            if task_type not in M.DUAL_PROTOCOL_TASKS:
                return ChemReasonBenchVerifyResponse(
                    **payload, reward=0.0, status="no_lm_protocol", harness_failure=True
                )
            prediction = to_prediction_lm(task_type, body.response.output_text, _first_output_logprobs(body.response))
            status = prediction.pop("status")
        else:
            raw = body.response.output_text or ""
            parsed, status = extract_json(raw)
            prediction = to_prediction(task_type, parsed, body.expected_step_ids, body.options, raw)
        scored = M.score_row(task_type, prediction, ground_truth)
        reward = float(scored.pop("reward"))

        return ChemReasonBenchVerifyResponse(
            **payload,
            reward=reward,
            status=status,
            contributions={k: float(v) for k, v in scored.items()},
            harness_failure=False,
        )

    # ---------------------------------------------------------------- metrics

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Reduce per-row contributions into the six published primary metrics.

        For the three discriminative tasks the published primary metric is the
        mean of the two protocols (paper appendix F.3.4, m_t = (m_gen+m_lm)/2),
        so gen and lm rows are reduced separately and then averaged. Scoring gen
        alone and calling the result Primary-Overall is NOT the paper's number.
        A task with only one protocol present falls back to that protocol, which
        is what makes a gen-only dataset still score.
        """
        by_key: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
        harness_failures = 0
        total = 0
        for task in tasks:
            for rollout in task:
                total += 1
                if rollout.get("harness_failure"):
                    harness_failures += 1
                    continue
                task_type = rollout.get("task_type")
                contributions = rollout.get("contributions")
                if task_type in M.TASK_TYPES and isinstance(contributions, dict):
                    by_key[(task_type, rollout.get("protocol") or "gen")].append(contributions)

        out: Dict[str, Any] = {}
        per_task: Dict[str, float] = {}
        for task_type in M.TASK_TYPES:
            present = {}
            for protocol in ("gen", "lm"):
                rows = by_key.get((task_type, protocol))
                if rows:
                    present[protocol] = M.reduce_task(task_type, rows)
                    out[f"{task_type}/{M.PRIMARY_METRIC_BY_TASK[task_type]}[{protocol}]"] = present[protocol] * 100.0
                    out[f"{task_type}/count[{protocol}]"] = float(len(rows))
            value = sum(present.values()) / len(present) if present else 0.0
            per_task[task_type] = value
            out[f"{task_type}/{M.PRIMARY_METRIC_BY_TASK[task_type]}"] = value * 100.0
            out[f"{task_type}/protocols"] = float(len(present))

        out["primary_overall"] = M.primary_overall(per_task) * 100.0
        # Published as a score, not filtered out silently. These rollouts also
        # score reward 0, so nothing is dropped from any denominator.
        out["harness_failure"] = M.safe_div(harness_failures, total)
        return out

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline set: Primary-Overall plus the six per-task primaries.

        The inherited implementation promotes every ``mean/*`` entry, which would
        make ``mean/reward`` -- an average of per-row rewards that matches no
        published quantity -- read as the benchmark score on a dashboard.
        Overriding both this and ``compute_metrics`` is what prevents that;
        overriding only ``compute_metrics`` leaves the wrong headline in place.
        """
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        for name in ("primary_overall", "harness_failure"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        for task_type in M.TASK_TYPES:
            name = f"{task_type}/{M.PRIMARY_METRIC_BY_TASK[task_type]}"
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        return key


if __name__ == "__main__":
    ChemReasonBenchResourcesServer.run_webserver()
