# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemReason-Bench resources server.

Single-turn and deterministic: one JSON object in, a pure-Python scorer against
gold. ``task_type`` selects one of six scorers; ``metrics.py`` holds the formulas.

``reward`` is a per-row signal, NOT the benchmark's metric -- four of the six
published metrics are only defined over a corpus, so ``compute_metrics`` reduces
contributions rather than averaging rewards. See the README.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Optional

import metrics as M
from pydantic import model_validator
from response_parsing import extract_json, to_prediction, to_prediction_lm

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class ChemReasonBenchResourcesServerConfig(BaseResourcesServerConfig):
    # Nothing is carried between verifications: scoring is a pure function of the
    # reply and the row. This is not a claim that arbitrary model output reproduces
    # -- only that this server keeps no state.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class ChemReasonBenchVerifyRequest(BaseVerifyRequest):
    # Prepared rows are flat; example.jsonl nests under `verifier_metadata`. Accept both.
    verifier_metadata: Optional[Dict[str, Any]] = None
    # Any, deliberately: a narrower type makes a wrong-typed row a 422 during request
    # validation, which simple_agent's raise_for_status turns into an aborted run.
    # verify() shape-checks instead and reports a harness_failure status.
    task_type: Any = None
    ground_truth: Any = None
    task_id: Any = None
    benchmark_id: Any = None
    # Question-side vocabulary upstream's post-processors need. Not gold.
    expected_step_ids: Any = None
    options: Any = None
    legend: Any = None
    # "gen" (JSON reply) or "lm" (bare decision token). Rows without it are gen,
    # so a dataset prepared before the lm protocol existed still scores.
    protocol: str = "gen"

    @model_validator(mode="before")
    @classmethod
    def _lift_verifier_metadata(cls, data: Any) -> Any:
        """Lift nested `verifier_metadata` to the top level; top level wins.

        Without this a flat row reaches verify() with task_type unset and every
        instance is charged to the harness.
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
    """Per-token logprobs of the first output text part, or None (lm rows only)."""
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            logprobs = getattr(part, "logprobs", None)
            if logprobs:
                return [lp if isinstance(lp, dict) else lp.model_dump() for lp in logprobs]
    return None


def _sanitize(value: Any) -> Optional[str]:
    """Coerce to a wire-safe string, dropping lone surrogates.

    Must accept any type: `task_id` is declared `Any` so a wrong-typed row reaches
    verify() as a status rather than a 422, which means a non-string can arrive here.
    A surrogate reaching the JSON encoder raises while the response is being built.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", "replace").decode("utf-8", "replace")


class ChemReasonBenchResourcesServer(SimpleResourcesServer):
    """Scores one ChemReason-Bench instance against its gold record."""

    ray_enabled = False  # pure-stdlib scorer; no Ray in the verify path

    config: ChemReasonBenchResourcesServerConfig

    async def verify(self, body: ChemReasonBenchVerifyRequest) -> ChemReasonBenchVerifyResponse:
        task_type = body.task_type
        ground_truth = body.ground_truth
        # Declared on the request, so already in model_dump(); re-passing them is a TypeError.
        payload = body.model_dump()
        payload["task_id"] = _sanitize(payload.get("task_id"))

        # A malformed row is a status, not a 500: a 500 ends the whole run. The 0.0
        # is not a measurement of the model, so mask_sample keeps it out of the
        # downstream score as well as out of compute_metrics.
        if not isinstance(task_type, str) or task_type not in M.TASK_TYPES:
            return ChemReasonBenchVerifyResponse(
                **payload, reward=0.0, status="bad_task_type", harness_failure=True, mask_sample=True
            )
        if not isinstance(ground_truth, dict):
            return ChemReasonBenchVerifyResponse(
                **payload, reward=0.0, status="bad_ground_truth", harness_failure=True, mask_sample=True
            )

        # Optional: a wrong-typed value degrades to empty rather than failing the row.
        expected_step_ids = body.expected_step_ids if isinstance(body.expected_step_ids, list) else []
        options = body.options if isinstance(body.options, list) else []
        legend = body.legend if isinstance(body.legend, dict) else {}

        if body.protocol == "lm":
            if task_type not in M.DUAL_PROTOCOL_TASKS:
                return ChemReasonBenchVerifyResponse(
                    **payload, reward=0.0, status="no_lm_protocol", harness_failure=True, mask_sample=True
                )
            prediction = to_prediction_lm(
                task_type, body.response.output_text, _first_output_logprobs(body.response), options
            )
            status = prediction.pop("status")
        else:
            raw = body.response.output_text or ""
            parsed, status = extract_json(raw)
            prediction = to_prediction(
                task_type, parsed, expected_step_ids, options, raw, legend, non_object=status == "non_object_json"
            )
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

        The three discriminative tasks average their gen and lm protocols
        (appendix F.3.4); a task with only one present falls back to it, so a
        gen-only dataset still scores rather than being halved.
        """
        by_key: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
        for task in tasks:
            for rollout in task:
                # Masked upstream of here; the aggregator reports them as coverage/masked_rollouts.
                if rollout.get("harness_failure"):
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
        return out

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Headline set: Primary-Overall plus the six per-task primaries.

        The inherited version promotes every ``mean/*`` entry, which would make
        ``mean/reward`` -- matching no published quantity -- read as the score.
        """
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        if "primary_overall" in agent_metrics:
            key["primary_overall"] = agent_metrics["primary_overall"]
        for task_type in M.TASK_TYPES:
            name = f"{task_type}/{M.PRIMARY_METRIC_BY_TASK[task_type]}"
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        return key


if __name__ == "__main__":
    ChemReasonBenchResourcesServer.run_webserver()
