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
"""The policy plays the agentic_turn_simulation (ATS) judge.

A row holds the exact prompt the ATS judge reads (system: harness frame, rubrics, tool definitions and output
schema; user: the rendered conversation with the turn under evaluation) and a reference label built from reference
judge verdicts. The policy writes the judge's JSON verdict; /verify reads it (`verdict.read_verdict`), takes its
correctness ladder (`addresses_concern == "no"` -> "no", else `correct`) and pays agreement with `gold_ladder`.
Verification is deterministic: no LLM is called. Output that cannot be read as a valid verdict scores 0 and is never
masked. Reasoning is expected in the model server's separate reasoning items (vLLM `--reasoning-parser`); only
assistant `message` items are read, with any leftover `<think>` block stripped.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.core import NeMoGymResponse
from nemo_gym.metrics import compute_pass_majority_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_turn_simulation_judge.verdict import (
    PARSE_ERRORS,
    VERDICT_FIELDS,
    ParseMode,
    read_verdict,
)


LOG = logging.getLogger(__name__)

Ladder = Literal["yes", "no"]
# parse_error of a /verify that failed outside parsing (logged; a bug, not the policy's output).
INTERNAL_ERROR = "internal"
UNSCORED = "unscored"
# Per-slice metrics: (metric name, path into the rollout dict). A rollout without the field counts as "none".
SLICES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source", ("meta", "source")),
    ("family", ("meta", "family")),
    ("harness", ("meta", "harness")),
    ("tier", ("label", "tier")),
)
KEY_METRICS = ("accuracy", "balanced_accuracy", "kappa", "parse_fail", "leniency_gap")
# Labels other than gold_ladder that metrics score against when a row carries them: (metric prefix, path).
OTHER_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hand", ("meta", "hand_ladder")),
    ("expect", ("meta", "expect")),
)


class FieldBonusConfig(BaseModel):
    """reward = ladder score * ((1 - weight) + weight * share of `fields` that equal the row's gold_fields). Gated
    by the ladder: matching fields never pay for a wrong ladder."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(ge=0.0, le=1.0)
    fields: list[str] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _known_fields(cls, fields: list[str]) -> list[str]:
        unknown = [f for f in fields if f not in VERDICT_FIELDS]
        if unknown:
            raise ValueError(f"field_bonus.fields {unknown} are not verdict fields {VERDICT_FIELDS}")
        return fields


class AgenticTurnSimulationJudgeConfig(BaseResourcesServerConfig):
    # /verify recomputes the reward from the request alone.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # strict: exactly one JSON object (one ```json fence allowed). lenient: ATS parse_verdict without its salvage.
    parse_mode: ParseMode = "strict"
    # Require every key of the evaluation schema and enum values it allows (off: only the ladder must be decided).
    validate_schema: bool = True
    # hard: 1 if the predicted ladder equals gold_ladder. soft: gold_p_yes for "yes", 1 - gold_p_yes for "no"
    # (rows without gold_p_yes fall back to hard).
    reward_mode: Literal["hard", "soft"] = "hard"
    field_bonus: Optional[FieldBonusConfig] = None
    # An `analysis` longer than this many words is a "too_long" output (scores 0). None: off.
    max_analysis_words: Optional[int] = Field(default=None, ge=1)


class AgenticTurnSimulationJudgeRunRequest(BaseRunRequest):
    """Row fields on the wire (see task_data.py for their meaning)."""

    model_config = ConfigDict(extra="allow")

    task: Literal["verdict"] = "verdict"
    row_id: Optional[str] = None
    group_id: Optional[str] = None
    gold_ladder: Ladder
    gold_p_yes: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gold_n: Optional[int] = Field(default=None, ge=0)
    gold_fields: Optional[dict[str, Optional[str]]] = None
    label: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = None


class AgenticTurnSimulationJudgeVerifyRequest(AgenticTurnSimulationJudgeRunRequest, BaseVerifyRequest):
    pass


class AgenticTurnSimulationJudgeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    gold_ladder: Ladder
    pred_ladder: Optional[Ladder] = None
    # 1.0 when pred_ladder == gold_ladder: the accuracy metrics read this, whatever reward_mode pays.
    ladder_match: float = 0.0
    parse_ok: bool = False
    parse_error: Optional[str] = None
    # Confusion one-hots with POSITIVE = ladder "no" (the judge catches a failed turn); all 0 when unparsed.
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    analysis_words: int = 0
    output_chars: int = 0
    # The parsed verdict's enum fields (no analysis/reason text); None when nothing parsed.
    pred_fields: Optional[dict[str, Any]] = None
    # Per gold_fields key: whether the prediction gives the same value (an unparsed output matches nothing).
    field_matches: dict[str, bool] = Field(default_factory=dict)
    field_match_share: Optional[float] = None


def assistant_text(response: Optional[NeMoGymResponse]) -> str:
    """The assistant `message` items' text, joined by newlines. Reasoning and other items are skipped."""
    texts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        content = getattr(item, "content", None) or []
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _confusion(pred: Optional[str], gold: str) -> dict[str, float]:
    if pred not in ("yes", "no"):
        return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    return {
        "tp": float(pred == "no" and gold == "no"),
        "fp": float(pred == "no" and gold == "yes"),
        "fn": float(pred == "yes" and gold == "no"),
        "tn": float(pred == "yes" and gold == "yes"),
    }


def score_output(
    text: str,
    *,
    gold_ladder: str,
    gold_p_yes: Optional[float],
    gold_fields: Optional[dict[str, Any]],
    config: AgenticTurnSimulationJudgeConfig,
) -> dict[str, Any]:
    """The verify-response fields (reward included) for one output text."""
    read = read_verdict(
        text,
        parse_mode=config.parse_mode,
        validate_schema=config.validate_schema,
        max_analysis_words=config.max_analysis_words,
    )
    pred = read.ladder
    match = float(pred is not None and pred == gold_ladder)
    if pred is None:
        base = 0.0
    elif config.reward_mode == "soft" and gold_p_yes is not None:
        base = gold_p_yes if pred == "yes" else 1.0 - gold_p_yes
    else:
        base = match

    predicted = read.verdict if pred is not None else {}
    gold = {k: v for k, v in (gold_fields or {}).items() if k in VERDICT_FIELDS and v is not None}
    field_matches = {k: predicted.get(k) == v for k, v in gold.items()}
    share = sum(field_matches.values()) / len(field_matches) if field_matches else None

    reward = base
    bonus = config.field_bonus
    if bonus is not None and base > 0.0:
        listed = [field_matches[f] for f in bonus.fields if f in field_matches]
        if listed:
            reward = base * ((1.0 - bonus.weight) + bonus.weight * sum(listed) / len(listed))

    return {
        "reward": reward,
        "pred_ladder": pred,
        "ladder_match": match,
        "parse_ok": read.error is None,
        "parse_error": read.error,
        **_confusion(pred, gold_ladder),
        "analysis_words": read.analysis_words,
        "output_chars": len(text),
        "pred_fields": {k: read.verdict.get(k) for k in VERDICT_FIELDS} if read.verdict is not None else None,
        "field_matches": field_matches,
        "field_match_share": share,
    }


# --- metrics --------------------------------------------------------------------------------------------------


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for part in path:
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _slice_name(value: Any) -> str:
    if value is None or value == "":
        return "none"
    return str(value).replace("/", "_")


def classification_stats(pairs: list[tuple[Optional[str], str]]) -> dict[str, float]:
    """Agreement stats over (pred_ladder, gold_ladder) pairs, POSITIVE = "no". An unparsed prediction (None) is a
    third class that matches no gold: it counts against accuracy and both recalls, never excluded. Rates are
    fractions; kappa (Cohen's, over yes/no/unparsed) is omitted when chance agreement is 1."""
    n = len(pairs)
    if n == 0:
        return {}
    gold_yes = sum(1 for _, g in pairs if g == "yes")
    gold_no = n - gold_yes
    pred_yes = sum(1 for p, _ in pairs if p == "yes")
    pred_no = sum(1 for p, _ in pairs if p == "no")
    hit_yes = sum(1 for p, g in pairs if p == g == "yes")
    hit_no = sum(1 for p, g in pairs if p == g == "no")
    accuracy = (hit_yes + hit_no) / n
    out = {"accuracy": accuracy, "n": float(n)}
    recalls = []
    if gold_no:
        out["fail_recall"] = hit_no / gold_no
        recalls.append(out["fail_recall"])
    if gold_yes:
        out["pass_recall"] = hit_yes / gold_yes
        recalls.append(out["pass_recall"])
    out["balanced_accuracy"] = sum(recalls) / len(recalls)
    chance = (pred_yes * gold_yes + pred_no * gold_no) / (n * n)
    if chance < 1.0:
        out["kappa"] = (accuracy - chance) / (1.0 - chance)
    out["pred_yes_rate"] = pred_yes / n
    out["gold_yes_rate"] = gold_yes / n
    # Leniency is judged on the verdicts that parsed (unparsed outputs would read as "strict"): the yes-rate of the
    # parsed predictions minus the gold yes-rate on those same rollouts.
    parsed = [(p, g) for p, g in pairs if p in ("yes", "no")]
    if parsed:
        out["parsed_pred_yes_rate"] = pred_yes / len(parsed)
        out["leniency_gap"] = out["parsed_pred_yes_rate"] - sum(1 for _, g in parsed if g == "yes") / len(parsed)
    return out


def _pairs(rows: list[dict[str, Any]], gold_path: tuple[str, ...] = ("gold_ladder",)) -> list[tuple[Any, str]]:
    pairs = []
    for row in rows:
        gold = _nested(row, gold_path)
        if gold in ("yes", "no"):
            pred = row.get("pred_ladder")
            pairs.append((pred if pred in ("yes", "no") else None, gold))
    return pairs


class AgenticTurnSimulationJudgeResourcesServer(SimpleResourcesServer):
    config: AgenticTurnSimulationJudgeConfig

    async def verify(self, body: AgenticTurnSimulationJudgeVerifyRequest) -> AgenticTurnSimulationJudgeVerifyResponse:
        try:
            outcome = score_output(
                assistant_text(body.response),
                gold_ladder=body.gold_ladder,
                gold_p_yes=body.gold_p_yes,
                gold_fields=body.gold_fields,
                config=self.config,
            )
        except Exception:  # noqa: BLE001 - the reward boundary: a verifier bug scores 0, it never crashes the server
            LOG.exception("agentic_turn_simulation_judge: verify failed for row %s", body.row_id)
            outcome = {"reward": 0.0, "parse_ok": False, "parse_error": INTERNAL_ERROR}
        # by_alias: aliased OpenAI fields (e.g. a json_schema text format's `schema`) must survive the round trip.
        return AgenticTurnSimulationJudgeVerifyResponse(**body.model_dump(by_alias=True), **outcome)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """pass@k / majority@k accuracy (Gym's, in percent, majority over pred_ladder), then pooled agreement stats
        over every rollout (fractions), parse failures by kind, accuracy per slice, against hand labels where rows
        have meta.hand_ladder, and per gold field."""
        rows = [r for task in tasks for r in task]
        if not rows:
            return {}
        metrics: dict[str, Any] = compute_pass_majority_metrics(
            tasks,
            score_fn=lambda r: {"accuracy": float(r.get("ladder_match") or 0.0)},
            answer_key="pred_ladder",
        )[0]
        metrics.update({k: v for k, v in classification_stats(_pairs(rows)).items() if k != "n"})

        # A rollout without verify fields (one that never ran) is an unscored failure, not dropped.
        errors = Counter(r.get("parse_error") or UNSCORED for r in rows if not r.get("parse_ok"))
        metrics["parse_fail"] = sum(errors.values()) / len(rows)
        for kind in sorted(set(PARSE_ERRORS) | set(errors)):
            metrics[f"parse_fail/{kind}"] = errors.get(kind, 0) / len(rows)

        for name, path in SLICES:
            # A task's rollouts share its row, so the first one names the slice.
            by_value: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
            for task in tasks:
                if task:
                    by_value[_slice_name(_nested(task[0], path))].append(task)
            if set(by_value) == {"none"}:
                continue  # no row carries this field
            for value, slice_tasks in sorted(by_value.items()):
                stats = classification_stats(_pairs([r for task in slice_tasks for r in task]))
                for stat in ("accuracy", "balanced_accuracy"):
                    if stat in stats:
                        metrics[f"{name}/{value}/{stat}"] = stats[stat]
                metrics[f"{name}/{value}/n"] = float(len(slice_tasks))

        # Labels independent of the reference judge, where rows carry them: hand labels, and the expected verdict of
        # a behaviour stratum (a BAD turn should fail, a GOOD one pass). Each is reported with the reference label's
        # own agreement on the same rows, the bar the policy is read against.
        for name, path in OTHER_LABELS:
            label_tasks = [task for task in tasks if task and _nested(task[0], path) in ("yes", "no")]
            if not label_tasks:
                continue
            label_rows = [r for task in label_tasks for r in task]
            policy = classification_stats(_pairs(label_rows, path))
            reference = classification_stats([(r.get("gold_ladder"), _nested(r, path)) for r in label_rows])
            metrics[f"{name}/accuracy"] = policy["accuracy"]
            metrics[f"{name}/balanced_accuracy"] = policy["balanced_accuracy"]
            metrics[f"{name}/gold_accuracy"] = reference["accuracy"]
            for key in ("fail_recall", "pass_recall"):
                if key in policy:
                    metrics[f"{name}/{key}"] = policy[key]
                    metrics[f"{name}/gold_{key}"] = reference[key]
            for prefix, stats in ((f"{name}/kappa", policy), (f"{name}/gold_kappa", reference)):
                if "kappa" in stats:
                    metrics[prefix] = stats["kappa"]
            metrics[f"{name}/n"] = float(len(label_tasks))

        per_field: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            # A rollout with no prediction (unparsed, or no verifier output) misses every known reference field.
            matches = dict(r.get("field_matches") or {})
            for field_name, value in (r.get("gold_fields") or {}).items():
                if field_name in VERDICT_FIELDS and value is not None:
                    matches.setdefault(field_name, False)
            for field_name, matched in matches.items():
                per_field[field_name].append(bool(matched))
        for field_name, matches in sorted(per_field.items()):
            metrics[f"field_acc/{field_name}"] = sum(matches) / len(matches)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = ("mean/reward", *KEY_METRICS)
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    AgenticTurnSimulationJudgeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AgenticTurnSimulationJudgeResourcesServer.run_webserver()  # noqa: F401
