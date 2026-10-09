# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""long_transduction resource server — long-context input-to-output transduction.

Supported sample types (dispatched on the row's `type` field):

  Arithmetic chain summing:
    - "unnumbered_streaming_sum" : plain "<expr>=<answer>" lines
    - "streaming_sum"            : "[N]<expr>=<answer>" lines, in order
    - "shuffled_streaming_sum"   : same numbered format, input lines shuffled

  Per-line UUID sorting:
    - "streaming_uuid_sort"          : "[N](u1),(u2),..." lines, in order;
                                       model outputs each line's UUIDs in
                                       ascending lexicographic order.
    - "shuffled_streaming_uuid_sort" : same as above but input lines are
                                       shuffled; model must still emit in
                                       ascending [N] order.
    - "unnumbered_uuid_sort"         : no index prefix; positional matching.

  Variable expansion (dereference hex-named vars to words):
    - "streaming_var_expand"          : "[N]aaa+bbb" lines in order; model
                                        emits "[N]<word> <word>" resolving each
                                        key against the shuffled var pool.
    - "shuffled_streaming_var_expand" : same but input lines are shuffled;
                                        model must still emit ascending [N].
    - "unnumbered_var_expand"         : no index prefix; positional matching.

  CSV tasks (scored by cell-level accuracy via score_csv_permutation):
    - "csv_permutation_homogeneous"   : permute rows/cols of 4-digit integer grid.
    - "csv_permutation_heterogeneous" : permute rows/cols of variable-length UUID grid.
    - "csv_kv_lookup"                 : resolve adjective+noun key expressions
                                        in each cell using provided lookup tables.

verify() looks up the scorer in _SCORERS_BY_TYPE (raises on unknown). Rows
without a `type` are treated as the default ("unnumbered_streaming_sum") so
pre-typed rollouts continue to score.

compute_metrics() reports accuracy by type, difficulty, target_tokens, and
type paired with difficulty or target_tokens. Difficulty for sum types =
max_operands; for uuid_sort types = uuids_per_line; for csv_permutation types =
perm_fraction; for csv_kv_lookup = vocab_fraction; for var_expand types = n_variables.
Arithmetic aggregate accuracy requires both correct expression copying and a
correct answer, matching gym-evals reporting. The reward remains answer-only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.long_transduction.parse import (
    score_csv_permutation,
    score_response,
    score_response_numbered,
    score_unnumbered_uuid_sort,
    score_uuid_sort,
    score_var_expand_numbered,
    score_var_expand_unnumbered,
)


def _score_sum(generation: str, body: LongTransductionVerifyRequest) -> list[list[bool]]:
    """Adapter: pick the right arithmetic scorer for body.type and return
    each per-expression score as a 3-bool list [copy, answer, self_consistent].
    """
    if body.type == "unnumbered_streaming_sum" or body.type is None:
        triples = score_response(generation, body.expressions or [])
    else:
        triples = score_response_numbered(generation, body.expressions or [])
    return [list(t) for t in triples]


def _score_uuid(generation: str, body: LongTransductionVerifyRequest) -> list[list[bool]]:
    """Adapter: per-line (copy_correct, answer_correct, self_consistent) for numbered UUID-sort."""
    return [list(t) for t in score_uuid_sort(generation, body.uuid_lines or [])]


def _score_unnumbered_uuid(generation: str, body: LongTransductionVerifyRequest) -> list[list[bool]]:
    """Adapter: positional (copy_correct, answer_correct, self_consistent) for unnumbered UUID-sort."""
    return [list(t) for t in score_unnumbered_uuid_sort(generation, body.uuid_lines or [])]


def _score_var_expand(generation: str, body: LongTransductionVerifyRequest) -> list[list[bool]]:
    """Adapter: pick the numbered vs positional variable-expansion scorer.

    Returns each item's [copy_correct, answer_correct, self_consistent] (the
    three are identical for this value-only task).
    """
    if body.type == "unnumbered_var_expand":
        triples = score_var_expand_unnumbered(generation, body.expressions or [])
    else:
        triples = score_var_expand_numbered(generation, body.expressions or [])
    return [list(t) for t in triples]


def _generation(body: LongTransductionVerifyRequest) -> str:
    """Extract the model's generated text from a verify request body."""
    parts = [
        item.text
        for output in body.response.output
        if output.type == "message"
        for item in output.content
        if item.type == "output_text"
    ]
    return "".join(parts)


# Parser dispatch by `type`. Keys are the source of truth for which sample
# types this server understands; verify() raises ValueError on unknowns.
_SCORERS_BY_TYPE = {
    "unnumbered_streaming_sum": _score_sum,
    "streaming_sum": _score_sum,
    "shuffled_streaming_sum": _score_sum,
    "unnumbered_uuid_sort": _score_unnumbered_uuid,
    "streaming_uuid_sort": _score_uuid,
    "shuffled_streaming_uuid_sort": _score_uuid,
    "unnumbered_var_expand": _score_var_expand,
    "streaming_var_expand": _score_var_expand,
    "shuffled_streaming_var_expand": _score_var_expand,
}
# Rows with no `type` set keep scoring against the legacy unnumbered parser.
_DEFAULT_TYPE = "unnumbered_streaming_sum"
_SUM_TYPES = {"unnumbered_streaming_sum", "streaming_sum", "shuffled_streaming_sum"}

# CSV types all use score_csv_permutation (cell-level string equality).
_CSV_TYPES = {
    "csv_permutation_homogeneous",
    "csv_permutation_heterogeneous",
    "csv_kv_lookup",
}


def _strip_reasoning(text: str) -> str:
    # Qwen3 / Nemotron / generic <think>...</think> reasoning models.
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip("\n")
    if "<think>" in text:
        return ""
    # gpt-oss harmony format: keep only the final channel.
    if "<|channel|>final<|message|>" in text:
        return text.rsplit("<|channel|>final<|message|>", 1)[1].lstrip("\n")
    if "<|channel|>analysis<|message|>" in text:
        return ""
    return text


class LongTransductionConfig(BaseResourcesServerConfig):
    strip_reasoning: bool = True


class LongTransductionRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # Sample variant. None falls back to _DEFAULT_TYPE; recognized values are
    # the keys of _SCORERS_BY_TYPE.
    type: Optional[str] = None
    # Approximate input-token budget used when generating the task.
    target_tokens: Optional[int] = None
    max_operands: Optional[int] = None
    n_expressions: Optional[int] = None
    # Arithmetic-chain payload (sum types).
    expressions: Optional[List[Dict[str, Any]]] = None
    # UUID-sort payload (uuid_sort types). Each inner list is one line's UUIDs
    # in their canonical (input-presentation) order; expected output for that
    # line is the same UUIDs sorted lexicographically.
    uuid_lines: Optional[List[List[str]]] = None
    uuids_per_line: Optional[int] = None
    n_lines: Optional[int] = None
    # CSV permutation payload.
    expected_output: Optional[str] = None
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    perm_fraction: Optional[float] = None
    # CSV KV lookup payload.
    vocab_fraction: Optional[float] = None
    n_vocab: Optional[int] = None
    # Variable-expansion payload (var_expand types). Reuses `expressions`
    # ({"expr": "aaa+bbb", "answer": "big dog"}); n_variables is the difficulty.
    n_variables: Optional[int] = None


class LongTransductionVerifyRequest(LongTransductionRunRequest, BaseVerifyRequest):
    pass


class LongTransductionVerifyResponse(LongTransductionVerifyRequest, BaseVerifyResponse):
    # Mean per-item correctness (used as the reward). For sum types this is
    # mean(answer_correct); for uuid_sort it's mean(whole_line_correct);
    # for CSV tasks it's cell accuracy.
    answer_correct: float
    n_items_scored: int
    # Per-item [copy_correct, answer_correct, self_consistent]. The three
    # signals are defined slightly differently per task type — see the scorer
    # docstrings — but the shape and reward semantics (index 1) are uniform.
    item_scores: List[List[bool]]
    # Count of items where copy_correct (index 0) is True.
    n_items_copy_correct: int
    # CSV-permutation diagnostics (None for non-CSV types).
    cell_accuracy: Optional[float] = None
    row_accuracy: Optional[float] = None
    col_accuracy: Optional[float] = None


class LongTransductionServer(SimpleResourcesServer):
    config: LongTransductionConfig

    async def verify(self, body: LongTransductionVerifyRequest) -> LongTransductionVerifyResponse:
        """Score task-specific output, treating missing answers as incorrect."""
        sample_type = body.type or _DEFAULT_TYPE

        generation = _generation(body)
        if self.config.strip_reasoning:
            generation = _strip_reasoning(generation)

        if sample_type in _CSV_TYPES:
            cell_item_scores, row_scores, col_scores = score_csv_permutation(
                generation,
                body.expected_output or "",
                body.n_rows or 0,
                body.n_cols or 0,
            )
            n_items = len(cell_item_scores)
            flat = [row[1] for row in cell_item_scores]
            reward = sum(flat) / n_items if n_items else 0.0
            return LongTransductionVerifyResponse(
                **body.model_dump(),
                reward=reward,
                answer_correct=reward,
                n_items_scored=n_items,
                n_items_copy_correct=sum(row[0] for row in cell_item_scores),
                item_scores=cell_item_scores,
                cell_accuracy=reward,
                row_accuracy=sum(row_scores) / len(row_scores) if row_scores else 0.0,
                col_accuracy=sum(col_scores) / len(col_scores) if col_scores else 0.0,
            )

        try:
            scorer = _SCORERS_BY_TYPE[sample_type]
        except KeyError as e:
            raise ValueError(
                f"Unsupported long_transduction sample type: {sample_type!r}. "
                f"Supported types: {sorted(_SCORERS_BY_TYPE) + sorted(_CSV_TYPES)}."
            ) from e

        item_scores = scorer(generation, body)

        # Reward is the mean "answer_correct" signal (index 1) across items.
        flat = [row[1] for row in item_scores] if item_scores else []
        n_items = len(flat)
        reward = (sum(flat) / n_items) if n_items else 0.0

        return LongTransductionVerifyResponse(
            **body.model_dump(),
            reward=reward,
            answer_correct=reward,
            n_items_scored=n_items,
            n_items_copy_correct=sum(row[0] for row in item_scores),
            item_scores=item_scores,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Average accuracy by type, difficulty, and context; arithmetic requires copy AND answer correctness."""
        by_difficulty: Dict[Any, List[float]] = defaultdict(list)
        by_type: Dict[str, List[float]] = defaultdict(list)
        by_type_difficulty: Dict[tuple, List[float]] = defaultdict(list)
        by_target_tokens: Dict[int, List[float]] = defaultdict(list)
        by_type_target_tokens: Dict[tuple[str, int], List[float]] = defaultdict(list)
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                if rollout.get("answer_correct") is None:
                    continue
                ttype = rollout.get("type") or _DEFAULT_TYPE
                # Pick the difficulty knob present on this row.
                diff = rollout.get("max_operands")
                if diff is None:
                    diff = rollout.get("uuids_per_line")
                if diff is None:
                    diff = rollout.get("perm_fraction")
                if diff is None:
                    diff = rollout.get("vocab_fraction")
                if diff is None:
                    diff = rollout.get("n_variables")
                if diff is None:
                    diff = "n/a"
                acc = rollout["answer_correct"]
                item_scores = rollout.get("item_scores")
                # Match gym-evals plots; legacy rows without item scores retain answer-only accuracy.
                if ttype in _SUM_TYPES and item_scores:
                    acc = sum(bool(scores[0] and scores[1]) for scores in item_scores) / len(item_scores)
                by_difficulty[diff].append(acc)
                by_type[ttype].append(acc)
                by_type_difficulty[(ttype, diff)].append(acc)
                target_tokens = rollout.get("target_tokens")
                if target_tokens is not None:
                    by_target_tokens[target_tokens].append(acc)
                    by_type_target_tokens[(ttype, target_tokens)].append(acc)

        metrics: Dict[str, Any] = {}
        all_vals: List[float] = []
        for diff in sorted(by_difficulty.keys(), key=lambda d: (d == "n/a", d)):
            vals = by_difficulty[diff]
            metrics[f"difficulty_{diff}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }
            if vals:
                all_vals.extend(vals)

        for ttype in sorted(by_type.keys()):
            vals = by_type[ttype]
            metrics[f"type_{ttype}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }

        for ttype, diff in sorted(by_type_difficulty.keys(), key=lambda k: (k[0], k[1] == "n/a", k[1])):
            vals = by_type_difficulty[(ttype, diff)]
            metrics[f"type_{ttype}_difficulty_{diff}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }

        for target_tokens, vals in sorted(by_target_tokens.items()):
            metrics[f"target_tokens_{target_tokens}"] = {"accuracy": sum(vals) / len(vals), "n": len(vals)}

        for (ttype, target_tokens), vals in sorted(by_type_target_tokens.items()):
            metrics[f"type_{ttype}_target_tokens_{target_tokens}"] = {
                "accuracy": sum(vals) / len(vals),
                "n": len(vals),
            }

        if all_vals:
            metrics["overall_accuracy"] = sum(all_vals) / len(all_vals)
        return metrics

    def get_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Expose the grouped accuracies for Gym's aggregate metric display."""
        return {k: v["accuracy"] for k, v in metrics.items() if isinstance(v, dict) and v.get("accuracy") is not None}


if __name__ == "__main__":
    LongTransductionServer.run_webserver()
