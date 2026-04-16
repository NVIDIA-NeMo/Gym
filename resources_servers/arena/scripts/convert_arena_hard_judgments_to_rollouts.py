#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Convert arena-hard-auto answers + judgments to NeMo-Gym rollout JSONL.

Produces rollouts compatible with ArenaResourcesServer.compute_metrics(), so that
NeMo-Gym metrics (win_rate, style-controlled BT, rollout_failure_rate) can be
recomputed from pre-existing arena-hard-auto evaluation data — without running
ng_collect_rollouts.

One rollout per question. Run once per model.

Usage (from the nemo-gym root):

    # lmarena-260311 (single baseline)
    python resources_servers/arena/scripts/convert_arena_hard_judgments_to_rollouts.py \
        --arena_data_dir /path/to/arena-hard-auto/data/lmarena-260311 \
        --judge gemini-3.1-pro-preview \
        --model gpt-4o \
        --baselines baseline \
        --output results/lmarena/gpt-4o_rollouts.jsonl

    # arena-hard-v2.0 (per-category baselines)
    python resources_servers/arena/scripts/convert_arena_hard_judgments_to_rollouts.py \
        --arena_data_dir /path/to/arena-hard-auto/data/arena-hard-v2.0 \
        --judge claude-opus-4-6 \
        --model gpt-4o \
        --baselines hard_prompt:o3-mini-2025-01-31 creative_writing:gemini-2.0-flash-001 \
        --output results/arena_hard_v2/gpt-4o_rollouts.jsonl

Then recompute NeMo-Gym metrics (see README § "Recompute metrics from saved rollouts"):
"""

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from resources_servers.arena.arena import (
    _AH_AUTO_LABEL_TO_VERDICT,
    _DEFAULT_VERDICT_WEIGHT,
    _strip_thinking_blocks,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)


def _get_answer_text(row: dict) -> str | None:
    """Extract the last assistant answer from a model_answer row.

    Content is a plain string, {"answer": str}, or occasionally {"answer": {"answer": str}}
    (double-nested, seen in some lmarena entries). Unwrap until we reach a string.
    Thinking blocks (<think>…</think>) are stripped so that style features match what
    the live verify() path produces (which also strips them before judge and style computation).
    """
    messages = row.get("messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return None
    content = assistant_msgs[-1]["content"]
    while isinstance(content, dict):
        content = content.get("answer")
    if not isinstance(content, str):
        return None
    return _strip_thinking_blocks(content) or None


def _load_answers(path: Path) -> dict[str, str]:
    """Load uid → answer text from a model_answer/*.jsonl file."""
    answers: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = _get_answer_text(row)
            if text:
                answers[row["uid"]] = text
    return answers


def _load_baseline_source_models(path: Path) -> dict[str, str]:
    """Load uid → model name from a model_answer/*.jsonl file.

    Used for mixed-model baselines (e.g. lmarena-260311 v0.2) to detect
    questions where the baseline answer was provided by the same model
    being evaluated, so those questions can be skipped.
    """
    source: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("model") and row.get("uid"):
                source[row["uid"]] = row["model"]
    return source


def main(
    arena_data_dir: str,
    judge: str,
    model: str,
    baselines: str | dict[str, str],
    output: str,
    verdict_weight: int,
) -> None:
    data_dir = Path(arena_data_dir)

    # ── Load questions ────────────────────────────────────────────────────────
    uid_to_idx: dict[str, int] = {}
    uid_to_category: dict[str, str | None] = {}
    with open(data_dir / "question.jsonl") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            uid_to_idx[row["uid"]] = i
            uid_to_category[row["uid"]] = row.get("category")
    print(f"Loaded {len(uid_to_idx)} questions")

    # ── Load baseline answers ─────────────────────────────────────────────────
    model_answers_dir = data_dir / "model_answer"

    def _require_answer_file(model_name: str) -> Path:
        path = model_answers_dir / f"{model_name}.jsonl"
        if not path.exists():
            available = sorted(p.stem for p in model_answers_dir.glob("*.jsonl"))
            print(f"ERROR: answer file not found: {path}", file=sys.stderr)
            print(f"Available: {available}", file=sys.stderr)
            sys.exit(1)
        return path

    # uid → which model actually provided the baseline answer (for self-comparison detection).
    # For a fixed baseline (single model or per-category model), the source is the model name
    # itself.  For mixed baselines (e.g. lmarena-260311 where baseline.jsonl aggregates
    # answers from several models), we load the per-uid model field.
    baseline_source: dict[str, str] = {}  # uid → model name that provided the baseline

    if isinstance(baselines, str):
        baseline_path = _require_answer_file(baselines)
        single_baseline = _load_answers(baseline_path)
        cat_baselines: dict[str, dict[str, str]] = {}
        print(f"Loaded {len(single_baseline)} baseline answers ({baselines})")
        baseline_source = _load_baseline_source_models(baseline_path)
    else:
        single_baseline = {}
        cat_baselines = {}
        for cat, baseline_model in baselines.items():
            cat_baselines[cat] = _load_answers(_require_answer_file(baseline_model))
            print(f"Category {cat!r}: loaded {len(cat_baselines[cat])} baseline answers ({baseline_model})")
            # For fixed per-category baselines, every uid in that category uses the same source model.
            # Populate lazily in the main loop using the fixed baseline_model name.

    # ── Validate judge directory ──────────────────────────────────────────────
    judgment_dir = data_dir / "model_judgment" / judge
    if not judgment_dir.is_dir():
        available = sorted(p.name for p in (data_dir / "model_judgment").iterdir() if p.is_dir())
        print(f"ERROR: judge not found: {judgment_dir}", file=sys.stderr)
        print(f"Available: {available}", file=sys.stderr)
        sys.exit(1)

    # ── Convert ───────────────────────────────────────────────────────────────
    judgment_path = judgment_dir / f"{model}.jsonl"
    if not judgment_path.exists():
        available = sorted(p.stem for p in judgment_dir.glob("*.jsonl"))
        print(f"ERROR: no judgment file for {model!r}", file=sys.stderr)
        print(f"Available: {available}", file=sys.stderr)
        sys.exit(1)

    policy_path = model_answers_dir / f"{model}.jsonl"
    if not policy_path.exists():
        print(f"WARNING: no model_answer file for {model!r}, policy_answer will be empty.")
    policy_answers = _load_answers(policy_path) if policy_path.exists() else {}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = n_self_comparison = 0
    with open(judgment_path) as f, open(out_path, "w") as out_f:
        for line in f:
            row = json.loads(line)
            uid = row["uid"]
            task_idx = uid_to_idx.get(uid)
            if task_idx is None:
                n_skipped += 1
                continue

            # Detect questions where the baseline answer was provided by the same model
            # being evaluated.  Write a marker rollout (self_comparison=True) so that
            # compute_metrics() can track the count and exclude it from the failure-rate
            # denominator, rather than silently dropping the task index.
            is_self_comparison = False
            if isinstance(baselines, str):
                bl_source = baseline_source.get(uid)
                if bl_source == model:
                    is_self_comparison = True
            else:
                cat_key = row.get("category") or uid_to_category.get(uid) or ""
                if baselines.get(cat_key) == model:
                    is_self_comparison = True

            if is_self_comparison:
                n_self_comparison += 1
                out_f.write(
                    json.dumps(
                        {
                            "_ng_task_index": task_idx,
                            "reward": 0.0,
                            "games": None,
                            "policy_answer": "",
                            "baseline_answer": "",
                            "category": uid_to_category.get(uid),
                            "question_id": uid,
                            "model": model,
                            "self_comparison": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            category = uid_to_category.get(uid)
            raw_games = row.get("games") or []

            # Resolve baseline answer for this question.
            if single_baseline:
                baseline_text = single_baseline.get(uid, "")
            else:
                cat_key = row.get("category") or category or ""
                baseline_text = cat_baselines.get(cat_key, {}).get(uid, "")

            # Map arena-hard-auto scores → NeMo-Gym verdict labels and compute reward.
            # In arena-hard-auto: raw_games[0]=model-as-B, raw_games[1]=model-as-A.
            # NeMo-Gym compute_metrics expects nemo_games[0]=policy-as-A, nemo_games[1]=policy-as-B,
            # so we store ah_games[1] as nemo_games[0] and ah_games[0] as nemo_games[1].
            nemo_games: list[dict] = []
            scores: list[float] = []
            for i, game in enumerate(raw_games[:2]):
                if game is None:
                    nemo_games.append({"verdict": None})
                    continue
                verdict = _AH_AUTO_LABEL_TO_VERDICT.get(game.get("score"))
                nemo_games.append({"verdict": verdict})
                if verdict:
                    # i=0 → model-as-B, i=1 → model-as-A
                    w = (
                        _weighted_scores_as_b(verdict, verdict_weight)
                        if i == 0
                        else _weighted_scores_as_a(verdict, verdict_weight)
                    )
                    scores.extend(w)
            # Swap to NeMo-Gym convention: [0]=policy-as-A (ah[1]), [1]=policy-as-B (ah[0])
            if len(nemo_games) == 2:
                nemo_games = [nemo_games[1], nemo_games[0]]

            reward = sum(scores) / len(scores) if scores else 0.0

            out_f.write(
                json.dumps(
                    {
                        "_ng_task_index": task_idx,
                        "reward": reward,
                        "games": nemo_games,
                        "policy_answer": policy_answers.get(uid, ""),
                        "baseline_answer": baseline_text,
                        "category": category,
                        "question_id": uid,
                        "model": model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(
        f"Wrote {n_written} rollouts → {out_path}  (skipped: {n_skipped}, self-comparisons excluded: {n_self_comparison})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arena_data_dir", required=True, help="Path to arena-hard-auto dataset directory.")
    parser.add_argument("--judge", required=True, help="Judge subdirectory name under model_judgment/.")
    parser.add_argument(
        "--model",
        required=True,
        help="Evaluated model name (JSONL stem under model_answer/ and model_judgment/<judge>/).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        metavar="MODEL_OR_CATEGORY:MODEL",
        help=(
            "Baseline model(s). Single name for all categories (e.g. 'baseline'), "
            "or CATEGORY:MODEL pairs (e.g. 'hard_prompt:o3-mini-2025-01-31 creative_writing:gemini-2.0-flash-001')."
        ),
    )
    parser.add_argument("--output", required=True, help="Output rollout JSONL path.")
    parser.add_argument(
        "--verdict_weight",
        type=int,
        default=_DEFAULT_VERDICT_WEIGHT,
        metavar="N",
        help=f"Weight for strong verdicts (>>). Default: {_DEFAULT_VERDICT_WEIGHT}.",
    )
    args = parser.parse_args()

    # Detect baseline mode: CATEGORY:MODEL pairs vs single model name.
    if any(":" in v for v in args.baselines):
        parsed_baselines: str | dict[str, str] = {}
        for item in args.baselines:
            if ":" not in item:
                parser.error(f"--baselines: mix of plain name and CATEGORY:MODEL pairs not allowed, got: {item!r}")
            cat, baseline_model = item.split(":", 1)
            parsed_baselines[cat] = baseline_model  # type: ignore[index]
    else:
        if len(args.baselines) > 1:
            parser.error("--baselines: pass a single model name or CATEGORY:MODEL pairs, not multiple plain names.")
        parsed_baselines = args.baselines[0]

    main(args.arena_data_dir, args.judge, args.model, parsed_baselines, args.output, args.verdict_weight)
