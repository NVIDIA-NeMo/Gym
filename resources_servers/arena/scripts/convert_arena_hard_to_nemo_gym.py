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
"""Convert an arena-hard-auto dataset to NeMo-Gym JSONL format.

Supports both lmarena-260311 (single-category, messages-based questions) and
arena-hard-v2.0 (multi-category, prompt-based questions with per-category baselines).

Usage (from the nemo-gym root):

    # lmarena-260311: single baseline model for all questions
    python resources_servers/arena/scripts/convert_arena_hard_to_nemo_gym.py \
        --data_dir /path/to/arena-hard-auto/data/lmarena-260311 \
        --baselines baseline \
        --output resources_servers/arena/data/lmarena_260311/lmarena_260311_validation.jsonl

    # arena-hard-v2.0: per-category baseline models (CATEGORY:MODEL pairs)
    python resources_servers/arena/scripts/convert_arena_hard_to_nemo_gym.py \
        --data_dir /path/to/arena-hard-auto/data/arena-hard-v2.0 \
        --baselines hard_prompt:o3-mini-2025-01-31 creative_writing:gemini-2.0-flash-001 \
        --output resources_servers/arena/data/arena_hard_v2/arena_hard_v2_validation.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def _flatten_messages(messages: list[dict]) -> str:
    """Flatten an OpenAI-style messages list into a single formatted string.

    Matches arena-hard-auto gen_judgment.py flatten_messages format:
        [User]: ...\n\n[Assistant]: ...\n\n[User]: ...
    """
    role_map = {"user": "User", "assistant": "Assistant", "system": "System"}
    parts = []
    for msg in messages:
        role = role_map.get(msg["role"], msg["role"])
        content = msg["content"]
        if isinstance(content, dict):
            content = content["answer"]
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def _get_question_input(row: dict) -> tuple[list[dict], str]:
    """Return (input_messages, question_text) for a question.jsonl row.

    arena-hard-v2.0: prompt-based — single user message, question = raw prompt.
    lmarena-260311: messages-based — full conversation as input; question = raw content
    for single-turn, flattened "[User]/[Assistant]" string for multi-turn.
    """
    if "prompt" in row:
        prompt = row["prompt"]
        return [{"role": "user", "content": prompt}], prompt
    messages = row.get("messages", [])
    if not any(m.get("role") == "user" for m in messages):
        raise ValueError(f"No user message found in row uid={row.get('uid')}")
    if len(messages) == 1:
        content = messages[0]["content"]
        return messages, content if isinstance(content, str) else content["answer"]
    return messages, _flatten_messages(messages)


def _get_answer_text(row: dict) -> str:
    """Extract assistant answer text from a model_answer/*.jsonl row.

    Content is either a plain string or {"answer": str}.
    """
    messages = row.get("messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        raise ValueError(f"No assistant message found in row uid={row.get('uid')}")
    content = assistant_msgs[-1]["content"]
    if isinstance(content, dict):
        return content["answer"]
    return content


def _load_model_answers(path: Path) -> dict[str, str]:
    """Load uid → answer text from a model_answer/*.jsonl file."""
    answers: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            try:
                answers[row["uid"]] = _get_answer_text(row)
            except (ValueError, KeyError) as e:
                print(f"WARNING: skipping uid={row.get('uid')}: {e}")
    return answers


def main(
    data_dir: str,
    baselines: str | dict[str, str],
    output: str,
    n_samples: int | None,
) -> None:
    data_path = Path(data_dir)

    # ── Load questions ────────────────────────────────────────────────────────
    questions: dict[str, dict] = {}
    with open(data_path / "question.jsonl") as f:
        for line in f:
            row = json.loads(line)
            questions[row["uid"]] = row
    print(f"Loaded {len(questions)} questions from {data_path / 'question.jsonl'}")

    # Determine if this is a single-category dataset (no category field in output).
    all_cats = {r.get("category") for r in questions.values()}
    is_single_category = len(all_cats) <= 1 or (len(all_cats) == 2 and None in all_cats)
    if is_single_category:
        print("Single-category dataset — 'category' field will be omitted from output.")
    else:
        print(f"Multi-category dataset — categories: {sorted(c for c in all_cats if c)}")

    # ── Load baseline answers ─────────────────────────────────────────────────
    model_answers_dir = data_path / "model_answer"

    if isinstance(baselines, str):
        # Single baseline for all questions.
        path = model_answers_dir / f"{baselines}.jsonl"
        if not path.exists():
            available = sorted(p.stem for p in model_answers_dir.glob("*.jsonl"))
            print(f"ERROR: baseline model file not found: {path}", file=sys.stderr)
            print(f"Available models: {available}", file=sys.stderr)
            sys.exit(1)
        single_answers = _load_model_answers(path)
        print(f"Loaded {len(single_answers)} baseline answers from {path}")
        cat_answers: dict[str, dict[str, str]] = {}  # unused for single-baseline path
    else:
        # Per-category baselines.
        single_answers = {}
        cat_answers = {}
        for cat, model in baselines.items():
            path = model_answers_dir / f"{model}.jsonl"
            if not path.exists():
                available = sorted(p.stem for p in model_answers_dir.glob("*.jsonl"))
                print(f"ERROR: baseline model file not found for category {cat!r}: {path}", file=sys.stderr)
                print(f"Available models: {available}", file=sys.stderr)
                sys.exit(1)
            cat_answers[cat] = _load_model_answers(path)
            print(f"Category {cat!r}: loaded {len(cat_answers[cat])} baseline answers from {path}")

    # ── Convert ───────────────────────────────────────────────────────────────
    uids = list(questions.keys())
    if n_samples is not None:
        uids = uids[:n_samples]

    skipped = 0
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as out_f:
        for uid in uids:
            q_row = questions[uid]
            cat = q_row.get("category")

            # Resolve baseline answer.
            if isinstance(baselines, str):
                if uid not in single_answers:
                    print(f"WARNING: no baseline answer for uid={uid}, skipping.")
                    skipped += 1
                    continue
                baseline_answer = single_answers[uid]
            else:
                if cat not in cat_answers:
                    print(f"WARNING: no baseline configured for category {cat!r}, skipping uid={uid}.")
                    skipped += 1
                    continue
                if uid not in cat_answers[cat]:
                    print(f"WARNING: no baseline answer for uid={uid} (category={cat!r}), skipping.")
                    skipped += 1
                    continue
                baseline_answer = cat_answers[cat][uid]

            try:
                input_messages, question_text = _get_question_input(q_row)
            except ValueError as e:
                print(f"WARNING: {e}, skipping.")
                skipped += 1
                continue

            record: dict = {
                "responses_create_params": {"input": input_messages},
                "question_id": uid,
                "question": question_text,
                "baseline_answer": baseline_answer,
            }
            if not is_single_category and cat:
                record["category"] = cat

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    written = len(uids) - skipped
    print(f"Written {written} records to {out_path}  (skipped: {skipped})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to arena-hard-auto dataset directory (e.g. data/lmarena-260311).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        metavar="MODEL_OR_CATEGORY:MODEL",
        help=(
            "Baseline model(s). Pass a single model name for all categories "
            "(e.g. 'baseline'), or CATEGORY:MODEL pairs for per-category baselines "
            "(e.g. 'hard_prompt:o3-mini-2025-01-31 creative_writing:gemini-2.0-flash-001')."
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit output to the first N questions (useful for example datasets).",
    )
    args = parser.parse_args()

    # Detect mode: if any value contains ':', treat all as CATEGORY:MODEL pairs.
    if any(":" in v for v in args.baselines):
        baselines: str | dict[str, str] = {}
        for item in args.baselines:
            if ":" not in item:
                parser.error(
                    f"--baselines: mix of plain model name and CATEGORY:MODEL pairs is not allowed, got: {item!r}"
                )
            cat, model = item.split(":", 1)
            baselines[cat] = model  # type: ignore[index]
    else:
        if len(args.baselines) > 1:
            parser.error("--baselines: pass a single model name or CATEGORY:MODEL pairs, not multiple plain names.")
        baselines = args.baselines[0]

    main(args.data_dir, baselines, args.output, args.n_samples)
