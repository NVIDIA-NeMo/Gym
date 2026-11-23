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
"""Build the LongMemEval Gym dataset from the released ``longmemeval_*.json``.

The split file is a JSON array of questions, each carrying its own multi-session
chat haystack. This script renders the haystack into the ``orig-session`` /
JSON-history / no-CoT prompt exactly as upstream
``src/generation/run_generation.prepare_prompt`` does, so the resources server
never needs the haystack at verify time. Everything the judge needs travels in
``verifier_metadata`` (scalars only — the native verify driver drops non-scalar
top-level row fields).

Deliberate deviation from upstream: upstream ``run_generation.py`` (lines
548-580) truncates the rendered history at the tokenizer level to fit the
generator's context window. That is dropped here — it would pull in a
tiktoken/transformers dependency purely for prompt shortening — and truncation
is delegated to the model's own context window. Use ``--topk-context`` to bound
the history by session count.

Splits are downloaded on demand from the public HuggingFace mirror
``xiaowu0162/longmemeval-cleaned`` into a cache directory
(``$XDG_CACHE_HOME/longmemeval`` by default), so nothing outside this server is
needed to build a dataset. ``--input`` reads an already-downloaded file instead.

Each split writes its own ``data/<split>.jsonl``, and ``--limit N`` writes
``data/<split>_limit<N>.jsonl``, so no default invocation clobbers another
split's production file or replaces a full build with a truncated one. An
explicit ``--output`` always wins — ``--split s --output .../oracle.jsonl``
overwrites, which is treated as deliberate intent.

Usage::

    python resources_servers/longmemeval/prepare_longmemeval.py
    python resources_servers/longmemeval/prepare_longmemeval.py --split s --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List


_OUT_DIR = Path(__file__).resolve().parent / "data"

# Public mirror of the released splits (post-2025/09 cleanup).
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
_CACHE_DIRNAME = "longmemeval"
_FETCH_TIMEOUT_S = 600.0

# Split alias → released filename.
_SPLIT_FILES: Dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


def _default_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / _CACHE_DIRNAME


def _fetch_split(split: str, cache_dir: Path) -> Path:
    """Return the local path of ``split``'s JSON, downloading it if absent.

    The download is atomic (write ``.part``, then rename), so an interrupted run
    can never leave a truncated file that a later run would happily parse.
    """
    filename = _SPLIT_FILES[split]
    dst = cache_dir / filename
    if dst.is_file():
        return dst

    url = f"{_HF_BASE}/{filename}"
    print(f"LongMemEval: downloading {url} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)
    return dst


ANSWER_PROMPT_TEMPLATE = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"  # noqa: E501


def _clean_session(session: Any) -> Any:
    """Mirror upstream's ``turn_entry.pop('has_answer')`` cleanup."""
    if not isinstance(session, list):
        return session
    cleaned: List[Any] = []
    for turn in session:
        if isinstance(turn, dict):
            cleaned.append({k: v for k, v in turn.items() if k != "has_answer"})
        else:
            cleaned.append(turn)
    return cleaned


def build_history_string(entry: Dict[str, Any], topk: int = 50) -> str:
    """Render the JSON-format session history for one upstream question entry."""
    pairs = list(zip(entry.get("haystack_dates") or [], entry.get("haystack_sessions") or []))
    if topk > 0:
        pairs = pairs[-topk:]
    pairs.sort(key=lambda pair: pair[0])

    parts: List[str] = []
    for idx, (date, session) in enumerate(pairs, start=1):
        sess_string = "\n" + json.dumps(_clean_session(session))
        parts.append(f"\n### Session {idx}:\nSession Date: {date}\nSession Content:\n{sess_string}\n")
    return "".join(parts)


def _rows_losing_evidence(data: List[Dict[str, Any]], topk: int) -> int:
    """Count entries where the ``topk`` slice would drop a gold evidence session.

    ``build_history_string`` keeps the last ``topk`` sessions in dataset order
    before date-sorting, so a row with more than ``topk`` sessions can lose the
    very session that holds the answer — the row then cannot be answered and
    scores 0.0 for a reason that has nothing to do with the model.
    """
    if topk <= 0:
        return 0
    losing = 0
    for entry in data:
        session_ids = entry.get("haystack_session_ids") or []
        if len(session_ids) <= topk:
            continue
        kept = set(session_ids[-topk:])
        if any(gold not in kept for gold in (entry.get("answer_session_ids") or [])):
            losing += 1
    return losing


def build_row(entry: Dict[str, Any], split: str = "oracle", topk: int = 50) -> Dict[str, Any]:
    """Convert one upstream LongMemEval question entry into a Gym task row."""
    question_id = str(entry.get("question_id", "") or "")
    question = str(entry.get("question", "") or "")
    question_date = str(entry.get("question_date", "") or "")
    history = build_history_string(entry, topk)
    if not history:
        # Upstream run_generation.py asserts history_string != ""; warn instead so
        # one malformed entry cannot abort a whole dataset build.
        warnings.warn(f"empty history for question_id={question_id!r}", RuntimeWarning, stacklevel=2)
    prompt = ANSWER_PROMPT_TEMPLATE.format(history, question_date, question)
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "verifier_metadata": {
            "question_id": question_id,
            "question_type": str(entry.get("question_type", "") or ""),
            "question": question,
            "answer": str(entry.get("answer", "") or ""),
            "question_date": question_date,
            "abstention": "_abs" in question_id,
            "split": split,
            "topk_context": topk,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the LongMemEval Gym dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local longmemeval_*.json to read instead of downloading the split.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where downloaded split files live (default: $XDG_CACHE_HOME/longmemeval).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path; defaults to data/<split>.jsonl, or data/<split>_limit<N>.jsonl with --limit.",
    )
    parser.add_argument("--split", choices=tuple(_SPLIT_FILES), default="oracle", help="Dataset split label.")
    parser.add_argument("--topk-context", type=int, default=50, help="Keep the last N sessions; <=0 keeps all.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Write at most N rows; <=0 writes all. A truncated build goes to "
            "data/<split>_limit<N>.jsonl so it can never overwrite the full data/<split>.jsonl."
        ),
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir if args.cache_dir is not None else _default_cache_dir()
    input_path = args.input if args.input is not None else _fetch_split(args.split, cache_dir)
    default_stem = f"{args.split}_limit{args.limit}" if args.limit > 0 else args.split
    output_path = args.output if args.output is not None else _OUT_DIR / f"{default_stem}.jsonl"

    with open(input_path, "r", encoding="utf-8") as reader:
        data = json.load(reader)
    if args.limit > 0:
        data = data[: args.limit]

    dropped_evidence = _rows_losing_evidence(data, args.topk_context)
    if dropped_evidence:
        warnings.warn(
            f"--topk-context {args.topk_context} drops a gold evidence session on "
            f"{dropped_evidence} of {len(data)} rows, making them unanswerable. "
            f"Use --topk-context 0 (keep all) for a scoreable dataset; upstream's "
            f"run script uses --topk_context 1000.",
            RuntimeWarning,
            stacklevel=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as writer:
        for entry in data:
            row = build_row(entry, split=args.split, topk=args.topk_context)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()
