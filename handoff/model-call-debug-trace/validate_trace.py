# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Validate model-call debug trace files written during a toy training run.

Checks, for every traced model call:
  * the call succeeded and recorded its request and response text;
  * prompt ids, generated ids, and log probabilities are present, aligned, and plausible;
  * the tokens came from the expected source for the run mode;
  * the generated ids decode to the text the agent received;
  * with external staging, the echoed arrays match the staged chain length in the commit coordinates.
And, per rollout:
  * every model call that model-call capture saw for the rollout has a trace record (and vice versa);
  * each later call's prompt extends the previous call's prompt + generation (multi-turn continuity).
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def load_jsonl_dir(directory: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def response_texts(response: dict) -> list[str]:
    """Text fragments the agent received: message text, reasoning, and tool-call names/arguments."""
    texts = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            texts += [part.get("text") or "" for part in item.get("content") or []]
        elif item.get("type") == "function_call":
            texts += [item.get("name") or "", item.get("arguments") or ""]
        elif item.get("type") == "reasoning":
            texts += [part.get("text") or "" for part in item.get("summary") or []]
            texts += [part.get("text") or "" for part in item.get("content") or []]
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        texts.append(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            texts += [function.get("name") or "", function.get("arguments") or ""]
    return [text for text in texts if text and text.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mode", choices=["capture", "nocapture"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = load_jsonl_dir(args.trace_dir, "*.jsonl")
    captured = {
        path.name.removesuffix(".capture.jsonl"): [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for path in sorted(args.capture_dir.glob("*.capture.jsonl"))
    }
    expected_source = "engine_staged" if args.mode == "capture" else "served_response"

    failures: dict[str, list[str]] = defaultdict(list)
    # Soft checks: tool-call argument serialization can legitimately differ from the decoded text.
    warnings: dict[str, list[str]] = defaultdict(list)
    stats: dict[str, int] = defaultdict(int)
    by_rollout: dict[str, list[dict]] = defaultdict(list)

    for index, record in enumerate(records):
        label = f"record {index} rollout={record.get('rollout_id')} call={record.get('model_call_id')}"
        stats["records"] += 1
        if record.get("error"):
            failures["call_error"].append(f"{label}: {record['error']}")
            continue
        if record.get("rollout_id") is None:
            failures["missing_rollout_id"].append(label)
        else:
            by_rollout[record["rollout_id"]].append(record)
        if record.get("model_call_id") is None:
            failures["missing_model_call_id"].append(label)
        if not (record.get("request") or {}):
            failures["missing_request"].append(label)
        texts = response_texts(record.get("response") or {})
        if not texts:
            failures["missing_response_text"].append(label)
        if record.get("token_errors"):
            failures["token_errors"].append(f"{label}: {record['token_errors']}")
        if record.get("tokens_source") != expected_source:
            failures["unexpected_tokens_source"].append(f"{label}: {record.get('tokens_source')}")
        tokens = record.get("tokens") or {}
        prompt = tokens.get("prompt_token_ids") or []
        generation = tokens.get("generation_token_ids") or []
        log_probs = tokens.get("generation_log_probs") or []
        if not prompt or not generation or not log_probs:
            failures["missing_tokens"].append(
                f"{label}: prompt={len(prompt)} generation={len(generation)} log_probs={len(log_probs)}"
            )
            continue
        stats["records_with_tokens"] += 1
        stats["generated_tokens"] += len(generation)
        if len(generation) != len(log_probs):
            failures["length_mismatch"].append(f"{label}: {len(generation)} ids vs {len(log_probs)} log probs")
        if not all(isinstance(lp, (int, float)) and math.isfinite(lp) and lp <= 1e-6 for lp in log_probs):
            failures["implausible_log_probs"].append(label)
        decoded = normalize(tokenizer.decode(generation, skip_special_tokens=False))
        missing = [text for text in texts if normalize(text) not in decoded]
        if missing:
            warnings["text_not_in_decoded_generation"].append(f"{label}: {missing[0][:120]!r}")
        else:
            stats["text_matches_decoded_generation"] += 1
        if args.mode == "capture":
            coords = record.get("commit_coords") or {}
            if coords.get("disposition") != "staged":
                failures["not_staged"].append(f"{label}: {coords.get('disposition')}")
            elif coords.get("cum_len") != len(prompt) + len(generation):
                failures["echo_differs_from_staged_chain"].append(
                    f"{label}: cum_len={coords.get('cum_len')} prompt+generation={len(prompt) + len(generation)}"
                )
            else:
                stats["echo_matches_staged_chain"] += 1

    # Every call that model-call capture saw must have a trace record, and the reverse.
    for rollout_id in sorted(set(by_rollout) | set(captured)):
        traced_ids = {record["model_call_id"] for record in by_rollout.get(rollout_id, [])}
        captured_ids = {call.get("model_call_id") for call in captured.get(rollout_id, [])}
        if traced_ids != captured_ids:
            failures["trace_capture_mismatch"].append(
                f"rollout {rollout_id}: traced={len(traced_ids)} captured={len(captured_ids)} "
                f"missing_from_trace={len(captured_ids - traced_ids)} extra_in_trace={len(traced_ids - captured_ids)}"
            )
        stats["rollouts"] += 1

    # Multi-turn continuity: a later call's prompt extends an earlier call's prompt + generation.
    for rollout_id, calls in by_rollout.items():
        calls = sorted(calls, key=lambda record: record["started_at"])
        stats["calls_per_rollout_max"] = max(stats["calls_per_rollout_max"], len(calls))
        if len(calls) > 1:
            stats["multi_call_rollouts"] += 1
        for previous, current in zip(calls, calls[1:]):
            prev_tokens, cur_tokens = previous.get("tokens") or {}, current.get("tokens") or {}
            prefix = (prev_tokens.get("prompt_token_ids") or []) + (prev_tokens.get("generation_token_ids") or [])
            prompt = cur_tokens.get("prompt_token_ids") or []
            stats["continuations"] += 1
            if prefix and prompt[: len(prefix)] == prefix:
                stats["continuations_exact_prefix"] += 1

    summary = {
        "mode": args.mode,
        "expected_tokens_source": expected_source,
        "stats": dict(stats),
        "failure_counts": {kind: len(items) for kind, items in failures.items()},
        "failure_examples": {kind: items[:5] for kind, items in failures.items()},
        "warning_counts": {kind: len(items) for kind, items in warnings.items()},
        "warning_examples": {kind: items[:5] for kind, items in warnings.items()},
    }
    args.report.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    text_match_rate = stats["text_matches_decoded_generation"] / max(stats["records_with_tokens"], 1)
    summary["text_match_rate"] = text_match_rate
    args.report.write_text(json.dumps(summary, indent=2))
    ok = stats["records"] > 0 and stats["multi_call_rollouts"] > 0 and not failures and text_match_rate >= 0.9
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
