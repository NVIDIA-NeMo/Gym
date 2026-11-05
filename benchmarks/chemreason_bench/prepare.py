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
"""Prepare ChemReason-Bench for the ``chemreason_bench`` resources server.

Joins upstream's ``prompts.jsonl`` and ``answers.jsonl`` on ``task_id`` at a
pinned commit, and renders the prompts ``predict/predict.py`` builds so a row's
``question`` is byte-identical to what the reference harness sent. See the
server README for the task families, the two protocols and the licensing.

    gym eval prepare --benchmark chemreason_bench
    gym eval prepare --benchmark chemreason_bench +prepare_script_args.limit=50
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


OUTPUT_FPATH = Path(__file__).absolute().parent / "data" / "chemreason_bench_benchmark.jsonl"

# Pinned upstream revision. No tagged release exists; the default branch is mutable.
GITHUB_REPO = "Khadaz/ChemReason-Bench"
GITHUB_REVISION = "c0b9ac2933708fcca47b1795492952cbf280e194"  # pragma: allowlist secret
_RAW = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_REVISION}"
PROMPTS_URL = f"{_RAW}/benchmark_data/prompts.jsonl"
ANSWERS_URL = f"{_RAW}/benchmark_data/answers.jsonl"

# prepare() refuses any other counts: a truncated download exits 0 and silently
# changes the denominator of every metric derived from it.
EXPECTED_TOTAL = 7306
EXPECTED_BY_TASK = {
    "step_completion": 1483,
    "ordering": 1266,
    "rationalization": 1215,
    "step_validation": 1148,
    "condition_validation": 1117,
    "contrastive_choice": 1077,
}

# predict/predict.py line 47. Rendered into the step-completion prompt sorted, as upstream does.
ALLOWED_ACTIONS = sorted(
    {
        "ADD",
        "WASH",
        "FILTER",
        "EXTRACT",
        "REFLUX",
        "QUENCH",
        "RECRYSTALLIZE",
        "PARTITION",
        "TRITURATE",
        "YIELD",
        "PH",
        "STIR",
        "CONCENTRATE",
        "MAKESOLUTION",
        "DRYSOLUTION",
        "COLLECTLAYER",
        "SETTEMPERATURE",
        "WAIT",
        "PHASESEPARATION",
        "DRYSOLID",
        "DEGAS",
        "MICROWAVE",
        "SONICATE",
        "COLUMN",
        "DISTILL",
        "EVAPORATE",
        "TRANSFER",
    }
)

# Upstream's default --ordering_key. The scorer accepts "order" as a fallback, but the
# prompt must ask for exactly one key or the request and the parser disagree.
ORDERING_KEY = "predicted_order"

# Paper appendix F.3.4: these three average two protocols, m_t = (m_gen + m_lm)/2.
# `lm` is a different PROMPT, not the same request with logprobs.
DUAL_PROTOCOL_TASKS = ("step_validation", "condition_validation", "contrastive_choice")
EXPECTED_LM_ROWS = sum(EXPECTED_BY_TASK[t] for t in DUAL_PROTOCOL_TASKS)

# vLLM's default `--max-logprobs` ceiling. Raise both together if a model reports many
# `lm_abstained` rows (Phi-3-mini needed 1000).
DEFAULT_LM_TOP_LOGPROBS = 20


def lm_responses_create_params(top_logprobs: int = DEFAULT_LM_TOP_LOGPROBS) -> Dict[str, Any]:
    """What an `lm` row must send to get a token distribution back.

    `logprobs` is not a Responses API field (extra="forbid") and `top_logprobs`
    alone is inert in vLLM, so both travel via the model server's per-request
    `metadata["extra_body"]`.
    """
    return {"metadata": {"extra_body": json.dumps({"logprobs": True, "top_logprobs": top_logprobs})}}


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# Prompt builders, transcribed from predict/predict.py lines 710-805.


def _prompt_ordering(r: Dict[str, Any]) -> str:
    steps = r.get("steps_to_order") or []
    ls = [f'- id="{s.get("step_id", "")}": {s.get("description", "")}' for s in steps]
    return f"""Context:
{r.get("context", "")}

Question: Arrange the operations in the correct experimental order.
Return JSON ONLY with EXACT keys:
{{"{ORDERING_KEY}": ["<id1>","<id2>","<id3>", ...]}}

Steps:
{chr(10).join(ls)}

Legend (placeholders):
{_dumps(r.get("legend", {}))}"""


def _prompt_contrastive(r: Dict[str, Any]) -> str:
    return f"""Context:
{r.get("context", "")}

Question: {r.get("question", "")}

Options (choose ONE; answer must be exactly one of these strings):
{_dumps(r.get("options") or [])}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Return JSON ONLY with EXACT keys:
{{"predicted_choice": "<one of options>",
  "predicted_option_idx": <int zero-based index>,
  "probs": [<p1>, <p2>, ...]
}}"""


def _prompt_binary(r: Dict[str, Any]) -> str:
    """Asks for a SCORE, not a label; the label is derived as ``score >= 0.5``."""
    return f"""Context:
{r.get("context", "")}

Question: {r.get("question", "")}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Return JSON ONLY with EXACT keys:
{{"score": <number between 0 and 1>}}"""


def _prompt_step_completion(r: Dict[str, Any]) -> str:
    q = r.get("question") or {}
    return f"""A reaction step is missing between two fragments. Fill it with ONE action from this set and minimal slots.
Allowed actions: {", ".join(ALLOWED_ACTIONS)}

CRITICAL RULES:
- Use ONLY placeholder ids for chemicals as given in Legend (e.g., "$1$", "$5$"). NEVER reveal names.
- If a quantity exists, split into "amount_value" (number) and "amount_unit" (string). Do NOT write "10 mL water".
- Tokens (e.g., "#6#", "@10@") are allowed for temperature/duration if present in Legend.
- Keep slots minimal and structured.

Before action: {q.get("before_action", "")}
Before text:
{q.get("before", "")}

After action: {q.get("after_action", "")}
After text:
{q.get("after", "")}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Return JSON ONLY with EXACT keys:
{{"action":"<one_of_allowed_actions>",
  "slots":{{"<slot>":"<value>", ...}}
}}"""


def _prompt_rationalization(r: Dict[str, Any]) -> str:
    return f"""Context:
{r.get("context", "")}

Question: {r.get("question", "")}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Return JSON ONLY with EXACT keys:
{{"gold_rationale":"<1-3 sentences in English only>"}}"""


def _prompt_contrastive_index_only(r: Dict[str, Any]) -> str:
    """lm-protocol prompt: a bare option index, so the decision is one token."""
    opts = r.get("options") or []
    lines = [f"{i}: {o}" for i, o in enumerate(opts)]
    return f"""Context:
{r.get("context", "")}

Question: {r.get("question", "")}

Options (choose the best index):
{chr(10).join(lines)}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Respond with ONLY the integer index (0 to {max(0, len(opts) - 1)}). No JSON. No extra text.
"""


def _prompt_binary_yesno_only(r: Dict[str, Any]) -> str:
    """lm-protocol prompt: a bare YES/NO, so the decision is one token."""
    return f"""Context:
{r.get("context", "")}

Question: {r.get("question", "")}

Legend (placeholders):
{_dumps(r.get("legend", {}))}

Respond with ONLY one token: YES or NO. No JSON. No extra text.
"""


LM_PROMPT_BUILDERS = {
    "step_validation": _prompt_binary_yesno_only,
    "condition_validation": _prompt_binary_yesno_only,
    "contrastive_choice": _prompt_contrastive_index_only,
}

PROMPT_BUILDERS = {
    "ordering": _prompt_ordering,
    "contrastive_choice": _prompt_contrastive,
    "step_validation": _prompt_binary,
    "condition_validation": _prompt_binary,
    "step_completion": _prompt_step_completion,
    "rationalization": _prompt_rationalization,
}


# ---------------------------------------------------------------------------
# Fetch and assemble
# ---------------------------------------------------------------------------


def _fetch_jsonl(url: str) -> List[Dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - pinned https
        payload = response.read().decode("utf-8")
    rows = []
    for line in payload.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _format_row(
    prompt_row: Dict[str, Any],
    answer_row: Dict[str, Any],
    protocol: str = "gen",
    top_logprobs: int = DEFAULT_LM_TOP_LOGPROBS,
) -> Dict[str, Any]:
    task_type = prompt_row["task_type"]
    builders = PROMPT_BUILDERS if protocol == "gen" else LM_PROMPT_BUILDERS
    suffix = "" if protocol == "gen" else "::lm"
    row: Dict[str, Any] = {
        "task_id": prompt_row["task_id"] + suffix,
        # Upstream publishes one split; both fields are provenance, so a row stays
        # identifiable after being merged into a mixed-benchmark file.
        "dataset_name": "chemreason_bench",
        "split": "test",
        "task_type": task_type,
        "protocol": protocol,
        "benchmark_id": prompt_row.get("benchmark_id"),
        "question": builders[task_type](prompt_row),
        # Gold travels with the row; the server reads it back out of verifier_metadata.
        # It is never rendered into `question`.
        "ground_truth": answer_row["ground_truth"],
    }
    # Question-side vocabulary upstream's post-processors need. Not gold.
    if task_type == "ordering":
        row["expected_step_ids"] = [str(s.get("step_id", "")) for s in (prompt_row.get("steps_to_order") or [])]
    elif task_type == "contrastive_choice":
        row["options"] = list(prompt_row.get("options") or [])
    elif task_type == "step_completion":
        # canonicalize_slots resolves reagent names to $n$ through this.
        row["legend"] = dict(prompt_row.get("legend") or {})
    if protocol == "lm":
        row["responses_create_params"] = lm_responses_create_params(top_logprobs)
    return row


def prepare(
    limit: Optional[int] = None,
    protocol: Optional[str] = None,
    top_logprobs: int = DEFAULT_LM_TOP_LOGPROBS,
    output_fpath: Path = OUTPUT_FPATH,
) -> Path:
    """Fetch both upstream files, join on ``task_id``, and write the Gym JSONL.

    Everything is loaded and validated before anything is written, so a failure
    leaves no truncated file behind for the next run to score.

    Arguments are checked here rather than only in ``main``, because Gym calls
    ``prepare`` directly from ``prepare_script_args``: argparse never runs, and a
    silently ignored ``protocol=Gen`` would write a full corpus that reads like a
    subset was requested.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    if not isinstance(top_logprobs, int) or isinstance(top_logprobs, bool) or top_logprobs < 1:
        raise ValueError(f"top_logprobs must be a positive integer, got {top_logprobs!r}")
    if protocol is not None and protocol not in ("gen", "lm"):
        raise ValueError(f"protocol must be 'gen' or 'lm', got {protocol!r}")

    print(f"Downloading {GITHUB_REPO} @ {GITHUB_REVISION[:7]} ...")
    prompts = _fetch_jsonl(PROMPTS_URL)
    answers = _fetch_jsonl(ANSWERS_URL)
    print(f"  loaded {len(prompts)} prompts, {len(answers)} answers")

    answers_by_id = {row["task_id"]: row for row in answers}
    if len(answers_by_id) != len(answers):
        raise ValueError(f"answers.jsonl has duplicate task_ids: {len(answers)} rows, {len(answers_by_id)} unique")

    rows: List[Dict[str, Any]] = []
    lm_rows: List[Dict[str, Any]] = []
    by_task: Dict[str, int] = {}
    for prompt_row in prompts:
        task_id = prompt_row["task_id"]
        answer_row = answers_by_id.get(task_id)
        if answer_row is None:
            raise ValueError(f"no gold for task_id {task_id!r}; refusing to write a partial corpus")
        task_type = prompt_row["task_type"]
        if task_type not in PROMPT_BUILDERS:
            raise ValueError(f"unknown task_type {task_type!r} for {task_id!r}")
        if answer_row.get("task_type") != task_type:
            raise ValueError(f"task_type mismatch for {task_id!r}: {task_type!r} vs {answer_row.get('task_type')!r}")
        rows.append(_format_row(prompt_row, answer_row))
        by_task[task_type] = by_task.get(task_type, 0) + 1
        if task_type in DUAL_PROTOCOL_TASKS:
            lm_rows.append(_format_row(prompt_row, answer_row, protocol="lm", top_logprobs=top_logprobs))

    # Count what parsed, not what the server listed: a complete download whose
    # contents are unreadable would otherwise sail through a byte-length check.
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} instances at this revision, assembled {len(rows)}")
    if by_task != EXPECTED_BY_TASK:
        raise ValueError(f"per-task counts changed at this revision: {by_task} != {EXPECTED_BY_TASK}")
    if len(lm_rows) != EXPECTED_LM_ROWS:
        raise ValueError(f"expected {EXPECTED_LM_ROWS} lm instances, assembled {len(lm_rows)}")
    print(f"  validated {len(rows)} gen instances across {len(by_task)} task types")
    print(f"  validated {len(lm_rows)} lm instances across {len(DUAL_PROTOCOL_TASKS)} discriminative tasks")
    # lm rows trail the gen rows so --limit N still yields a gen-only smoke subset.
    # `--protocol lm` exists because that ordering otherwise makes the lm path
    # unreachable from a limited run, which is how a broken lm request survived
    # all the way to a four-hour job.
    if protocol == "gen":
        pass
    elif protocol == "lm":
        rows = lm_rows
    else:
        rows = rows + lm_rows

    if limit is not None:
        rows = rows[:limit]
        print(f"  --limit {limit}: writing a deliberate subset, NOT the scored population")
    if protocol:
        print(f"  --protocol {protocol}: one protocol only, NOT the scored population")

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} instances to {output_fpath}")
    return output_fpath


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("gen", "lm"),
        default=None,
        help="Write only one protocol's rows. For smoke tests; not a scored population.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Write only the first N instances. For smoke tests; not a scored population.",
    )
    # Validate arguments before the fetch, so a bad value fails fast rather than
    # after several megabytes have been pulled.
    args = parser.parse_args(list(argv) if argv is not None else None)
    prepare(limit=args.limit, protocol=args.protocol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
