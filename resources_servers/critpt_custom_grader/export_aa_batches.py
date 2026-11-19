# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the accepted ``finish`` answers of agentic-recipe rollouts as Artificial Analysis grading batches.

The agentic recipe grades locally and calls no external grading service. To have the same answers graded by the
Artificial Analysis CritPt API, this tool writes one batch per repeat in the format that
``python -m resources_servers.critpt.replay --cache-dir <dir>`` sends:

    <out-dir>/repeat-<r>/submissions.jsonl   one line per task: {"submission_id", "submission", "ts"}
    <out-dir>/repeat-<r>/sources.jsonl       which rollout each submission came from, and its extraction category
    <out-dir>/manifest.json                  per-repeat counts, problem ids and file digests

``r`` is the rollouts' ``_ng_rollout_index``. Each answer is taken from the trajectory with the grader's own rule
(``submission.extract_finish_source``): the last fenced block of the accepted ``finish.answer`` whose info string is
empty or ``python``, else the whole answer. The local grading verdict is not used, so an answer whose local reference
preflight failed is still exported. A scored no-answer (no accepted ``finish``, an empty answer, an exhausted turn
budget) is exported as an empty code block, which the API grades as incorrect, as the recipe scores it zero.

A repeat is written only when it holds exactly one exportable rollout for every task in the file. A rollout whose
outcome is unscorable (``response_incomplete`` or ``submission_invalid``, such as a provider failure or a wall-clock
stop) or a missing one leaves its repeat unwritten, because the API grades only full batches and an infrastructure
failure must not be graded as a wrong answer; regenerate those rollouts and export again. Exit code 0 means every
repeat was written, 3 that at least one was not (each is reported with its reasons).

    python -m resources_servers.critpt_custom_grader.export_aa_batches \\
        --rollouts results/rollouts.jsonl --out-dir results/aa_batches --batch-size 70
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from resources_servers.critpt_custom_grader.submission import extract_finish_source


EXPORTED_CATEGORIES = ("candidate_ready", "candidate_no_answer")


def generated_code(source: str | None) -> str:
    """Wrap a source the way the CritPt resources server submits it."""
    return f"```python\n{source}\n```" if source else "```python\n```"


def _problem_id(row: dict[str, Any]) -> str:
    problem_id = row.get("problem_id")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError(f"rollout task {row.get('_ng_task_index')} has no problem_id")
    return problem_id


def load_rollouts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            for key in ("_ng_task_index", "_ng_rollout_index"):
                if not isinstance(row.get(key), int):
                    raise ValueError(f"{path}:{line_number} has no integer {key}")
            rows.append(row)
    return rows


def plan_batches(rows: list[dict[str, Any]], batch_size: int) -> dict[int, dict[str, Any]]:
    """Group rollouts by repeat, extract each answer, and decide which repeats form a full batch."""
    tasks: dict[int, str] = {}
    for row in rows:
        problem_id = _problem_id(row)
        if tasks.setdefault(row["_ng_task_index"], problem_id) != problem_id:
            raise ValueError(f"task {row['_ng_task_index']} has rollouts for two problem ids")
    if len(set(tasks.values())) != len(tasks):
        raise ValueError("two task indexes share a problem id; a batch may hold each problem once")
    if len(tasks) != batch_size:
        raise ValueError(f"the rollouts cover {len(tasks)} tasks; a batch needs exactly {batch_size}")

    by_repeat: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_repeat[row["_ng_rollout_index"]][row["_ng_task_index"]].append(row)

    plans = {}
    for repeat in sorted(by_repeat):
        entries, problems = [], []
        for task_index in sorted(tasks):
            candidates = by_repeat[repeat].get(task_index, [])
            if len(candidates) != 1:
                problems.append({"task_index": task_index, "reason": f"{len(candidates)} rollouts"})
                continue
            row = candidates[0]
            response = row.get("response")
            submission = extract_finish_source(response) if isinstance(response, dict) else None
            category = submission.category if submission is not None else "response_missing"
            if category not in EXPORTED_CATEGORIES:
                problems.append({"task_index": task_index, "reason": category})
                continue
            entries.append(
                {
                    "task_index": task_index,
                    "problem_id": tasks[task_index],
                    "category": category,
                    "source": submission.source,
                    "local_reward": row.get("reward"),
                    "local_category": row.get("category"),
                }
            )
        plans[repeat] = {"entries": entries, "problems": problems}
    return plans


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_batches(plans: dict[int, dict[str, Any]], out_dir: Path, model: str) -> dict[str, Any]:
    for name in ("manifest.json", *(f"repeat-{repeat}" for repeat in plans)):
        if (out_dir / name).exists():
            raise FileExistsError(f"{out_dir / name} exists; export into a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"model": model, "repeats": {}}
    for repeat, plan in plans.items():
        record: dict[str, Any] = {"written": not plan["problems"], "problems": plan["problems"]}
        if not plan["problems"]:
            repeat_dir = out_dir / f"repeat-{repeat}"
            repeat_dir.mkdir()
            now = time.time()
            with (repeat_dir / "submissions.jsonl").open("w") as subs, (repeat_dir / "sources.jsonl").open("w") as src:
                for submission_id, entry in enumerate(plan["entries"]):
                    submission = {
                        "problem_id": entry["problem_id"],
                        "generated_code": generated_code(entry["source"]),
                        "model": model,
                        "generation_config": {},
                    }
                    subs.write(
                        json.dumps({"submission_id": submission_id, "submission": submission, "ts": now}) + "\n"
                    )
                    source = {
                        "submission_id": submission_id,
                        "task_index": entry["task_index"],
                        "rollout_index": repeat,
                        "problem_id": entry["problem_id"],
                        "category": entry["category"],
                        "local_reward": entry["local_reward"],
                        "local_category": entry["local_category"],
                    }
                    src.write(json.dumps(source) + "\n")
            record.update(
                submissions=len(plan["entries"]),
                no_answer=sum(e["category"] == "candidate_no_answer" for e in plan["entries"]),
                submissions_sha256=_sha256(repeat_dir / "submissions.jsonl"),
            )
        manifest["repeats"][str(repeat)] = record
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export agentic-recipe finish answers as Artificial Analysis CritPt grading batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--rollouts", type=Path, required=True, help="Rollouts JSONL written by gym eval run.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for repeat-<r>/ batch folders.")
    parser.add_argument("--batch-size", type=int, default=70, help="Tasks per batch; the API accepts exactly 70.")
    parser.add_argument("--model", default="unknown", help="Model name recorded in each submission.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plans = plan_batches(load_rollouts(args.rollouts), args.batch_size)
        manifest = write_batches(plans, args.out_dir, args.model)
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for repeat, record in manifest["repeats"].items():
        if record["written"]:
            print(f"repeat {repeat}: {record['submissions']} submissions, {record['no_answer']} without an answer")
        else:
            reasons = defaultdict(int)
            for problem in record["problems"]:
                reasons[problem["reason"]] += 1
            print(f"repeat {repeat}: not written ({dict(reasons)})", file=sys.stderr)
    return 0 if all(record["written"] for record in manifest["repeats"].values()) else 3


if __name__ == "__main__":
    sys.exit(main())
