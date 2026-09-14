# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the Gym FACTS Parametric verifier against the official starter implementation.

Two controls, both written to ``<out>/upstream-vs-gym.jsonl`` and summarised in ``<out>/summary.json``:

``replay``
    Re-parses every grader receipt from a Gym run with the starter's verbatim ``extract_classification`` and
    recomputes the starter's per-example score (``calculate_score``: all three grades CORRECT); the Gym labels,
    reward (mean of three grades) and ``all_correct`` must agree case by case. This isolates the parsing and
    aggregation logic from grader sampling.

``live``
    Re-runs the starter's grading loop (same GRADER_TEMPLATE, three grader calls with seed 0, 1, 2, provider
    defaults) through the grader endpoint for a stratified subset of the Gym run's answers, and compares the labels
    and scores with the Gym receipts. The grader samples, so this measures label stability rather than exact
    equality; disagreements are listed with both sets of grades.

Endpoint settings for ``live`` come from the same environment variables the Gym judge model server reads
(``FACTS_PARAMETRIC_JUDGE_BASE_URL``, ``FACTS_PARAMETRIC_JUDGE_API_KEY``, ``FACTS_PARAMETRIC_JUDGE_MODEL``).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.facts_parametric.upstream_control import calculate_score, extract_classification
from resources_servers.facts_parametric.app import GRADER_TEMPLATE_PATH, LABELS, load_grader_template


STARTER_LABEL_TO_GYM = {
    "CORRECT": "correct",
    "INCORRECT": "incorrect",
    "MISTAKE": "incorrect",
    "NOT_ATTEMPTED": "not_attempted",
    "UNKNOWN": "unknown",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replay(rollouts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    agreements = 0
    for row in rollouts:
        starter_labels = [
            extract_classification(receipt.get("content") or "") for receipt in row.get("judge_receipts", [])
        ]
        starter_gym = [STARTER_LABEL_TO_GYM[label] for label in starter_labels]
        starter_score = calculate_score(starter_labels)
        gym_labels = row.get("judge_labels_starter", [])
        gym_reward = row.get("reward")
        paper_mean = sum(1 for label in starter_gym if label == "correct") / len(starter_gym) if starter_gym else 0.0
        agree = (
            starter_gym == gym_labels
            and abs(paper_mean - float(gym_reward)) < 1e-9
            and starter_score == float(row.get("all_correct", 0.0))
        )
        agreements += agree
        cases.append(
            {
                "case_id": f"{row.get('id')}/replay",
                "control": "replay",
                "task_id": row.get("id"),
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream_labels": starter_labels,
                "upstream_score_all_correct": starter_score,
                "upstream_mean_correct": paper_mean,
                "gym_labels": gym_labels,
                "gym_reward": gym_reward,
                "gym_all_correct": row.get("all_correct"),
                "agree": agree,
            }
        )
    summary = {
        "control": "replay",
        "cases": len(cases),
        "agreement": agreements,
        "disagreements": len(cases) - agreements,
    }
    return cases, summary


class Grader:
    def __init__(self) -> None:
        self.base_url = os.environ["FACTS_PARAMETRIC_JUDGE_BASE_URL"].rstrip("/")
        self.api_key = os.environ["FACTS_PARAMETRIC_JUDGE_API_KEY"]
        self.model = os.environ.get("FACTS_PARAMETRIC_JUDGE_MODEL", "gemini-2.5-pro")

    def grade(self, prompt: str, seed: int) -> dict[str, Any]:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "seed": seed}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "user-agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode())
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        return {
            "model": payload.get("model"),
            "response_id": payload.get("id"),
            "content": content,
            "usage": payload.get("usage"),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }


# Some gateways sit behind a WAF that rejects urllib's default user agent (Cloudflare error 1010).
USER_AGENT = "nemo-gym-facts-parametric-calibrate/1.0"
LIVE_WORKERS = int(os.environ.get("FACTS_PARAMETRIC_JUDGE_CONCURRENCY", "6"))


def stratified_subset(rollouts: list[dict[str, Any]], size: int, seed: int = 20251211) -> list[dict[str, Any]]:
    """Deterministic stratified sample over the Gym label pattern (all correct / mixed / all mistake / hedge / unknown)."""

    def stratum(row):
        labels = row.get("judge_labels", [])
        if row.get("generation_truncated") or row.get("generation_empty"):
            return "truncated_or_empty"
        if all(label == "correct" for label in labels):
            return "all_correct"
        if "not_attempted" in labels:
            return "hedge"
        if "unknown" in labels:
            return "unknown"
        if all(label == "incorrect" for label in labels):
            return "all_mistake"
        return "mixed"

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rollouts:
        groups[stratum(row)].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    per_group = max(1, size // max(1, len(groups)))
    for name in sorted(groups):
        pool = sorted(groups[name], key=lambda row: str(row.get("id")))
        rng.shuffle(pool)
        picked.extend(pool[:per_group])
    remaining = [row for row in rollouts if row not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, size - len(picked))])
    return picked[:size]


def live(rollouts: list[dict[str, Any]], size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template = load_grader_template()
    grader = Grader()
    cases = []
    exact = 0
    score_agree = 0
    label_agree = 0
    total_labels = 0
    subset = stratified_subset(rollouts, size)

    def regrade(row: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = template.format(
            question=row.get("question", ""),
            gold_answer=row.get("expected_answer", ""),
            prediction=row.get("generation", ""),
        )
        assert hashlib.sha256(prompt.encode()).hexdigest() == row.get("judge_prompt_sha256"), (
            "prompt reconstruction drifted from the Gym prompt"
        )
        return [grader.grade(prompt, seed) for seed in range(3)]

    with ThreadPoolExecutor(max_workers=LIVE_WORKERS) as pool:
        regraded = list(pool.map(regrade, subset))
    for row, receipts in zip(subset, regraded):
        upstream_labels = [extract_classification(receipt["content"]) for receipt in receipts]
        upstream_gym = [STARTER_LABEL_TO_GYM[label] for label in upstream_labels]
        gym_labels = row.get("judge_labels_starter", [])
        same_multiset = collections.Counter(upstream_gym) == collections.Counter(gym_labels)
        upstream_mean = sum(1 for label in upstream_gym if label == "correct") / 3
        exact += same_multiset
        score_agree += abs(upstream_mean - float(row.get("reward", 0.0))) < 1e-9
        label_agree += sum(1 for a, b in zip(sorted(upstream_gym), sorted(gym_labels)) if a == b)
        total_labels += 3
        cases.append(
            {
                "case_id": f"{row.get('id')}/live",
                "control": "live",
                "task_id": row.get("id"),
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream_labels": upstream_labels,
                "upstream_mean_correct": upstream_mean,
                "upstream_score_all_correct": calculate_score(upstream_labels),
                "upstream_receipts": receipts,
                "gym_labels": gym_labels,
                "gym_reward": row.get("reward"),
                "agree": same_multiset,
            }
        )
    summary = {
        "control": "live",
        "cases": len(cases),
        "agreement": exact,
        "disagreements": len(cases) - exact,
        "score_agreement": score_agree,
        "label_agreement_rate": label_agree / total_labels if total_labels else None,
    }
    return cases, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--live-size", type=int, default=0, help="Stratified subset size for the live re-grade (0 = replay only)"
    )
    args = parser.parse_args(argv)
    rollouts = _read_jsonl(args.rollouts)
    args.out.mkdir(parents=True, exist_ok=True)
    replay_cases, replay_summary = replay(rollouts)
    cases = list(replay_cases)
    summaries = [replay_summary]
    if args.live_size:
        live_cases, live_summary = live(rollouts, args.live_size)
        cases += live_cases
        summaries.append(live_summary)
    with (args.out / "upstream-vs-gym.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    normalized = {
        "method": "replay of every grader receipt through the starter's verbatim extract_classification/calculate_score"
        + ("; plus a live stratified re-grade with the starter loop" if args.live_size else ""),
        "cases": sum(s["cases"] for s in summaries),
        "agreement": sum(s["agreement"] for s in summaries),
        "disagreements": sum(s["disagreements"] for s in summaries),
        "notes": [
            f"{s['control']}: {s['agreement']} of {s['cases']} cases agree"
            + (
                f"; label-level agreement {s['label_agreement_rate']:.1%}; score agreement {s['score_agreement']} of {s['cases']}"
                if s["control"] == "live"
                else ""
            )
            for s in summaries
        ],
        "details": {
            "controls": summaries,
            "case_ids": [case["case_id"] for case in cases if not case["agree"]][:20]
            or [case["case_id"] for case in cases[:3]],
            "grader_template_sha256": hashlib.sha256(GRADER_TEMPLATE_PATH.read_bytes()).hexdigest(),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps({"controls": summaries, "normalized": normalized}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    assert set(LABELS) == {"correct", "incorrect", "not_attempted", "unknown"}


if __name__ == "__main__":
    main()
