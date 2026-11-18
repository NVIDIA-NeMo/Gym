# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report Arena-Elo by language against the exact prepared benchmark population."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from nemo_gym.rollouts import RolloutStore
from nemo_gym.utils import failures_path_for
from resources_servers.arena_judge.app import ArenaJudgeConfig, ArenaJudgeServer


INDIC_LANGUAGES = ("bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur")
GROUPS = {
    "Information & advice": "info_advice",
    "Work writing & text tasks": "work_writing",
    "Creative writing / role-play": "creative_roleplay",
    "Planning & brainstorming": "planning",
    "Coding / technical": "coding",
    "Math / reasoning": "math",
    "Casual chat / unclear asks": "casual",
}
SLICES = ("all", *GROUPS.values(), "single_turn", "multi_turn")
VERDICTS = {"A>>B", "A>B", "A=B", "B>A", "B>>A"}
DEFAULT_INPUTS = Path(__file__).parent / "data/arena_proxy_v3_benchmark.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def _flatten(row: dict) -> dict:
    """Accept flat rows and the materialized task / episode reply containers."""
    fields = {key: value for key, value in row.items() if key not in {"task_input", "task_data", "result"}}
    for container in ("task_input", "task_data", "result"):
        child = row.get(container)
        if child is None:
            continue
        if not isinstance(child, dict):
            raise ValueError(f"{container} must be a JSON object")
        for key, value in _flatten(child).items():
            if key in fields and fields[key] != value:
                raise ValueError(f"Conflicting field {key!r} in {container}")
            fields[key] = value
    return fields


def _identity(row: dict) -> str:
    uid = row.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("Every prepared input and judged rollout must carry its nonempty uid")
    return uid


def _slices(row: dict) -> tuple[str, str, str]:
    group = row.get("group")
    if group not in GROUPS:
        raise ValueError(f"{_identity(row)}: unknown chat400 group {group!r}")
    turns = row.get("num_user_turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise ValueError(f"{_identity(row)}: num_user_turns must be a positive integer")
    return "all", GROUPS[group], "single_turn" if turns == 1 else "multi_turn"


def _failed(row: dict) -> bool:
    return bool(
        row.get("_ng_failure_class")
        or row.get("failure")
        or row.get("failure_kind")
        or row.get("mask_sample")
        or row.get("_ng_no_persist")
        or row.get("verification_skipped")
        or row.get("evaluation_completed") is False
    )


def _finite_metrics(metrics: dict) -> dict:
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in metrics.items()
    }


def _population_scope(inputs: Path, expected: dict[str, dict]) -> str:
    report_path = inputs.parent / "preparation_report.json"
    if not report_path.exists():
        return "unverified_inputs"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("prepared_sha256") != hashlib.sha256(inputs.read_bytes()).hexdigest():
        raise ValueError("Prepared inputs do not match preparation_report.json; prepare the dataset again")
    counts = Counter(row["language"] for row in expected.values())
    coverage = report.get("coverage", {})
    for language, count in counts.items():
        entry = coverage.get(language, {})
        if entry.get("prepared_rows") != count:
            raise ValueError(f"{language}: prepared row count disagrees with preparation_report.json")
        excluded = entry.get("excluded_untranslated", [])
        if entry.get("source_rows") != 400 or entry.get("eligible_rows") != 400 - len(set(excluded)):
            raise ValueError(f"{language}: invalid source coverage in preparation_report.json")
    if report.get("limit_per_language") is not None or any(
        coverage[language]["prepared_rows"] != coverage[language]["eligible_rows"] for language in counts
    ):
        return "subset"
    return "full_benchmark"


def summarize(path: Path, *, inputs: Path = DEFAULT_INPUTS, languages: list[str] | None = None) -> dict:
    """Compute the original 100-resample Arena-Elo protocol without a partial headline.

    Inputs define the exact eligible population after translation exclusions. Each task
    must have one judged rollout; repeats and duplicate identities are rejected. Managed
    Gym runs use their selected outcomes, so superseded attempts cannot inflate scores.
    A missing or failed task suppresses its language's scores and the 12-language mean.
    The headline also requires a matching preparation report proving the population
    was not limited to a smoke subset. Unverified inputs still get language scores.
    Successfully returned but unparseable verdicts retain the Arena scorer's invalid
    verdict accounting; they are not silently converted into losses.
    """
    expected = {}
    for raw in _read_jsonl(inputs):
        row = _flatten(raw)
        uid = _identity(row)
        if uid in expected:
            raise ValueError(f"Duplicate prepared uid {uid!r}; this protocol requires one rollout per task")
        if row.get("language") not in ("en", *INDIC_LANGUAGES):
            raise ValueError(f"{uid}: unsupported language {row.get('language')!r}")
        _slices(row)
        expected[uid] = row
    if not expected:
        raise ValueError(f"No prepared tasks in {inputs}")
    population_scope = _population_scope(inputs, expected)
    all_expected = expected
    available = {row["language"] for row in expected.values()}
    selected = languages if languages is not None else sorted(available)
    if not selected or len(set(selected)) != len(selected) or set(selected) - available:
        raise ValueError(f"languages must be unique and present in prepared inputs: {sorted(available)}")
    expected = {uid: row for uid, row in expected.items() if row["language"] in selected}

    store = RolloutStore.read(path, import_legacy=False)
    if store is not None:
        results = store.selected("success")
        failures = store.selected("failure") + store.selected("omitted")
    else:
        results = _read_jsonl(path)
        sidecar = failures_path_for(path)
        failures = _read_jsonl(sidecar) if sidecar.exists() else []

    judged, failed = {}, set()
    for raw in results:
        row = _flatten(raw)
        uid = _identity(row)
        if row.get("language") not in selected and row.get("language") in available:
            continue
        if uid not in expected:
            raise ValueError(f"Unexpected rollout uid {uid!r}; use the matching prepared inputs")
        if uid in judged or uid in failed:
            raise ValueError(f"Duplicate rollout uid {uid!r}; this protocol requires one rollout per task")
        for key in (
            "language",
            "group",
            "num_user_turns",
            "question",
            "baseline_answer",
            "category",
            "messages_sha256",
            "baseline_model",
        ):
            if key not in expected[uid]:
                continue
            if row.get(key) != expected[uid][key]:
                raise ValueError(f"{uid}: rollout {key} disagrees with prepared inputs")
        if _failed(row):
            failed.add(uid)
            continue
        for suffix in ("gen_base", "base_gen"):
            key = f"verdict_{suffix}"
            if key not in row or row[key] is not None and row[key] not in VERDICTS:
                raise ValueError(f"{uid}: missing or unsupported {key}; pass judged Gym rollouts")
            if f"invalid_{suffix}" in row and row[f"invalid_{suffix}"] != (row[key] is None):
                raise ValueError(f"{uid}: inconsistent invalid_{suffix} flag")
        judged[uid] = row
    for raw in failures:
        row = _flatten(raw)
        if row.get("uid") not in all_expected:
            raise ValueError("A failure sidecar row lacks a known uid; resolve failures before publishing metrics")
        if row.get("uid") in expected:
            failed.add(row["uid"])

    # Reuse Gym's scorer directly: strict verdict weight 3, MLE logistic fit,
    # battle-level bootstrap with the original 100 draws and seed 42.
    scorer = ArenaJudgeServer.model_construct(
        config=ArenaJudgeConfig.model_construct(arena_elo_bootstrap_rounds=100, arena_elo_bootstrap_seed=42)
    )
    metrics = {}
    for language in selected:
        population = {uid: row for uid, row in expected.items() if row["language"] == language}
        records = [judged[uid] for uid in population if uid in judged and uid not in failed]
        missing = sorted(set(population) - {row["uid"] for row in records})
        complete = not missing
        entry = {
            "rows": len(population),
            "judged": len(records),
            "complete": complete,
            "missing_or_failed_uids": missing,
            "failed": len(failed & population.keys()),
        }
        for name in SLICES:
            subset = [row for row in records if name in _slices(row)]
            scores = [[row["verdict_gen_base"], row["verdict_base_gen"]] for row in subset]
            aggregate = _finite_metrics(scorer._aggregate_arena_elo(scores))
            if not complete:
                aggregate.update(score=None, ci_lower=None, ci_upper=None)
            entry[name] = aggregate | {
                "verdicts_gen_base": dict(Counter(row["verdict_gen_base"] or "invalid" for row in subset))
            }
        metrics[language] = entry

    indic_complete = population_scope == "full_benchmark" and all(
        language in metrics and metrics[language]["complete"] for language in INDIC_LANGUAGES
    )
    averages = {}
    for name in SLICES:
        values = [metrics[language][name]["score"] for language in INDIC_LANGUAGES if language in metrics]
        averages[name] = (
            sum(values) / len(INDIC_LANGUAGES)
            if indic_complete and all(value is not None for value in values)
            else None
        )
    metrics["_avg"] = averages | {"languages": len(INDIC_LANGUAGES), "complete": indic_complete}
    metrics["_coverage"] = {
        "complete": all(metrics[language]["complete"] for language in selected),
        "languages": selected,
        "population_scope": population_scope,
        "expected": len(expected),
        "failure_records": len(failures),
        "headline_available": averages["all"] is not None,
        "english_reported_separately": "en" in metrics,
    }
    return metrics


def main() -> int:
    """Write strict JSON metrics and exit nonzero for incomplete selected coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument(
        "--inputs", type=Path, default=DEFAULT_INPUTS, help="Prepared benchmark JSONL defining coverage"
    )
    parser.add_argument("--languages", nargs="+", choices=("en", *INDIC_LANGUAGES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = summarize(args.rollouts, inputs=args.inputs, languages=args.languages)
    report = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if metrics["_coverage"]["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
