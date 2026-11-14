# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report identity outcomes and coverage."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DIMENSIONS = ("tactic", "tactic_kind", "strategy", "locale", "mode")
CORRECTNESS_METRICS = ("correct_developer", "correct_model")
# Keep metric names here so reporting does not need to import the UserSim-dependent scorer.
METRICS = (
    "passed",
    *CORRECTNESS_METRICS,
    "false_attribution",
    "other_mistake",
    "contradiction",
    "human_claim",
    "denial",
    "wrong_answer",
    "uncertain",
    "unrecognized_name",
)
# TODO: Replace this list if UserSim labels whether each opening asks for the developer or model identity.
# UserSim openings that do not ask who the model is ("What is your knowledge cutoff?").
# Identity disclosure is optional, but any identity claims must still be correct.
OPENINGS_NOT_ASKING_IDENTITY = frozenset(
    {"indirect_knowledge", "indirect_version", "indirect_open_source", "human_or_machine"}
)


# A policy failure (e.g. a disabled-tool call) has no scoring result: count it as 0 in
# pass/correctness rates and exclude it from mistake rates. A graded reply with no identity
# claim is different: it has 0 correctness but can pass an indirect question.
FAILED_BY_POLICY_FAILURE = ("passed", *CORRECTNESS_METRICS)


def validate_correctness_metrics(outcome: dict) -> None:
    """Reject incomplete scoring results instead of silently reporting zero correctness."""
    missing = [metric for metric in CORRECTNESS_METRICS if metric not in (outcome.get("metrics") or {})]
    if missing:
        raise ValueError(
            f"Missing correctness metrics: {', '.join(missing)}. "
            "Rescore stored rollouts with python -m benchmarks.model_identity.scoring <rollouts.jsonl>."
        )


def _summarize(records: list[dict]) -> dict:
    """Rates over judged samples; unscored (masked) and missing samples are counted, never rated.

    ``unscored_rate`` is the share of expected samples UserSim could not score, nearly always
    because its extractor's reply could not be parsed (``unscored_reasons``). A high rate
    leaves the other rates resting on fewer samples, and possibly on the easier ones.
    """
    scored = [r for r in records if r["status"] == "scored"]
    failed = [r for r in records if r["status"] == "policy_failure"]
    counts = Counter(r["status"] for r in records)

    def rate(metric: str) -> float | None:
        pool = scored + failed if metric in FAILED_BY_POLICY_FAILURE else scored
        return (
            sum(r["metrics"].get(metric, 0.0) for r in pool if r["status"] == "scored") / len(pool) if pool else None
        )

    flagged = [r for r in scored if r.get("needs_review")]
    return {
        "expected": len(records),
        "scored": len(scored),
        # Samples UserSim flags for a person to check (dropped claims, unrecognized names,
        # contradictions, an extractor from the developer under test), and why.
        "needs_review_rate": len(flagged) / len(scored) if scored else None,
        "review_reasons": dict(Counter(reason for r in flagged for reason in r["review_reasons"]).most_common()),
        "policy_failure": len(failed),
        "unscored": counts["unscored"],
        "unscored_rate": counts["unscored"] / len(records) if records else None,
        "unscored_reasons": dict(Counter(r["unscored_reason"] for r in records if r["status"] == "unscored")),
        "missing": counts["missing"],
        "coverage": (len(scored) + len(failed)) / len(records) if records else None,
        **{f"{metric}_rate": rate(metric) for metric in METRICS},
    }


def _reason_kind(reason: str) -> str:
    """A review reason without its counts, turns and names: "2 claim(s) dropped: ..." -> "N claim(s) dropped"."""
    return re.sub(r"\d+", "N", reason.split(":")[0]).strip()


def _row_digest(row: dict) -> str:
    """The digest the UserSim verifier records for the row it was seeded with."""
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sampling_mismatch(role_request_params: dict, invocations: list) -> bool:
    """Whether any model call sent a setting other than the one the panel row gives its role."""
    return any(
        invocation.get("request", {}).get(name) != value
        for invocation in invocations
        for name, value in role_request_params.get(invocation.get("role"), {}).items()
    )


def _top_false_attributions(records: list[dict]) -> dict:
    """How many scored samples attribute themselves to each developer and to each model name."""
    developers, models = Counter(), Counter()
    for record in records:
        found = record.get("false_attributions") or []
        developers.update({f["developer"] for f in found if f["developer"]})
        models.update({f["model"] for f in found if f["model"]})
    return {"developers": developers.most_common(), "models": models.most_common()}


def build_report(
    panel: list[dict], rollouts: list[dict], *, num_repeats: int = 1, expected_identity: dict | None = None
) -> dict:
    """Join on stable task ids, taking the latest attempt for each task/repeat.

    Rejects rollouts run on a different preparation: a different row, or, given the panel's
    ``expected_identity``, graded against a different identity. Rows alone do not record
    the identity, so a panel prepared for another model has the same rows.
    """
    if num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    by_id = {
        row["task_id"]["task_id"] if isinstance(row["task_id"], dict) else row["task_id"]: (
            row.get("task_input") or row
        )
        for row in panel
    }
    if len(by_id) != len(panel):
        raise ValueError("Panel contains duplicate task ids")
    latest = {}
    for row in rollouts:
        identity = row.get("_ng_task_id") or {}
        task_id = identity.get("task_id") if isinstance(identity, dict) else identity
        if task_id not in by_id:
            raise ValueError(f"Rollout task {task_id!r} does not belong to this panel")
        repeat = row.get("_ng_rollout_index", 0)
        if not isinstance(repeat, int) or not 0 <= repeat < num_repeats:
            raise ValueError("Unexpected repeat index; supply the run's --num-repeats")
        key = task_id, repeat
        if key not in latest or row.get("_ng_attempt_index", 0) >= latest[key].get("_ng_attempt_index", 0):
            latest[key] = row
    records = []
    groups = {dimension: defaultdict(list) for dimension in DIMENSIONS}
    if not by_id:
        raise ValueError("Expected a nonempty panel")
    for task_id, entry in by_id.items():
        task = entry["resolved_row"]
        for repeat in range(num_repeats):
            row = latest.get((task_id, repeat))
            record = {
                "task_id": task_id,
                "repeat": repeat,
                "status": "missing",
            }
            if row is not None:
                verification = row.get("verification") or {}
                data = verification.get("verifier_data") or {}
                if data.get("resolved_row_sha256") not in (None, _row_digest(task)):
                    raise ValueError(f"Rollout {task_id!r} ran on a different preparation of this panel")
                if _sampling_mismatch(entry.get("role_request_params") or {}, row.get("invocations") or []):
                    raise ValueError(f"Rollout {task_id!r} ran with different sampling settings than this panel")
                outcome = (data.get("scorer_result") or {}).get("identity_outcome")
                if row.get("mask_sample"):
                    reason = (data.get("scorer_result") or {}).get("error") or row.get("failure_reason")
                    record.update(status="unscored", unscored_reason=str(reason or "unknown"))
                elif data.get("scorer_state") != "ok" or not outcome:
                    record["status"] = "policy_failure"
                else:
                    validate_correctness_metrics(outcome)
                    if expected_identity is not None and outcome.get("expected") != expected_identity:
                        raise ValueError(
                            f"Rollout {task_id!r} was graded against {outcome.get('expected')}, "
                            f"but the panel expects {expected_identity}"
                        )
                    scorer = data.get("scorer_result") or {}
                    record.update(
                        status="scored",
                        needs_review=bool(scorer.get("needs_review")),
                        review_reasons=sorted({_reason_kind(reason) for reason in scorer.get("review_reasons") or []}),
                        metrics=outcome["metrics"],
                        false_attributions=outcome["false_attributions"],
                        other_mistakes=outcome["other_mistakes"],
                        unrecognized_names=outcome.get("unrecognized_names") or [],
                    )
            records.append(record)
            for dimension in DIMENSIONS:
                groups[dimension][str(task["identity_case"][dimension])].append(record)
    return {
        "benchmark": "model_identity",
        "num_repeats": num_repeats,
        "overall": _summarize(records),
        **_per_row(records, num_repeats),
        "breakdowns": {
            dimension: {key: _summarize(values) for key, values in sorted(group.items())}
            for dimension, group in groups.items()
        },
        "top_false_attributions": _top_false_attributions(records),
        "false_attributions": [
            {"task_id": r["task_id"], "repeat": r["repeat"], **found}
            for r in records
            for found in r.get("false_attributions") or []
        ],
        "other_mistakes": [
            {"task_id": r["task_id"], "repeat": r["repeat"], **found}
            for r in records
            for found in r.get("other_mistakes") or []
        ],
        # Model names outside UserSim's vocabulary, with their quotes to check.
        "unrecognized_names": [
            {"task_id": r["task_id"], "repeat": r["repeat"], **found}
            for r in records
            for found in r.get("unrecognized_names") or []
        ],
    }


def _per_row(records: list[dict], num_repeats: int) -> dict:
    """With several repeats, how often each row passes, since a wrong identity may be an occasional sample.

    Counts each row's scored repeats and its policy failures, which fail as they do in the pass
    rate; unscored and missing repeats are left out, and so is a row with none counted.
    """
    if num_repeats < 2:
        return {}
    passes: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["status"] == "scored":
            passes[record["task_id"]].append(record["metrics"]["passed"])
        elif record["status"] == "policy_failure":
            passes[record["task_id"]].append(0.0)
    rates = {task_id: sum(values) / len(values) for task_id, values in passes.items()}
    return {
        "rows": {
            "scored": len(rates),
            # Rows that passed every scored repeat, and rows that failed every one.
            "always_passed_rate": sum(rate == 1 for rate in rates.values()) / len(rates) if rates else None,
            "never_passed_rate": sum(rate == 0 for rate in rates.values()) / len(rates) if rates else None,
        },
        "row_pass_rates": [
            {"task_id": task_id, "pass_rate": rate, "repeats_counted": len(passes[task_id])}
            for task_id, rate in sorted(rates.items(), key=lambda item: (item[1], item[0]))
        ],
    }


def _load(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--panel", type=Path, default=Path("benchmarks/model_identity/data/benchmark.jsonl"))
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.panel.with_suffix(".manifest.json")
    expected_identity = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(args.panel.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("Panel dataset checksum does not match its manifest")
        if "settings" in manifest:
            settings = manifest["settings"]
            expected_identity = {
                "developer": settings["expected_developer"],
                "model_names": settings["expected_model_names"],
            }
    report = build_report(
        _load(args.panel), _load(args.rollouts), num_repeats=args.num_repeats, expected_identity=expected_identity
    )
    output = args.output or args.rollouts.with_name(args.rollouts.stem + "_identity_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["overall"], indent=2))
    print(f"Identity metrics: {output}")


if __name__ == "__main__":
    main()
