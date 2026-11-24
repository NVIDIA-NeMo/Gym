# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a native-path run with a Harbor-path run of the same tasks, and give a verdict.

Statistical mode (a model in the loop): tasks are paired across the paths, each task's reward is the mean over its
scored repeats, and the overall mean is the mean over paired tasks. The interval of the difference is a paired
bootstrap over tasks: pairing cancels task difficulty, so what is left is repeat-to-repeat (run-to-run) noise of both
paths plus any real per-task change. PASS when every task was scored on both paths, the masked and failed rates
are within `--max-rate-gap` of each other, and the interval of the difference contains 0 (or lies within
`--tolerance`). Fewer than `--min-tasks` paired tasks, an unscored task or an unfinished run is INCONCLUSIVE.
Supplementary lines, which never change the verdict, help read it at the edges: each path's task-level standard
error (the power of the run), a note when the interval is degenerate because no task differs, an exact sign test on
the tasks that differ, and each path's own repeat-to-repeat spread beside the between-path difference.

Strict mode (model-free checks: reference solution, negative control): every rollout must be scored on both paths
and every reward of a task, on both paths, must be the same value (and equal `--expect` when given).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.harbor_parity.reading import ParityInputError, SideRun, TaskCounts, read_side


PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
EXIT_CODES = {PASS: 0, FAIL: 1, INCONCLUSIVE: 3}
EXIT_INPUT_ERROR = 2
DEFAULT_HARBOR_KEY = "_ng_task_id.task_id"
PATHS = (("native", "native"), ("harbor", "Harbor"))  # (report key, title)
_EPS = 1e-9

STATISTICAL_RULE = (
    "PASS when every task has a scored rollout on both paths, no run is unfinished, the masked rate and the failed "
    "rate of the two paths differ by at most {max_rate_gap}, and the {confidence:.0%} paired bootstrap interval over "
    "tasks of mean(harbor) - mean(native) contains 0{within}. FAIL when the interval excludes that, or the rates "
    "differ by more. INCONCLUSIVE when a task is unscored on a path, a run is unfinished, tasks were left unmatched, "
    "or fewer than {min_tasks} tasks are paired."
)
STRICT_RULE = (
    "Strict (model-free) mode: PASS when every rollout of every task is scored on both paths and all rewards of a task "
    "on both paths are the same value{expect}. FAIL on any differing reward. INCONCLUSIVE when a rollout is masked, "
    "failed or missing and no reward differs."
)


@dataclass
class Settings:
    strict: bool = False
    expect: Optional[float] = None
    min_tasks: int = 10
    max_rate_gap: float = 0.05
    tolerance: float = 0.0
    confidence: float = 0.95
    bootstrap: int = 10000
    seed: int = 0
    allow_unmatched: bool = False


def match_tasks(native: SideRun, harbor: SideRun) -> tuple[list[str], list[str], list[str]]:
    """(tasks on both sides, native-only, Harbor-only), each sorted."""
    both = sorted(set(native.tasks) & set(harbor.tasks))
    return both, sorted(set(native.tasks) - set(harbor.tasks)), sorted(set(harbor.tasks) - set(native.tasks))


def unmatched_message(native: SideRun, harbor: SideRun, native_only: list[str], harbor_only: list[str]) -> str:
    def listing(names: list[str]) -> str:
        shown = ", ".join(names[:20])
        return shown + (f", ... ({len(names) - 20} more)" if len(names) > 20 else "")

    lines = [
        f"tasks do not match (native key {native.key!r}, Harbor key {harbor.key!r}; "
        f"{len(set(native.tasks) & set(harbor.tasks))} matched):"
    ]
    if native_only:
        lines.append(f"  only on the native path ({len(native_only)}): {listing(native_only)}")
    if harbor_only:
        lines.append(f"  only on the Harbor path ({len(harbor_only)}): {listing(harbor_only)}")
    lines.append(
        "Check --native-key/--harbor-key, map ids with --id-map, or run both paths on the same task subset "
        "(--allow-unmatched compares the matched tasks only and makes the verdict INCONCLUSIVE)."
    )
    return "\n".join(lines)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics."""
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap_intervals(
    native_means: list[float], harbor_means: list[float], *, resamples: int, confidence: float, seed: int
) -> dict[str, list[float]]:
    """Percentile intervals of mean(native), mean(harbor) and their difference, resampling paired tasks."""
    count = len(native_means)
    rng = random.Random(seed)
    native_draws, harbor_draws, diff_draws = [], [], []
    for _ in range(resamples):
        picks = [rng.randrange(count) for _ in range(count)]
        native_mean = math.fsum(native_means[i] for i in picks) / count
        harbor_mean = math.fsum(harbor_means[i] for i in picks) / count
        native_draws.append(native_mean)
        harbor_draws.append(harbor_mean)
        diff_draws.append(harbor_mean - native_mean)
    alpha = (1.0 - confidence) / 2.0
    result = {}
    for name, draws in (("native", native_draws), ("harbor", harbor_draws), ("difference", diff_draws)):
        draws.sort()
        result[name] = [_quantile(draws, alpha), _quantile(draws, 1.0 - alpha)]
    return result


def _standard_error(values: list[float]) -> Optional[float]:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def sign_test_p(higher: int, lower: int) -> float:
    """Exact two-sided sign test: the probability, with each differing task equally likely to go either way, of a
    split at least as uneven as `higher` vs `lower`. 1.0 when no task differs."""
    count = higher + lower
    tail = sum(math.comb(count, k) for k in range(min(higher, lower) + 1))
    return min(1.0, 2 * tail / 2**count)


def _repeat_means(tasks: list[TaskCounts]) -> dict[str, Any]:
    """Each repeat's mean reward over the tasks scored at that rollout index, and the spread (largest minus smallest
    repeat mean): how far one path moves against itself between repeats. Rows without a rollout index are left out."""
    by_repeat: dict[Any, list[float]] = {}
    for task in tasks:
        for repeat, reward in task.rewards_by_repeat.items():
            by_repeat.setdefault(repeat, []).append(reward)
    means = [
        {"repeat": repeat, "tasks": len(rewards), "mean_reward": math.fsum(rewards) / len(rewards)}
        for repeat, rewards in sorted(by_repeat.items(), key=lambda item: (type(item[0]).__name__, item[0]))
    ]
    values = [entry["mean_reward"] for entry in means]
    return {"repeat_means": means, "repeat_spread": max(values) - min(values) if len(values) > 1 else None}


def _repeat_noise_se(tasks: list[TaskCounts]) -> Optional[float]:
    """Standard error, due to repeat-to-repeat variation alone, of the mean over `tasks` (the paired tasks, the same
    population as the reported mean): sqrt(sum(var_i / k_i)) / n, where n is the number of tasks and a task with one
    scored repeat contributes 0. None when no task has 2+ scored repeats. Informational: shows how much one path
    moves between runs."""
    if not any(t.scored > 1 for t in tasks):
        return None
    total = math.fsum(statistics.variance(t.rewards) / t.scored for t in tasks if t.scored > 1)
    return math.sqrt(total) / len(tasks)


def _side_totals(side: SideRun, tasks: list[str]) -> dict[str, Any]:
    counts = [side.tasks[name] for name in tasks]
    rollouts = sum(c.rollouts for c in counts)
    masked = sum(c.masked for c in counts)
    failed = sum(c.failed for c in counts)
    missing = sum(c.missing for c in counts)
    kinds: dict[str, int] = {}
    for c in counts:
        for kind, number in c.failure_kinds.items():
            kinds[kind] = kinds.get(kind, 0) + number
    return {
        "rollouts_path": str(side.rollouts_path),
        "failures_path": str(side.failures_path) if side.failures_path else None,
        "materialized_path": str(side.materialized_path) if side.materialized_path else None,
        "key": side.key,
        "agent": ((side.summary or {}).get("agent") or {}).get("instance"),
        "target": (side.summary or {}).get("target"),
        "tasks": len(counts),
        "rollouts": rollouts,
        "scored": sum(c.scored for c in counts),
        "masked": masked,
        "failed": failed,
        "missing": missing,
        "measured_failures": sum(c.measured_failures for c in counts),
        "agent_errors": sum(c.agent_errors for c in counts),
        "masked_rate": masked / rollouts if rollouts else 0.0,
        "failed_rate": failed / rollouts if rollouts else 0.0,
        "repeats": sorted({c.rollouts for c in counts}),
        "failure_kinds": dict(sorted(kinds.items())),
        "reward_keys": dict(sorted(side.reward_keys.items())),
    }


def _task_row(name: str, native: Optional[TaskCounts], harbor: Optional[TaskCounts]) -> dict[str, Any]:
    def side(c: Optional[TaskCounts]) -> Optional[dict[str, Any]]:
        if c is None:
            return None
        return {
            "mean_reward": c.mean,
            "rollouts": c.rollouts,
            "scored": c.scored,
            "masked": c.masked,
            "failed": c.failed,
            "missing": c.missing,
            "rewards": c.rewards,
            "failure_kinds": dict(sorted(c.failure_kinds.items())),
        }

    row: dict[str, Any] = {"task": name, "native": side(native), "harbor": side(harbor), "difference": None}
    if native is not None and harbor is not None and native.mean is not None and harbor.mean is not None:
        row["difference"] = harbor.mean - native.mean
    return row


def _strict_findings(tasks: list[dict[str, Any]], expect: Optional[float]) -> tuple[list[str], list[str]]:
    """(tasks whose rewards differ, tasks with an unscored rollout)."""
    differing, unscored = [], []
    for row in tasks:
        native, harbor = row["native"], row["harbor"]
        if native is None or harbor is None:
            continue
        values = native["rewards"] + harbor["rewards"]
        reference = expect if expect is not None else (values[0] if values else None)
        if reference is not None and any(not math.isclose(v, reference, rel_tol=0.0, abs_tol=_EPS) for v in values):
            differing.append(row["task"])
        elif native["scored"] < native["rollouts"] or harbor["scored"] < harbor["rollouts"]:
            unscored.append(row["task"])
    return differing, unscored


def compare(native: SideRun, harbor: SideRun, settings: Settings) -> dict[str, Any]:
    """The full comparison as a JSON-ready dict. Raises ParityInputError on unmatched tasks unless allowed."""
    both, native_only, harbor_only = match_tasks(native, harbor)
    if (native_only or harbor_only) and not settings.allow_unmatched:
        raise ParityInputError(unmatched_message(native, harbor, native_only, harbor_only))
    if not both:
        raise ParityInputError(unmatched_message(native, harbor, native_only, harbor_only))

    tasks = [_task_row(name, native.tasks[name], harbor.tasks[name]) for name in both]
    paired = [row for row in tasks if row["difference"] is not None]
    native_means = [row["native"]["mean_reward"] for row in paired]
    harbor_means = [row["harbor"]["mean_reward"] for row in paired]
    differences = [row["difference"] for row in paired]
    totals = {"native": _side_totals(native, both), "harbor": _side_totals(harbor, both)}
    paired_names = [row["task"] for row in paired]
    for side_name, side in (("native", native), ("harbor", harbor)):
        paired_counts = [side.tasks[name] for name in paired_names]
        totals[side_name].update(_repeat_means(paired_counts))
        totals[side_name]["repeat_noise_se"] = _repeat_noise_se(paired_counts)

    overall: dict[str, Any] = {"paired_tasks": len(paired)}
    if paired:
        intervals = bootstrap_intervals(
            native_means,
            harbor_means,
            resamples=settings.bootstrap,
            confidence=settings.confidence,
            seed=settings.seed,
        )
        harbor_higher = sum(1 for d in differences if d > _EPS)
        native_higher = sum(1 for d in differences if d < -_EPS)
        overall.update(
            {
                "native_mean": math.fsum(native_means) / len(paired),
                "harbor_mean": math.fsum(harbor_means) / len(paired),
                "difference": math.fsum(differences) / len(paired),
                "native_interval": intervals["native"],
                "harbor_interval": intervals["harbor"],
                "difference_interval": intervals["difference"],
                "difference_standard_error": _standard_error(differences),
                # Spread of per-task means over tasks, per path: the size of shift a run of this many tasks resolves.
                "native_task_standard_error": _standard_error(native_means),
                "harbor_task_standard_error": _standard_error(harbor_means),
                "tasks_differing": harbor_higher + native_higher,
                "tasks_harbor_higher": harbor_higher,
                "tasks_native_higher": native_higher,
                "sign_test_p": sign_test_p(harbor_higher, native_higher),
                # No paired task differs, so every resample gives 0: the interval measures no noise at all.
                "interval_degenerate": harbor_higher + native_higher == 0,
            }
        )

    native_kinds, harbor_kinds = set(totals["native"]["failure_kinds"]), set(totals["harbor"]["failure_kinds"])
    native_rkeys, harbor_rkeys = set(totals["native"]["reward_keys"]), set(totals["harbor"]["reward_keys"])
    one_sided = {
        "failure_kinds_native_only": sorted(native_kinds - harbor_kinds),
        "failure_kinds_harbor_only": sorted(harbor_kinds - native_kinds),
        # Reward keys come from `verifier_rewards` (the Harbor verifier's reward file); native rows rarely carry it.
        "reward_keys_native_only": sorted(native_rkeys - harbor_rkeys) if native_rkeys else [],
        "reward_keys_harbor_only": sorted(harbor_rkeys - native_rkeys) if native_rkeys else [],
    }

    notes = []
    agents = {totals[s]["agent"] for s in ("native", "harbor") if totals[s]["agent"]}
    if len(agents) > 1:
        notes.append(f"the run summaries name different agents: {sorted(agents)}")
    if totals["native"]["repeats"] != totals["harbor"]["repeats"]:
        notes.append(
            f"repeats per task differ: native {totals['native']['repeats']}, Harbor {totals['harbor']['repeats']}"
        )

    fail: list[str] = []
    inconclusive: list[str] = []
    if native_only or harbor_only:
        inconclusive.append(
            f"{len(native_only)} task(s) only on the native path and {len(harbor_only)} only on the Harbor path "
            "were left out (--allow-unmatched)"
        )
    for side_name in ("native", "harbor"):
        if totals[side_name]["missing"]:
            inconclusive.append(
                f"the {side_name} run is unfinished: {totals[side_name]['missing']} expected rollout(s) left no "
                "record; resume it"
            )

    if settings.strict:
        rule = STRICT_RULE.format(expect=f" and equal {settings.expect}" if settings.expect is not None else "")
        differing, unscored = _strict_findings(tasks, settings.expect)
        if differing:
            fail.append(f"{len(differing)} task(s) have differing rewards: {', '.join(differing[:20])}")
        if unscored:
            inconclusive.append(f"{len(unscored)} task(s) have a masked or failed rollout: {', '.join(unscored[:20])}")
    else:
        rule = STATISTICAL_RULE.format(
            max_rate_gap=settings.max_rate_gap,
            confidence=settings.confidence,
            within=f" or lies within +/-{settings.tolerance}" if settings.tolerance > 0 else "",
            min_tasks=settings.min_tasks,
        )
        unpaired = [row["task"] for row in tasks if row["difference"] is None]
        if unpaired:
            inconclusive.append(
                f"{len(unpaired)} task(s) have no scored rollout on a path: {', '.join(unpaired[:20])}"
            )
        for rate in ("masked_rate", "failed_rate"):
            gap = abs(totals["native"][rate] - totals["harbor"][rate])
            if gap > settings.max_rate_gap + _EPS:
                fail.append(
                    f"{rate.replace('_', ' ')} differs by {gap:.3f} (native {totals['native'][rate]:.3f}, "
                    f"Harbor {totals['harbor'][rate]:.3f}; allowed {settings.max_rate_gap})"
                )
        if paired:
            low, high = overall["difference_interval"]
            contains_zero = low <= _EPS and high >= -_EPS
            within_tolerance = low >= -settings.tolerance - _EPS and high <= settings.tolerance + _EPS
            if not (contains_zero or within_tolerance):
                fail.append(
                    f"mean reward differs by {overall['difference']:+.4f}: the {settings.confidence:.0%} paired "
                    f"bootstrap interval over tasks [{low:+.4f}, {high:+.4f}] excludes 0"
                    + (f" and is not within +/-{settings.tolerance}" if settings.tolerance else "")
                )
        if len(paired) < settings.min_tasks:
            if fail:
                inconclusive.append("would FAIL (" + "; ".join(fail) + ") but evidence is too thin")
                fail = []
            inconclusive.append(f"only {len(paired)} paired task(s), fewer than --min-tasks {settings.min_tasks}")

    verdict = FAIL if fail else INCONCLUSIVE if inconclusive else PASS
    return {
        "verdict": verdict,
        "reasons": fail + inconclusive,
        "rule": rule,
        "mode": "strict" if settings.strict else "statistical",
        "settings": {
            "expect": settings.expect,
            "min_tasks": settings.min_tasks,
            "max_rate_gap": settings.max_rate_gap,
            "tolerance": settings.tolerance,
            "confidence": settings.confidence,
            "bootstrap": settings.bootstrap,
            "seed": settings.seed,
        },
        "unmatched": {"native_only": native_only, "harbor_only": harbor_only},
        "overall": overall,
        "sides": totals,
        "one_sided": one_sided,
        "notes": notes,
        "tasks": tasks,
    }


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_interval(interval: Optional[list[float]], digits: int = 3, signed: bool = False) -> str:
    if not interval:
        return "-"
    sign = "+" if signed else ""
    return f"[{interval[0]:{sign}.{digits}f}, {interval[1]:{sign}.{digits}f}]"


def _repeat_text(title: str, side: dict[str, Any]) -> str:
    means = ", ".join(f"repeat {entry['repeat']} {entry['mean_reward']:.3f}" for entry in side["repeat_means"])
    spread = side["repeat_spread"]
    return f"{title} {'-' if spread is None else f'{spread:.4f}'} ({means})"


def _failed_cell(side: dict[str, Any]) -> str:
    return f"{side['failed']} (+{side['missing']} missing)" if side["missing"] else str(side["failed"])


def render_markdown(report: dict[str, Any]) -> str:
    overall, sides = report["overall"], report["sides"]
    lines = [
        "# Harbor parity",
        "",
        f"**Verdict: {report['verdict']}** ({report['mode']} mode)",
        "",
    ]
    lines += [f"- {reason}" for reason in report["reasons"]] or ["- every check passed"]
    lines += ["", f"Rule: {report['rule']}", "", "## Overall", ""]
    confidence = report["settings"]["confidence"]
    lines += [
        "| Path | Tasks | Rollouts | Scored | Masked | Failed | Missing | Mean reward "
        f"| {confidence:.0%} bootstrap interval | Task-level SE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, title in PATHS:
        side = sides[name]
        mean = overall.get(f"{name}_mean")
        lines.append(
            f"| {title} | {side['tasks']} | {side['rollouts']} | {side['scored']} | {side['masked']} "
            f"({side['masked_rate']:.1%}) | {side['failed']} ({side['failed_rate']:.1%}) | {side['missing']} | "
            f"{_fmt(mean)} | {_fmt_interval(overall.get(f'{name}_interval'))} | "
            f"{_fmt(overall.get(f'{name}_task_standard_error'), 4)} |"
        )
    lines.append("")
    if overall.get("paired_tasks"):
        se = overall["difference_standard_error"]
        lines.append(
            f"Difference (Harbor - native) over {overall['paired_tasks']} paired tasks: "
            f"{overall['difference']:+.4f}, {confidence:.0%} paired bootstrap interval "
            f"{_fmt_interval(overall['difference_interval'], 4, True)}"
            + (f", standard error {se:.4f}" if se is not None else "")
            + "."
        )
        if overall["interval_degenerate"]:
            task_se = [
                f"{title} {overall[f'{name}_task_standard_error']:.4f}"
                for name, title in PATHS
                if overall[f"{name}_task_standard_error"] is not None
            ]
            lines.append(
                "The interval is degenerate: no paired task's mean reward differs between the paths, so every "
                "resample gives a difference of 0. It shows that no task changed, not that run-to-run noise is "
                "small, and it is not a tight bound on a shift"
                + (f"; the task-level standard error of each path's mean ({', '.join(task_se)}) " if task_se else "")
                + ("shows the size of shift this run can resolve." if task_se else ".")
            )
        lines += [
            "",
            "Supplementary (not part of the verdict):",
            "",
            f"- Tasks whose mean differs: {overall['tasks_differing']} (Harbor higher {overall['tasks_harbor_higher']}"
            f", native higher {overall['tasks_native_higher']}); exact two-sided sign test on them: "
            f"p = {overall['sign_test_p']:.3f}.",
        ]
        spreads = [_repeat_text(title, sides[name]) for name, title in PATHS if sides[name]["repeat_means"]]
        if spreads:
            lines.append(
                f"- Between-path difference {overall['difference']:+.4f}; each path's own repeat-to-repeat spread "
                f"(largest minus smallest repeat mean, by rollout index): {'; '.join(spreads)}."
            )
    else:
        lines.append("No task has a scored rollout on both paths.")
    noise = [
        f"{title} {sides[name]['repeat_noise_se']:.4f}"
        for name, title in PATHS
        if sides[name]["repeat_noise_se"] is not None
    ]
    if noise:
        lines.append(
            "- Repeat-to-repeat standard error of each path's mean over the paired tasks (a task with one scored "
            f"repeat counts as 0): {', '.join(noise)}."
        )
    lines.append("")
    for name, title in PATHS:
        side = sides[name]
        lines.append(
            f"- {title}: `{side['rollouts_path']}`, key `{side['key']}`"
            + (f", agent `{side['agent']}`" if side["agent"] else "")
            + (f", failure kinds {side['failure_kinds']}" if side["failure_kinds"] else "")
            + (f", reward keys {side['reward_keys']}" if side["reward_keys"] else "")
        )
    one_sided = {label: kinds for label, kinds in report["one_sided"].items() if kinds}
    if one_sided:
        lines += ["", "Seen on one path only:"]
        lines += [f"- {label.replace('_', ' ')}: {', '.join(kinds)}" for label, kinds in one_sided.items()]
    if report["notes"]:
        lines += ["", "Notes:"] + [f"- {note}" for note in report["notes"]]
    unmatched = report["unmatched"]
    if unmatched["native_only"] or unmatched["harbor_only"]:
        lines += [
            "",
            f"Unmatched: native only {unmatched['native_only']}, Harbor only {unmatched['harbor_only']}",
        ]
    lines += [
        "",
        "## Per task",
        "",
        "| Task | Native mean | Native scored/rollouts | Native masked | Native failed | Harbor mean "
        "| Harbor scored/rollouts | Harbor masked | Harbor failed | Difference |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["tasks"]:
        cells = [row["task"].replace("|", "\\|")]
        for name in ("native", "harbor"):
            side = row[name]
            cells += [
                _fmt(side["mean_reward"]),
                f"{side['scored']}/{side['rollouts']}",
                str(side["masked"]),
                _failed_cell(side),
            ]
        difference = row["difference"]
        cells.append("-" if difference is None else f"{difference:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _load_id_map(path: str) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParityInputError(f"cannot read --id-map {path}: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ParityInputError(f"--id-map {path}: expected a JSON object of native id -> Harbor task id strings")
    return {str(k): v for k, v in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.harbor_parity",
        description="Compare a native-path run with a Harbor-path run of the same tasks (same harness, model and "
        "repeats) and give a PASS, FAIL or INCONCLUSIVE verdict. Exit code: 0 PASS, 1 FAIL, 2 input error, "
        "3 INCONCLUSIVE.",
    )
    parser.add_argument("--native", required=True, help="Native run: its rollouts JSONL or the directory holding it.")
    parser.add_argument(
        "--native-key",
        required=True,
        help="Dotted field naming the task on a native row (e.g. verifier_metadata.instance_id). A row without it "
        "is keyed through its materialized input row (same _ng_task_index).",
    )
    parser.add_argument("--harbor", required=True, help="Harbor run: results/harbor/<taskset>/ or its rollouts JSONL.")
    parser.add_argument(
        "--harbor-key",
        default=DEFAULT_HARBOR_KEY,
        help=f"Dotted field naming the task on a Harbor row (default {DEFAULT_HARBOR_KEY}, the task folder name).",
    )
    parser.add_argument(
        "--id-map",
        help="JSON object mapping a native task id to its Harbor task id, for converters that rename tasks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Model-free check (reference solution or negative control): rewards must match exactly per task.",
    )
    parser.add_argument("--expect", type=float, help="With --strict: the reward every rollout must have (1 or 0).")
    parser.add_argument("--min-tasks", type=int, default=10, help="Fewer paired tasks is INCONCLUSIVE (default 10).")
    parser.add_argument(
        "--max-rate-gap",
        type=float,
        default=0.05,
        help="Largest allowed gap between the paths' masked rates, and between their failed rates (default 0.05).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Also PASS when the whole difference interval lies within +/- this (default 0: noise only). Choose and "
        "record it before the run, from the smallest shift in mean reward that would matter; see the README.",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="Interval coverage (default 0.95).")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples (default 10000).")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default 0).")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Compare only the tasks on both sides instead of failing on unmatched ones (verdict INCONCLUSIVE).",
    )
    parser.add_argument("--json", dest="json_path", help="Write the JSON report here.")
    parser.add_argument("--markdown", dest="markdown_path", help="Write the markdown report here (also printed).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expect is not None and not args.strict:
        print("error: --expect needs --strict", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not 0.0 < args.confidence < 1.0 or args.bootstrap < 1 or args.min_tasks < 1:
        print("error: --confidence must be in (0, 1), --bootstrap and --min-tasks at least 1", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.max_rate_gap < 0 or args.tolerance < 0:
        print("error: --max-rate-gap and --tolerance must be at least 0", file=sys.stderr)
        return EXIT_INPUT_ERROR
    settings = Settings(
        strict=args.strict,
        expect=args.expect,
        min_tasks=args.min_tasks,
        max_rate_gap=args.max_rate_gap,
        tolerance=args.tolerance,
        confidence=args.confidence,
        bootstrap=args.bootstrap,
        seed=args.seed,
        allow_unmatched=args.allow_unmatched,
    )
    try:
        id_map = _load_id_map(args.id_map) if args.id_map else None
        native = read_side(args.native, args.native_key, label="native", id_map=id_map)
        harbor = read_side(args.harbor, args.harbor_key, label="harbor")
        report = compare(native, harbor, settings)
    except ParityInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    markdown = render_markdown(report)
    try:
        if args.json_path:
            Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.markdown_path:
            Path(args.markdown_path).write_text(markdown, encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write the report: {exc.strerror or exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    sys.stdout.write(markdown)
    return EXIT_CODES[report["verdict"]]
