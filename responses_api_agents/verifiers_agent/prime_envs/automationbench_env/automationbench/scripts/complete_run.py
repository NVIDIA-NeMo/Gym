# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Re-run aborted tasks in a run file and merge them back — guaranteeing zero aborts.

An *abort* is a rollout cut off mid-work: the transcript ends on an assistant turn
that requested tool calls which were never executed, while step budget still
remained (``steps < cap - 2``). These are infra/transient terminations (a dropped
connection, a gateway 5xx, an un-retried rate limit) — not model decisions and not
the step cap. They score 0, so they drag a run's pass rate below the truth.

This tool:
  1. detects aborted tasks in an existing run file,
  2. re-runs ONLY those tasks with the run's own config (model / reasoning effort /
     toolset / domains) — the hardened clients now retry transient errors, so the
     re-run completes,
  3. merges the fresh per-task records back in (keeping id + position),
  4. recomputes every aggregate (scores, pass rate, token totals, cost), and
  5. loops until zero aborts remain (or --max-rounds is hit, which is reported loudly).

By default the original is preserved as ``<name>-UNPATCHED-<ts>.json`` and the merged
file takes the canonical name. It doubles as:
  * the post-run completeness gate (``eval.py --ensure-complete`` calls heal_run_file), and
  * resume-by-task: anything aborted is simply re-run and spliced in.

Usage:
    # Patch an existing run file in visualizer/runs/local (renames original to UNPATCHED)
    uv run python -m automationbench.scripts.complete_run <run.json> \
        [--base-url URL] [--api-key-var VAR] \
        [--api auto|anthropic|responses|chat_completions|gemini_interactions] \
        [--max-concurrent N] [--max-rounds N] [--step-cap 50]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from automationbench.console import console
from automationbench.domains import DOMAIN_ALIASES
from automationbench.export import STEP_CAP_MARGIN, is_aborted
from automationbench.usage import format_cost

__all__ = ["STEP_CAP_MARGIN", "is_aborted", "find_aborted", "heal_run", "heal_run_file"]


def find_aborted(run: dict[str, Any], step_cap: int) -> list[str]:
    return [t["name"] for t in run["tasks"] if is_aborted(t, step_cap)]


def recompute_aggregates(run: dict[str, Any], step_cap: int = 50) -> None:
    """Recompute summary + usage_by_task in place, mirroring export.export_results."""
    tasks = run["tasks"]
    n = len(tasks)
    costs = [t.get("cost") for t in tasks]
    total_cost = sum(c for c in costs) if (n and all(c is not None for c in costs)) else None

    summary = run.setdefault("summary", {})
    summary["avg_score"] = sum(float(t["score"]) for t in tasks) / n if n else 0.0
    summary["pass_rate"] = sum(1 for t in tasks if t["passed"]) / n if n else 0.0
    summary["passed_count"] = sum(1 for t in tasks if t["passed"])
    summary["failed_count"] = sum(1 for t in tasks if not t["passed"])
    summary["total_input_tokens"] = sum(int(t.get("input_tokens", 0) or 0) for t in tasks)
    summary["total_output_tokens"] = sum(int(t.get("output_tokens", 0) or 0) for t in tasks)
    summary["total_cost"] = total_cost
    if total_cost is not None:
        summary["cost_formatted"] = format_cost(total_cost)
    summary.setdefault("cost_formatted", "N/A")
    summary["tasks_with_empty_responses"] = sum(1 for t in tasks if t.get("empty_responses"))
    summary["tasks_with_errors"] = sum(1 for t in tasks if t.get("errors"))
    summary["tasks_with_zero_output"] = sum(
        1 for t in tasks if (t.get("output_tokens", 0) or 0) == 0
    )
    # Self-describing partial-success marker: non-empty means infra cut these
    # tasks off mid-work and healing didn't (or couldn't) replace them.
    summary["aborted_tasks"] = find_aborted(run, step_cap)

    run["usage_by_task"] = [
        {
            "task_id": t["id"],
            "task_name": t["name"],
            "input_tokens": t.get("input_tokens", 0),
            "output_tokens": t.get("output_tokens", 0),
            "total_tokens": (t.get("input_tokens", 0) or 0) + (t.get("output_tokens", 0) or 0),
            "cost": t.get("cost"),
        }
        for t in tasks
    ]


def merge_rerun(run: dict[str, Any], rerun_export: dict[str, Any], step_cap: int = 50) -> int:
    """Replace each re-run task's record in `run` by name (keeping id + position),
    then recompute aggregates. Returns the number of tasks replaced."""
    by_name = {t["name"]: t for t in rerun_export["tasks"]}
    replaced = 0
    for i, t in enumerate(run["tasks"]):
        new = by_name.get(t["name"])
        if new is None:
            continue
        merged = dict(new)
        merged["id"] = t["id"]  # preserve original index/position
        run["tasks"][i] = merged
        replaced += 1
    recompute_aggregates(run, step_cap)
    return replaced


def _domains_from_meta(meta: dict[str, Any]) -> list[str]:
    """Expand the run's recorded domains through aliases for dataset construction."""
    domains: list[str] = []
    for d in meta.get("domains", []):
        domains.extend(DOMAIN_ALIASES.get(d, [d]))
    return domains


async def _rerun(
    meta,
    task_names,
    export_path,
    *,
    base_url,
    api_key_var,
    api,
    max_concurrent,
    step_cap,
    extra_headers=None,
    extra_body=None,
    batch=None,
    batch_linger_seconds=20.0,
    batch_poll_seconds=30.0,
    model=None,
) -> None:
    # Imported lazily to avoid an import cycle (eval imports this module's gate).
    from automationbench.scripts.eval import run_evaluation

    await run_evaluation(
        model=model or meta["model"],
        domains=_domains_from_meta(meta),
        base_url=base_url,
        api_key_var=api_key_var or "OPENAI_API_KEY",
        num_examples=-1,
        max_turns=step_cap,
        export_json=str(export_path),
        tasks=",".join(task_names),
        reasoning_effort=meta.get("reasoning_effort"),
        extra_headers=extra_headers,
        extra_body=extra_body,
        api=api,
        toolset=meta.get("toolset", "api"),
        max_concurrent=max_concurrent,
        batch=bool(meta.get("batch")) if batch is None else batch,
        batch_linger_seconds=batch_linger_seconds,
        batch_poll_seconds=batch_poll_seconds,
    )


def heal_run(
    run: dict[str, Any],
    *,
    base_url=None,
    api_key_var=None,
    api="auto",
    max_concurrent=100,
    max_rounds=3,
    step_cap=50,
    extra_headers=None,
    extra_body=None,
    batch=None,
    batch_linger_seconds=20.0,
    batch_poll_seconds=30.0,
    model=None,
    scratch_dir: Path | None = None,
    label: str = "run",
) -> int:
    """Re-run aborted tasks and merge until none remain. Mutates `run` in place.

    Returns the count of aborts still remaining (0 = fully healed)."""
    scratch_dir = scratch_dir or Path("visualizer/runs/local")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for rnd in range(1, max_rounds + 1):
        aborted = find_aborted(run, step_cap)
        if not aborted:
            console.print(f"[success]✓ {label}: no aborted tasks remain[/success]")
            return 0
        console.print(
            f"[key]{label}: round {rnd}/{max_rounds} — re-running {len(aborted)} aborted task(s)[/key]"
        )
        tmp = scratch_dir / f".rerun-{label}-r{rnd}.json"
        asyncio.run(
            _rerun(
                run["meta"],
                aborted,
                tmp,
                base_url=base_url,
                api_key_var=api_key_var,
                api=api,
                max_concurrent=max_concurrent,
                step_cap=step_cap,
                extra_headers=extra_headers,
                extra_body=extra_body,
                batch=batch,
                batch_linger_seconds=batch_linger_seconds,
                batch_poll_seconds=batch_poll_seconds,
                model=model,
            )
        )
        rerun_export = json.load(open(tmp))
        replaced = merge_rerun(run, rerun_export, step_cap)
        tmp.unlink(missing_ok=True)
        console.print(f"[key]  merged {replaced} re-run record(s)[/key]")

    remaining = find_aborted(run, step_cap)
    if remaining:
        console.print(
            f"[bold red]✗ {label}: {len(remaining)} task(s) STILL aborted after "
            f"{max_rounds} rounds — endpoint may be unhealthy. Tasks: {remaining}[/bold red]"
        )
    return len(remaining)


def heal_run_file(path: str | Path, *, rename_original: bool = True, **kwargs) -> int:
    """Heal a run JSON file on disk. If rename_original, the pre-patch file is kept
    as <name>-UNPATCHED-<ts>.json and the healed run takes the canonical name.

    Returns remaining abort count (0 = success)."""
    path = Path(path)
    run = json.load(open(path))
    step_cap = kwargs.get("step_cap", 50)

    before = find_aborted(run, step_cap)
    if not before:
        console.print(f"[success]✓ {path.name}: already clean (0 aborts) — nothing to do[/success]")
        return 0
    console.print(f"[key]{path.name}: {len(before)} aborted task(s) detected[/key]")

    remaining = heal_run(run, label=path.stem, **kwargs)

    # Write the healed run, preserving the original as UNPATCHED.
    tmp_out = path.with_suffix(".patched.tmp")
    with open(tmp_out, "w") as f:
        json.dump(run, f, indent=2)
    if rename_original:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        unpatched = path.with_name(f"{path.stem}-UNPATCHED-{ts}{path.suffix}")
        shutil.move(str(path), str(unpatched))
        console.print(f"[key]original preserved as:[/key] {unpatched.name}")
    shutil.move(str(tmp_out), str(path))
    sm = run["summary"]
    console.print(
        f"[success]✓ {path.name}: pass {sm['pass_rate']:.1%}  "
        f"avg {sm['avg_score']:.1%}  ({sm['passed_count']}/{len(run['tasks'])} passed)[/success]"
    )
    return remaining


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Re-run aborted tasks and patch a run file.")
    parser.add_argument("run_file", help="Path to the run JSON to patch")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (e.g. the Vercel AI Gateway /v1, or an alpha endpoint)",
    )
    parser.add_argument("--api-key-var", default=None, help="Env var holding the API key")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model id from the run's meta (e.g. to hit a "
        "provider directly instead of via a proxy alias)",
    )
    parser.add_argument(
        "--api",
        default="auto",
        choices=["auto", "anthropic", "chat_completions", "responses", "gemini_interactions"],
    )
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--step-cap", type=int, default=50)
    parser.add_argument(
        "--headers",
        action="append",
        metavar="KEY=VALUE",
        help="Custom HTTP header for re-run requests (repeatable), "
        "e.g. --headers x-custom-header=value",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="JSON object merged into every re-run request body — pass the same "
        "value as the original run so healed tasks keep their gateway "
        "providerTimeouts and FinOps tags "
        '(e.g. \'{"providerOptions":{"gateway":{"tags":[...]}}}\')',
    )
    parser.add_argument(
        "--no-rename", action="store_true", help="Heal in place without writing a -UNPATCHED copy"
    )
    args = parser.parse_args()

    extra_headers = None
    if args.headers:
        extra_headers = {}
        for h in args.headers:
            if "=" not in h:
                parser.error(f"Invalid header format: {h}. Use KEY=VALUE.")
            k, v = h.split("=", 1)
            extra_headers[k] = v

    remaining = heal_run_file(
        args.run_file,
        rename_original=not args.no_rename,
        base_url=args.base_url,
        api_key_var=args.api_key_var,
        api=args.api,
        max_concurrent=args.max_concurrent,
        max_rounds=args.max_rounds,
        step_cap=args.step_cap,
        extra_headers=extra_headers,
        extra_body=args.extra_body,
        model=args.model,
    )
    raise SystemExit(1 if remaining else 0)


if __name__ == "__main__":
    main()
