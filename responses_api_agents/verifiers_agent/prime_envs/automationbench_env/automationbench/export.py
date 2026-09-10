# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Export evaluation results to JSON for visualization."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from automationbench.task_contract import TASK_CONTRACT_SCHEMA, task_contract_sha256
from automationbench.usage import RunUsage, format_cost

STEP_CAP_MARGIN = 2


def is_aborted(task: dict[str, Any], step_cap: int) -> bool:
    """True if a task's transcript ends on an unexecuted assistant tool-call turn
    with step budget left (``steps < step_cap - STEP_CAP_MARGIN``) — the signature
    of a rollout cut off mid-work by an infra/transient termination rather than a
    model decision or the step cap. See automationbench.scripts.complete_run for
    the healing loop that re-runs these."""
    msgs = task.get("messages") or []
    if not msgs:
        return False
    last = msgs[-1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return int(task.get("steps", 0)) < step_cap - STEP_CAP_MARGIN
    return False


def _read_benchmark_version() -> str:
    """Read benchmark version from pyproject.toml — the single source of truth."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    try:
        content = pyproject.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "unknown"


def export_results(
    outputs: list[dict[str, Any]],
    usage: RunUsage,
    model: str,
    domains: list[str],
    output_path: Path | str | None = None,
    duration_seconds: float | None = None,
    toolset: str | None = None,
    reasoning_effort: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    step_cap: int = 50,
) -> dict[str, Any]:
    """
    Export evaluation results to a JSON structure for visualization.

    Args:
        outputs: Per-task output dicts from the evaluation run
        usage: Token usage and cost data
        model: Model name used for evaluation
        domains: List of domains evaluated
        output_path: Optional path to save JSON file
        duration_seconds: Wall-clock duration of the evaluation run
        extra_meta: Optional extra key/values merged into the meta block
            (e.g. {"batch": True} for Batch API transport runs)

    Returns:
        Dictionary with structured results ready for visualization
    """
    # Extract task results
    task_results = []
    for i, output in enumerate(outputs):
        reward = output.get("reward", 0.0)
        task_info = output.get("info") or {}
        if isinstance(task_info, str):
            import json as _json

            task_info = _json.loads(task_info)
        task_name = task_info.get("task_name") or output.get("task", f"task_{i + 1}")

        # Extract assertions if available
        assertions = task_info.get("assertions", [])

        # Get messages/conversation — serialize vf.Message Pydantic objects to plain dicts
        def _serialize_msg(m):
            if hasattr(m, "model_dump"):
                return m.model_dump()
            return m

        prompt = output.get("prompt") or []
        completion = output.get("completion") or []
        messages = [
            _serialize_msg(m)
            for m in (prompt + completion if isinstance(prompt, list) else completion)
        ]

        # Get token usage - prefer _usage accumulated in add_model_response (via state_columns)
        task_usage = output.get("_usage") or {}
        if task_usage.get("input_tokens", 0) or task_usage.get("output_tokens", 0):
            input_tokens = int(task_usage.get("input_tokens", 0) or 0)
            output_tokens = int(task_usage.get("output_tokens", 0) or 0)
        else:
            token_usage = output.get("token_usage") or {}
            if token_usage.get("input_tokens", 0) or token_usage.get("output_tokens", 0):
                input_tokens = int(token_usage.get("input_tokens", 0) or 0)
                output_tokens = int(token_usage.get("output_tokens", 0) or 0)
            else:
                # Estimate from messages (~4 characters per token)
                input_tokens = 0
                output_tokens = 0
                for msg in messages:
                    content = msg.get("content", "") or ""
                    if isinstance(content, list):
                        content = str(content)
                    char_count = len(content)
                    if msg.get("role") == "assistant":
                        output_tokens += char_count // 4
                        if msg.get("tool_calls"):
                            output_tokens += len(str(msg.get("tool_calls"))) // 4
                    else:
                        input_tokens += char_count // 4

        # Count model response steps (what --max-steps caps)
        steps = sum(1 for msg in messages if msg.get("role") == "assistant")

        # Get per-task cost from usage data
        task_cost = None
        if i < len(usage.tasks):
            task_cost = usage.tasks[i].total_cost

        # Get debug info (finish_reasons, empty_responses, errors)
        debug_info = output.get("_debug") or {}
        finish_reasons = debug_info.get("finish_reasons", [])
        # Raw provider stop reasons, one per model call including the episode's
        # final call (finish_reasons is captured at env_response time and misses
        # it, which is exactly where refusal-classifier stops land).
        stop_reasons = debug_info.get("stop_reasons", [])
        empty_responses = debug_info.get("empty_responses", [])
        errors = debug_info.get("errors", [])

        # Per-assertion results stored by rubric as a side effect
        assertion_results = output.get("_assertion_results") or []
        assertions_passed = (
            sum(1 for r in assertion_results if r.get("passed") and not r.get("excluded"))
            if assertion_results
            else int(reward * len(assertions))
            if assertions
            else 0
        )
        assertions_total = (
            sum(1 for r in assertion_results if not r.get("excluded"))
            if assertion_results
            else len(assertions)
        )

        contract_fields: dict[str, str] = {}
        precomputed_schema = output.get("_task_contract_schema")
        precomputed_sha256 = output.get("_task_contract_sha256")
        if (
            precomputed_schema == TASK_CONTRACT_SCHEMA
            and isinstance(precomputed_sha256, str)
            and len(precomputed_sha256) == 64
        ):
            contract_fields = {
                "task_contract_schema": TASK_CONTRACT_SCHEMA,
                "task_contract_sha256": precomputed_sha256,
            }
        elif (
            output.get("example_id") is not None
            and isinstance(prompt, list)
            and isinstance(task_info.get("assertions"), list)
            and isinstance(task_info.get("initial_state"), dict)
            and isinstance(task_info.get("zapier_tools"), list)
        ):
            contract_fields = {
                "task_contract_schema": TASK_CONTRACT_SCHEMA,
                "task_contract_sha256": task_contract_sha256(
                    example_id=output["example_id"], prompt=prompt, info=task_info
                ),
            }

        # Performance metrics (model/tool latency, tool-call count, cache split).
        # Captured per task in state["_perf"] by the clients + env. Absent on
        # older runs; default to a zeroed block so downstream readers are uniform.
        perf = output.get("_perf") or {}
        cached_input_tokens = int(perf.get("cached_input_tokens", 0) or 0)
        # Anthropic usage.input_tokens excludes cache read/write tokens; the client
        # accumulates those in extra_input_tokens so the input total is comparable
        # to OpenAI-style accounting (where cached is a subset of prompt tokens).
        input_tokens += int(perf.get("extra_input_tokens", 0) or 0)

        task_result = {
            "id": i + 1,
            "name": task_name,
            "score": float(reward),
            "passed": reward == 1.0,
            "assertions_total": assertions_total,
            "assertions_passed": assertions_passed,
            "assertion_results": assertion_results,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": max(0, input_tokens - cached_input_tokens),
            "reasoning_tokens": int(perf.get("reasoning_tokens", 0) or 0),
            "num_tool_calls": int(perf.get("tool_calls", 0) or 0),
            "num_model_calls": int(perf.get("model_calls", 0) or 0),
            "model_time_s": round(float(perf.get("model_time_s", 0.0) or 0.0), 3),
            "tool_time_s": round(float(perf.get("tool_time_s", 0.0) or 0.0), 3),
            "cost": task_cost,
            "steps": steps,
            "messages": messages,  # Include full chat completion
            **contract_fields,
            "end_state": output.get("_end_state"),
        }

        # Add debug info if there were any issues
        if empty_responses:
            task_result["empty_responses"] = empty_responses
        if errors:
            task_result["errors"] = errors
        if finish_reasons:
            task_result["finish_reasons"] = finish_reasons
        if stop_reasons:
            task_result["stop_reasons"] = stop_reasons
            # A refusal on the final call means the safety classifier killed the
            # completion and the episode died there. Scored as a normal failure,
            # but flagged so flake-adjusted reporting and re-runs can find it.
            task_result["died_on_refusal"] = stop_reasons[-1] == "refusal"

        task_results.append(task_result)

    # Build export structure
    export_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "toolset": toolset,
            "domains": domains,
            "total_tasks": len(task_results),
            "duration_seconds": duration_seconds,
            "benchmark_version": _read_benchmark_version(),
            "reasoning_effort": reasoning_effort,
            "max_steps": step_cap,
            **(extra_meta or {}),
        },
        "summary": {
            "avg_score": sum(float(o.get("reward", 0.0)) for o in outputs) / len(outputs)
            if outputs
            else 0.0,
            "pass_rate": (
                sum(1 for t in task_results if t["passed"]) / len(task_results)
                if task_results
                else 0.0
            ),
            "passed_count": sum(1 for t in task_results if t["passed"]),
            "failed_count": sum(1 for t in task_results if not t["passed"]),
            # Sum the per-task input_tokens (which add back cache read/write via
            # extra_input_tokens) rather than usage.total_input_tokens, which carries
            # Anthropic's cache-EXCLUDING input_tokens and undercounts once prompt
            # caching is on. Keeps the summary consistent with the per-task rows.
            "total_input_tokens": sum(t["input_tokens"] for t in task_results),
            "total_output_tokens": usage.total_output_tokens,
            "total_cached_input_tokens": sum(t.get("cached_input_tokens", 0) for t in task_results),
            "total_reasoning_tokens": sum(t.get("reasoning_tokens", 0) for t in task_results),
            "total_tool_calls": sum(t.get("num_tool_calls", 0) for t in task_results),
            "total_model_time_s": round(sum(t.get("model_time_s", 0.0) for t in task_results), 1),
            "total_tool_time_s": round(sum(t.get("tool_time_s", 0.0) for t in task_results), 1),
            "total_cost": usage.total_cost,
            "cost_formatted": format_cost(usage.total_cost),
            # Debug stats
            "tasks_with_empty_responses": sum(1 for t in task_results if t.get("empty_responses")),
            "tasks_with_errors": sum(1 for t in task_results if t.get("errors")),
            "tasks_with_zero_output": sum(
                1 for t in task_results if t.get("output_tokens", 0) == 0
            ),
            "tasks_refusal_stopped": sum(1 for t in task_results if t.get("died_on_refusal")),
            # Tasks cut off mid-work by infra (scored 0, not model failures).
            # Non-empty means the run is a partial success — the console warning
            # from the --ensure-complete gate is no longer the only record.
            "aborted_tasks": [t["name"] for t in task_results if is_aborted(t, step_cap)],
        },
        "tasks": task_results,
        "usage_by_task": [
            {
                "task_id": t["id"],
                "task_name": t["name"],
                "input_tokens": t["input_tokens"],
                "output_tokens": t["output_tokens"],
                "total_tokens": t["input_tokens"] + t["output_tokens"],
                "cost": t["cost"],
            }
            for t in task_results
        ],
    }

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    return export_data


def export_partial_results(
    partial_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Convert partial results saved by verifiers to our visualizer format.

    Useful for recovering data from interrupted runs.

    Args:
        partial_path: Path to the partial results directory (contains metadata.json + results.jsonl)
        output_path: Optional path to save converted JSON file

    Returns:
        Dictionary with structured results ready for visualization
    """
    partial_path = Path(partial_path)

    # Load metadata
    metadata_file = partial_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No metadata.json found in {partial_path}")

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Load results
    results_file = partial_path / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.jsonl found in {partial_path}")

    task_results = []
    with open(results_file) as f:
        for i, line in enumerate(f):
            result = json.loads(line)
            reward = result.get("reward", 0.0)
            info = result.get("info", {})
            if isinstance(info, str):
                info = json.loads(info)
            task_name = info.get("task_name") or result.get("task", f"task_{i + 1}")
            assertions = info.get("assertions", [])

            # Calculate tokens from completion
            completion = result.get("completion", [])
            input_tokens = 0
            output_tokens = 0
            steps = 0
            for msg in completion:
                if msg.get("role") == "assistant":
                    steps += 1
                    # Rough token estimate: 4 chars per token
                    content = msg.get("content", "") or ""
                    output_tokens += len(content) // 4

            task_results.append(
                {
                    "id": i + 1,
                    "name": task_name,
                    "score": float(reward),
                    "passed": reward == 1.0,
                    "assertions_total": len(assertions),
                    "assertions_passed": int(reward * len(assertions)) if assertions else 0,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "steps": steps,
                }
            )

    # Build export structure
    total_tasks = len(task_results)
    passed_count = sum(1 for t in task_results if t["passed"])
    avg_score = sum(t["score"] for t in task_results) / total_tasks if total_tasks > 0 else 0.0

    export_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model": metadata.get("model", "unknown"),
            "domains": ["partial"],  # Can't determine from partial data
            "total_tasks": total_tasks,
            "partial": True,  # Flag that this is from partial results
            "benchmark_version": _read_benchmark_version(),
        },
        "summary": {
            "avg_score": avg_score,
            "passed_count": passed_count,
            "failed_count": total_tasks - passed_count,
            "total_input_tokens": sum(t["input_tokens"] for t in task_results),
            "total_output_tokens": sum(t["output_tokens"] for t in task_results),
            "total_cost": 0.0,  # Can't calculate without pricing info
            "cost_formatted": "N/A (partial)",
        },
        "tasks": task_results,
        "usage_by_task": [
            {
                "task_id": t["id"],
                "task_name": t["name"],
                "input_tokens": t["input_tokens"],
                "output_tokens": t["output_tokens"],
                "total_tokens": t["input_tokens"] + t["output_tokens"],
            }
            for t in task_results
        ],
    }

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    return export_data
