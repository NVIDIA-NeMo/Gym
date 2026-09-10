# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Token usage extraction and cost calculation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from verifiers.types import GenerateOutputs, State

from automationbench.console import console, create_usage_table, print_usage_total
from automationbench.pricing import PricingDatabase


@dataclass
class TaskUsage:
    """Token usage for a single task."""

    task_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float | None  # None if pricing unknown
    output_cost: float | None
    total_cost: float | None


@dataclass
class RunUsage:
    """Token usage for an entire evaluation run."""

    tasks: list[TaskUsage]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float | None
    model: str
    pricing_source: str  # "llm-prices", "cli-override", "fallback", "unknown"


def extract_usage_from_state(state: State) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a task's responses.

    Sums usage from all model responses (multi-turn interactions).
    Includes reasoning tokens (from xAI, OpenAI o-series, etc.) in output count.
    """
    input_tokens = 0
    output_tokens = 0

    # verifiers 0.1.10+ stores raw responses in state["responses"]
    responses = state.get("responses", [])
    for response in responses:
        if response is None:
            continue

        # Handle both ChatCompletion and Completion response types
        usage = getattr(response, "usage", None)
        if usage is None:
            continue

        input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        output_tokens += getattr(usage, "completion_tokens", 0) or 0

        # Add reasoning tokens if present (xAI Grok, OpenAI o-series, etc.)
        # These are billed as output tokens but reported separately
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            output_tokens += reasoning_tokens

    return input_tokens, output_tokens


def calculate_run_usage(
    results: GenerateOutputs,
    model: str,
    pricing_db: PricingDatabase,
) -> RunUsage:
    """Calculate usage metrics for an entire evaluation run."""
    pricing = pricing_db.get_pricing(model)

    tasks: list[TaskUsage] = []
    total_input = 0
    total_output = 0

    # Support Pydantic model (attribute access), legacy dict (parallel lists), and
    # new dict format (results["outputs"] list of RolloutOutput dicts)
    task_list: Any
    state_list: Any
    if "outputs" in results:

        def _task_name(o: Any) -> str:
            info = o.get("info") or {}
            if isinstance(info, str):
                info = json.loads(info)
            return info.get("task_name") or o.get("task", "unknown")

        task_list = [_task_name(o) for o in results["outputs"]]
        # In new format, usage/token data lives directly on each RolloutOutput dict
        state_list = results["outputs"]
    else:
        task_list = getattr(results, "task", None) or results["task"]  # type: ignore[index]
        state_list = getattr(results, "state", None) or results["state"]  # type: ignore[index]
    rollout_items = list(zip(task_list, state_list))

    for task_name, state in rollout_items:
        # Prefer the _usage field we accumulated in add_model_response (via state_columns).
        # Fall back to the framework's token_usage if available, then to trajectory scanning.
        custom_usage = state.get("_usage") or {}
        if custom_usage.get("input_tokens", 0) or custom_usage.get("output_tokens", 0):
            input_tokens = int(custom_usage.get("input_tokens", 0) or 0)
            output_tokens = int(custom_usage.get("output_tokens", 0) or 0)
        else:
            token_usage = state.get("token_usage") or {}
            input_tokens = int(token_usage.get("input_tokens", 0) or 0)
            output_tokens = int(token_usage.get("output_tokens", 0) or 0)

        total_input += input_tokens
        total_output += output_tokens

        # Calculate costs if pricing available
        if pricing is not None:
            input_cost = input_tokens * pricing.input_cost_per_token
            output_cost = output_tokens * pricing.output_cost_per_token
            total_cost = input_cost + output_cost
        else:
            input_cost = None
            output_cost = None
            total_cost = None

        tasks.append(
            TaskUsage(
                task_name=task_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
            )
        )

    # Calculate totals
    if pricing is not None:
        total_cost = pricing.calculate_cost(total_input, total_output)
        pricing_source = pricing.source
    else:
        total_cost = None
        pricing_source = "unknown"

    return RunUsage(
        tasks=tasks,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost=total_cost,
        model=model,
        pricing_source=pricing_source,
    )


def format_cost(cost: float | None) -> str:
    """Format cost for display."""
    if cost is None:
        return "N/A"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_tokens(count: int) -> str:
    """Format token count with thousands separator."""
    return f"{count:,}"


def print_usage_report(usage: RunUsage) -> None:
    """Print a formatted usage report to stdout."""
    table = create_usage_table()

    for i, task in enumerate(usage.tasks, 1):
        breakdown = (
            f"{format_tokens(task.input_tokens)} in / {format_tokens(task.output_tokens)} out"
        )
        table.add_row(
            f"{i}. {task.task_name}",
            format_tokens(task.total_tokens),
            breakdown,
            format_cost(task.total_cost),
        )

    console.print()
    console.print(table)

    # Print totals
    total_tokens = format_tokens(usage.total_tokens)
    breakdown = f"{format_tokens(usage.total_input_tokens)} in / {format_tokens(usage.total_output_tokens)} out"
    cost = format_cost(usage.total_cost)

    if usage.pricing_source == "unknown":
        pricing_info = None
        console.print("[muted]Use --input-cost and --output-cost to specify pricing[/muted]")
    else:
        pricing_info = f"{usage.pricing_source}/{usage.model}"

    print_usage_total(total_tokens, breakdown, cost, pricing_info)
