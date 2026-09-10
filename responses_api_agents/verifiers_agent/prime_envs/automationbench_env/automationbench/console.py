# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Orange-themed console output for AutomationBench CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Zapier orange theme
ZAPIER_ORANGE = "#FF4A00"

theme = Theme(
    {
        "orange": ZAPIER_ORANGE,
        "header": f"bold {ZAPIER_ORANGE}",
        "success": "bold green",
        "failure": "bold red",
        "muted": "dim",
        "key": ZAPIER_ORANGE,
        "value": "white",
    }
)

console = Console(theme=theme)


def print_banner() -> None:
    """Print the AutomationBench banner."""
    banner = Panel(
        "[header]AutomationBench[/header]\n[muted]AI Business Workflow Benchmark by Zapier[/muted]",
        border_style="orange",
        padding=(0, 2),
    )
    console.print(banner)


def print_config(model: str, domains: list[str], base_url: str | None = None) -> None:
    """Print evaluation configuration."""
    console.print()
    console.print("[key]Model:[/key]", model)
    console.print("[key]Domains:[/key]", ", ".join(domains))
    if base_url:
        console.print("[key]Base URL:[/key]", base_url)
    console.print()


def print_results_header() -> None:
    """Print results section header."""
    console.print()
    console.rule("[header]Results[/header]", style="orange")


def print_avg_reward(reward: float, pass_rate: float | None = None) -> None:
    """Print the average reward score (partial credit) and optional binary pass rate."""
    pct = f"{reward:.0%}"
    style = "success" if reward >= 0.5 else "failure"
    console.print(f"[key]Average score (partial credit):[/key] [{style}]{pct}[/{style}]")
    if pass_rate is not None:
        pr_pct = f"{pass_rate:.0%}"
        pr_style = "success" if pass_rate >= 0.5 else "failure"
        console.print(f"[key]Pass rate (100% only):[/key] [{pr_style}]{pr_pct}[/{pr_style}]")


def print_task_result(task_num: int, task_name: str, passed: bool) -> None:
    """Print a single task result."""
    status = "[success]PASS[/success]" if passed else "[failure]FAIL[/failure]"
    console.print(f"  Task {task_num} [muted]({task_name})[/muted]: {status}")


def create_usage_table() -> Table:
    """Create a table for token usage display."""
    table = Table(
        title="Token Usage",
        title_style="header",
        border_style="orange",
        header_style="orange",
    )
    table.add_column("Task", style="white")
    table.add_column("Tokens", justify="right")
    table.add_column("In/Out", justify="right", style="muted")
    table.add_column("Cost", justify="right", style="orange")
    return table


def print_usage_total(
    total_tokens: str,
    breakdown: str,
    cost: str,
    pricing_info: str | None = None,
) -> None:
    """Print the usage totals."""
    console.print()
    console.print(f"[key]Total:[/key] {total_tokens} [muted]({breakdown})[/muted]")
    if pricing_info:
        console.print(
            f"[key]Estimated cost:[/key] [orange]{cost}[/orange] [muted]({pricing_info})[/muted]"
        )
    else:
        console.print(f"[key]Estimated cost:[/key] [orange]{cost}[/orange]")
