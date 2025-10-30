# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward profiling analysis script for GRL Sokoban.
Generates comprehensive statistics and metrics required for CONTRIBUTING.md.
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd


def load_rollouts(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load rollouts from JSONL file."""
    rollouts = []
    with open(jsonl_path) as f:
        for line in f:
            rollouts.append(json.loads(line))
    return rollouts


def compute_reward_statistics(rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute basic reward statistics."""
    rewards = [r["reward"] for r in rollouts]
    rewards_sorted = sorted(rewards)
    n = len(rewards_sorted)

    return {
        "total_rollouts": n,
        "min": min(rewards),
        "max": max(rewards),
        "mean": sum(rewards) / n,
        "median": rewards_sorted[n // 2] if n % 2 else (rewards_sorted[n // 2 - 1] + rewards_sorted[n // 2]) / 2,
    }


def compute_success_rate(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute success rate."""
    total = len(rollouts)
    successes = sum(1 for r in rollouts if r.get("success", False))
    return {
        "total": total,
        "successes": successes,
        "success_rate": successes / total if total > 0 else 0,
    }


def compute_reward_distribution(rollouts: List[Dict[str, Any]]) -> Dict[float, int]:
    """Compute reward distribution histogram."""
    distribution = defaultdict(int)
    for r in rollouts:
        distribution[r["reward"]] += 1
    return dict(sorted(distribution.items()))


def compute_tool_call_metrics(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute tool call statistics."""
    tool_call_counts = []
    for r in rollouts:
        # Handle nested structure: response.output
        output = r.get("response", {}).get("output", [])
        count = sum(1 for item in output if item.get("type") == "function_call")
        tool_call_counts.append(count)

    if not tool_call_counts:
        return {
            "avg_tool_calls": 0,
            "min_tool_calls": 0,
            "max_tool_calls": 0,
        }

    return {
        "avg_tool_calls": sum(tool_call_counts) / len(tool_call_counts),
        "min_tool_calls": min(tool_call_counts),
        "max_tool_calls": max(tool_call_counts),
    }


def compute_tool_call_correlation(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute correlation between tool calls and rewards."""
    data = []
    for r in rollouts:
        # Handle nested structure: response.output
        output = r.get("response", {}).get("output", [])
        tool_calls = sum(1 for item in output if item.get("type") == "function_call")
        data.append({"tool_calls": tool_calls, "reward": r["reward"]})

    if not data:
        return {"correlation": 0, "tool_call_breakdown": {}}

    df = pd.DataFrame(data)
    correlation = df["tool_calls"].corr(df["reward"]) if len(df) > 1 else 0

    # Group by tool call count
    breakdown = df.groupby("tool_calls")["reward"].agg(["mean", "count"]).to_dict("index")

    return {
        "correlation": correlation,
        "tool_call_breakdown": breakdown,
    }


def generate_report(
    rollouts_path: str,
    model_name: str = "Qwen3-30B-A3B",
    output_path: str = None,
) -> str:
    """Generate complete reward profiling report."""
    print(f"Loading rollouts from {rollouts_path}...")
    rollouts = load_rollouts(rollouts_path)

    print("Computing statistics...")
    reward_stats = compute_reward_statistics(rollouts)
    success_stats = compute_success_rate(rollouts)
    reward_dist = compute_reward_distribution(rollouts)
    tool_call_metrics = compute_tool_call_metrics(rollouts)
    tool_call_corr = compute_tool_call_correlation(rollouts)

    # Generate report
    report = f"""
# Reward Profiling Report: {model_name}

## Dataset Overview
- **Rollouts file**: `{rollouts_path}`
- **Total rollouts**: {reward_stats["total_rollouts"]:,}

## Reward Distribution

### Summary Statistics
- **Min reward**: {reward_stats["min"]:.4f}
- **Max reward**: {reward_stats["max"]:.4f}
- **Mean reward**: {reward_stats["mean"]:.4f}
- **Median reward**: {reward_stats["median"]:.4f}

### Success Rate
- **Successful rollouts**: {success_stats["successes"]:,} / {success_stats["total"]:,}
- **Success rate**: {success_stats["success_rate"]:.2%}

### Reward Histogram
"""

    # Add reward distribution
    for reward, count in sorted(reward_dist.items(), key=lambda x: -x[1])[:20]:
        report += f"- Reward {reward:.4f}: {count:,} occurrences ({count / reward_stats['total_rollouts']:.1%})\n"

    if len(reward_dist) > 20:
        report += f"... and {len(reward_dist) - 20} more unique reward values\n"

    # Tool call metrics
    report += f"""
## Tool Call Metrics

### Overall Statistics
- **Average tool calls per rollout**: {tool_call_metrics["avg_tool_calls"]:.2f}
- **Min tool calls**: {tool_call_metrics["min_tool_calls"]}
- **Max tool calls**: {tool_call_metrics["max_tool_calls"]}

### Correlation with Reward
- **Pearson correlation (tool calls ↔ reward)**: {tool_call_corr["correlation"]:.4f}

### Mean Reward by Tool Call Count
"""

    for tool_calls, stats in sorted(tool_call_corr["tool_call_breakdown"].items()):
        report += f"- {tool_calls} tool calls: mean reward = {stats['mean']:.4f} ({stats['count']} rollouts)\n"

    report += """
---
*Generated by analyze_rewards.py for CONTRIBUTING.md reward profiling requirements*
"""

    # Save report if output path specified
    if output_path:
        print(f"Saving report to {output_path}...")
        with open(output_path, "w") as f:
            f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze rollout rewards for CONTRIBUTING.md requirements")
    parser.add_argument(
        "--rollouts-path",
        type=str,
        required=True,
        help="Path to rollouts JSONL file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3-30B-A3B",
        help="Model name for the report header",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the report (defaults to stdout)",
    )

    args = parser.parse_args()

    report = generate_report(
        rollouts_path=args.rollouts_path,
        model_name=args.model_name,
        output_path=args.output,
    )

    if not args.output:
        print(report)
    else:
        print(f"✓ Report saved to {args.output}")


if __name__ == "__main__":
    main()
