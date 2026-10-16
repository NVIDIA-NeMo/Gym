#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write an SRT diagnostic recipe while retaining serving flags and workload arguments.

Example (run from the Gym checkout, then submit the generated recipe normally):
  python benchmarks/nemotron_3.5_super/diagnostics/prepare_recipe.py \
    benchmarks/nemotron_3.5_super/sglang_configs/2P2D.yaml \
    --out /tmp/2P2D-diagnostics.yaml --profile

Omit --profile when nsys/CUPTI is active. Worker sidecars require a shared /logs
mount and access to all allocated GPUs. Metrics run during minutes 5–10 after
Gym starts by default; --delay/--seconds adjust the window. No job is submitted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml


RUNNER = "/opt/Gym/benchmarks/nemotron_3.5_super/diagnostics/run.py"
ROOT = "/logs/nemotron-diag"


def instrument(recipe: dict, *, profile: bool, delay: float, seconds: float, steps: int) -> dict:
    """Add a controller wrapper and worker sidecars without changing engine settings."""
    result = copy.deepcopy(recipe)
    if result.get("engine") != "sglang" or result.get("benchmark", {}).get("type") != "custom":
        raise ValueError("Expected an SGLang recipe with a custom Gym benchmark")
    command = result["benchmark"].get("command", "")
    if command.count("gym eval run") != 1 or '"$inference_metrics_config"' not in command:
        raise ValueError("Expected one gym eval run and the recipe-generated inference_metrics_config")
    if "SGLANG_DIAG_SOURCE_RECIPE_JSON" in result["benchmark"].get("env", {}):
        raise ValueError("Recipe is already instrumented")
    if "/opt/Gym" not in result.get("container_mounts", {}).values():
        raise ValueError("Recipe must mount this Gym checkout at /opt/Gym in worker and benchmark containers")
    if profile and any(
        "nsys" in str(recipe.get(key, "")).lower() for key in ("profiling", "profiler", "benchmark", "roles")
    ):
        raise ValueError("Recipe mentions nsys; omit --profile to avoid concurrent CUPTI captures")
    env = result["benchmark"].setdefault("env", {})
    env["SGLANG_DIAG_SOURCE_RECIPE_JSON"] = json.dumps(recipe)
    wrapper = (
        f"python3 {RUNNER} benchmark --root {ROOT} "
        f'--endpoints "$inference_metrics_config" --delay {delay:g} --seconds {seconds:g} --steps {steps}'
        + (" --profile" if profile else "")
        + " -- gym eval run"
    )
    result["benchmark"]["command"] = command.replace("gym eval run", wrapper)
    result.setdefault("services", []).append(
        {
            "name": "nemotron-diag-gpu",
            "type": "generic",
            "placement": {"node": "workers"},
            "start": "before_workers",
            "critical": False,
            "command": ["python3", RUNNER],
            "args": ["worker", "--root", ROOT],
        }
    )
    result["name"] = result.get("name", "sglang") + "-diagnostics"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profile", action="store_true", help="Request 20 scheduler iterations after the metric window"
    )
    parser.add_argument("--delay", type=float, default=300)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if (
        not math.isfinite(args.delay)
        or not math.isfinite(args.seconds)
        or args.delay < 0
        or args.seconds <= 0
        or args.steps <= 0
    ):
        parser.error("delay must be finite and nonnegative; seconds and steps must be positive")
    recipe = yaml.safe_load(args.recipe.read_text())
    result = instrument(recipe, profile=args.profile, delay=args.delay, seconds=args.seconds, steps=args.steps)
    with args.out.open("x") as output:
        yaml.safe_dump(result, output, sort_keys=False)
    print(args.out)


if __name__ == "__main__":
    main()
