#!/usr/bin/env python3
"""Evaluate one system-prompt module candidate through live NeMo Gym servers.

Terminal 1:
    gym env start \
      "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml]" \
      ++observability_enabled=true \
      ++debug_mode=false

Terminal 2:
    uv run --no-project --python .venv/bin/python gym harness optimize \
      --config scripts/agent_harness_optimization_live_example.yaml

The command exercises:
SystemPromptCandidate -> SystemPromptConfigAdapter -> RolloutCollectionHelper ->
Agent /run -> Resources /verify -> native rollout evidence -> HarnessOptimizer.

Running this module directly evaluates only the seed candidate.
"""

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from nemo_gym.agent_harness_optimization import (
    SystemPromptCandidate,
)
from nemo_gym.harness_optimization import (
    HarnessOptimizationRecipe,
    build_system_prompt_evaluator,
    load_source_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="scripts/agent_harness_optimization_live_example.yaml",
        help="Harness-evaluation recipe YAML.",
    )
    return parser.parse_args()


def load_recipe(path: str) -> HarnessOptimizationRecipe:
    with Path(path).open() as handle:
        return HarnessOptimizationRecipe.model_validate(yaml.safe_load(handle))


async def main() -> None:
    args = parse_args()
    recipe = load_recipe(args.config)
    source_rows = load_source_rows(recipe.train_jsonl_fpath, recipe.train_limit)

    initial_system = recipe.target.prompt_config.system
    assert initial_system is not None  # Validated by SystemPromptTargetConfig.
    candidate = SystemPromptCandidate.from_system(
        system=initial_system,
    )
    evaluator = build_system_prompt_evaluator(recipe, source_rows)

    rollouts = await evaluator.evaluate(candidate)
    print(
        json.dumps(
            {
                "recipe": recipe.model_dump(mode="json"),
                "candidate": candidate.model_dump(mode="json"),
                "rollouts": rollouts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
