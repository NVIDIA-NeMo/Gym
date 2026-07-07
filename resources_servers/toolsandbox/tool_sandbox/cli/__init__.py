# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Entry point: drive the ToolSandbox scenarios from an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.logging import logging_redirect_tqdm

from tool_sandbox.cli.utils import (
    TEST_SCENARIO_NAMES,
    get_category_summary,
    get_category_to_scenario_count,
    get_necessary_tool_name_to_scenario_count,
    resolve_scenarios,
    run_scenario,
)
from tool_sandbox.common.execution_context import ScenarioCategories
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.openai_api_agent import OpenAIRoleConfig

LOGGER = logging.getLogger(__name__)


def _write_result_summary(
    result_summary: list[dict],
    category_summary: dict[str, dict[str, list[float]]],
    output_directory: Path,
) -> None:
    """Write ``result_summary.json`` matching the upstream schema."""
    payload = {
        "per_scenario_results": result_summary,
        "category_aggregated_results": {
            category: {k: sum(v) / len(v) for k, v in aggregation.items()}
            for category, aggregation in category_summary.items()
        },
    }
    out = output_directory / "result_summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4, ensure_ascii=False)
    LOGGER.info("Wrote %s", out)


def _resolve_api_key(cli_value: Optional[str], env_name: str) -> str:
    if cli_value:
        return cli_value
    return os.environ.get(env_name, "").strip()


async def _bounded_run(
    sem: asyncio.Semaphore,
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_config: OpenAIRoleConfig,
    user_config: OpenAIRoleConfig,
    output_directory: Path,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Run one scenario under the concurrency cap.

    asyncio.gather already wraps each coroutine in a Task with its own
    ContextVar copy, so per-scenario ``set_current_context`` mutations stay
    local to this task.
    """
    async with sem:
        return await run_scenario(
            name_and_scenario,
            agent_config=agent_config,
            user_config=user_config,
            output_directory=output_directory,
            index=index,
            total=total,
        )


async def _run_sandbox(
    *,
    agent_config: OpenAIRoleConfig,
    user_config: OpenAIRoleConfig,
    name_to_scenario: dict[str, Scenario],
    parallel: int,
    output_base_dir: Path,
) -> None:
    """Play every scenario concurrently, aggregate, and write result_summary.json."""
    pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
    pl.Config.set_tbl_formatting("ASCII_FULL")

    timestamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_directory = (
        Path(output_base_dir)
        / f"agent_{agent_config.model}_user_{user_config.model}_{timestamp}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Storing outputs to %s", output_directory)
    LOGGER.info(
        "Agent: model=%s base_url=%s temperature=%s top_p=%s max_tokens=%s",
        agent_config.model,
        agent_config.base_url,
        agent_config.temperature,
        agent_config.top_p,
        agent_config.max_tokens,
    )
    LOGGER.info(
        "User:  model=%s base_url=%s temperature=%s top_p=%s max_tokens=%s",
        user_config.model,
        user_config.base_url,
        user_config.temperature,
        user_config.top_p,
        user_config.max_tokens,
    )

    category_counter: Counter[Union[ScenarioCategories, str]] = (
        get_category_to_scenario_count(name_to_scenario)
    )
    LOGGER.debug(
        "Number of test cases per category: %s",
        json.dumps(
            {str(k): v for k, v in category_counter.most_common(len(category_counter))},
            indent=4,
            ensure_ascii=False,
        ),
    )
    necessary_tool_counter: Counter[Union[ScenarioCategories, str]] = (
        get_necessary_tool_name_to_scenario_count(name_to_scenario)
    )
    LOGGER.debug(
        "Number of test cases per necessary tool name: %s",
        json.dumps(
            {
                str(k): v
                for k, v in necessary_tool_counter.most_common(
                    len(necessary_tool_counter)
                )
            },
            indent=4,
            ensure_ascii=False,
        ),
    )

    name_and_scenario_list = list(name_to_scenario.items())
    random.shuffle(name_and_scenario_list)
    num_scenarios = len(name_and_scenario_list)
    LOGGER.debug("Running %d scenarios with parallel=%d", num_scenarios, parallel)

    sem = asyncio.Semaphore(max(1, parallel))
    coros = [
        _bounded_run(
            sem,
            item,
            agent_config=agent_config,
            user_config=user_config,
            output_directory=output_directory,
            index=i,
            total=num_scenarios,
        )
        for i, item in enumerate(name_and_scenario_list, start=1)
    ]
    result_summary: list[dict[str, Any]] = await tqdm_asyncio.gather(
        *coros, desc="Scenarios", mininterval=5.0
    )

    category_summary = get_category_summary(result_summary)
    overall = category_summary.get("ALL_CATEGORIES", {})
    if overall.get("similarity"):
        mean_sim = sum(overall["similarity"]) / len(overall["similarity"])
        LOGGER.info("ALL_CATEGORIES mean similarity: %.4f over %d scenarios", mean_sim, num_scenarios)
    _write_result_summary(
        result_summary=result_summary,
        category_summary=category_summary,
        output_directory=output_directory,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--agent-base-url", required=True, help="OpenAI-compatible base URL for the agent under test.")
    parser.add_argument("--agent-model", required=True, help="Agent model id sent in the ``model`` request field.")
    parser.add_argument("--agent-api-key", default=None, help="Agent API key (or set TOOLSANDBOX_AGENT_API_KEY).")
    parser.add_argument("--agent-temperature", type=float, default=None, help="Sampling temperature for the agent.")
    parser.add_argument("--agent-top-p", type=float, default=None, help="Nucleus sampling top_p for the agent.")
    parser.add_argument("--agent-max-tokens", type=int, default=None, help="Max output tokens for the agent.")
    parser.add_argument(
        "--agent-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Toggle reasoning/thinking mode for the agent. For OpenAI reasoning "
            "models (gpt-5+, o-series), sends top-level reasoning_effort=high|low; "
            "for vLLM/Qwen-style models, sends "
            "extra_body.chat_template_kwargs.enable_thinking. Omit to leave the "
            "server-side default."
        ),
    )

    parser.add_argument("--user-base-url", required=True, help="OpenAI-compatible base URL for the user simulator.")
    parser.add_argument("--user-model", required=True, help="User simulator model id.")
    parser.add_argument("--user-api-key", default=None, help="User simulator API key (or set TOOLSANDBOX_USER_API_KEY).")
    parser.add_argument("--user-temperature", type=float, default=None, help="Sampling temperature for the user simulator.")
    parser.add_argument("--user-top-p", type=float, default=None, help="Nucleus sampling top_p for the user simulator.")
    parser.add_argument("--user-max-tokens", type=int, default=None, help="Max output tokens for the user simulator.")
    parser.add_argument(
        "--user-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Toggle reasoning/thinking mode for the user simulator. For OpenAI "
            "reasoning models (gpt-5+, o-series), sends top-level "
            "reasoning_effort=high|low; for vLLM/Qwen-style models, sends "
            "extra_body.chat_template_kwargs.enable_thinking. Omit to leave the "
            "server-side default."
        ),
    )

    parser.add_argument(
        "--preferred-tool-backend",
        default="DEFAULT",
        choices=[str(t) for t in ToolBackend],
        help="Preferred tool backend on name conflict.",
    )
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "-t", "--test-mode", action="store_true",
        help="Run only the small smoke-test subset of scenarios.",
    )
    scenario_group.add_argument(
        "-s", "--scenarios", nargs="*", default=None,
        help="Specific scenario names to run (default: all).",
    )
    parser.add_argument(
        "-p", "--parallel", type=int, default=16,
        help="Max concurrent scenarios (asyncio semaphore size).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("data"),
        help="Output base directory; a timestamped run subdir is created inside.",
    )
    return parser


def _list_scenarios(test_mode: bool) -> None:
    """Print a JSON array of scenario names (smoke subset if ``test_mode``)."""
    if test_mode:
        names = list(TEST_SCENARIO_NAMES)
    else:
        names = sorted(
            resolve_scenarios(
                desired_scenario_names=None,
                preferred_tool_backend=ToolBackend.DEFAULT,
            ).keys()
        )
    json.dump(names, sys.stdout)
    sys.stdout.write("\n")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    random.seed(42)
    if "--list-scenarios" in sys.argv:
        _list_scenarios(test_mode=("--test-mode" in sys.argv or "-t" in sys.argv))
        return 0
    args = _build_parser().parse_args()

    agent_config = OpenAIRoleConfig(
        base_url=args.agent_base_url,
        model=args.agent_model,
        api_key=_resolve_api_key(args.agent_api_key, "TOOLSANDBOX_AGENT_API_KEY"),
        temperature=args.agent_temperature,
        top_p=args.agent_top_p,
        max_tokens=args.agent_max_tokens,
        enable_thinking=args.agent_enable_thinking,
    )
    user_config = OpenAIRoleConfig(
        base_url=args.user_base_url,
        model=args.user_model,
        api_key=_resolve_api_key(args.user_api_key, "TOOLSANDBOX_USER_API_KEY"),
        temperature=args.user_temperature,
        top_p=args.user_top_p,
        max_tokens=args.user_max_tokens,
        enable_thinking=args.user_enable_thinking,
    )

    scenario_names = TEST_SCENARIO_NAMES if args.test_mode else args.scenarios
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=scenario_names,
        preferred_tool_backend=args.preferred_tool_backend,
    )
    with logging_redirect_tqdm():
        asyncio.run(
            _run_sandbox(
                agent_config=agent_config,
                user_config=user_config,
                name_to_scenario=name_to_scenario,
                parallel=args.parallel,
                output_base_dir=args.output_dir,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
