# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Scenario orchestration for OpenAI-compatible agent + user simulator runs."""
from __future__ import annotations

import logging
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Union

from tool_sandbox.common.execution_context import (
    RoleType,
    ScenarioCategories,
    get_current_context,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent, OpenAIRoleConfig
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser
from tool_sandbox.scenarios import named_scenarios

LOGGER = logging.getLogger(__name__)

# Smoke-test subset — same names upstream used with the ``--test_mode`` flag.
TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
]


def resolve_scenarios(
    desired_scenario_names: Optional[list[str]],
    preferred_tool_backend: ToolBackend,
) -> dict[str, Scenario]:
    """Return either all scenarios or just the ones whose names are listed."""
    if desired_scenario_names is None:
        return named_scenarios(preferred_tool_backend=preferred_tool_backend)

    name_to_scenario = {
        name: scenario
        for name, scenario in named_scenarios(
            preferred_tool_backend=preferred_tool_backend
        ).items()
        if name in desired_scenario_names
    }
    missing = set(desired_scenario_names) - set(name_to_scenario.keys())
    if missing:
        raise KeyError(f"The following desired scenarios do not exist: {sorted(missing)}")
    return name_to_scenario


async def run_scenario(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_config: OpenAIRoleConfig,
    user_config: OpenAIRoleConfig,
    output_directory: Path,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Play and evaluate a single scenario.

    Each invocation runs inside its own :class:`contextvars.Context` copy, so
    the per-scenario ``ExecutionContext`` doesn't leak between concurrent
    asyncio tasks.
    """
    name, scenario = name_and_scenario
    LOGGER.info("(%d/%d) [%s] start", index, total, name)
    roles = {
        RoleType.USER: OpenAIAPIUser(user_config),
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        RoleType.AGENT: OpenAIAPIAgent(agent_config),
    }
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        result = await scenario.play_and_evaluate(
            roles=roles,
            output_directory=output_directory,
            scenario_name=name,
            progress_desc=f"({index}/{total}) {name}",
        )
        LOGGER.info(
            "(%d/%d) [%s] done sim=%.4f turns=%d",
            index,
            total,
            name,
            result.evaluation_result.similarity,
            result.evaluation_result.turn_count,
        )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": None,
            "exception_type": None,
            "milestone_similarity": result.evaluation_result.milestone_similarity,
            "minefield_similarity": result.evaluation_result.minefield_similarity,
            "similarity": result.evaluation_result.similarity,
            "turn_count": result.evaluation_result.turn_count,
            "milestone_mapping": result.evaluation_result.milestone_mapping,
            "minefield_mapping": result.evaluation_result.minefield_mapping,
        }
    except Exception as exc:
        # Per-scenario failures are expected (model hallucinates a tool name,
        # emits malformed JSON args, non-retriable API error, etc.). The run
        # continues with similarity=0 for this scenario, and the full traceback
        # is preserved in result_summary.json — no need to spam stderr.
        LOGGER.warning(
            "(%d/%d) [%s] failed: %s: %s",
            index,
            total,
            name,
            type(exc).__name__,
            exc,
        )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": traceback.format_exc(),
            "exception_type": type(exc).__name__,
            "milestone_similarity": 0,
            "minefield_similarity": 0,
            "similarity": 0,
            "turn_count": scenario.max_messages,
            "milestone_mapping": {},
            "minefield_mapping": {},
        }
    finally:
        for role in roles.values():
            await role.teardown()


# Categories on which the augmented (distractor / scrambled) scenarios are stacked.
# When a scenario carries one of these tags it should not also count toward
# THREE_DISTRACTION_TOOLS — otherwise that bucket double-counts.
_AUGMENTATION_TAGS: frozenset[ScenarioCategories] = frozenset(
    {
        ScenarioCategories.TOOL_NAME_SCRAMBLED,
        ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
        ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
        ScenarioCategories.ARG_TYPE_SCRAMBLED,
        ScenarioCategories.ARG_NAME_SCRAMBLED,
    }
)


def get_category_summary(
    result_summary: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Bucket per-scenario results into per-category similarity / turn_count lists."""
    category_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for current in result_summary:
        categories = set(current["categories"])
        for category in current["categories"]:
            if (
                category == ScenarioCategories.THREE_DISTRACTION_TOOLS
                and categories & _AUGMENTATION_TAGS
            ):
                continue
            category_summary[category]["similarity"].append(current["similarity"])
            category_summary[category]["turn_count"].append(current["turn_count"])
        category_summary["ALL_CATEGORIES"]["similarity"].append(current["similarity"])
        category_summary["ALL_CATEGORIES"]["turn_count"].append(current["turn_count"])
    return category_summary


def get_category_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Per-category scenario counts, with the same augmentation-bucket rule as scoring."""
    counter: Counter[Union[ScenarioCategories, str]] = Counter()
    for scenario in name_to_scenario.values():
        categories = set(scenario.categories)
        for category in scenario.categories:
            if (
                category == ScenarioCategories.THREE_DISTRACTION_TOOLS
                and categories & _AUGMENTATION_TAGS
            ):
                continue
            counter[category] += 1
        counter["ALL_CATEGORIES"] += 1
    return counter


def get_necessary_tool_name_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count scenarios per necessary tool name, weighted by augmentation variants."""
    counter: Counter[Union[ScenarioCategories, str]] = Counter(
        {tool_name: 0 for tool_name in get_current_context().get_available_tools(
            scrambling_allowed=False
        )}
    )
    augmentation_categories: set[Union[ScenarioCategories, str]] = set()
    for scenario in name_to_scenario.values():
        if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
            assert scenario.starting_context.tool_allow_list is not None
            for necessary_tool in scenario.starting_context.tool_allow_list:
                counter[necessary_tool] += 1
        augmentation_categories |= (
            {
                ScenarioCategories.NO_DISTRACTION_TOOLS,
                ScenarioCategories.THREE_DISTRACTION_TOOLS,
                ScenarioCategories.TEN_DISTRACTION_TOOLS,
                ScenarioCategories.ALL_TOOLS_AVAILABLE,
            }
            | _AUGMENTATION_TAGS
        ) & set(scenario.categories)
    for tool_name in counter:
        counter[tool_name] *= len(augmentation_categories)
    return counter
