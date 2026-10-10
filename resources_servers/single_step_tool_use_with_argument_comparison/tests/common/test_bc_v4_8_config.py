# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavioural contract for configs/bc_v4_8.yaml.

The `search.queries` threshold sits at 0.5, the ceiling of the word-count similarity
`|intersection| / (|expected| + |actual|)`. Only identical word bags reach it, so any deleted,
added or substituted word fails, while word order still does not matter.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pytest import fixture

from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ActionComparator,
    FunctionCallAction,
    FunctionCallBatchAction,
    StepRewardCategory,
    ToolCallArgumentComparisonOverride,
    ToolCallComparatorConfig,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "bc_v4_8.yaml"

GOLDEN_QUERY = "2024 PGA Awards Outstanding Producer Limited Anthology Series Television winner"
GOLDEN_TOKENS = GOLDEN_QUERY.split()


def _raw() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _comparator_config() -> dict[str, Any]:
    return _raw()["bc_v4_8_rs"]["resources_servers"]["single_step_tool_use_with_argument_comparison"][
        "tool_call_comparator_config"
    ]


def _call(name: str, arguments: dict) -> FunctionCallAction:
    return FunctionCallAction(type="function_call", name=name, arguments=json.dumps(arguments))


def _score(comparator: ActionComparator, name: str, expected: dict, actual: dict) -> float:
    return comparator.compare_action(_call(name, expected), _call(name, actual)).reward


@fixture
def v48() -> ActionComparator:
    return ActionComparator(config=ToolCallComparatorConfig(**_comparator_config()))


def test_every_config_key_is_a_known_option() -> None:
    # Pydantic ignores unknown keys, so a misspelled option would silently fall back to its default.
    config = _comparator_config()
    assert set(config) <= set(ToolCallComparatorConfig.model_fields)
    for tool_overrides in config["argument_comparison_overrides"].values():
        for override in tool_overrides.values():
            assert set(override) <= set(ToolCallArgumentComparisonOverride.model_fields)


def test_wiring() -> None:
    raw = _raw()
    assert set(raw) == {"bc_v4_8_rs", "bc_v4_8_agent", "bc_v4_8_environment_server"}
    agent = raw["bc_v4_8_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert agent["resources_server"]["name"] == "bc_v4_8_rs"
    assert agent["model_server"]["name"] == "policy_model"
    environment = raw["bc_v4_8_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["agent_server"]["name"] == "bc_v4_8_agent"


def test_threshold_sits_at_the_metric_ceiling() -> None:
    queries = _comparator_config()["argument_comparison_overrides"]["search"]["queries"]
    assert queries["word_count_similarity_threshold"] == 0.5


def test_exact_golden_scores_one(v48: ActionComparator) -> None:
    arguments = {"queries": [GOLDEN_QUERY]}
    assert _score(v48, "search", arguments, arguments) == 1.0


def test_any_deletion_fails(v48: ActionComparator) -> None:
    expected = {"queries": [GOLDEN_QUERY]}
    assert _score(v48, "search", expected, {"queries": [" ".join(GOLDEN_TOKENS[:9])]}) == 0.0
    assert _score(v48, "search", expected, {"queries": [" ".join(GOLDEN_TOKENS[:-1])]}) == 0.0


def test_any_addition_fails(v48: ActionComparator) -> None:
    assert _score(v48, "search", {"queries": [GOLDEN_QUERY]}, {"queries": [GOLDEN_QUERY + " extra"]}) == 0.0


def test_wrong_entity_substitution_fails(v48: ActionComparator) -> None:
    # One token of ten replaced gives sim = 9/20 = 0.45, which a 0.45 threshold accepts.
    substituted = " ".join(["2025"] + GOLDEN_TOKENS[1:])
    assert _score(v48, "search", {"queries": [GOLDEN_QUERY]}, {"queries": [substituted]}) == 0.0


def test_word_order_does_not_matter(v48: ActionComparator) -> None:
    reordered = " ".join(reversed(GOLDEN_TOKENS))
    assert _score(v48, "search", {"queries": [GOLDEN_QUERY]}, {"queries": [reordered]}) == 1.0


def test_bash_command_scores_on_tool_name_alone(v48: ActionComparator) -> None:
    assert _score(v48, "bash_command", {"command": "ls -la"}, {"command": "pwd"}) == 1.0


def test_update_progress_scores_on_tool_name_alone(v48: ActionComparator) -> None:
    assert _score(v48, "update_progress", {"board": "a"}, {"board": "totally different"}) == 1.0


def test_browse_urls_are_exact(v48: ActionComparator) -> None:
    expected = {"urls": ["https://example.com/a"]}
    assert _score(v48, "browse", expected, expected) == 1.0
    assert _score(v48, "browse", expected, {"urls": ["https://example.com/b"]}) == 0.0


def test_browse_goal_is_not_compared(v48: ActionComparator) -> None:
    urls = ["https://example.com/a"]
    expected = {"urls": urls, "goal": "find the founding year"}
    assert _score(v48, "browse", expected, {"urls": urls, "goal": "something else entirely"}) == 1.0
    # `goal` is optional in the browse tool schema, so a call that omits it is still valid.
    assert _score(v48, "browse", expected, {"urls": urls}) == 1.0


def test_several_calls_for_one_expected_call_score_zero(v48: ActionComparator) -> None:
    expected = _call("search", {"queries": [GOLDEN_QUERY]})
    actual = FunctionCallBatchAction(type="function_call_batch", calls=[expected, expected])
    result = v48.compare_action(expected, actual)
    assert result.reward == 0.0
    assert result.category == StepRewardCategory.FUNCTION_CALL_BATCH_LENGTH_DIFFERENT
