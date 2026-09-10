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
"""Behavioural contract for bc_v4_6.

bc_v4_6 differs from bc_v4_5 in EXACTLY ONE value: the ``search.queries``
override moves ``word_count_similarity_threshold`` 0.30 -> 0.40. It copies
venkats' 2026-08-15 change to his own ``search_pivot`` config, measured in this
directory's notes.

WHY THIS IS A RECALL CHANGE, NOT A SIMILARITY CHANGE
----------------------------------------------------
``sim = |intersection| / (|expected| + |actual|)``, so ``sim == F1 / 2`` and the
ceiling is 0.5. For a model query that is a pure SUBSET of the golden (precision
1.0, recall r)::

    sim = r / (1 + r)   >= t   <=>   r >= t / (1 - t)

    t = 0.30  ->  r >= 0.4286   (delete up to 57% of the golden's words)
    t = 0.40  ->  r >= 0.6667   (delete up to 33%)

So the single knob acts as a recall floor of 0.667, stricter than the config's
own ``word_count_min_recall: 0.5``, which it therefore supersedes for subsets.
Padding is barely touched: adding j junk words gives ``sim = L / (2L + j)``,
needing ``j <= L/3`` at 0.40 versus ``j <= 4L/3`` at 0.30 -- and this config
already caps padding much harder via ``word_count_max_unmatched_actual_words:
4`` and ``word_count_max_actual_to_expected_ratio: 1.25``.

Everything else -- the list gates, ``browse``, and the name-only scoring of
``bash_command`` / ``update_progress`` -- is unchanged from bc_v4_5.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pytest import fixture

from nemo_gym.openai_utils import NeMoGymResponseFunctionToolCall
from resources_servers.single_step_tool_use_with_argument_comparison.common.verification_utils import (
    ExpectedFunctionCall,
    ToolCallComparator,
    ToolCallComparatorConfig,
)


CONFIGS_DIR = Path(__file__).parents[2] / "configs"

# A 10-token golden query. Ten tokens makes the recall arithmetic exact: keeping
# the first k tokens gives recall k/10, so the 0.667 boundary falls cleanly
# between k=6 (fails at 0.40) and k=7 (passes).
GOLDEN_QUERY = "2024 PGA Awards Outstanding Producer Limited Anthology Series Television winner"
GOLDEN_TOKENS = GOLDEN_QUERY.split()


def _raw(config_filename: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / config_filename).read_text())


def _comparator_config(config_filename: str, root_key: str) -> dict[str, Any]:
    return _raw(config_filename)[root_key]["resources_servers"][
        "single_step_tool_use_with_argument_comparison"
    ]["tool_call_comparator_config"]


def _comparator(config_filename: str, root_key: str) -> ToolCallComparator:
    return ToolCallComparator(config=ToolCallComparatorConfig(**_comparator_config(config_filename, root_key)))


def _score(comparator: ToolCallComparator, name: str, expected: dict[str, Any], actual: dict[str, Any]) -> float:
    reward, _ = comparator.compare_tool_call(
        ExpectedFunctionCall(type="function_call", name=name, arguments=json.dumps(expected)),
        NeMoGymResponseFunctionToolCall(call_id="c", name=name, arguments=json.dumps(actual)),
    )
    return reward


def _first_k(k: int) -> str:
    return " ".join(GOLDEN_TOKENS[:k])


@fixture
def v45() -> ToolCallComparator:
    return _comparator("bc_v4_5.yaml", "bc_v4_5_rs")


@fixture
def v46() -> ToolCallComparator:
    return _comparator("bc_v4_6.yaml", "bc_v4_6_rs")


# ---------------------------------------------------------------------------
# Structural: the config differs from bc_v4_5 in exactly one value.
# ---------------------------------------------------------------------------


def test_comparator_config_differs_only_in_search_queries_similarity_threshold() -> None:
    """The whole point of the ablation: one number, nothing else."""
    v45_cfg = _comparator_config("bc_v4_5.yaml", "bc_v4_5_rs")
    v46_cfg = _comparator_config("bc_v4_6.yaml", "bc_v4_6_rs")

    assert v45_cfg["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.3
    assert v46_cfg["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.4

    # Patch the one value back and the two blocks must be indistinguishable.
    patched = json.loads(json.dumps(v46_cfg))
    patched["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] = 0.3
    assert patched == v45_cfg


def test_agent_wiring_points_at_the_v4_6_resources_server() -> None:
    """A row whose agent_ref is bc_v4_6_agent must reach the stricter server."""
    raw = _raw("bc_v4_6.yaml")
    assert set(raw) == {"bc_v4_6_rs", "bc_v4_6_agent"}
    agent = raw["bc_v4_6_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert agent["resources_server"]["name"] == "bc_v4_6_rs"
    assert agent["model_server"]["name"] == "policy_model"


def test_agent_dataset_block_matches_bc_v4_5() -> None:
    """Training reads data.train.data_path, but keep the block identical anyway."""
    v45_agent = _raw("bc_v4_5.yaml")["bc_v4_5_agent"]["responses_api_agents"]["tool_simulation_agent"]
    v46_agent = _raw("bc_v4_6.yaml")["bc_v4_6_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert v46_agent["datasets"] == v45_agent["datasets"]
    assert v46_agent["entrypoint"] == v45_agent["entrypoint"]


# ---------------------------------------------------------------------------
# Behavioural: what the change actually kills.
# ---------------------------------------------------------------------------


def test_exact_golden_still_scores_one(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """The ceiling must not move. sim is exactly 0.5 here."""
    args = {"queries": [GOLDEN_QUERY]}
    assert _score(v45, "search", args, args) == 1.0
    assert _score(v46, "search", args, args) == 1.0


def test_truncation_to_60_percent_is_killed(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """THE headline change. recall 0.6 -> sim 6/16 = 0.375, under 0.40."""
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [_first_k(6)]}
    assert _score(v45, "search", expected, actual) == 1.0
    assert _score(v46, "search", expected, actual) == 0.0


def test_truncation_to_half_is_killed(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """recall 0.5 sits exactly on bc_v4_5's word_count_min_recall floor."""
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [_first_k(5)]}
    assert _score(v45, "search", expected, actual) == 1.0
    assert _score(v46, "search", expected, actual) == 0.0


def test_recall_boundary_is_two_thirds(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """r >= t/(1-t) = 0.667, so k=7 survives and k=6 does not."""
    expected = {"queries": [GOLDEN_QUERY]}
    assert _score(v46, "search", expected, {"queries": [_first_k(7)]}) == 1.0
    assert _score(v46, "search", expected, {"queries": [_first_k(6)]}) == 0.0
    # bc_v4_5 accepts both, which is what makes this the discriminating case.
    assert _score(v45, "search", expected, {"queries": [_first_k(7)]}) == 1.0
    assert _score(v45, "search", expected, {"queries": [_first_k(6)]}) == 1.0


def test_dropping_one_token_still_passes(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """Benign variation must survive. recall 0.9 -> sim 9/19 = 0.474."""
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [" ".join(GOLDEN_TOKENS[:-1])]}
    assert _score(v45, "search", expected, actual) == 1.0
    assert _score(v46, "search", expected, actual) == 1.0


def test_wrong_year_still_scores_one_under_both(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """Documented blind spot: this change does NOT close the year hack.

    9 of 10 tokens shared -> sim 9/20 = 0.45, above both thresholds. Recorded so
    a future reader does not mistake bc_v4_6 for a fix to entity errors.
    """
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [GOLDEN_QUERY.replace("2024", "2025")]}
    assert _score(v45, "search", expected, actual) == 1.0
    assert _score(v46, "search", expected, actual) == 1.0


def test_token_order_still_ignored(v46: ToolCallComparator) -> None:
    """Word bags, not sequences. Unchanged by the threshold."""
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [" ".join(reversed(GOLDEN_TOKENS))]}
    assert _score(v46, "search", expected, actual) == 1.0


def test_padding_behaviour_is_unchanged(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    """Two junk words: sim 10/22 = 0.455, above both. The ratio and
    max_unmatched_actual_words caps already govern padding, not the threshold."""
    expected = {"queries": [GOLDEN_QUERY]}
    actual = {"queries": [GOLDEN_QUERY + " additional details"]}
    assert _score(v45, "search", expected, actual) == 1.0
    assert _score(v46, "search", expected, actual) == 1.0


# ---------------------------------------------------------------------------
# The other three tools must be untouched.
# ---------------------------------------------------------------------------


def test_bash_command_still_scores_on_tool_name_alone(v46: ToolCallComparator) -> None:
    expected = {"duration": 5, "keystrokes": "grep -i -A 5 'step 12' pages/0013.txt"}
    actual = {"duration": 99, "keystrokes": "cat /etc/hostname"}
    assert _score(v46, "bash_command", expected, actual) == 1.0


def test_update_progress_still_scores_on_tool_name_alone(v46: ToolCallComparator) -> None:
    expected = {"board": "found the 2024 winner, verifying the anthology category"}
    actual = {"board": "totally unrelated text"}
    assert _score(v46, "update_progress", expected, actual) == 1.0


def test_browse_urls_stay_exact(v45: ToolCallComparator, v46: ToolCallComparator) -> None:
    expected = {"urls": ["https://example.org/a"], "goal": "find the producer"}
    same = {"urls": ["https://example.org/a"], "goal": "a completely different goal"}
    other = {"urls": ["https://example.org/b"], "goal": "find the producer"}
    for c in (v45, v46):
        # goal is skipped by compare_tool_call_arguments; the URL is what counts.
        assert _score(c, "browse", expected, same) == 1.0
        assert _score(c, "browse", expected, other) == 0.0


def test_wrong_tool_name_still_scores_zero(v46: ToolCallComparator) -> None:
    reward, _ = v46.compare_tool_call(
        ExpectedFunctionCall(type="function_call", name="search", arguments=json.dumps({"queries": [GOLDEN_QUERY]})),
        NeMoGymResponseFunctionToolCall(call_id="c", name="browse", arguments=json.dumps({"urls": ["x"]})),
    )
    assert reward == 0.0
