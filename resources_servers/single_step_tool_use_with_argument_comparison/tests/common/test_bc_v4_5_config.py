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
"""Behavioural contract for the bc_v4_5 BrowseComp verifier config.

bc_v4_5 differs from bc_v4_tight_search in exactly two ways:

1. ``list_f1_relaxed_min_expected_len`` moves 3 -> 4 on ``search.queries``, so the
   relaxed miss/extra branch only fires when the golden holds >= 4 queries.
2. ``bash_command`` and ``update_progress`` score 1.0 on a tool-name match alone.
   Their arguments are filtered away before comparison, mirroring how any chat
   message already scores 1.0 in ``app.py``.

``search`` and ``browse`` behaviour is unchanged from v4.
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


def _comparator(config_filename: str, root_key: str) -> ToolCallComparator:
    raw = yaml.safe_load((CONFIGS_DIR / config_filename).read_text())
    comparator_config = raw[root_key]["resources_servers"]["single_step_tool_use_with_argument_comparison"][
        "tool_call_comparator_config"
    ]
    return ToolCallComparator(config=ToolCallComparatorConfig(**comparator_config))


def _score(comparator: ToolCallComparator, name: str, expected: dict[str, Any], actual: dict[str, Any]) -> float:
    reward, _ = comparator.compare_tool_call(
        ExpectedFunctionCall(type="function_call", name=name, arguments=json.dumps(expected)),
        NeMoGymResponseFunctionToolCall(call_id="c", name=name, arguments=json.dumps(actual)),
    )
    return reward


@fixture
def v4() -> ToolCallComparator:
    return _comparator("bc_v4_tight_search.yaml", "bc_v4_tight_rs")


@fixture
def v45() -> ToolCallComparator:
    return _comparator("bc_v4_5.yaml", "bc_v4_5_rs")


class TestRelaxedSplitMovesToFour:
    """Change 1: the relaxed branch fires at >= 4 goldens instead of >= 3."""

    THREE = ["alpha bravo charlie delta", "echo foxtrot golf hotel", "india juliet kilo lima"]
    FOUR = THREE + ["mike november oscar papa"]
    EXTRA = "quebec romeo sierra tango"

    def test_three_goldens_plus_one_extra_is_relaxed_under_v4(self, v4: ToolCallComparator) -> None:
        expected = {"queries": self.THREE}
        actual = {"queries": self.THREE + [self.EXTRA]}
        assert _score(v4, "search", expected, actual) == 1.0

    def test_three_goldens_plus_one_extra_is_strict_under_v45(self, v45: ToolCallComparator) -> None:
        expected = {"queries": self.THREE}
        actual = {"queries": self.THREE + [self.EXTRA]}
        assert _score(v45, "search", expected, actual) == 0.0

    def test_four_goldens_plus_one_extra_stays_relaxed_under_v45(self, v45: ToolCallComparator) -> None:
        expected = {"queries": self.FOUR}
        actual = {"queries": self.FOUR + [self.EXTRA]}
        assert _score(v45, "search", expected, actual) == 1.0

    def test_exact_match_holds_at_three_goldens_under_v45(self, v45: ToolCallComparator) -> None:
        expected = {"queries": self.THREE}
        assert _score(v45, "search", expected, {"queries": list(self.THREE)}) == 1.0


class TestNameOnlyToolsScoreOne:
    """Change 2: bash_command and update_progress ignore argument content."""

    def test_bash_command_ignores_keystrokes_and_duration(self, v45: ToolCallComparator) -> None:
        expected = {"keystrokes": "grep -i stevens pages/0011_browse.txt | head -20", "duration": 5}
        actual = {"keystrokes": "wc -l notes.txt", "duration": 999}
        assert _score(v45, "bash_command", expected, actual) == 1.0

    def test_update_progress_ignores_board(self, v45: ToolCallComparator) -> None:
        expected = {"board": "Step 1 complete. Next: verify the 2024 filing."}
        actual = {"board": "totally unrelated text"}
        assert _score(v45, "update_progress", expected, actual) == 1.0

    def test_v4_still_graded_bash_command_content(self, v4: ToolCallComparator) -> None:
        expected = {"keystrokes": "grep -i stevens pages/0011_browse.txt | head -20", "duration": 5}
        actual = {"keystrokes": "wc -l notes.txt", "duration": 999}
        assert _score(v4, "bash_command", expected, actual) == 0.0

    def test_wrong_tool_name_still_scores_zero(self, v45: ToolCallComparator) -> None:
        reward, _ = v45.compare_tool_call(
            ExpectedFunctionCall(
                type="function_call", name="bash_command", arguments=json.dumps({"keystrokes": "ls"})
            ),
            NeMoGymResponseFunctionToolCall(call_id="c", name="update_progress", arguments=json.dumps({"board": "x"})),
        )
        assert reward == 0.0


class TestSearchAndBrowseUnchanged:
    """search and browse must behave identically under v4 and v4_5."""

    def test_browse_url_must_still_match(self, v4: ToolCallComparator, v45: ToolCallComparator) -> None:
        expected = {"urls": ["https://example.com/a"], "goal": "find the 2024 winner"}
        same = {"urls": ["https://example.com/a"], "goal": "a completely different goal"}
        other = {"urls": ["https://example.com/b"], "goal": "find the 2024 winner"}
        for comparator in (v4, v45):
            assert _score(comparator, "browse", expected, same) == 1.0
            assert _score(comparator, "browse", expected, other) == 0.0

    def test_search_padding_still_rejected(self, v4: ToolCallComparator, v45: ToolCallComparator) -> None:
        golden = ["alpha bravo charlie delta", "echo foxtrot golf hotel"]
        padded = ["alpha bravo charlie delta one two three four five", "echo foxtrot golf hotel"]
        for comparator in (v4, v45):
            assert _score(comparator, "search", {"queries": golden}, {"queries": list(golden)}) == 1.0
            assert _score(comparator, "search", {"queries": golden}, {"queries": padded}) == 0.0


class TestResolvedConfigDiff:
    """Guard against drift: v4_5 must differ from v4 in only the intended fields."""

    def test_only_relaxed_len_differs_on_search_override(
        self, v4: ToolCallComparator, v45: ToolCallComparator
    ) -> None:
        a = v4._argument_comparison_override("search", "queries")
        b = v45._argument_comparison_override("search", "queries")
        assert a is not None and b is not None
        differing = {f for f in type(a).model_fields if getattr(a, f) != getattr(b, f)}
        assert differing == {"list_f1_relaxed_min_expected_len"}
        assert a.list_f1_relaxed_min_expected_len == 3
        assert b.list_f1_relaxed_min_expected_len == 4

    def test_v45_filters_exactly_two_tools(self, v45: ToolCallComparator) -> None:
        filters = v45.config.argument_filters
        assert filters is not None
        assert set(filters) == {"bash_command", "update_progress"}
        for tool_filter in filters.values():
            assert tool_filter.included_argument_names == []

    def test_v4_has_no_argument_filters(self, v4: ToolCallComparator) -> None:
        assert v4.config.argument_filters is None
