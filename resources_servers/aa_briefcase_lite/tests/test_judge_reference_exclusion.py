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
"""Per-reference judge exclusion and skipped-reference records."""

from __future__ import annotations

import pytest

from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseReference,
    _judges_for_reference,
    _skipped_reference,
)
from resources_servers.gdpval.judge_panel import ResolvedJudge


PANEL = ["claude-opus-4.8", "gemini-3.1-pro", "gpt-5.5"]


def _panel() -> list[ResolvedJudge]:
    return [ResolvedJudge(name=name, base_url="http://j", model=name) for name in PANEL]


def _kept(reference: AABriefcaseReference) -> list[str]:
    return [judge.name for judge in _judges_for_reference(reference, _panel())]


@pytest.mark.parametrize("excluded", PANEL)
def test_an_excluded_judge_is_dropped(excluded: str) -> None:
    assert _kept(AABriefcaseReference(exclude_judges=[excluded])) == [name for name in PANEL if name != excluded]


def test_no_exclusion_keeps_the_whole_panel() -> None:
    assert _kept(AABriefcaseReference()) == PANEL


def test_an_unknown_judge_name_is_ignored() -> None:
    assert _kept(AABriefcaseReference(exclude_judges=["not-on-the-panel"])) == PANEL


def test_excluding_every_judge_leaves_an_empty_panel() -> None:
    assert _kept(AABriefcaseReference(exclude_judges=PANEL)) == []


def test_a_skipped_reference_has_a_reason_and_no_vote_counts() -> None:
    record = _skipped_reference("o3", AABriefcaseReference(exclude_judges=["gpt-5.5"]), "no_submission_for_task")

    assert record == {"reference_id": "o3", "skipped_reason": "no_submission_for_task", "excluded_judges": ["gpt-5.5"]}
