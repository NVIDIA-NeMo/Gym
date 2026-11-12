# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A gymnasium environment grades in /step and has no verifier: its trial is observed through the reward record
gymnasium_agent saves on the row (``result.reward_info.reward``), and only while that record equals the row reward."""

import ast
import copy

import pytest

from nemo_gym.orchestration.collect_evidence import COLLECT_SCRIPT
from tests.unit_tests.orchestration.test_tool_trial_coverage import fixture
from tests.unit_tests.orchestration.test_verifier_recorded_trial_coverage import _gate


def _collector_functions():
    """The collector's own definitions (imports and functions, not its main code), as it runs on the host."""
    tree = ast.parse(COLLECT_SCRIPT)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<collect_evidence>", "exec"), namespace)
    return namespace


grading_facts = _collector_functions()["grading_facts"]


def _gymnasium_run(*, saved_record, mutate=None):
    """The healthy fixture as a gymnasium environment run saves it: no verifier invocation, and the grading facts the
    collector projects from the row (with ``result`` as gymnasium_agent writes it, or without it)."""
    collected = fixture("repeats")
    for row in collected.results:
        row["ng_trial_observations"]["verifier_invocations"] = []
        source = {"reward": row["reward"]}
        if saved_record:
            source["result"] = {"reward_info": {"reward": row["reward"]}}
        if mutate is not None:
            mutate(source)
        row["_ng_collected_grading_facts"] = grading_facts(source)
    return collected


def test_a_gymnasium_trial_without_a_saved_reward_record_is_unobserved():
    passed, reason, counts, covered = _gate(_gymnasium_run(saved_record=False))
    assert covered is False and passed is None, reason
    assert counts["unobserved"] == 2


def test_a_gymnasium_trial_with_gymnasium_agents_reward_record_is_observed():
    passed, reason, counts, covered = _gate(_gymnasium_run(saved_record=True))
    assert covered is True and passed is True, reason
    assert counts["unobserved"] == 0


@pytest.mark.parametrize("recorded", [0.5, -1.0, 2.0])
def test_a_reward_record_that_disagrees_with_the_row_reward_does_not_observe_the_trial(recorded):
    def disagree(source):
        source["result"] = copy.deepcopy(source["result"])
        source["result"]["reward_info"]["reward"] = source["reward"] + recorded

    passed, reason, counts, covered = _gate(_gymnasium_run(saved_record=True, mutate=disagree))
    assert covered is False and passed is None, reason
    assert counts["unobserved"] == 2
