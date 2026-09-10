# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioural contract for bc_v4_8.

bc_v4_8 differs from bc_v4_7 in EXACTLY ONE value: the ``search.queries`` override moves
``word_count_similarity_threshold`` 0.45 -> 0.50.

WHY 0.50 IS EXACT MATCH
-----------------------
``sim = |intersection| / (|expected| + |actual|)``, so ``sim == F1 / 2`` and the CEILING is
0.5, reached only when the two word bags are identical. Putting the threshold AT the
ceiling means every deviation fails::

    delete a word     -> |intersection| falls  -> sim < 0.5
    add a word        -> |actual| grows        -> sim < 0.5
    substitute a word -> both                  -> sim < 0.5

WHAT THIS FIXES
---------------
At 0.45, substituting one token of ten left ``sim = 9/20 = 0.45``, which satisfied the
``>=`` comparison -- the year / wrong-entity hack. At 0.50 the same substitution fails.
See ``test_wrong_entity_no_longer_survives``.

WHAT "EXACT" DOES NOT MEAN
--------------------------
Word bags are unordered and deduplicated, so a reordered query still scores 1.0. This is
an exact SET match, not an exact STRING match. See ``test_word_order_still_does_not_matter``.
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

GOLDEN_QUERY = "2024 PGA Awards Outstanding Producer Limited Anthology Series Television winner"
GOLDEN_TOKENS = GOLDEN_QUERY.split()


def _raw(fn: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / fn).read_text())


def _cmp_cfg(fn: str, root: str) -> dict[str, Any]:
    return _raw(fn)[root]["resources_servers"]["single_step_tool_use_with_argument_comparison"][
        "tool_call_comparator_config"
    ]


def _comparator(fn: str, root: str) -> ToolCallComparator:
    return ToolCallComparator(config=ToolCallComparatorConfig(**_cmp_cfg(fn, root)))


def _score(c: ToolCallComparator, name: str, expected: dict, actual: dict) -> float:
    reward, _ = c.compare_tool_call(
        ExpectedFunctionCall(type="function_call", name=name, arguments=json.dumps(expected)),
        NeMoGymResponseFunctionToolCall(call_id="c", name=name, arguments=json.dumps(actual)),
    )
    return reward


def _first_k(k: int) -> str:
    return " ".join(GOLDEN_TOKENS[:k])


@fixture
def v47() -> ToolCallComparator:
    return _comparator("bc_v4_7.yaml", "bc_v4_7_rs")


@fixture
def v48() -> ToolCallComparator:
    return _comparator("bc_v4_8.yaml", "bc_v4_8_rs")


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------


def test_differs_from_v4_7_only_in_search_queries_threshold() -> None:
    a = _cmp_cfg("bc_v4_7.yaml", "bc_v4_7_rs")
    b = _cmp_cfg("bc_v4_8.yaml", "bc_v4_8_rs")
    assert a["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.45
    assert b["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.5
    patched = json.loads(json.dumps(b))
    patched["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] = 0.45
    assert patched == a


def test_threshold_sits_at_the_metric_ceiling() -> None:
    """0.5 is the maximum sim can ever be; anything higher would reject everything."""
    cfg = _cmp_cfg("bc_v4_8.yaml", "bc_v4_8_rs")
    assert cfg["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.5


def test_agent_wiring() -> None:
    raw = _raw("bc_v4_8.yaml")
    assert set(raw) == {"bc_v4_8_rs", "bc_v4_8_agent"}
    agent = raw["bc_v4_8_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert agent["resources_server"]["name"] == "bc_v4_8_rs"
    assert agent["model_server"]["name"] == "policy_model"


def test_dataset_block_matches_v4_7() -> None:
    a = _raw("bc_v4_7.yaml")["bc_v4_7_agent"]["responses_api_agents"]["tool_simulation_agent"]
    b = _raw("bc_v4_8.yaml")["bc_v4_8_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert b["datasets"] == a["datasets"]


# --------------------------------------------------------------------------
# Behavioural — exact match
# --------------------------------------------------------------------------


def test_exact_golden_scores_one(v48) -> None:
    args = {"queries": [GOLDEN_QUERY]}
    assert _score(v48, "search", args, args) == 1.0


def test_any_deletion_now_fails(v47, v48) -> None:
    """v4_7 tolerated dropping up to 18% of the words; v4_8 tolerates none."""
    exp = {"queries": [GOLDEN_QUERY]}
    assert _score(v47, "search", exp, {"queries": [_first_k(9)]}) == 1.0
    assert _score(v48, "search", exp, {"queries": [_first_k(9)]}) == 0.0
    assert _score(v48, "search", exp, {"queries": [_first_k(len(GOLDEN_TOKENS) - 1)]}) == 0.0


def test_any_addition_fails(v48) -> None:
    exp = {"queries": [GOLDEN_QUERY]}
    assert _score(v48, "search", exp, {"queries": [GOLDEN_QUERY + " extra"]}) == 0.0


def test_wrong_entity_no_longer_survives(v47, v48) -> None:
    """The hack bc_v4_7 could not close: substitute one token of ten."""
    exp = {"queries": [GOLDEN_QUERY]}
    hacked = " ".join(["2025"] + GOLDEN_TOKENS[1:])
    assert _score(v47, "search", exp, {"queries": [hacked]}) == 1.0
    assert _score(v48, "search", exp, {"queries": [hacked]}) == 0.0


def test_word_order_still_does_not_matter(v48) -> None:
    """Exact SET match, not exact STRING match."""
    exp = {"queries": [GOLDEN_QUERY]}
    reordered = " ".join(reversed(GOLDEN_TOKENS))
    assert _score(v48, "search", exp, {"queries": [reordered]}) == 1.0


# --------------------------------------------------------------------------
# Behavioural — everything else is unchanged from v4_7
# --------------------------------------------------------------------------


def test_bash_command_still_scores_on_tool_name_alone(v48) -> None:
    assert _score(v48, "bash_command", {"command": "ls -la"}, {"command": "rm -rf /"}) == 1.0


def test_update_progress_still_scores_on_tool_name_alone(v48) -> None:
    assert _score(v48, "update_progress", {"board": "a"}, {"board": "totally different"}) == 1.0


def test_browse_urls_remain_exact(v47, v48) -> None:
    exp = {"urls": ["https://example.com/a"]}
    assert _score(v48, "browse", exp, exp) == 1.0
    assert _score(v48, "browse", exp, {"urls": ["https://example.com/b"]}) == 0.0
    assert _score(v47, "browse", exp, {"urls": ["https://example.com/b"]}) == 0.0
