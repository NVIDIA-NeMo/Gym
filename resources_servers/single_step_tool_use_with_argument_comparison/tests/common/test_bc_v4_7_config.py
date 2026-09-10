# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioural contract for bc_v4_7.

bc_v4_7 differs from bc_v4_6 in EXACTLY ONE value: the ``search.queries`` override moves
``word_count_similarity_threshold`` 0.40 -> 0.45.

HOW STRICT THAT IS
------------------
``sim = |intersection| / (|expected| + |actual|)``, so ``sim == F1 / 2`` and the ceiling is
0.5 (identical word bags). 0.45 therefore means **F1 >= 0.9**, i.e. only a hair below exact
match. For a model query that is a pure SUBSET of the golden (precision 1.0, recall r)::

    sim = r / (1 + r) >= t   <=>   r >= t / (1 - t)

    t = 0.30  ->  r >= 0.4286    delete up to 57% of the golden's words
    t = 0.40  ->  r >= 0.6667    delete up to 33%
    t = 0.45  ->  r >= 0.8182    delete up to 18%

On a 10-token golden that puts the cliff between keeping 8 tokens (fails) and 9 (passes).

WHAT IT STILL DOES NOT FIX
--------------------------
Substituting one token of ten leaves ``sim = 9/20 = 0.45`` exactly, which satisfies ``>=``.
So even at 0.45 the year / wrong-entity hack survives -- now precisely on the knife edge.
See ``test_wrong_entity_survives_even_at_0_45``.
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
def v46() -> ToolCallComparator:
    return _comparator("bc_v4_6.yaml", "bc_v4_6_rs")


@fixture
def v47() -> ToolCallComparator:
    return _comparator("bc_v4_7.yaml", "bc_v4_7_rs")


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------


def test_differs_from_v4_6_only_in_search_queries_threshold() -> None:
    a = _cmp_cfg("bc_v4_6.yaml", "bc_v4_6_rs")
    b = _cmp_cfg("bc_v4_7.yaml", "bc_v4_7_rs")
    assert a["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.4
    assert b["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] == 0.45
    patched = json.loads(json.dumps(b))
    patched["argument_comparison_overrides"]["search"]["queries"]["word_count_similarity_threshold"] = 0.4
    assert patched == a


def test_agent_wiring() -> None:
    raw = _raw("bc_v4_7.yaml")
    assert set(raw) == {"bc_v4_7_rs", "bc_v4_7_agent"}
    agent = raw["bc_v4_7_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert agent["resources_server"]["name"] == "bc_v4_7_rs"
    assert agent["model_server"]["name"] == "policy_model"


def test_dataset_block_matches_v4_6() -> None:
    a = _raw("bc_v4_6.yaml")["bc_v4_6_agent"]["responses_api_agents"]["tool_simulation_agent"]
    b = _raw("bc_v4_7.yaml")["bc_v4_7_agent"]["responses_api_agents"]["tool_simulation_agent"]
    assert b["datasets"] == a["datasets"]


# --------------------------------------------------------------------------
# Behavioural
# --------------------------------------------------------------------------


def test_exact_golden_still_scores_one(v46, v47) -> None:
    args = {"queries": [GOLDEN_QUERY]}
    assert _score(v46, "search", args, args) == 1.0
    assert _score(v47, "search", args, args) == 1.0


def test_recall_cliff_moves_to_nine_of_ten(v46, v47) -> None:
    """r >= 0.818, so k=9 survives and k=8 does not."""
    exp = {"queries": [GOLDEN_QUERY]}
    assert _score(v47, "search", exp, {"queries": [_first_k(9)]}) == 1.0
    assert _score(v47, "search", exp, {"queries": [_first_k(8)]}) == 0.0


def test_k8_and_k7_are_the_discriminating_cases(v46, v47) -> None:
    """These pass under bc_v4_6 and fail under bc_v4_7 -- the whole delta."""
    exp = {"queries": [GOLDEN_QUERY]}
    for k in (7, 8):
        assert _score(v46, "search", exp, {"queries": [_first_k(k)]}) == 1.0, k
        assert _score(v47, "search", exp, {"queries": [_first_k(k)]}) == 0.0, k


def test_dropping_one_token_still_passes(v47) -> None:
    """Benign variation must survive: sim 9/19 = 0.474."""
    exp = {"queries": [GOLDEN_QUERY]}
    assert _score(v47, "search", exp, {"queries": [" ".join(GOLDEN_TOKENS[:-1])]}) == 1.0


def test_wrong_entity_survives_even_at_0_45(v47) -> None:
    """DOCUMENTED BLIND SPOT, now knife-edge: sim = 9/20 = 0.45 exactly, and the
    comparison is >=. Tightening to 0.45 still does not punish a wrong year."""
    exp = {"queries": [GOLDEN_QUERY]}
    act = {"queries": [GOLDEN_QUERY.replace("2024", "2025")]}
    assert _score(v47, "search", exp, act) == 1.0


def test_token_order_still_ignored(v47) -> None:
    exp = {"queries": [GOLDEN_QUERY]}
    assert _score(v47, "search", exp, {"queries": [" ".join(reversed(GOLDEN_TOKENS))]}) == 1.0


# --------------------------------------------------------------------------
# Other tools untouched
# --------------------------------------------------------------------------


def test_bash_command_still_name_only(v47) -> None:
    assert _score(v47, "bash_command", {"duration": 5, "keystrokes": "grep -i x y.txt"},
                  {"duration": 9, "keystrokes": "cat /etc/hostname"}) == 1.0


def test_update_progress_still_name_only(v47) -> None:
    assert _score(v47, "update_progress", {"board": "a b c"}, {"board": "totally different"}) == 1.0


def test_browse_urls_stay_exact(v47) -> None:
    exp = {"urls": ["https://example.org/a"], "goal": "find the producer"}
    assert _score(v47, "browse", exp, {"urls": ["https://example.org/a"], "goal": "other"}) == 1.0
    assert _score(v47, "browse", exp, {"urls": ["https://example.org/b"], "goal": "find the producer"}) == 0.0
