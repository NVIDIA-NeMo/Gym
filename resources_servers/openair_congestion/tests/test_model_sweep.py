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
#
# The capability sweep is the pre-training environment check: scripted
# anchors of known quality must land in their known order over the real HTTP
# surface, and the report must expose the profile a reviewer needs. The LLM
# path runs end-to-end against a stub chat-completions server, so request
# shape, parsing, and failure accounting are covered without a network.
import asyncio
import json
import random

import pytest
from aiohttp import web

from resources_servers.openair_congestion.client import _free_port
from resources_servers.openair_congestion.model_sweep import (
    _ANCHOR_CONSTRAINTS,
    _ANCHOR_ORDER,
    ModelSpec,
    _evaluate_model_ordering,
    _load_example_rows,
    _make_random_valid,
    _parse_tool_call,
    _parse_topology,
    _profile_task_rows,
    _repeat_task_rows,
    _require_compliance_models,
    _sweep,
    _validate_compliance_rows,
)


def test_anchor_sweep_orders_policies_and_reports_profile():
    # Ordering is a cross-regime gate; evaluating only the first one or two
    # rows can correctly tie relief with noop when neither state needs action.
    rows = _profile_task_rows(_load_example_rows(), task_count=5)
    report = asyncio.run(_sweep(rows, repeats=1, specs=[], concurrency=2))

    assert report["episodes_per_policy"] == 5
    by_policy = {row["policy"]: row for row in report["profile"]}
    assert set(by_policy) == set(_ANCHOR_ORDER)
    assert all(row["episodes"] == 5 for row in report["profile"])

    # The known quality ladder must hold on the deterministic replay tasks,
    # and the report must say so itself.
    assert report["anchor_ordering_ok"] is True
    assert all(
        by_policy[better]["mean_return"] > by_policy[worse]["mean_return"] for better, worse in _ANCHOR_CONSTRAINTS
    )
    assert len(report["anchor_ordering_comparisons"]) == len(_ANCHOR_CONSTRAINTS)
    assert all(
        comparison["status"] == "PASS" and comparison["ci95_low"] > 0.0
        for comparison in report["anchor_ordering_comparisons"]
    )

    # Catastrophic play is rejected every step; valid play never is.
    assert by_policy["anchor:catastrophic"]["rejection_rate"] == 1.0
    assert by_policy["anchor:random-valid"]["rejection_rate"] == 0.0
    assert by_policy["anchor:noop"]["noop_rate"] == 1.0
    assert by_policy["anchor:relief"]["return_distribution"]["median"] is not None
    assert by_policy["anchor:relief"]["tool_metrics"]
    assert set(by_policy["anchor:relief"]["returns_by_scenario"]) == {row["scenario_id"] for row in rows}


def test_profile_rows_cover_every_example_regime_and_repeat_exact_prompts():
    base_rows = _load_example_rows()
    assert len(base_rows) == 5

    prompts = _profile_task_rows(base_rows, task_count=10)
    assert [row["scenario_id"] for row in prompts] == [
        "prb_exhaustion",
        "bursty",
        "interference",
        "prach_storm",
        "qos_competition",
    ] * 2
    assert len({row["seed"] for row in prompts}) == 10

    repeated = _repeat_task_rows(prompts, repeats=3)
    assert len(repeated) == 30
    for prompt_idx in range(10):
        group = repeated[prompt_idx * 3 : (prompt_idx + 1) * 3]
        assert {row["seed"] for row in group} == {prompts[prompt_idx]["seed"]}
        assert [row["_profile_response_index"] for row in group] == [0, 1, 2]


def test_model_ordering_requires_ranked_models_to_improve_monotonically():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    passing = [
        {
            "policy": "model:small",
            "mean_return": -4.0,
            "episodes": 8,
            "infra_errors": 0,
            "parse_failures": 0,
            "invalid_calls": 0,
            "episode_records": [{"pair_key": str(index), "return": -4.0, "usable": True} for index in range(8)],
        },
        {
            "policy": "model:frontier",
            "mean_return": -3.0,
            "episodes": 8,
            "infra_errors": 0,
            "parse_failures": 0,
            "invalid_calls": 0,
            "episode_records": [{"pair_key": str(index), "return": -3.0, "usable": True} for index in range(8)],
        },
    ]
    assert _evaluate_model_ordering(passing, specs, expected_episodes=8)["status"] == "PASS"

    failing = [
        dict(passing[0]),
        dict(
            passing[1],
            mean_return=-5.0,
            episode_records=[{"pair_key": str(index), "return": -5.0, "usable": True} for index in range(8)],
        ),
    ]
    result = _evaluate_model_ordering(failing, specs, expected_episodes=8)
    assert result["status"] == "FAIL"
    assert result["expected"] == ["model:small", "model:frontier"]


def test_all_parse_failures_are_not_evaluable():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = [
        {
            "policy": f"model:{label}",
            "mean_return": -1.0 + rank,
            "episodes": 500,
            "infra_errors": 0,
            "parse_failures": 500,
            "invalid_calls": 0,
            "episode_records": [
                {
                    "pair_key": str(index),
                    "return": -1.0 + rank,
                    "usable": False,
                }
                for index in range(500)
            ],
        }
        for rank, label in enumerate(("small", "frontier"))
    ]

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=500,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "parse" in result["reason"]


def test_partial_parse_failures_are_not_evaluable():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = []
    for rank, label in enumerate(("small", "frontier")):
        records = [
            {
                "pair_key": str(index),
                "return": float(rank),
                "usable": not (label == "frontier" and index == 0),
            }
            for index in range(8)
        ]
        profile.append(
            {
                "policy": f"model:{label}",
                "mean_return": float(rank),
                "episodes": 8,
                "infra_errors": 0,
                "parse_failures": int(label == "frontier"),
                "invalid_calls": 0,
                "episode_records": records,
            }
        )

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=8,
    )

    assert result["status"] == "NOT_EVALUABLE"


def test_noncompliance_failure_ceiling_uses_only_common_usable_pairs():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = []
    for rank, label in enumerate(("small", "frontier")):
        records = [
            {
                "pair_key": str(index),
                "return": float(rank),
                "usable": not (label == "frontier" and index == 0),
            }
            for index in range(8)
        ]
        profile.append(
            {
                "policy": f"model:{label}",
                "mean_return": float(rank),
                "episodes": 8,
                "infra_errors": 0,
                "parse_failures": int(label == "frontier"),
                "invalid_calls": 0,
                "episode_records": records,
            }
        )

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=8,
        compliance=False,
        failure_rate_ceiling=0.125,
    )

    assert result["status"] == "PASS"
    assert result["valid_paired_episodes"] == 7
    assert result["failure_rates"]["model:frontier"] == pytest.approx(0.125)
    assert result["comparisons"][0]["paired_episodes"] == 7


def test_noncompliance_failure_ceiling_rejects_profiles_above_limit():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = [
        {
            "policy": f"model:{label}",
            "mean_return": float(rank),
            "episodes": 4,
            "infra_errors": 0,
            "parse_failures": 1 if label == "frontier" else 0,
            "invalid_calls": 0,
            "episode_records": [
                {
                    "pair_key": str(index),
                    "return": float(rank),
                    "usable": not (label == "frontier" and index == 0),
                }
                for index in range(4)
            ],
        }
        for rank, label in enumerate(("small", "frontier"))
    ]

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=4,
        compliance=False,
        failure_rate_ceiling=0.20,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "failure-rate ceiling" in result["reason"]


def test_positive_mean_with_interval_crossing_zero_fails():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    deltas = [-0.1, 0.1, 0.1, -0.05]
    weak_records = [{"pair_key": str(index), "return": 0.0, "usable": True} for index in range(len(deltas))]
    strong_records = [
        {
            "pair_key": str(index),
            "return": delta,
            "usable": True,
        }
        for index, delta in enumerate(deltas)
    ]
    profile = [
        {
            "policy": "model:small",
            "mean_return": 0.0,
            "episodes": len(deltas),
            "infra_errors": 0,
            "parse_failures": 0,
            "invalid_calls": 0,
            "episode_records": weak_records,
        },
        {
            "policy": "model:frontier",
            "mean_return": sum(deltas) / len(deltas),
            "episodes": len(deltas),
            "infra_errors": 0,
            "parse_failures": 0,
            "invalid_calls": 0,
            "episode_records": strong_records,
        },
    ]

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=len(deltas),
    )

    assert result["status"] == "FAIL"
    assert result["comparisons"][0]["ci95_low"] <= 0.0


def test_compliance_profile_requires_two_preranked_models():
    with pytest.raises(ValueError, match="at least two real models"):
        _require_compliance_models([])

    unranked = [
        ModelSpec(label="small", model="small", base_url="http://x"),
        ModelSpec(label="frontier", model="frontier", base_url="http://x"),
    ]
    with pytest.raises(ValueError, match="capability_rank"):
        _require_compliance_models(unranked)

    ranked = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    _require_compliance_models(ranked)

    duplicate_identity = [
        ModelSpec(label="small", model="same-model", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="same-model", base_url="http://x", capability_rank=2),
    ]
    with pytest.raises(ValueError, match="distinct model identities"):
        _require_compliance_models(duplicate_identity)

    missing_identity = [
        ModelSpec(label="small", model="", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    with pytest.raises(ValueError, match="non-empty"):
        _require_compliance_models(missing_identity)


def test_compliance_profile_requires_exact_one_hot_regime_coverage():
    rows = _load_example_rows()
    _validate_compliance_rows(rows)

    duplicate = json.loads(json.dumps(rows))
    duplicate[-1] = json.loads(json.dumps(duplicate[0]))
    with pytest.raises(ValueError, match="exactly once"):
        _validate_compliance_rows(duplicate)

    mismatched = json.loads(json.dumps(rows))
    mismatched[0]["regime_mix"] = {"bursty": 1.0}
    with pytest.raises(ValueError, match="one-hot regime_mix"):
        _validate_compliance_rows(mismatched)

    extra = json.loads(json.dumps(rows))
    extra.append(json.loads(json.dumps(rows[0])))
    extra[-1]["scenario_id"] = "unknown"
    extra[-1]["regime_mix"] = {"unknown": 1.0}
    with pytest.raises(ValueError, match="exactly once"):
        _validate_compliance_rows(extra)


def test_random_valid_never_repeats_within_rate_limit_window():
    policy = _make_random_valid()
    observation = (
        "- Cell 0: DL PRB util p50=34%, p99=41%; 0 SLA violation(s) in last 5s.\n"
        "    UE 0 (5QI 9): offered 8.0 Mbps.\n"
        "    UE 1 (5QI 9): offered 8.0 Mbps.\n"
        "- Cell 1: DL PRB util p50=24%, p99=30%; 0 SLA violation(s) in last 5s.\n"
        "    UE 0 (5QI 9): offered 8.0 Mbps.\n"
    )
    rng = random.Random(0)
    keys = [json.dumps(policy(observation, i, rng), sort_keys=True) for i in range(64)]
    # The guardrail rejects an identical action within its 2 s window (two
    # logical steps); the policy's dedupe must keep adjacent pairs distinct.
    assert all(keys[i] != keys[i - 1] for i in range(1, len(keys)))
    assert all(keys[i] != keys[i - 2] for i in range(2, len(keys)))


def test_parse_topology_reads_cells_and_ues():
    observation = "- Cell 0: stuff\n    UE 0 (5QI 9): x\n    UE 1 (5QI 9): x\n- Cell 1: stuff\n    UE 0 (5QI 9): x\n"
    assert _parse_topology(observation) == {0: [0, 1], 1: [0]}
    # Unparseable text degrades to a safe single-cell fallback, never a crash.
    assert _parse_topology("") == {0: [0]}


def test_llm_policies_end_to_end_against_mock_endpoint():
    # A stub chat-completions server proves the whole LLM path with no
    # network and no key. Four model personas cover the failure taxonomy:
    # "cooperative" calls noop natively; "rambler" replies prose only (a
    # terminal protocol failure); "hallucinator" calls a tool that does not
    # exist (a terminal invalid call); "dead" 500s every request (episodes dropped as
    # infra errors). "dead" runs FIRST so the test also proves a failing
    # model cannot starve the replay pool for the models after it.
    seen_payloads: list[dict] = []

    async def chat_completions(request: web.Request) -> web.Response:
        payload = await request.json()
        seen_payloads.append(payload)
        if payload["model"] == "dead":
            return web.json_response({"error": "upstream unavailable"}, status=500)
        if payload["model"] == "cooperative":
            message = {"tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}], "content": None}
        elif payload["model"] == "hallucinator":
            message = {"tool_calls": [{"function": {"name": "restart_gnb", "arguments": "{}"}}], "content": None}
        else:
            message = {"content": "The network seems fine; monitoring further."}
        return web.json_response({"choices": [{"message": message}]})

    async def run() -> dict:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        try:
            base_url = f"http://127.0.0.1:{port}/v1"
            specs = [
                ModelSpec(label="dead", model="dead", base_url=base_url),
                ModelSpec(label="cooperative", model="cooperative", base_url=base_url),
                ModelSpec(label="hallucinator", model="hallucinator", base_url=base_url),
                ModelSpec(label="rambler", model="rambler", base_url=base_url),
            ]
            rows = _profile_task_rows(_load_example_rows(), task_count=1)
            return await _sweep(rows, repeats=1, specs=specs, concurrency=1)
        finally:
            await runner.cleanup()

    report = asyncio.run(run())
    by_policy = {row["policy"]: row for row in report["profile"]}

    dead = by_policy["model:dead"]
    assert dead["episodes"] == 0
    assert dead["infra_errors"] == 1
    assert dead["mean_return"] is None and dead["vs_relief"] is None

    # The models after the dead one still complete: its drained episode did
    # not leak a replay-pool slot or bleed errors into their rows.
    cooperative = by_policy["model:cooperative"]
    assert cooperative["episodes"] == 1
    assert cooperative["parse_failures"] == 0
    assert cooperative["noop_rate"] == 1.0
    assert cooperative["infra_errors"] == 0

    hallucinator = by_policy["model:hallucinator"]
    assert hallucinator["episodes"] == 1
    assert hallucinator["invalid_calls"] > 0
    assert hallucinator["parse_failures"] == 0  # the call parses; the env refuses it

    rambler = by_policy["model:rambler"]
    assert rambler["episodes"] == 1
    assert rambler["parse_failures"] == 1
    assert rambler["noop_rate"] == 0.0
    assert rambler["tool_metrics"]["<parse_failure>"]["calls"] == 1
    assert rambler["usable_episodes"] == 0
    # A terminal protocol penalty and a multi-step valid return are not
    # comparable on magnitude; usability, not score ordering, excludes the
    # malformed response from capability comparisons.
    assert rambler["episode_records"][0]["usable"] is False

    # LLM rows carry the reward-ceiling reference; noop-playing models must
    # match the noop anchor's return on the same paired task.
    assert cooperative["mean_return"] == by_policy["anchor:noop"]["mean_return"]
    assert cooperative["vs_relief"] is not None

    # The request carried the row's own messages plus the observation.
    payload = seen_payloads[0]
    assert payload["messages"][0]["role"] == "system" and payload["messages"][0]["content"]
    assert payload["messages"][-1]["role"] == "user" and "Cell 0" in payload["messages"][-1]["content"]
    assert any(t["function"]["name"] == "set_prb_cap" for t in payload["tools"])
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False


def test_sweep_rejects_bad_config():
    with pytest.raises(ValueError, match="repeats"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=0, specs=[]))
    dup = [
        ModelSpec(label="same", model="a", base_url="http://127.0.0.1:1/v1"),
        ModelSpec(label="same", model="b", base_url="http://127.0.0.1:1/v1"),
    ]
    with pytest.raises(ValueError, match="duplicate model labels"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=1, specs=dup))


def test_sweep_refuses_named_but_unset_api_key_env(monkeypatch):
    monkeypatch.delenv("SWEEP_TEST_MISSING_KEY", raising=False)
    spec = ModelSpec(label="x", model="x", base_url="http://127.0.0.1:1/v1", api_key_env="SWEEP_TEST_MISSING_KEY")
    with pytest.raises(ValueError, match="SWEEP_TEST_MISSING_KEY"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=1, specs=[spec]))


def test_parse_tool_call_handles_tool_calls_content_json_and_garbage():
    native = {
        "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
        "content": None,
    }
    assert _parse_tool_call(native) == {"name": "noop", "arguments": {}}

    content = {"content": 'Sure: {"name": "set_scheduler_policy", "arguments": {"cell_id": 0, "policy": "PF"}}'}
    assert _parse_tool_call(content) == {
        "name": "set_scheduler_policy",
        "arguments": {"cell_id": 0, "policy": "PF"},
    }

    assert _parse_tool_call({"content": "I would consider the network first."}) is None
    assert _parse_tool_call({"tool_calls": [{"function": {"name": "noop", "arguments": "{not json"}}]}) is None


def test_parse_tool_call_rejects_multiple_native_tool_calls():
    message = {
        "tool_calls": [
            {"function": {"name": "noop", "arguments": "{}"}},
            {
                "function": {
                    "name": "set_scheduler_policy",
                    "arguments": '{"cell_id": 0, "policy": "PF"}',
                }
            },
        ]
    }

    assert _parse_tool_call(message) is None


@pytest.mark.parametrize(
    "arguments",
    [
        '{"cell_id": 0, "weights": {"1": NaN}}',
        '{"cell_id": 0, "cell_id": 1, "policy": "PF"}',
    ],
)
def test_parse_tool_call_rejects_nonstandard_or_ambiguous_json(arguments):
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "set_qos_weights",
                    "arguments": arguments,
                }
            }
        ]
    }

    assert _parse_tool_call(message) is None
