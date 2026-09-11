# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen parity with the policies that produced the published OSWorld scores.

Releases b1 through b11p were measured with ``fixed`` and ``hysteresis``. Any
change that moves a rendered prompt by one byte, or moves a ``policy_id``,
silently invalidates the comparison between a new run and those records -- and
does so without failing anything, because the benchmark would still produce a
plausible number.

The constants below were computed from Gym ``a4471b59``, the commit that ran
b8/b9/b10p/b11p. They are goldens: a diff here means the published records and
a new run are no longer measuring the same thing. Regenerate them only
together with a deliberate, documented release boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

import pytest

from responses_api_agents.osworld_agent.adapter_agents import NemotronV3NanoOmniAgent
from responses_api_agents.osworld_agent.history_policy import (
    HistoryPolicySpec,
    HistoryPolicyState,
    plan_history,
)

# policy_id feeds run records and the exact-trace contract; it must not move.
PUBLISHED_POLICY_IDS = {
    "fixed(1)": (HistoryPolicySpec.fixed(1), "osworld-history-policy-b7220985bd046580251603b7"),
    "fixed(3)": (HistoryPolicySpec.fixed(3), "osworld-history-policy-d171b7730c73e7c384ab2b25"),
    "fixed(10)": (HistoryPolicySpec.fixed(10), "osworld-history-policy-c6aa3cc4908990e7e48aa92a"),
    "hysteresis(3,10)": (
        HistoryPolicySpec.hysteresis(low_water=3, high_water=10),
        "osworld-history-policy-df73e1447137486cb16fa3a6",
    ),
    "hysteresis(1,5)": (
        HistoryPolicySpec.hysteresis(low_water=1, high_water=5),
        "osworld-history-policy-c4f9c60cd41592bec97923c8",
    ),
    "hysteresis(5,20)": (
        HistoryPolicySpec.hysteresis(low_water=5, high_water=20),
        "osworld-history-policy-abfafbcac32b574d98e2ef69",
    ),
}

# sha256 over every prompt the adapter renders for a deterministic trajectory.
PUBLISHED_PROMPT_DIGESTS = {
    "fixed-3": (
        {"max_image_history_length": 3},
        25,
        "10be484ce3a314fd655235412f2fb455401e7b3ba88cf1826959ac4b9ab6611b",
    ),
    "fixed-1": (
        {"max_image_history_length": 1},
        15,
        "1ded2c44eb5e75008fc58fd3910ca72e7c4ca77b665e47f0c9cab5087bcef38d",
    ),
    "fixed-10": (
        {"max_image_history_length": 10},
        30,
        "12d7e595e029032e80ef9806a8d5447728d8903d59a2a5909ef2c1a680794bb2",
    ),
    "hysteresis-3-10": (
        {"max_image_history_length": 3, "max_live_images": 10},
        35,
        "14f1207a6e35c2b9ca15fa541c1118ecc3ba9885b7df2fbff6cf56691b9dbc4f",
    ),
    "hysteresis-1-5": (
        {"max_image_history_length": 1, "max_live_images": 5},
        20,
        "af37a0c96763486cc4903d381018a882de4d5f7428446bd36ab4b37483b00b48",
    ),
    "hysteresis-5-20": (
        {"max_image_history_length": 5, "max_live_images": 20},
        45,
        "456a84ac58f9e545c1d7b68bbf2599c2a02eacf3b03dacd984fef2f0b2cd546c",
    ),
}

GOOD_REPLY = "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"


def _render_digest(agent_kwargs: Dict[str, Any], steps: int) -> str:
    agent = NemotronV3NanoOmniAgent(
        model="policy-under-test", max_steps=40, parse_retries=1, **agent_kwargs
    )
    prompts: List[Any] = []

    def call_llm(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
        prompts.append(payload["messages"])
        return {
            "content": GOOD_REPLY,
            "reasoning_content": f"Thought {len(prompts)}",
            "finish_reason": "stop",
        }

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for index in range(steps):
        agent.predict("Complete the task.", {"screenshot": f"png-{index + 1}".encode()})

    blob = json.dumps(prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("label", sorted(PUBLISHED_POLICY_IDS))
def test_published_policy_ids_are_frozen(label: str) -> None:
    spec, expected = PUBLISHED_POLICY_IDS[label]
    assert spec.policy_id == expected, (
        f"{label} policy_id moved; every run record naming the old id now describes "
        "a policy this code no longer implements"
    )


@pytest.mark.parametrize("label", sorted(PUBLISHED_PROMPT_DIGESTS))
def test_published_prompt_rendering_is_byte_identical(label: str) -> None:
    agent_kwargs, steps, expected = PUBLISHED_PROMPT_DIGESTS[label]
    assert _render_digest(agent_kwargs, steps) == expected, (
        f"{label} renders different bytes than the commit that produced the published "
        "scores; a new run is no longer comparable to those records"
    )


def test_explicit_policy_config_renders_the_same_bytes_as_the_legacy_arguments() -> None:
    """The two configuration surfaces must not drift apart."""

    legacy = _render_digest({"max_image_history_length": 3}, 25)
    explicit = _render_digest(
        {"history_policy": {"schema_version": 1, "name": "fixed", "params": {"keep_images": 3}}}, 25
    )
    assert legacy == explicit == PUBLISHED_PROMPT_DIGESTS["fixed-3"][2]

    legacy_hyst = _render_digest({"max_image_history_length": 3, "max_live_images": 10}, 35)
    explicit_hyst = _render_digest(
        {
            "history_policy": {
                "schema_version": 1,
                "name": "hysteresis",
                "params": {"low_water": 3, "high_water": 10},
            }
        },
        35,
    )
    assert legacy_hyst == explicit_hyst == PUBLISHED_PROMPT_DIGESTS["hysteresis-3-10"][2]


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("fixed", 1, 1),
        ("fixed", 2, 2),
        ("fixed", 3, 3),
        ("fixed", 5, 5),
        ("fixed", 10, 10),
        ("fixed", 50, 50),
        ("hysteresis", 1, 2),
        ("hysteresis", 1, 5),
        ("hysteresis", 2, 7),
        ("hysteresis", 3, 10),
        ("hysteresis", 5, 20),
        ("hysteresis", 10, 11),
    ],
)
def test_shipped_policies_still_select_a_contiguous_suffix(name: str, low: int, high: int) -> None:
    """Property the published policies had, restated so a refactor cannot lose it.

    Both shipped policies fold a prefix into text and keep one unbroken run of
    live images ending at the current turn. The scalar accessor stays defined
    for them, which is what lets the sink_window change be additive.
    """

    spec = HistoryPolicySpec(name=name, low_water=low, high_water=high)
    state = HistoryPolicyState()
    for completed in range(200):
        plan = plan_history(spec, state, completed_turns=completed)
        assert len(plan.image_intervals) == 1
        start = plan.image_window_start
        assert plan.image_turns == tuple(range(start, completed + 1))
        assert plan.text_turns == tuple(range(start))
        assert not plan.dropped_turns
        assert plan.spec.sink == 0
        assert len(plan.image_turns) <= high
        assert plan.image_budget_clamped is False
        state = plan.next_state


def test_legacy_construction_never_produces_a_sink() -> None:
    for keep, live in [(3, None), (3, 3), (3, 10), (1, None), (5, 20)]:
        spec = HistoryPolicySpec.from_legacy(keep_images=keep, max_live_images=live)
        assert spec.sink == 0
        assert spec.name in {"fixed", "hysteresis"}
        assert "sink" not in spec.to_config()["params"]


def test_shipped_telemetry_keys_survive_and_new_keys_are_additive() -> None:
    """Downstream readers of snapshot_* must keep working unchanged."""

    spec = HistoryPolicySpec.hysteresis(low_water=3, high_water=10)
    plan = plan_history(spec, HistoryPolicyState(), completed_turns=4)
    telemetry = plan.telemetry()

    for key in (
        "prompt_snapshot_count",
        "snapshot_window_start",
        "snapshot_compaction_triggered",
        "snapshot_window_min",
        "snapshot_window_max",
        "history_policy_id",
        "history_policy_name",
        "history_policy_schema_version",
        "history_policy_compaction_epoch",
        "history_selection_append_expected",
    ):
        assert key in telemetry, f"shipped telemetry key {key} disappeared"

    # For a contiguous policy the scalar keeps its old meaning exactly.
    assert telemetry["snapshot_window_start"] == plan.image_window_start
    assert telemetry["snapshot_sink_size"] == 0
    assert telemetry["snapshot_image_budget_clamped"] is False
    assert telemetry["snapshot_image_intervals"] == [[plan.image_window_start, 5]]
