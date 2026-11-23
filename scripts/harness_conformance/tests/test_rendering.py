# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation requires all checks and groups failures at the first failing tier."""

from scripts.harness_conformance.rendering import render_table


def scenario(name, **statuses):
    return {
        "scenario": name,
        # Deliberately passing: a P0-only runner verdict must not hide higher-tier failures.
        "verdict": "fulfilled",
        "checks": [
            {"id": f"check.{tier.lower()}", "kind": "behavioral", "tier": tier, "status": status}
            for tier, status in statuses.items()
        ],
    }


def render(scenarios):
    return render_table({"gym_commit": "a" * 40, "harnesses": {"pi": {"scenarios": scenarios}}})


def test_each_scenario_appears_once_with_all_check_results():
    text = render(
        [
            scenario("tool_success", P0="pass"),
            scenario("judge_verifier", P0="pass", P1="pass"),
            scenario("context_compaction", P0="pass", P1="fail"),
            scenario("tool_failure", P0="fail", P1="fail"),
            scenario("efficiency", P0="pass", P1="pass", P2="fail"),
        ]
    )
    assert (
        '| `pi` | <Badge intent="error" minimal>2/5</Badge> | '
        '<Badge intent="error" minimal>1</Badge> | <Badge intent="error" minimal>1</Badge> | '
        '<Badge intent="error" minimal>1</Badge> |'
    ) in text
    details = text.split("### `pi`", 1)[1].split("| Check |", 1)[0]
    for name, result in {
        "tool_success": "✓ (P0:✓)",
        "judge_verifier": "✓ (P0:✓, P1:✓)",
        "context_compaction": "✗ (P0:✓, P1:✗)",
        "tool_failure": "✗ (P0:✗, P1:✗)",
        "efficiency": "✗ (P0:✓, P1:✓, P2:✗)",
    }.items():
        assert details.count(f"`{name}`") == 1
        assert f"- `{name}`: {result}" in details
    assert "| `hermes` | — | — | — | — |" in text


def test_no_applicable_checks_is_unmeasured_not_passed():
    text = render([scenario("model_error", P0="not_applicable")])
    assert "| `pi` | — | — | — | — |" in text
    assert "<Badge" not in text
    assert "`model_error`: — (no applicable checks)" in text


def test_fully_passing_scenarios_use_brand_color():
    text = render([scenario("tool_success", P0="pass", P1="not_applicable")])
    assert '| `pi` | <Badge intent="launch" minimal>1/1</Badge> | 0 | 0 | 0 |' in text
    assert "`tool_success`: ✓ (P0:✓)" in text


def test_first_failure_uses_priority_order_and_does_not_require_lower_tiers():
    text = render([scenario("context_compaction", P2="fail", P1="fail")])
    assert (
        '| `pi` | <Badge intent="error" minimal>0/1</Badge> | 0 | <Badge intent="error" minimal>1</Badge> | 0 |'
        in text
    )
    assert "`context_compaction`: ✗ (P1:✗, P2:✗)" in text
