# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario counts use each check's priority and applicability, not the overall P0 verdict."""

from scripts.harness_conformance.rendering import render_table


def test_summary_counts_scenarios_independently_in_each_applicable_tier():
    rows = {
        "pi": {
            "verdict": "not_fulfilled",
            "scenarios": [
                {
                    "scenario": "tool_success",
                    "verdict": "fulfilled",
                    "checks": [
                        {"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "pass"},
                        {"id": "calls.usage", "kind": "schema", "tier": "P0", "status": "pass"},
                        {"id": "efficiency.context", "kind": "behavioral", "tier": "P1", "status": "fail"},
                    ],
                },
                {
                    "scenario": "tool_failure",
                    "verdict": "not_fulfilled",
                    "checks": [
                        {"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "fail"},
                        {"id": "efficiency.context", "kind": "behavioral", "tier": "P1", "status": "not_applicable"},
                    ],
                },
                {
                    "scenario": "retry_429",
                    "verdict": "fulfilled",
                    "checks": [
                        {"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "pass"},
                        {"id": "efficiency.context", "kind": "behavioral", "tier": "P1", "status": "pass"},
                        {"id": "efficiency.cache", "kind": "behavioral", "tier": "P2", "status": "fail"},
                    ],
                },
                {
                    "scenario": "model_error",
                    "verdict": "not_fulfilled",
                    "checks": [
                        {"id": "efficiency.cache", "kind": "behavioral", "tier": "P2", "status": "not_applicable"},
                    ],
                },
            ],
        }
    }

    text = render_table({"gym_commit": "a" * 40, "harnesses": rows})

    assert "| Harness | P0 | P1 | P2 |" in text
    assert (
        '| `pi` | <Badge intent="error" minimal>2/3</Badge> | '
        '<Badge intent="error" minimal>1/2</Badge> | <Badge intent="error" minimal>0/1</Badge> |'
    ) in text
    assert "| `hermes` | — | — | — |" in text
    details = text.split("### `pi`", 1)[1].split("| Check |", 1)[0]
    assert "**P0**\n\n- **Passing:** `tool_success`, `retry_429`\n- **Not passing:** `tool_failure`" in details
    assert "**P1**\n\n- **Passing:** `retry_429`\n- **Not passing:** `tool_success`" in details
    assert "**P2**\n\n- **Passing:** None\n- **Not passing:** `retry_429`" in details
    assert "model_error" not in details
    assert "Passing scenarios" not in text
    assert "Passed" not in text and "Failed" not in text and "Unavailable" not in text


def test_tier_without_applicable_measurements_has_no_count():
    rows = {
        "pi": {
            "verdict": "not_fulfilled",
            "scenarios": [
                {
                    "scenario": "model_error",
                    "verdict": "not_fulfilled",
                    "checks": [
                        {"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "not_applicable"},
                    ],
                }
            ],
        }
    }

    text = render_table({"gym_commit": "a" * 40, "harnesses": rows})
    assert "| `pi` | — | — | — |" in text
    assert "<Badge" not in text
    assert "**Passing:**" not in text and "**Not passing:**" not in text


def test_fully_passing_tier_uses_brand_color_and_names_its_scenarios():
    rows = {
        "simple": {
            "verdict": "fulfilled",
            "scenarios": [
                {
                    "scenario": "tool_success",
                    "verdict": "fulfilled",
                    "checks": [{"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "pass"}],
                }
            ],
        }
    }

    text = render_table({"gym_commit": "a" * 40, "harnesses": rows})

    assert '| `simple` | <Badge intent="launch" minimal>1/1</Badge> | — | — |' in text
    details = text.split("### `simple`", 1)[1].split("| Check |", 1)[0]
    assert "**P0**\n\n- **Passing:** `tool_success`\n- **Not passing:** None" in details
    assert "**P1**" not in details and "**P2**" not in details
