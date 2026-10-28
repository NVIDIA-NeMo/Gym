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
    assert "| `pi` | 2/3 | 1/2 | 0/1 |" in text
    assert "| `hermes` | — | — | — |" in text
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

    assert "| `pi` | — | — | — |" in render_table({"gym_commit": "a" * 40, "harnesses": rows})
