# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from responses_api_agents.nooa_agent.context_diagnostic import context_difference


def test_reports_exact_first_difference_without_contents():
    report = context_difference(
        [{"role": "user", "content": "PRIVATE-original"}], [{"role": "user", "content": "PRIVATE-changed"}]
    )
    assert report["first_item"] == 0 and set(report["fields"]) == {"content"}
    assert report["fields"]["content"]["before_sha256"] != report["fields"]["content"]["after_sha256"]
    assert "PRIVATE" not in json.dumps(report)


def test_reports_removal_and_null_vs_absent():
    report = context_difference(
        [{"role": "user"}, {"type": "reasoning", "content": None}], [{"role": "user"}, {"type": "reasoning"}]
    )
    assert report["first_item"] == 1 and report["fields"]["content"]["before_present"] is True
    assert report["fields"]["content"]["after_present"] is False
    report = context_difference([{"role": "user"}], [])
    assert report["previous_items"] == 1 and report["current_items"] == 0 and report["first_item"] == 0
