# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.web.native_webvoyager import adapt_native_webvoyager_record, native_webvoyager_tools


def test_native_webvoyager_tools_returns_independent_copy() -> None:
    first = native_webvoyager_tools()
    second = native_webvoyager_tools()

    first[0]["name"] = "changed"
    assert second[0]["name"] == "navigate"


@pytest.mark.parametrize(
    ("record", "expected_urls", "expected_sites", "expected_intent"),
    [
        (
            {"id": "A--0", "web": "https://one.example |AND| https://two.example", "web_name": "A", "ques": "q"},
            ["https://one.example", "https://two.example"],
            ["A"],
            "q",
        ),
        (
            {"id": 7, "start_url": ["https://one.example", "", 2], "intent": "fallback"},
            ["https://one.example", "2"],
            [],
            "fallback",
        ),
        ({"id": "scalar", "start_url": 123}, ["123"], [], ""),
        ({"id": "empty"}, [], [], ""),
    ],
)
def test_adapt_native_webvoyager_record_normalizes_public_rows(
    record, expected_urls, expected_sites, expected_intent
) -> None:
    row = adapt_native_webvoyager_record(record)

    assert row["web_task"]["task_id"] == str(record["id"])
    assert row["web_task"]["start_urls"] == expected_urls
    assert row["web_task"]["sites"] == expected_sites
    assert row["web_task"]["intent"] == expected_intent
    assert row["responses_create_params"]["metadata"]["task_id"] == str(record["id"])
    assert row["responses_create_params"]["tools"] == native_webvoyager_tools()


def test_adapt_native_webvoyager_record_requires_id() -> None:
    with pytest.raises(ValueError, match="requires id"):
        adapt_native_webvoyager_record({"ques": "missing identity"})
