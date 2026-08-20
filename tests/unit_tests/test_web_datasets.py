# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_gym.web.datasets import (
    adapt_visualwebarena_records,
    adapt_webarena_record,
    adapt_webvoyager_record,
    write_jsonl,
)
from nemo_gym.web.models import WebTask


def test_webarena_record_preserves_source_and_splits_multi_page_start():
    record = {
        "task_id": 7,
        "intent": "Compare the pages",
        "sites": ["reddit", "gitlab"],
        "start_url": "__REDDIT__ |AND| __GITLAB__",
        "eval": {"reference_answers": {"exact_match": "secret"}},
    }

    row = adapt_webarena_record(record)
    task = WebTask.model_validate(row["web_task"])

    assert task.start_urls == ["__REDDIT__", "__GITLAB__"]
    assert task.original_metadata == record
    assert row["responses_create_params"]["metadata"]["task_id"] == "7"


def test_null_storage_state_does_not_become_string_auth_profile():
    row = adapt_webarena_record(
        {
            "task_id": 8,
            "intent": "Inspect the map",
            "require_login": True,
            "storage_state": None,
        }
    )

    task = WebTask.model_validate(row["web_task"])
    assert task.auth_profile is None


def test_visualwebarena_partitions_are_globally_reindexed():
    rows = adapt_visualwebarena_records(
        [
            ("classifieds", [{"task_id": 0, "intent": "c", "sites": ["classifieds"]}]),
            (
                "reddit",
                [
                    {"task_id": 0, "intent": "r0", "sites": ["reddit"]},
                    {"task_id": 1, "intent": "r1", "sites": ["wikipedia"]},
                ],
            ),
            ("shopping", [{"task_id": 0, "intent": "s", "sites": ["shopping"]}]),
        ]
    )

    tasks = [WebTask.model_validate(row["web_task"]) for row in rows]
    assert [task.task_id for task in tasks] == ["0", "1", "2", "3"]
    assert tasks[-1].original_metadata["_source_task_id"] == 0
    assert tasks[-1].original_metadata["_source_partition"] == "shopping"


def test_webvoyager_uses_legacy_action_surface_over_browsergym():
    row = adapt_webvoyager_record(
        {
            "web_name": "Allrecipes",
            "id": "Allrecipes--0",
            "ques": "Find a recipe",
            "web": "https://www.allrecipes.com/",
        }
    )
    task = WebTask.model_validate(row["web_task"])

    assert task.runtime_profile.value == "browsergym"
    assert task.action_profile.value == "webvoyager_legacy"
    assert task.start_urls == ["https://www.allrecipes.com/"]


def test_write_jsonl_is_utf8_and_newline_delimited(tmp_path):
    output = tmp_path / "rows.jsonl"
    assert write_jsonl([{"text": "中文"}, {"text": "English"}], output) == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"text": "中文"},
        {"text": "English"},
    ]
