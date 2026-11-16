# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.web.datasets import (
    adapt_webarena_record,
    adapt_webarena_records,
    adapt_webvoyager_record,
    load_json_records,
    write_jsonl,
)
from nemo_gym.web.models import WebTask


def test_webarena_rows_use_the_shared_visual_browser_contract():
    source = {
        "id": "webarena-17",
        "ques": "Compare both pages",
        "web_name": ["reddit", "gitlab"],
        "web": "__REDDIT__ |AND| __GITLAB__",
        "eval": {"eval_types": ["string_match"]},
    }

    row = adapt_webarena_record(source)
    task = WebTask.model_validate(row["web_task"])

    assert task.task_id == "17"
    assert task.runtime_profile.value == "visual_browser"
    assert task.action_profile.value == "computer_use"
    assert task.observation_profile.value == "screenshot"
    assert task.start_urls == ["__REDDIT__", "__GITLAB__"]
    assert task.original_metadata == source
    assert row["responses_create_params"]["input"] == []
    assert "tools" not in row["responses_create_params"]


def test_webarena_batch_preserves_cross_task_collision_plan():
    rows = adapt_webarena_records(
        [
            {
                "id": f"webarena-{index}",
                "ques": "Update the shared profile",
                "web_name": ["gitlab"],
                "web": ["__GITLAB__/byteblaze"],
                "eval": {
                    "eval_types": ["program_html"],
                    "program_html": [
                        {
                            "url": "__GITLAB__/byteblaze",
                            "locator": "document.body.innerText",
                            "required_contents": {"must_include": ["updated"]},
                        }
                    ],
                },
            }
            for index in range(2)
        ]
    )

    for row in rows:
        targets = row["web_task"]["task_kwargs"]["collision_plan"]["snapshot_adapters"]["program_html"]["targets"]
        assert len(targets) == 1


def test_webvoyager_rows_use_the_visual_browser_contract():
    row = adapt_webvoyager_record(
        {
            "web_name": "Allrecipes",
            "id": "Allrecipes--0",
            "ques": "Find a recipe",
            "web": "https://www.allrecipes.com/",
        }
    )
    task = WebTask.model_validate(row["web_task"])

    assert task.runtime_profile.value == "visual_browser"
    assert task.action_profile.value == "computer_use"
    assert task.observation_profile.value == "screenshot"
    assert task.start_urls == ["https://www.allrecipes.com/"]
    assert row["responses_create_params"]["input"] == []
    assert "tools" not in row["responses_create_params"]


def test_write_jsonl_is_utf8_and_newline_delimited(tmp_path):
    output = tmp_path / "rows.jsonl"
    assert write_jsonl([{"text": "中文"}, {"text": "English"}], output) == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"text": "中文"},
        {"text": "English"},
    ]


def test_load_json_records_accepts_json_and_jsonl_and_rejects_non_objects(tmp_path):
    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    invalid_path = tmp_path / "invalid.json"
    json_path.write_text('[{"id": 1}]', encoding="utf-8")
    jsonl_path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    invalid_path.write_text('[{"id": 1}, 2]', encoding="utf-8")

    assert load_json_records(json_path) == [{"id": 1}]
    assert load_json_records(jsonl_path) == [{"id": 1}, {"id": 2}]
    with pytest.raises(ValueError, match="JSON array or JSONL stream"):
        load_json_records(invalid_path)
