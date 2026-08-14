# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.apex_agents.prepare import convert_task, prefetch_worlds


def test_convert_task_keeps_gold_in_verifier_metadata() -> None:
    task = {
        "task_id": "task-1",
        "task_name": "demo",
        "world_id": "world-1",
        "domain": "Finance",
        "prompt": "Build the model",
        "expected_output": "make_new_sheet",
        "gold_response": "secret gold",
        "gold_response_type": "text",
        "rubric": [{"verifier_id": "v1", "criteria": "Correct"}],
    }
    world = {
        "apps": [
            {"service_name": "FMP"},
            {"service_name": "Edgar SEC"},
            {"service_name": "Mail"},
        ]
    }

    row = convert_task(task, world)

    assert row["responses_create_params"]["input"][0]["content"] == "Build the model"
    assert row["foundry_services"] == ["edgar", "fmp"]
    assert "expected_output" not in row
    assert "gold_response" not in row["responses_create_params"]
    assert row["verifier_metadata"]["expected_output"] == "make_new_sheet"
    assert row["verifier_metadata"]["gold_response"] == "secret gold"
    assert row["verifier_metadata"]["rubric"][0]["grading_target"] == {
        "scope": "files",
        "expected_file_type": "Spreadsheets (.xlsx, .xls, .xlsm)",
        "extensions": [".csv", ".xls", ".xlsm", ".xlsx"],
    }


def test_prefetch_worlds_downloads_each_unique_world_once(monkeypatch, tmp_path) -> None:
    worlds_path = tmp_path / "worlds.json"
    worlds_path.write_text(
        json.dumps([{"world_id": "world-b"}, {"world_id": "world-a"}, {"world_id": "world-a"}]),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: calls.append(kwargs))

    count = prefetch_worlds(worlds_path, cache_dir=tmp_path / "cache", hf_token="hf_test_token")

    assert count == 2
    assert calls == [
        {
            "repo_id": "mercor/apex-agents",
            "filename": "world_files_zipped/world-a.zip",
            "repo_type": "dataset",
            "cache_dir": str(tmp_path / "cache"),
            "token": "hf_test_token",
        },
        {
            "repo_id": "mercor/apex-agents",
            "filename": "world_files_zipped/world-b.zip",
            "repo_type": "dataset",
            "cache_dir": str(tmp_path / "cache"),
            "token": "hf_test_token",
        },
    ]
