# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from benchmarks.apex_agents.prepare import (
    EXCLUDED_WORLD_IDS,
    convert_task,
    prefetch_task_files,
    prefetch_worlds,
    prepare_rows,
)


def test_convert_task_keeps_gold_in_verifier_metadata() -> None:
    task = {
        "task_id": "task-1",
        "task_name": "demo",
        "world_id": "world-1",
        "task_input_files": "snap_0123456789abcdef0123456789abcdef",
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
    assert row["task_input_files"] == "snap_0123456789abcdef0123456789abcdef"
    assert "expected_output" not in row
    assert "gold_response" not in row["responses_create_params"]
    assert row["verifier_metadata"]["expected_output"] == "make_new_sheet"
    assert row["verifier_metadata"]["gold_response"] == "secret gold"
    assert row["verifier_metadata"]["rubric"][0]["grading_target"] == {
        "scope": "files",
        "expected_file_type": "Spreadsheets (.xlsx, .xls, .xlsm, .ods)",
        "extensions": [".xls", ".xlsm", ".xlsx", ".ods"],
    }


def test_prefetch_worlds_downloads_each_unique_world_once(monkeypatch, tmp_path) -> None:
    worlds_path = tmp_path / "worlds.json"
    worlds_path.write_text(
        json.dumps(
            [
                {"world_id": "world-b"},
                {"world_id": "world-a"},
                {"world_id": "world-a"},
                *({"world_id": world_id} for world_id in EXCLUDED_WORLD_IDS),
            ]
        ),
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


def test_prefetch_task_files_downloads_all_needed_attachments_once(monkeypatch, tmp_path) -> None:
    excluded_world_id = next(iter(EXCLUDED_WORLD_IDS))
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "task_b",
                    "world_id": "world-kept",
                    "task_input_files": "snap_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                },
                {
                    "task_id": "task_a",
                    "world_id": "world-kept",
                    "task_input_files": "snap_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                },
                {"task_id": "task_without_files", "world_id": "world-kept", "task_input_files": None},
                {
                    "task_id": "task_excluded",
                    "world_id": excluded_world_id,
                    "task_input_files": "snap_cccccccccccccccccccccccccccccccc",
                },
            ]
        ),
        encoding="utf-8",
    )
    snapshot_root = tmp_path / "snapshot"
    for task_id in ("task_a", "task_b"):
        task_dir = snapshot_root / "task_files" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "input.docx").write_bytes(b"input")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    count = prefetch_task_files(tasks_path, cache_dir=tmp_path / "cache", hf_token="hf_test_token")

    assert count == 2
    assert calls == [
        {
            "repo_id": "mercor/apex-agents",
            "repo_type": "dataset",
            "cache_dir": str(tmp_path / "cache"),
            "allow_patterns": ["task_files/task_a/**", "task_files/task_b/**"],
            "token": "hf_test_token",
        }
    ]


def test_prepare_rows_excludes_external_dependency_worlds_before_limit(tmp_path) -> None:
    excluded_world_id = next(iter(EXCLUDED_WORLD_IDS))
    tasks_path = tmp_path / "tasks.json"
    worlds_path = tmp_path / "worlds.json"
    output = tmp_path / "output.jsonl"
    tasks_path.write_text(
        json.dumps(
            [
                {"task_id": "excluded", "world_id": excluded_world_id, "prompt": "skip", "rubric": []},
                {"task_id": "included", "world_id": "world-kept", "prompt": "keep", "rubric": []},
            ]
        ),
        encoding="utf-8",
    )
    worlds_path.write_text(
        json.dumps([{"world_id": excluded_world_id}, {"world_id": "world-kept"}]),
        encoding="utf-8",
    )

    count = prepare_rows(tasks_path, worlds_path, output, limit=1)

    assert count == 1
    assert json.loads(output.read_text(encoding="utf-8"))["task_id"] == "included"
