# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json


def test_importer_emits_catalog_envelope_without_hidden_prompts(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.swe_together import prepare
    from environment_servers.swe_together import task
    from nemo_gym.base_resources_server import BaseRunRequest

    directory = tmp_path / "source/tasks/fixture"
    directory.mkdir(parents=True)
    toml = directory / "task.toml"
    toml.write_text('[environment]\ndocker_image = "ghcr.io/official/task:pin"\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"tasks": [{"task_id": "fixture", "files": {"task.toml": hashlib.sha256(toml.read_bytes()).hexdigest()}}]}
        )
    )
    monkeypatch.setattr(task, "MANIFEST", manifest)
    monkeypatch.setattr(prepare, "MANIFEST", manifest)
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                "fixture": {
                    "image": "ghcr.io/official/task:pin",
                    "image_digest": "sha256:" + "a" * 64,
                    "workdir": "/workspace",
                }
            }
        )
    )
    output = prepare.prepare(tmp_path / "source", images, tmp_path / "tasks.jsonl")
    row = json.loads(output.read_text())
    request = BaseRunRequest.model_validate(row)
    assert request.responses_create_params.input == []
    assert row["task_input"]["task_data"]["task_id"] == row["task_id"]["task_id"] == "fixture"
