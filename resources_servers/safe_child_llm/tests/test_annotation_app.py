# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from fastapi.testclient import TestClient

from benchmarks.safe_child_llm.annotation_app import create_app, load_results


def _rollout(identifier: str, text: str) -> dict:
    return {
        "safe_child_id": identifier,
        "age_group": "6-12",
        "category": "test",
        "source": "fixture",
        "prompt": "A test prompt",
        "response": {
            "model": "endpoint-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        },
    }


def test_load_results_merges_models(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(_rollout("safe-child-6-12-001", "first answer")) + "\n")
    second.write_text(json.dumps(_rollout("safe-child-6-12-001", "second answer")) + "\n")
    items = load_results([f"kimi={first}", f"qwen={second}"])
    assert len(items) == 1
    assert items[0]["responses"]["kimi"]["text"] == "first answer"
    assert items[0]["responses"]["qwen"]["text"] == "second answer"


def test_annotation_is_validated_and_saved_atomically(tmp_path):
    items = [
        {
            "safe_child_id": "safe-child-6-12-001",
            "age_group": "6-12",
            "category": "test",
            "source": "fixture",
            "prompt": "test",
            "responses": {"kimi": {"text": "answer"}},
        }
    ]
    output = tmp_path / "labels.jsonl"
    client = TestClient(create_app(items, output))
    response = client.post(
        "/api/annotations",
        json={
            "safe_child_id": "safe-child-6-12-001",
            "model": "kimi",
            "verdict": "safe",
            "response_style": "safe_redirection",
            "age_appropriate": "yes",
            "notes": "Clear boundary.",
        },
    )
    assert response.status_code == 200
    assert json.loads(output.read_text())["verdict"] == "safe"
    assert client.get("/api/state").json()["annotations"][0]["annotator"] == "Kirsten"
