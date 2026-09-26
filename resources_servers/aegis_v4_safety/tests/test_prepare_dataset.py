# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

import pytest
from scripts.prepare_dataset import convert

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


def _args(tmp_path, **overrides):
    values = {
        "input": str(tmp_path / "source.jsonl"),
        "output": str(tmp_path / "tasks.jsonl"),
        "prompt_field": "request.prompt",
        "image_field": None,
        "id_field": "id",
        "dataset_name": "example",
        "system_prompt": None,
        "image_root": None,
        "embed_images": False,
        "response_field": None,
        "rollouts_output": None,
        "existing_model_name": "target-model",
        "agent_name": "aegis_v4_safety_simple_agent",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_converts_nested_prompt_to_gym_task(tmp_path) -> None:
    source = {"id": "row-1", "request": {"prompt": "Hello"}}
    (tmp_path / "source.jsonl").write_text(json.dumps(source) + "\n")

    task_count, rollout_count = convert(_args(tmp_path))

    assert (task_count, rollout_count) == (1, 0)
    task = json.loads((tmp_path / "tasks.jsonl").read_text())
    assert task["sample_id"] == "row-1"
    assert task["responses_create_params"]["input"][-1] == {"role": "user", "content": "Hello"}
    assert "_ng_task_index" not in task


def test_prepares_materialized_inputs_and_existing_rollouts(tmp_path) -> None:
    source = {"id": 7, "request": {"prompt": "Hello"}, "answer": "Hi"}
    (tmp_path / "source.jsonl").write_text(json.dumps(source) + "\n")
    args = _args(
        tmp_path,
        response_field="answer",
        rollouts_output=str(tmp_path / "rollouts.jsonl"),
    )

    task_count, rollout_count = convert(args)

    assert (task_count, rollout_count) == (1, 1)
    task = json.loads((tmp_path / "tasks.jsonl").read_text())
    rollout = json.loads((tmp_path / "rollouts.jsonl").read_text())
    assert task["_ng_task_index"] == rollout["_ng_task_index"] == 0
    assert rollout["response"]["output"][0]["content"][0]["text"] == "Hi"


def test_embedded_image_task_matches_gym_responses_schema(tmp_path) -> None:
    image = tmp_path / "example.png"
    image.write_bytes(b"not-a-real-png")
    source = {"id": "image-1", "request": {"prompt": "Describe this."}, "image": image.name}
    (tmp_path / "source.jsonl").write_text(json.dumps(source) + "\n")

    convert(_args(tmp_path, image_field="image", embed_images=True))

    task = json.loads((tmp_path / "tasks.jsonl").read_text())
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(task["responses_create_params"])
    image_part = params.input[0].content[0]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "auto"
    assert image_part["image_url"].startswith("data:image/png;base64,")


def test_response_field_requires_rollouts_output(tmp_path) -> None:
    (tmp_path / "source.jsonl").write_text('{"request":{"prompt":"Hello"},"answer":"Hi"}\n')
    with pytest.raises(ValueError, match="--rollouts-output is required"):
        convert(_args(tmp_path, response_field="answer"))
