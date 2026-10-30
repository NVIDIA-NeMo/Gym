# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate construction of the complete multimodal benchmark artifact."""

import base64
import json

import pytest

from benchmarks.vision_arena_v1 import prepare as preparation
from benchmarks.vision_arena_v1.build_validation import build_validation
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.arena.task_data import TaskData


pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")


@pytest.fixture
def inputs(tmp_path):
    prompts = [
        {
            "question_id": str(i),
            "cluster_name": "example",
            "turns": [[{"content": "What is shown?"}]],
            "images": [{"bytes": b"image", "path": "image.png"}],
        }
        for i in range(500)
    ]
    questions = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist(prompts), questions)
    rows = [
        {
            "question_id": str(i),
            "model": "baseline",
            "response": {
                "id": str(i),
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<think>private</think>A tree.", "reasoning_content": "secret"},
                    }
                ],
            },
        }
        for i in range(500)
    ]
    return questions, rows


def write_responses(tmp_path, rows):
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_build_embeds_images_and_only_final_answers(tmp_path, inputs):
    questions, rows = inputs
    # A failed retry must not erase a usable answer already saved for this question.
    rows.append({"question_id": "0", "error": "timeout"})
    output = tmp_path / "validation.jsonl"
    count = build_validation(questions=questions, responses=write_responses(tmp_path, rows), output=output)
    assert count == 500
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(result) == 500
    for row in result:
        assert row["baseline_answer"] == "A tree."
        TaskData.model_validate(row)
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
        image_url = params.input[0].content[1]["image_url"]
        assert base64.b64decode(image_url.split(",", 1)[1]) == b"image"
    assert "secret" not in output.read_text()
    with pytest.raises(FileExistsError):
        build_validation(questions=questions, responses=tmp_path / "responses.jsonl", output=output)


@pytest.mark.parametrize("problem", ["missing", "truncated", "empty", "error"])
def test_rejects_incomplete_baseline_set(tmp_path, inputs, problem):
    questions, rows = inputs
    if problem == "missing":
        rows.pop()
    elif problem == "truncated":
        rows[0]["response"]["choices"][0]["finish_reason"] = "length"
    elif problem == "empty":
        rows[0]["response"]["choices"][0]["message"]["content"] = ""
    else:
        rows[0]["error"] = "failed"
    output = tmp_path / "validation.jsonl"
    with pytest.raises(ValueError, match="Missing usable baseline"):
        build_validation(questions=questions, responses=write_responses(tmp_path, rows), output=output)
    assert not output.exists()


def test_rejects_wrong_prompt_ids(tmp_path, inputs):
    questions, rows = inputs
    rows[0]["question_id"] = "unknown"
    with pytest.raises(ValueError, match="absent from prompts"):
        build_validation(
            questions=questions, responses=write_responses(tmp_path, rows), output=tmp_path / "validation.jsonl"
        )


def test_malformed_second_prompt_does_not_publish_partial_dataset(tmp_path, inputs):
    questions, rows = inputs
    prompts = pq.read_table(questions).to_pylist()
    prompts[1]["turns"][0][0]["content"] = None
    pq.write_table(pa.Table.from_pylist(prompts), questions)
    responses = write_responses(tmp_path, rows)
    output = tmp_path / "validation.jsonl"
    with pytest.raises(TypeError):
        build_validation(questions=questions, responses=responses, output=output)
    assert not output.exists()
    assert {p.name for p in tmp_path.iterdir()} == {questions.name, responses.name}

    prompts[1]["turns"][0][0]["content"] = "Fixed question"
    pq.write_table(pa.Table.from_pylist(prompts), questions)
    assert build_validation(questions=questions, responses=responses, output=output) == 500


def test_explicit_partial_build_excludes_unusable_answers(tmp_path, inputs):
    questions, rows = inputs
    rows.pop()
    rows[0]["response"]["choices"][0]["finish_reason"] = "length"
    output = tmp_path / "validation.jsonl"
    count = build_validation(
        questions=questions, responses=write_responses(tmp_path, rows), output=output, allow_partial=True
    )
    assert count == 498
    ids = {json.loads(line)["question_id"] for line in output.read_text().splitlines()}
    assert "0" not in ids
    assert "499" not in ids


@pytest.mark.parametrize(
    "answer", ["A tree.", "<think>private analysis</think>A tree.", "<thinking>private</thinking>A tree."]
)
def test_prepare_local_answers_without_gitlab(tmp_path, inputs, monkeypatch, answer):
    questions, _ = inputs
    rows = [
        {
            "question_id": str(i),
            "baseline_model": "baseline",
            "baseline_answer": "A tree.",
            "baseline_response_id": f"response-{i}",
            "source": preparation.DATASET_ID,
            "source_revision": preparation.DATASET_REVISION,
        }
        for i in range(486)
    ]
    rows[0]["baseline_answer"] = answer
    responses = write_responses(tmp_path, rows)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(questions)

    monkeypatch.setattr(preparation, "hf_hub_download", download)
    output = tmp_path / "validation.jsonl"
    assert preparation.prepare(baseline_responses=responses, output_path=output) == output
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(result) == 486
    assert result[0]["metadata"]["baseline_response_id"] == "response-0"
    assert result[0]["baseline_answer"] == "A tree."
    assert "private analysis" not in output.read_text()
    assert calls[0]["revision"] == preparation.DATASET_REVISION
    with pytest.raises(FileExistsError):
        preparation.prepare(baseline_responses=responses, output_path=output)

    rows[0]["source_revision"] = "wrong"
    responses = write_responses(tmp_path, rows)
    with pytest.raises(ValueError, match="provenance"):
        preparation.prepare(baseline_responses=responses, output_path=tmp_path / "wrong.jsonl")
    rows.pop()
    responses = write_responses(tmp_path, rows)
    with pytest.raises(ValueError, match="486 distinct"):
        preparation.prepare(baseline_responses=responses, output_path=tmp_path / "short.jsonl")


def test_minimal_answers_reject_duplicate_ids(tmp_path, inputs):
    questions, _ = inputs
    rows = [{"question_id": "0", "baseline_model": "baseline", "baseline_answer": "A tree."}] * 2
    with pytest.raises(ValueError, match="Duplicate baseline"):
        build_validation(
            questions=questions,
            responses=write_responses(tmp_path, rows),
            output=tmp_path / "validation.jsonl",
            allow_partial=True,
        )


@pytest.mark.parametrize("answer", ["<think>unfinished", "<thinking>unfinished", "<think>reasoning only</think>"])
def test_answers_only_rejects_unusable_final_answer(tmp_path, inputs, monkeypatch, answer):
    questions, _ = inputs
    rows = [
        {
            "question_id": str(i),
            "baseline_model": "baseline",
            "baseline_answer": "A tree.",
            "source": preparation.DATASET_ID,
            "source_revision": preparation.DATASET_REVISION,
        }
        for i in range(486)
    ]
    rows[1]["baseline_answer"] = answer
    monkeypatch.setattr(preparation, "hf_hub_download", lambda **kwargs: str(questions))
    output = tmp_path / "validation.jsonl"
    with pytest.raises(ValueError, match="Invalid baseline final answer"):
        preparation.prepare(baseline_responses=write_responses(tmp_path, rows), output_path=output)
    assert not output.exists()
