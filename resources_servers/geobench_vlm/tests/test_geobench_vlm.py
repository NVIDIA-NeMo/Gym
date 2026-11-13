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
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.geobench_vlm.app import (
    GeoBenchVLMResourcesServer,
    GeoBenchVLMResourcesServerConfig,
    GeoBenchVLMVerifyRequest,
)
from resources_servers.geobench_vlm.scoring import predicted_letter, score_answer
from resources_servers.geobench_vlm.task_data import (
    PERCEPTION_COUNTS,
    PERCEPTION_TASKS,
    dataset_row,
    example_rows,
    load_qa_tree,
    materialize,
    materialize_perception,
    perception_row,
)


def test_letter_match_accepts_the_official_short_answer() -> None:
    scored = score_answer("B", "B", "Harbor")
    assert scored["reward"] == 1.0
    assert scored["geobench_vlm_first_letter_correct"] is True


def test_sentence_answer_counts_when_the_letter_is_explicit() -> None:
    scored = score_answer("The choice is C.", "C.", "3")
    assert scored["predicted_letter"] == "C"
    assert scored["reward"] == 1.0


def test_wrong_letter_scores_zero() -> None:
    scored = score_answer("A. Forest", "B", "Harbor")
    assert scored["reward"] == 0.0
    assert scored["gold_letter"] == "B"


def test_first_character_rule_rejects_a_leading_word() -> None:
    scored = score_answer("Answer: D", "D", "Drought")
    assert scored["geobench_vlm_correct"] is True
    assert scored["geobench_vlm_first_letter_correct"] is False


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("\nB", "B"),
        ("**B**", "B"),
        ("(C)", "C"),
        ("[D]", "D"),
        ("C)", "C"),
        ("b", "B"),
        ("E. Nimitz-class aircraft carrier", "E"),
        ("The answer is **D**.", "D"),
        ("The answer is C, not option A.", "C"),
        ("The answer is not A. Final answer: C", "C"),
        ("I would pick option B", "B"),
        ("Considering the shapes, E.", "E"),
        ("<think>option A looks likely</think>C", "C"),
        ("reasoning without an opening tag </think>\nE", "E"),
        ("The answer is a harbor.", None),
        ("Based on the image, it is a harbor.", None),
        ("A harbor.", None),
        ("Answer: Harbor", None),
        ("", None),
    ],
)
def test_predicted_letter_reads_formatting_but_not_words(text: str, letter: str | None) -> None:
    assert predicted_letter(text) == letter


def test_example_rows_cover_single_and_temporal_prompts() -> None:
    rows = example_rows()
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"single", "temporal"}
    temporal = next(row for row in rows if row["split"] == "temporal")
    types = [block["type"] for block in temporal["responses_create_params"]["input"][0]["content"]]
    assert types.count("input_image") == 2
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(rows[0]["responses_create_params"])
    assert params.temperature == 0
    text = params.input[0].content[-1]["text"]
    assert "A., B., C., D. or E." in text


def test_qa_tree_expands_every_prompt(tmp_path) -> None:
    single = tmp_path / "Single"
    single.mkdir()
    (single / "qa.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 9,
                    "task": "Scene Classification",
                    "image_path": "images/a.jpg",
                    "prompts": ["one", "two"],
                    "options": "A. Forest B. Harbor",
                    "ground_truth": "Harbor",
                    "ground_truth_option": "B",
                    "cls_description": "overhead",
                }
            ]
        )
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"not-a-real-jpeg")
    rows = load_qa_tree(tmp_path)
    assert len(rows) == 2
    assert rows[0]["prompt_index"] == 0
    assert rows[1]["prompt_index"] == 1
    assert "one" in rows[0]["responses_create_params"]["input"][0]["content"][-1]["text"]


def test_materialize_writes_the_example_set(tmp_path) -> None:
    path = materialize(tmp_path / "example.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["1", "2", "3", "4", "5", "5"]
    assert dataset_row({"prompts": ["q"], "ground_truth_option": "A", "task": "t"})["task"] == "t"


def test_perception_row_puts_text_first_and_blind_drops_the_image() -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="BMP")
    record = {
        "question_id": 7,
        "task": PERCEPTION_TASKS[2],
        "prompts": ["What type of ship is visible in this image?", "unused"],
        "options": "A. Tanker   B. Yacht",
        "ground_truth": "Yacht",
        "ground_truth_option": "B",
        "image": {"bytes": buffer.getvalue()},
    }
    row = perception_row(record)
    content = row["responses_create_params"]["input"][0]["content"]
    assert [block["type"] for block in content] == ["input_text", "input_image"]
    assert content[0]["text"].endswith("A. Tanker   B. Yacht\nAnswer with only the option letter.")
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,/9j/")
    assert row["responses_create_params"]["max_output_tokens"] == 8192
    blind = perception_row(record, blind=True)
    assert [block["type"] for block in blind["responses_create_params"]["input"][0]["content"]] == ["input_text"]
    assert len(PERCEPTION_TASKS) == 8


def test_perception_row_rejects_a_question_without_an_image() -> None:
    record = {
        "question_id": 8,
        "task": PERCEPTION_TASKS[0],
        "prompts": ["q"],
        "options": "A. x",
        "ground_truth": "x",
        "ground_truth_option": "A",
    }
    with pytest.raises(ValueError, match="no image bytes"):
        perception_row(record)
    assert perception_row(record, blind=True)["question_id"] == "8"


def test_perception_builder_checks_the_pinned_file_and_task_counts(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = tmp_path / "Single.parquet"
    pq.write_table(pa.table({"task": [PERCEPTION_TASKS[0]], "question_id": [1]}), parquet)
    with pytest.raises(ValueError, match="expected 81bb88e1"):
        materialize_perception(parquet, tmp_path / "data")
    with pytest.raises(ValueError, match="task counts"):
        materialize_perception(parquet, tmp_path / "data", check_sha256=False)
    assert not (tmp_path / "data").exists()
    assert sum(PERCEPTION_COUNTS.values()) == 1557


def test_qa_tree_without_question_files_fails(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Single/qa.json"):
        load_qa_tree(tmp_path)


async def test_verify_scores_an_example_row() -> None:
    row = example_rows()[0]
    server = GeoBenchVLMResourcesServer(
        config=GeoBenchVLMResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "\nB", "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    result = await server.verify(GeoBenchVLMVerifyRequest(**row, response=response))
    assert result.reward == 1.0
    assert result.predicted_letter == "B"
    assert result.task == "Scene Classification"
    assert result.split == "single"
