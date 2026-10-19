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

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.geobench_vlm.app import (
    GeoBenchVLMResourcesServer,
    GeoBenchVLMResourcesServerConfig,
    GeoBenchVLMVerifyRequest,
)
from resources_servers.geobench_vlm.scoring import score_answer
from resources_servers.geobench_vlm.task_data import (
    PERCEPTION_TASKS,
    dataset_row,
    example_rows,
    load_qa_tree,
    materialize,
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
