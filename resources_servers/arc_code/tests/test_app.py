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
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.arc_code.app import (
    ArcCodeResourcesServer,
    ArcCodeResourcesServerConfig,
    ArcCodeVerifyRequest,
)
from resources_servers.arc_code.grid import is_grid, normalized_grid, parse_answer, unreadable_reason


EXAMPLE_PATH = Path(__file__).parent.parent / "data" / "example.jsonl"


def make_server() -> ArcCodeResourcesServer:
    config = ArcCodeResourcesServerConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="arc_code")
    return ArcCodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def verify_request(row: dict, assistant_texts: list) -> ArcCodeVerifyRequest:
    output = [
        {
            "id": f"m{i}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        for i, text in enumerate(assistant_texts)
    ]
    response = {
        "id": "r",
        "created_at": 1,
        "model": "m",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    return ArcCodeVerifyRequest.model_validate({**row, "response": response})


def grid_block(grid: list) -> str:
    return "<answer>\n" + "\n".join(" ".join(map(str, r)) for r in grid) + "\n</answer>"


ROWS = [json.loads(line) for line in EXAMPLE_PATH.read_text().splitlines()]


def test_example_rows_have_the_agreed_schema() -> None:
    assert len(ROWS) >= 2
    for row in ROWS:
        assert set(row) == {"task_id", "examples", "test_input", "expected_output", "responses_create_params"}
        assert is_grid(row["test_input"]) and is_grid(row["expected_output"])
        assert all(is_grid(p["input"]) and is_grid(p["output"]) for p in row["examples"])


@pytest.mark.asyncio
@pytest.mark.parametrize("row", ROWS)
async def test_true_answer_scores_one_and_perturbed_scores_zero(row: dict) -> None:
    server = make_server()
    good = await server.verify(
        verify_request(row, ["thinking...", f"The result:\n{grid_block(row['expected_output'])}"])
    )
    assert good.reward == 1.0 and good.extraction_successful and good.predicted_output == row["expected_output"]

    wrong = [r[:] for r in row["expected_output"]]
    wrong[0][0] = (wrong[0][0] + 1) % 10
    bad = await server.verify(verify_request(row, [grid_block(wrong)]))
    assert bad.reward == 0.0 and bad.extraction_successful


@pytest.mark.asyncio
async def test_only_the_last_assistant_message_and_last_block_count() -> None:
    row = ROWS[0]
    server = make_server()
    wrong = [[0]]
    text = grid_block(wrong) + "\nno wait\n" + grid_block(row["expected_output"])
    assert (await server.verify(verify_request(row, [text]))).reward == 1.0
    assert (
        await server.verify(verify_request(row, [grid_block(row["expected_output"]), "no answer here"]))
    ).reward == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "",
        "no block",
        "<answer></answer>",
        "<answer>\n1 2\n3\n</answer>",  # ragged
        "<answer>\n12 3\n</answer>",  # not single digits
        "<answer>\n1,2\n</answer>",
        "<answer>\n1 2\n\n3 4\n</answer>",  # blank line between rows
    ],
)
async def test_malformed_or_missing_answer_is_not_correct(text: str) -> None:
    row = {**ROWS[0], "expected_output": [[1, 2], [3, 4]]}
    result = await make_server().verify(verify_request(row, [text]))
    assert result.reward == 0.0


def test_grid_helpers() -> None:
    assert parse_answer("<answer>\n1 2\n3 4\n</answer>") == [[1, 2], [3, 4]]
    assert parse_answer("<answer>\n1 2\n3\n</answer>") is None
    assert normalized_grid([[1, 2], [3, 4]]) == "12\n34"
    assert unreadable_reason("<answer>\n1 2\n3\n</answer>") == "the rows have different lengths (1, 2 cells)"
    assert unreadable_reason("no block") is None
    assert not is_grid([[1, 10]]) and not is_grid([[1], [1, 2]]) and not is_grid([])
