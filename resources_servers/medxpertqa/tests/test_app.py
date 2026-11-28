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

from benchmarks.medxpertqa_text.prepare import convert
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.medxpertqa.app import (
    MedXpertQAResourcesServer,
    MedXpertQAResourcesServerConfig,
    answer_trigger,
    extract_answer,
)


LETTERS = list("ABCDEFGHIJ")
EXAMPLE_FPATH = Path(__file__).parents[1] / "data" / "example.jsonl"

# Replies and the letter TsinghuaC3I/MedXpertQA@5f1db162 eval/utils.py answer_cleansing
# ("zero_shot", the CoT trigger, options A-J) returns for them, recorded by running it.
UPSTREAM_PARITY_CASES = [
    ("E", "E"),
    ("E.", "E"),
    ("(E)", "E"),
    ("**E**", "E"),
    ("Therefore, among A through J, the answer is E.", "E"),
    ("Therefore, among A through J, the answer is (G) Myasthenia gravis.", "G"),
    ("The answer is C.", "C"),
    ("Based on the analysis, the correct answer is J", "J"),
    ("I understand. Among the choices, B is best.", "B"),
    ("The answer is A through J", None),
    ("Answer: D", "D"),
    ("Option H is correct; option A is wrong.", "H"),
    ("Esophageal varices are the cause.", None),
    ("e", None),
    ("", None),
    ("I cannot determine the answer.", "I"),
    ("Between B and C, I choose C.", "B"),
    ("The answer is not A, it is B.", "A"),
    ("Therefore, among A through J, the answer is\n\nF", "F"),
    ("K", None),
]


def _server(answer_format: str = "trigger") -> MedXpertQAResourcesServer:
    config = MedXpertQAResourcesServerConfig(host="", port=0, entrypoint="", name="", answer_format=answer_format)
    return MedXpertQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text: str, status: str = "completed") -> NeMoGymResponse:
    output = []
    if text:
        output.append(
            {
                "id": "msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        )
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
    )


def _row() -> dict:
    return json.loads(EXAMPLE_FPATH.read_text().splitlines()[0])


async def _episode(server: MedXpertQAResourcesServer, row: dict, rationale: str, answer: str) -> tuple:
    await server.reset(row, "sid")
    trigger_step = await server.step(_response(rationale), row, "sid")
    return trigger_step, await server.step(_response(answer), row, "sid")


class TestExtraction:
    @pytest.mark.parametrize(("reply", "expected"), UPSTREAM_PARITY_CASES)
    def test_matches_upstream_answer_cleansing(self, reply: str, expected: str | None) -> None:
        assert extract_answer(reply, LETTERS, "trigger") == expected

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("Reasoning.\\boxed{C} and later \\boxed{D}", "C"),
            ("\\boxed{K}", None),
            ("The answer is C.", None),
            (" B ", "B"),
            ("", None),
        ],
    )
    def test_boxed_matches_upstream_deepseek_r1_branch(self, reply: str, expected: str | None) -> None:
        assert extract_answer(reply, LETTERS, "boxed") == expected

    def test_unparsed_thinking_is_graded_like_upstream(self) -> None:
        assert extract_answer("<think>the answer is A</think>D", LETTERS, "trigger") == "A"

    def test_trigger_names_the_option_range(self) -> None:
        assert answer_trigger(10, "trigger") == "Therefore, among A through J, the answer is"
        assert answer_trigger(5, "boxed") == (
            "Put your final answer within \\boxed{}. Therefore, among A through E, the answer is"
        )


class TestEpisode:
    @pytest.mark.asyncio
    async def test_two_rounds_then_grade(self) -> None:
        server, row = _server(), _row()
        (observation, reward, terminated, _, _), final = await _episode(
            server, row, "Let me think...", f"Therefore, among A through J, the answer is {row['label']}."
        )
        assert (observation, reward, terminated) == ("Therefore, among A through J, the answer is", 0.0, False)
        _, reward, terminated, truncated, info = final
        assert (reward, terminated, truncated) == (1.0, True, False)
        assert info["question_id"] == row["id"]
        assert info["expected_answer"] == info["extracted_answer"] == row["label"]
        assert {key: info[key] for key in ("question_type", "medical_task", "body_system")} == {
            key: row[key] for key in ("question_type", "medical_task", "body_system")
        }
        assert "mask_sample" not in info

    @pytest.mark.asyncio
    async def test_wrong_and_absent_answers_score_zero_unmasked(self) -> None:
        server, row = _server(), _row()
        wrong = next(letter for letter in LETTERS if letter != row["label"])
        for answer, extracted in ((wrong, wrong), ("", None), ("Esophagitis.", None)):
            _, (_, reward, terminated, _, info) = await _episode(server, row, "", answer)
            assert (reward, terminated, info["extracted_answer"]) == (0.0, True, extracted)
            assert "mask_sample" not in info

    @pytest.mark.asyncio
    async def test_records_turn_statuses(self) -> None:
        server, row = _server(), _row()
        await server.reset(row, "sid")
        await server.step(_response("", status="incomplete"), row, "sid")
        _, _, _, _, info = await server.step(_response(row["label"]), row, "sid")
        assert info["turn_statuses"] == ["incomplete", "completed"]

    @pytest.mark.asyncio
    async def test_boxed_format_sends_boxed_trigger(self) -> None:
        server, row = _server("boxed"), _row()
        (observation, *_), (_, reward, *_) = await _episode(server, row, "", f"\\boxed{{{row['label']}}}")
        assert observation.startswith("Put your final answer within \\boxed{}.")
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_retried_first_step_resends_the_trigger(self) -> None:
        server, row = _server(), _row()
        _, info = await server.reset(row, "sid")
        assert info == {"supports_step_idempotency": True}
        first = row | {"_ng_step_request_id": "step-1"}
        assert (await server.step(_response("Thinking."), first, "sid"))[2] is False
        observation, _, terminated, _, _ = await server.step(_response("Thinking."), first, "sid")
        assert (observation, terminated) == ("Therefore, among A through J, the answer is", False)
        final = row | {"_ng_step_request_id": "step-2"}
        _, reward, terminated, _, info = await server.step(_response(row["label"]), final, "sid")
        assert (reward, terminated, info["turn_statuses"]) == (1.0, True, ["completed", "completed"])

    @pytest.mark.asyncio
    async def test_retried_final_step_replays_its_result(self) -> None:
        server, row = _server(), _row()
        await server.reset(row, "sid")
        await server.step(_response("Thinking."), row | {"_ng_step_request_id": "step-1"}, "sid")
        final = row | {"_ng_step_request_id": "step-2"}
        first = await server.step(_response(row["label"]), final, "sid")
        server.session_state.pop("sid")  # what _step_endpoint does after a terminal step
        assert await server.step(_response(row["label"]), final, "sid") == first
        other = await server.step(_response(row["label"]), row | {"_ng_step_request_id": "step-3"}, "sid")
        assert other[4]["failure_kind"] == "session_lost"

    @pytest.mark.asyncio
    async def test_lost_session_is_masked(self) -> None:
        _, reward, terminated, _, info = await _server().step(_response("A"), _row(), "unknown")
        assert (reward, terminated) == (0.0, True)
        assert (info["mask_sample"], info["failure_kind"]) == (True, "session_lost")

    @pytest.mark.asyncio
    async def test_row_without_label_is_masked(self) -> None:
        server, row = _server(), _row() | {"label": None}
        await server.reset(row, "sid")
        _, reward, terminated, _, info = await server.step(_response("A"), row, "sid")
        assert (reward, terminated, info["mask_sample"], info["failure_kind"]) == (0.0, True, True, "verifier_error")


class TestMetrics:
    def test_overall_and_subset_accuracy(self) -> None:
        def rollout(reward: float, question_type: str, answer: str | None) -> dict:
            info = {"question_type": question_type, "medical_task": "Diagnosis", "body_system": "Nervous"}
            return {"reward": reward, "info": info | {"extracted_answer": answer}}

        tasks = [
            [rollout(1.0, "Reasoning", "A")],
            [rollout(0.0, "Reasoning", None)],
            [rollout(1.0, "Understanding", "B")],
        ]
        server = _server()
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1/accuracy"] == pytest.approx(200 / 3)
        assert metrics["pass@1/no_answer"] == pytest.approx(100 / 3)
        assert metrics["question_type/Reasoning/pass@1/accuracy"] == 50.0
        assert metrics["question_type/Understanding/pass@1/accuracy"] == 100.0
        assert metrics["medical_task/Diagnosis/pass@1/accuracy"] == pytest.approx(200 / 3)
        key = server.get_key_metrics(metrics)
        assert key["pass@1/accuracy"] == metrics["pass@1/accuracy"]
        assert key["question_type/Reasoning/pass@1/accuracy"] == 50.0


class TestPrepare:
    def test_rejects_other_content(self) -> None:
        with pytest.raises(ValueError, match="SHA-256"):
            convert(b'{"id": "Text-0"}\n')
