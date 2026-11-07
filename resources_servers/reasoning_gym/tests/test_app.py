# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest
import reasoning_gym

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.reasoning_gym.app import (
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
)


class TestApp:
    @pytest.fixture
    def config(self) -> ReasoningGymResourcesServerConfig:
        return ReasoningGymResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @pytest.fixture
    def server(self, config) -> ReasoningGymResourcesServer:
        return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_response(self, text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=msg_id,
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=text,
                            annotations=[],
                        )
                    ],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_correct_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text=entry["answer"], msg_id="test_msg_1")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward >= 0.9, f"Expected high reward for correct answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_incorrect_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text="This is completely wrong", msg_id="test_msg_2")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward <= 0.1, f"Expected low reward for incorrect answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_multiple_tasks(self, server):
        tasks_to_test = ["knights_knaves", "leg_counting", "basic_arithmetic"]

        for task_name in tasks_to_test:
            dataset = reasoning_gym.create_dataset(task_name, size=1, seed=42)
            entry = dataset[0]

            response = self._create_response(text=entry["answer"], msg_id=f"test_msg_{task_name}")

            verify_request = ReasoningGymVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[{"role": "user", "content": entry["question"]}]
                ),
                response=response,
                question=entry["question"],
                answer=entry["answer"],
                metadata=entry["metadata"],
            )

            verify_response = await server.verify(verify_request)

            assert verify_response.reward >= 0.9, f"Task {task_name} failed: reward={verify_response.reward}"
            assert verify_response.task_name == task_name


def _knights_knaves_entry() -> tuple[dict, str, str]:
    """A fixed knights_knaves entry with its gold answer and a fully inverted (wrong) answer."""
    entry = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)[0]
    gold = entry["answer"]
    terms = entry["metadata"]["knight_knave_terms"]
    wrong = (
        gold.replace(terms["a_knight"], "\0")
        .replace(terms["a_knave"], terms["a_knight"])
        .replace("\0", terms["a_knave"])
    )
    assert wrong != gold
    return entry, gold, wrong


def _rotate_matrix_entry() -> tuple[dict, str]:
    """A fixed rotate_matrix entry: its gold answer is a grid of newline-separated rows, scored by exact match."""
    entry = reasoning_gym.create_dataset("rotate_matrix", size=1, seed=42)[0]
    gold = entry["answer"]
    assert "\n" in gold
    return entry, gold


def _entry_with_gold(name: str, predicate, max_seed: int = 400) -> tuple[dict, str]:
    """The first entry of dataset ``name`` (seeds 0..max_seed) whose gold answer satisfies ``predicate``."""
    for seed in range(max_seed):
        entry = reasoning_gym.create_dataset(name, size=1, seed=seed)[0]
        if predicate(entry["answer"]):
            return entry, entry["answer"]
    raise AssertionError(f"no {name} gold in {max_seed} seeds satisfies the predicate")


def _message(index: int, text: str) -> dict:
    return {
        "id": f"msg_{index}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _tool_round_trip(index: int, status: str) -> list[dict]:
    """A DeepAgents ``write_todos`` call and its output, as nemo_fabric_agent emits them between messages."""
    return [
        {
            "id": f"fc_{index}",
            "type": "function_call",
            "call_id": f"call_{index}",
            "name": "write_todos",
            "arguments": f'{{"todos": [{{"content": "Analyze the puzzle", "status": "{status}"}}]}}',
            "status": "completed",
        },
        {
            "id": f"fco_{index}",
            "type": "function_call_output",
            "call_id": f"call_{index}",
            "output": f"Updated todo list to [{{'content': 'Analyze the puzzle', 'status': '{status}'}}]",
            "status": "completed",
        },
    ]


def _verify_request(entry: dict, output: list[dict]) -> ReasoningGymVerifyRequest:
    response = NeMoGymResponse(
        id="resp_fixed",
        created_at=0.0,
        model="fixed-text-control",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return ReasoningGymVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": entry["question"]}]
        ),
        response=response,
        question=entry["question"],
        answer=entry["answer"],
        metadata=entry["metadata"],
    )


class TestAnswerExtraction:
    """Deterministic trajectories from NVBug 6900506: the answer the agent ended on must be the one scored."""

    @pytest.fixture
    def server(self) -> ReasoningGymResourcesServer:
        config = ReasoningGymResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @staticmethod
    def _narration(gold: str, wrong: str) -> str:
        return f"{wrong}\n\nReasoning:\n- Suppose otherwise ... contradiction.\n\nSo only: {gold}\n\n"

    @pytest.mark.asyncio
    async def test_multi_message_trajectory_scores_the_final_message(self, server):
        # Case A: nemo_fabric_agent's item order for a DeepAgents run that plans with write_todos.
        entry, gold, wrong = _knights_knaves_entry()
        output = [
            _message(0, "\n\n"),
            *_tool_round_trip(1, "in_progress"),
            _message(2, self._narration(gold, wrong)),
            *_tool_round_trip(3, "completed"),
            _message(4, gold),
        ]
        result = await server.verify(_verify_request(entry, output))
        assert result.extracted_answer == gold
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_multipart_final_message_keeps_every_content_part(self, server):
        # One assistant message whose answer is split over two output_text parts: the parts are fragments of one
        # reply, so the whole message is scored, not its last part.
        entry, gold, wrong = _knights_knaves_entry()
        split = len(gold) // 2
        message = _message(0, gold[:split])
        message["content"].append({"type": "output_text", "text": gold[split:], "annotations": []})
        result = await server.verify(_verify_request(entry, [message]))
        assert result.extracted_answer == gold
        assert result.reward == 1.0

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A single untagged message with reasoning before the answer has no delimiter the verifier can use; "
            "the environment must ask for <answer> tags (environments/nemo_fabric_deepagents_reasoning_gym/config.yaml)."
        ),
    )
    async def test_single_message_reasoning_then_answer_line_is_not_recoverable(self, server):
        # Case B: Gym's shipped nemo_fabric_deepagents_reasoning_gym example_rollouts have this shape.
        entry, gold, wrong = _knights_knaves_entry()
        output = [_message(0, f"Suppose {wrong[:-1]}. Then the first statement is false: contradiction.\n\n{gold}")]
        result = await server.verify(_verify_request(entry, output))
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_multi_message_trajectory_with_tagged_final_answer(self, server):
        # Case C: the same trajectory with the tag instruction the sibling environments use.
        entry, gold, wrong = _knights_knaves_entry()
        output = [
            _message(0, "\n\n"),
            *_tool_round_trip(1, "in_progress"),
            _message(2, self._narration(gold, wrong)),
            *_tool_round_trip(3, "completed"),
            _message(4, f"<answer>{gold}</answer>"),
        ]
        result = await server.verify(_verify_request(entry, output))
        assert result.extracted_answer == gold
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_bare_gold_control(self, server):
        # Case D.
        entry, gold, _ = _knights_knaves_entry()
        result = await server.verify(_verify_request(entry, [_message(0, gold)]))
        assert result.extracted_answer == gold
        assert result.reward == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label, build",
        [
            ("answer_tags", lambda gold, wrong: [_message(0, f"<answer>{gold}</answer>")]),
            ("boxed_plain", lambda gold, wrong: [_message(0, f"\\boxed{{{gold}}}")]),
            ("boxed_latex_spaces", lambda gold, wrong: [_message(0, "\\boxed{" + gold.replace(" ", "\\ ") + "}")]),
            ("boxed_text_wrapper", lambda gold, wrong: [_message(0, f"$$\n\\boxed{{\\text{{{gold}}}}}\n$$")]),
            ("boxed_dollar_inside", lambda gold, wrong: [_message(0, f"\\boxed{{${gold}$}}")]),
            (
                "boxed_draft_then_final_same_message",
                lambda gold, wrong: [_message(0, f"Draft: \\boxed{{{wrong}}}. Final: \\boxed{{{gold}}}")],
            ),
            (
                "boxed_draft_message_then_final_message",
                lambda gold, wrong: [
                    _message(0, f"Draft: \\boxed{{{wrong}}}"),
                    _message(1, f"Final: \\boxed{{{gold}}}"),
                ],
            ),
        ],
    )
    async def test_boxed_answers_score_like_tagged_answers(self, server, label, build):
        # Hermes and 11 sibling environments tell the model it may answer in \boxed{...}.
        entry, gold, wrong = _knights_knaves_entry()
        result = await server.verify(_verify_request(entry, build(gold, wrong)))
        assert result.extracted_answer == gold, label
        assert result.reward == 1.0, label

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label, build",
        [
            ("boxed_rows", lambda gold: f"\\boxed{{{gold}}}"),
            (
                "boxed_rows_indented_block",
                lambda gold: "\\boxed{\n" + "\n".join("    " + row for row in gold.splitlines()) + "\n}",
            ),
            ("boxed_rows_latex_spaces", lambda gold: "\\boxed{" + gold.replace(" ", "\\ ") + "}"),
        ],
    )
    async def test_boxed_grid_answers_keep_their_rows(self, server, label, build):
        # rotate_matrix, binary_matrix and pool_matrix oracles are newline-separated rows scored by exact match, so
        # the box normalization may remove a common indent but never join rows.
        entry, gold = _rotate_matrix_entry()
        result = await server.verify(_verify_request(entry, [_message(0, build(gold))]))
        assert result.extracted_answer == gold, label
        assert result.reward == 1.0, label

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name, predicate",
        [
            ("futoshiki", lambda gold: "   " in gold),  # cells and ∧/∨ markers aligned with runs of spaces
            ("emoji_mystery", lambda gold: "  " in gold),  # a sentence with a double space
        ],
    )
    async def test_boxed_answers_keep_runs_of_spaces(self, server, name, predicate):
        # Runs of spaces inside a line are part of these oracles: a boxed gold must score like the tagged gold.
        entry, gold = _entry_with_gold(name, predicate)
        boxed = await server.verify(_verify_request(entry, [_message(0, f"\\boxed{{{gold}}}")]))
        tagged = await server.verify(_verify_request(entry, [_message(0, f"<answer>{gold}</answer>")]))
        assert tagged.reward == 1.0, name
        assert boxed.reward == tagged.reward, name

    def test_last_balanced_box_wins_and_latex_text_is_normalized(self, server):
        extract = server._extract_answer_from_response
        gold = "Zoey is a fool, and Riley is a sage."

        def response_of(*texts: str):
            return NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[_message(i, t) for i, t in enumerate(texts)],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )

        assert extract(response_of("\\boxed{\\textbf{\\text{" + gold + "}}}")) == gold
        assert extract(response_of("\\boxed{x} then \\boxed{\\mathrm{" + gold + "}}")) == gold
        assert extract(response_of("\\boxed{unbalanced", gold)) == gold
        # \, is a space between words but a thousands separator between digits.
        assert extract(response_of("\\boxed{" + gold.replace(" ", "\\,") + "}")) == gold
        assert extract(response_of("\\boxed{1\\,000}")) == "1000"
        # Dedent and right-strip only: runs of spaces inside a line (codeio string outputs such as "L   L") and the
        # relative indent of later lines are part of the answer.
        assert extract(response_of("\\boxed{\n    L\n    L L\n    L   L\n}")) == "L\nL L\nL   L"
        assert extract(response_of("\\boxed{a  b}")) == "a  b"
        assert extract(response_of("\\boxed{\n    x\n      y\n}")) == "x\n  y"
        assert extract(response_of("first message", "", "   ")) == "first message"
