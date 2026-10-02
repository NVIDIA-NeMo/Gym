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
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.longbench_v2_48k.app import (
    LongbenchResourcesServer,
    LongbenchResourcesServerConfig,
    LongbenchVerifyRequest,
    LongbenchVerifyResponse,
    extract_letter,
)
from resources_servers.longbench_v2_48k.prepare_longbench import (
    DEFAULT_MAX_PROMPT_TOKENS,
    REQUIRED_COLUMNS,
    check_columns,
    render_prompt,
    to_task,
    truncate_token_ids,
)


EXAMPLE_JSONL = Path(__file__).resolve().parent.parent / "data" / "example.jsonl"
EXAMPLE_ROLLOUTS_JSONL = EXAMPLE_JSONL.with_name("example_rollouts.jsonl")


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_request(text: str, *, expected_answer: str = "B", **fields: Any) -> LongbenchVerifyRequest:
    payload: dict[str, Any] = {
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]).model_dump(),
        "response": _make_response(text).model_dump(),
        "expected_answer": expected_answer,
    }
    payload.update(fields)
    return LongbenchVerifyRequest.model_validate(payload)


@pytest.fixture
def server() -> LongbenchResourcesServer:
    return LongbenchResourcesServer(
        config=LongbenchResourcesServerConfig(host="0.0.0.0", port=8071, entrypoint="", name="longbench"),
        server_client=MagicMock(spec=ServerClient),
    )


async def _verify(server: LongbenchResourcesServer, text: str, **fields: Any) -> LongbenchVerifyResponse:
    return await server.verify(_make_request(text, **fields))


def test_extracts_parens_form() -> None:
    assert extract_letter("The correct answer is (C).") == "C"


@pytest.mark.parametrize("text", ["*The correct answer is (D)*", "**The correct answer is (D)**"])
def test_extracts_parens_form_with_asterisks(text: str) -> None:
    assert extract_letter(text) == "D"


def test_extracts_no_parens_fallback() -> None:
    assert extract_letter("The correct answer is A, because the text says so.") == "A"


def test_leftmost_parens_match_wins_over_a_later_parens_match() -> None:
    text = "The correct answer is (A). On reflection, The correct answer is (D)."
    assert extract_letter(text) == "A"


def test_parens_form_beats_an_earlier_bare_form() -> None:
    text = "The correct answer is A. On reflection, The correct answer is (B)."
    assert extract_letter(text) == "B"


def test_no_match_returns_none() -> None:
    assert extract_letter("I think it is probably option C, but I am unsure.") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Option B seems right.", "B"])
async def test_unparseable_output_scores_zero_without_raising(server: LongbenchResourcesServer, text: str) -> None:
    result = await _verify(server, text, expected_answer="B")
    assert result.reward == 0.0
    assert result.extracted_answer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   \n\t "])
async def test_empty_output_scores_zero_no_raise(server: LongbenchResourcesServer, text: str) -> None:
    result = await _verify(server, text, expected_answer="A")
    assert result.reward == 0.0
    assert result.extracted_answer is None
    assert result.expected_answer == "A"


@pytest.mark.asyncio
async def test_correct_letter_scores_one(server: LongbenchResourcesServer) -> None:
    result = await _verify(server, "The correct answer is (B).", expected_answer=" b ")
    assert result.reward == 1.0
    assert result.extracted_answer == "B"
    assert result.expected_answer == "B"


@pytest.mark.asyncio
async def test_wrong_letter_scores_zero(server: LongbenchResourcesServer) -> None:
    result = await _verify(server, "The correct answer is (C).", expected_answer="B")
    assert result.reward == 0.0
    assert result.extracted_answer == "C"


@pytest.mark.asyncio
async def test_missing_expected_answer_scores_zero(server: LongbenchResourcesServer) -> None:
    result = await _verify(server, "The correct answer is (C).", expected_answer="")
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_metadata_echoed(server: LongbenchResourcesServer) -> None:
    result = await _verify(
        server,
        "The correct answer is (B).",
        _id="row-42",
        domain="Single-Document QA",
        sub_domain="Financial",
        difficulty="hard",
        length="medium",
        verifier_metadata={"choices": [{"A": "a"}, {"B": "b"}, {"C": "c"}, {"D": "d"}]},
    )
    dumped = result.model_dump()
    assert dumped["_id"] == "row-42"
    assert "row_id" not in dumped
    assert "choices" not in dumped
    assert dumped["domain"] == "Single-Document QA"
    assert dumped["sub_domain"] == "Financial"
    assert dumped["difficulty"] == "hard"
    assert dumped["length"] == "medium"
    assert dumped["verifier_metadata"]["choices"][1] == {"B": "b"}


def _example_rows() -> list[dict[str, Any]]:
    with EXAMPLE_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestGoldWithoutTopLevelExpectedAnswer:
    """A caller may forward every row field except the top-level ``expected_answer``."""

    @staticmethod
    def _request_without_top_level_gold(row: dict[str, Any], text: str) -> LongbenchVerifyRequest:
        body = {k: v for k, v in row.items() if k != "expected_answer"}
        body["response"] = _make_response(text).model_dump()
        return LongbenchVerifyRequest.model_validate(body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", _example_rows(), ids=lambda row: str(row["_id"]))
    async def test_gold_is_read_from_verifier_metadata(
        self, server: LongbenchResourcesServer, row: dict[str, Any]
    ) -> None:
        gold = row["verifier_metadata"]["expected_answer"]
        wrong = next(letter for letter in "ABCD" if letter != gold)

        right = await server.verify(self._request_without_top_level_gold(row, f"The correct answer is ({gold})."))
        miss = await server.verify(self._request_without_top_level_gold(row, f"The correct answer is ({wrong})."))

        assert right.reward == 1.0
        assert right.expected_answer == gold
        assert miss.reward == 0.0

    @pytest.mark.asyncio
    async def test_top_level_gold_wins_when_both_are_present(self, server: LongbenchResourcesServer) -> None:
        result = await _verify(
            server, "The correct answer is (A).", expected_answer="A", verifier_metadata={"expected_answer": "C"}
        )
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_no_gold_anywhere_scores_zero(self, server: LongbenchResourcesServer) -> None:
        result = await _verify(server, "The correct answer is (A).", expected_answer=None, verifier_metadata={})
        assert result.reward == 0.0

    def test_every_example_row_carries_gold_in_both_places(self) -> None:
        for row in _example_rows():
            assert row["verifier_metadata"]["expected_answer"] == row["expected_answer"]


class TestExampleData:
    def test_rows_present(self) -> None:
        assert len(_example_rows()) == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", _example_rows(), ids=lambda row: str(row["_id"]))
    async def test_gold_letter_scores_one(self, server: LongbenchResourcesServer, row: dict[str, Any]) -> None:
        gold = row["expected_answer"]
        fields = {key: row[key] for key in ("_id", "domain", "sub_domain", "difficulty", "length")}
        result = await _verify(
            server,
            f"The correct answer is ({gold}).",
            expected_answer=gold,
            verifier_metadata=row["verifier_metadata"],
            **fields,
        )
        assert result.reward == 1.0
        assert result.extracted_answer == gold

    def test_tag_coverage(self) -> None:
        rows = _example_rows()
        difficulties = {row["difficulty"] for row in rows}
        lengths = {row["length"] for row in rows}
        domains = {row["domain"] for row in rows}
        golds = {row["expected_answer"] for row in rows}
        assert {"easy", "hard"} <= difficulties
        assert len(lengths & {"short", "medium", "long"}) >= 2
        assert len(domains) >= 2
        assert len(golds) >= 2

    @pytest.mark.asyncio
    async def test_committed_rollouts_reproduce_under_verify(self, server: LongbenchResourcesServer) -> None:
        with EXAMPLE_ROLLOUTS_JSONL.open(encoding="utf-8") as fh:
            rollouts = [json.loads(line) for line in fh if line.strip()]
        assert len(rollouts) == 5
        assert {r["reward"] for r in rollouts} == {0.0, 1.0}
        for rollout in rollouts:
            result = await server.verify(LongbenchVerifyRequest.model_validate(rollout))
            assert result.reward == rollout["reward"]
            assert result.extracted_answer == rollout["extracted_answer"]

    def test_prompt_is_single_user_message(self) -> None:
        for row in _example_rows():
            messages: list[dict[str, Optional[str]]] = row["responses_create_params"]["input"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"].startswith("Please read the following text")
            assert messages[0]["content"].endswith('"The correct answer is (insert answer here)".')


EXAMPLE_SOURCE_ROWS: list[dict[str, Any]] = [
    {
        "_id": "example-single-doc-qa-001",
        "context": "The Voyager 1 probe launched on 5 September 1977. It crossed the heliopause in August 2012, becoming the first human-made object to enter interstellar space. Its twin, Voyager 2, launched earlier that year but on a slower trajectory.",
        "question": "Which probe was the first to enter interstellar space?",
        "choice_A": "Voyager 2",
        "choice_B": "Voyager 1",
        "choice_C": "Pioneer 10",
        "choice_D": "New Horizons",
        "answer": "B",
        "domain": "Single-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": "short",
    },
    {
        "_id": "example-single-doc-qa-002",
        "context": "Quarterly report. Revenue rose from 412 million to 508 million year over year. Operating expenses grew from 380 million to 401 million. A one-off legal settlement of 22 million is recorded under other expenses and is excluded from operating expenses. Net income before tax was 85 million.",
        "question": "By how much did operating expenses grow year over year?",
        "choice_A": "21 million",
        "choice_B": "43 million",
        "choice_C": "96 million",
        "choice_D": "22 million",
        "answer": "A",
        "domain": "Single-Document QA",
        "sub_domain": "Financial",
        "difficulty": "hard",
        "length": "medium",
    },
    {
        "_id": "example-multi-doc-qa-003",
        "context": "Article one: the city council approved the tram extension on Tuesday, with work starting in March.\n\nArticle two: the transport minister said federal funding for the tram extension would cover 60 percent of the cost.\n\nArticle three: local businesses on the route asked for compensation during construction, which the council has not yet agreed to.",
        "question": "Which claim is supported by the articles taken together?",
        "choice_A": "The council agreed to compensate businesses.",
        "choice_B": "Federal funding covers the entire cost.",
        "choice_C": "Construction is funded in part federally and starts in March.",
        "choice_D": "The tram extension was rejected by the council.",
        "answer": "C",
        "domain": "Multi-Document QA",
        "sub_domain": "Multi-news",
        "difficulty": "hard",
        "length": "long",
    },
    {
        "_id": "example-long-dialogue-004",
        "context": "User: book me a table for four on Friday.\nAssistant: booked at Osteria, 7pm.\nUser: actually make it six people.\nAssistant: updated to six at 7pm.\nUser: and move it to Saturday.\nAssistant: moved to Saturday, six people, 7pm.",
        "question": "What is the final state of the booking?",
        "choice_A": "Friday, four people, 7pm",
        "choice_B": "Saturday, six people, 7pm",
        "choice_C": "Saturday, four people, 8pm",
        "choice_D": "Friday, six people, 7pm",
        "answer": "B",
        "domain": "Long Dialogue History Understanding",
        "sub_domain": "Agent history QA",
        "difficulty": "easy",
        "length": "medium",
    },
    {
        "_id": "example-code-repo-005",
        "context": "def load(path):\n    with open(path) as fh:\n        return [json.loads(line) for line in fh]\n\ndef summarize(rows):\n    return sum(r['n'] for r in rows) / len(rows)\n\ndef main(path):\n    rows = load(path)\n    if not rows:\n        return 0.0\n    return summarize(rows)",
        "question": "What does main return for an empty file?",
        "choice_A": "It raises ZeroDivisionError.",
        "choice_B": "It raises FileNotFoundError.",
        "choice_C": "It returns 0.0.",
        "choice_D": "It returns None.",
        "answer": "C",
        "domain": "Long-context Code Understanding",
        "sub_domain": "Code repo QA",
        "difficulty": "hard",
        "length": "short",
    },
]


class TestPreparePipeline:
    @pytest.mark.parametrize("source", EXAMPLE_SOURCE_ROWS, ids=lambda row: str(row["_id"]))
    def test_example_prompt_is_byte_identical_to_render_prompt(self, source: dict[str, Any]) -> None:
        committed = {row["_id"]: row for row in _example_rows()}[source["_id"]]
        rendered = render_prompt(source)
        assert rendered == committed["responses_create_params"]["input"][0]["content"]

    def test_render_prompt_strips_its_substitutions(self) -> None:
        source = dict(EXAMPLE_SOURCE_ROWS[0])
        padded = {key: f"  {value}  " if key != "_id" else value for key, value in source.items()}
        assert render_prompt(padded) == render_prompt(source)

    def test_render_prompt_leaves_no_placeholder_behind(self) -> None:
        rendered = render_prompt(EXAMPLE_SOURCE_ROWS[0])
        for placeholder in ("$DOC$", "$Q$", "$C_A$", "$C_B$", "$C_C$", "$C_D$"):
            assert placeholder not in rendered

    def test_to_task_uppercases_gold_at_top_level_and_nests_choices(self) -> None:
        source = dict(EXAMPLE_SOURCE_ROWS[0], answer=" b ")
        task = to_task(source)
        assert task["expected_answer"] == "B"
        assert "choices" not in task
        assert task["verifier_metadata"]["choices"] == [
            {"A": source["choice_A"]},
            {"B": source["choice_B"]},
            {"C": source["choice_C"]},
            {"D": source["choice_D"]},
        ]

    def test_to_task_matches_the_committed_example_row(self) -> None:
        source = EXAMPLE_SOURCE_ROWS[0]
        committed = {row["_id"]: row for row in _example_rows()}[source["_id"]]
        assert to_task(source) == committed

    def test_to_task_uses_a_supplied_prompt_instead_of_rendering(self) -> None:
        source = EXAMPLE_SOURCE_ROWS[0]
        task = to_task(source, "already truncated")
        assert task["responses_create_params"]["input"][0]["content"] == "already truncated"

    def test_check_columns_accepts_the_full_column_set(self) -> None:
        check_columns(list(REQUIRED_COLUMNS) + ["extra"])

    def test_check_columns_exits_on_a_missing_column(self) -> None:
        columns = [name for name in REQUIRED_COLUMNS if name != "context"]
        with pytest.raises(SystemExit) as excinfo:
            check_columns(columns)
        assert "context" in str(excinfo.value)


class TestTruncation:
    def test_prompt_within_budget_is_returned_unchanged(self) -> None:
        ids = list(range(DEFAULT_MAX_PROMPT_TOKENS))
        assert truncate_token_ids(ids, DEFAULT_MAX_PROMPT_TOKENS) == ids

    def test_oversized_prompt_keeps_both_ends_and_drops_the_middle(self) -> None:
        ids = list(range(300_000))
        out = truncate_token_ids(ids, DEFAULT_MAX_PROMPT_TOKENS)
        half = DEFAULT_MAX_PROMPT_TOKENS // 2
        assert len(out) == DEFAULT_MAX_PROMPT_TOKENS
        assert out[:half] == ids[:half]
        assert out[half:] == ids[-half:]

    @pytest.mark.parametrize("max_len", [1, 2, 3, 7, 8, 119_799, 119_800, 119_801])
    def test_result_is_exactly_max_len_long_for_odd_and_even_budgets(self, max_len: int) -> None:
        """The tail slice negates before dividing, so an odd budget keeps the extra token."""
        out = truncate_token_ids(list(range(1_000_000)), max_len)
        assert len(out) == max_len

    def test_a_row_one_token_over_budget_is_truncated(self) -> None:
        ids = list(range(DEFAULT_MAX_PROMPT_TOKENS + 1))
        out = truncate_token_ids(ids, DEFAULT_MAX_PROMPT_TOKENS)
        assert len(out) == DEFAULT_MAX_PROMPT_TOKENS
        assert out != ids
