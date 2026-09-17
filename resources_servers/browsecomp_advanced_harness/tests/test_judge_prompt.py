# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The judge prompt's contract with the parser and the ordering its correctness rests on.

A prompt edit that drops `correct:` grades every row WRONG rather than failing loudly,
because the reply then does not parse. These pin the pieces that carry that risk.
"""

import os
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _server() -> TavilySearchResourcesServer:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider="exa",
        exa_api_key="test_exa_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
    )
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestTemplateShape:
    def test_formats_with_exactly_the_three_documented_fields(self) -> None:
        out = JUDGE_PROMPT_TEMPLATE.format(question="Q?", response="R", correct_answer="A")
        assert "Q?" in out and "R" in out and "A" in out
        assert "{" not in out and "}" not in out, "an unescaped brace would break .format at judge time"

    def test_asks_for_every_field_the_row_consumes(self) -> None:
        for field in ("answer_type:", "extracted_final_answer:", "reasoning:", "correct:", "confidence:"):
            assert field in JUDGE_PROMPT_TEMPLATE


class TestRuleOrdering:
    """R1/R2 must be evaluated before R3: accepting a spelling variant must not be able
    to admit a partial answer (R1) or a different entity (R2)."""

    def test_all_five_rules_are_present_and_in_order(self) -> None:
        positions = [JUDGE_PROMPT_TEMPLATE.index(f"R{n}.") for n in range(1, 6)]
        assert positions == sorted(positions)

    def test_missing_component_and_wrong_entity_precede_surface_form(self) -> None:
        assert JUDGE_PROMPT_TEMPLATE.index("R1.") < JUDGE_PROMPT_TEMPLATE.index("R3.")
        assert JUDGE_PROMPT_TEMPLATE.index("R2.") < JUDGE_PROMPT_TEMPLATE.index("R3.")

    def test_first_match_wins_is_stated(self) -> None:
        assert "IN ORDER" in JUDGE_PROMPT_TEMPLATE
        assert "FIRST rule" in JUDGE_PROMPT_TEMPLATE


class TestParserContract:
    """The prompt and `_parse_judge` are one unit: the reply shape the prompt asks for
    must be the shape the parser accepts."""

    def test_a_reply_in_the_requested_shape_parses(self) -> None:
        reply = (
            "answer_type: full birth name of a person\n"
            "extracted_final_answer: Jennifer Tour Chayes\n"
            "reasoning: R3 applies -- same person, the reference omits the middle name's accent.\n"
            "correct: yes\n"
            "confidence: 95%"
        )
        is_correct, extracted, parsed_ok = _server()._parse_judge(reply)
        assert parsed_ok is True
        assert is_correct is True
        assert extracted == "Jennifer Tour Chayes"

    def test_a_no_verdict_parses_as_incorrect(self) -> None:
        reply = "extracted_final_answer: Someone Else\nreasoning: R2 applies.\ncorrect: no\nconfidence: 80%"
        is_correct, _, parsed_ok = _server()._parse_judge(reply)
        assert parsed_ok is True and is_correct is False

    def test_a_reply_without_the_correct_line_does_not_parse(self) -> None:
        """Why the `correct:` field cannot be dropped from the template: the row would
        grade wrong rather than raising."""
        reply = "answer_type: a name\nextracted_final_answer: X\nreasoning: R1 applies.\nconfidence: 90%"
        _, _, parsed_ok = _server()._parse_judge(reply)
        assert parsed_ok is False
