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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.spartqa.app import (
    SpartqaResourcesServer,
    SpartqaResourcesServerConfig,
    SpartqaVerifyRequest,
    _extract_answer,
    _normalize,
    _response_text,
    _strip_reasoning,
)


_EXAMPLE_JSONL = Path(__file__).resolve().parent.parent / "data" / "example.jsonl"


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


def _config() -> SpartqaResourcesServerConfig:
    return SpartqaResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="spartqa")


def _make_request(
    text: str,
    *,
    target: str = "",
    all_targets: list[str] | None = None,
    verifier_metadata: dict | None = None,
) -> SpartqaVerifyRequest:
    return SpartqaVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        target=target,
        all_targets=all_targets or [],
        verifier_metadata=verifier_metadata,
    )


def _server() -> SpartqaResourcesServer:
    return SpartqaResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


# ── Pure helpers ─────────────────────────────────────────────────────────


class TestExtractAnswer:
    def test_pulls_phrase_after_final_answer(self) -> None:
        assert _extract_answer("Reasoning here.\nFinal answer: below the circle") == (
            "below the circle"
        )

    def test_strips_think_reasoning(self) -> None:
        text = "<think>the star is north</think>Final answer: yes"
        assert _extract_answer(text) == "yes"

    def test_thinking_process_prefix_returns_last_line(self) -> None:
        # No explicit "Final answer:" marker; a multi-line "thinking process"
        # prefix means the answer is the last non-empty line.
        text = "Thinking process: I consider the layout\ngreen cup"
        assert _extract_answer(text) == "green cup"

    def test_empty_returns_empty(self) -> None:
        assert _extract_answer("   ") == ""


class TestStripReasoning:
    def test_removes_think_block(self) -> None:
        assert _strip_reasoning("<think>hidden</think>visible") == "visible"

    def test_passthrough(self) -> None:
        assert _strip_reasoning("plain") == "plain"


class TestNormalize:
    def test_lowercases_strips_punctuation_collapses_ws(self) -> None:
        assert _normalize("  The  Big, BLACK Square!! ") == "the big black square"

    def test_empty(self) -> None:
        assert _normalize("") == ""


# ── verify() ─────────────────────────────────────────────────────────────


class TestVerify:
    async def test_exact_match(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: green cup", target="green cup")
        )
        assert result.reward == approx(1.0)
        assert result.exact is True
        assert result.parsed is True
        assert result.extracted == "green cup"

    async def test_contains_match_not_exact(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: the green cup on the table", target="green cup")
        )
        assert result.reward == approx(1.0)
        assert result.exact is False

    async def test_no_match(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: red ball", target="green cup")
        )
        assert result.reward == approx(0.0)
        assert result.exact is False
        assert result.parsed is True

    async def test_empty_output_reward_zero_no_raise(self) -> None:
        result = await _server().verify(_make_request("   ", target="green cup"))
        assert result.reward == approx(0.0)
        assert result.parsed is False

    async def test_all_targets_empty_falls_back_to_target(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: yes", target="yes", all_targets=[])
        )
        assert result.reward == approx(1.0)

    async def test_multiple_targets_any_match(self) -> None:
        result = await _server().verify(
            _make_request(
                "Final answer: under the circle",
                target="below the circle",
                all_targets=["below the circle", "under the circle"],
            )
        )
        assert result.reward == approx(1.0)
        assert result.exact is True

    async def test_empty_targets_are_skipped(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: yes", target="", all_targets=["", "   "])
        )
        assert result.reward == approx(0.0)

    async def test_all_targets_from_verifier_metadata(self) -> None:
        # The native driver drops the top-level ``all_targets`` list; the full
        # accepted set must be recoverable from verifier_metadata.
        result = await _server().verify(
            _make_request(
                "Final answer: under the circle",
                target="below the circle",
                all_targets=[],
                verifier_metadata={
                    "target": "below the circle",
                    "all_targets": ["below the circle", "under the circle"],
                },
            )
        )
        assert result.reward == approx(1.0)
        assert result.exact is True

    async def test_target_from_verifier_metadata_when_top_level_missing(self) -> None:
        result = await _server().verify(
            _make_request(
                "Final answer: yes",
                target="",
                all_targets=[],
                verifier_metadata={"target": "yes"},
            )
        )
        assert result.reward == approx(1.0)


# ── compute_metrics() / get_key_metrics() ──────────────────────────────────


class TestComputeMetrics:
    def test_mean_and_rates(self) -> None:
        tasks = [
            [{"reward": 1.0, "exact": True, "parsed": True}],
            [{"reward": 0.0, "exact": False, "parsed": True}],
            [{"reward": 1.0, "exact": False, "parsed": True}],
            [{"reward": 0.0, "exact": False, "parsed": False}],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["mean_reward"] == approx(0.5)
        assert metrics["count"] == 4
        assert metrics["exact_match_rate"] == approx(0.25)
        assert metrics["parse_rate"] == approx(0.75)

    def test_empty(self) -> None:
        assert _server().compute_metrics([]) == {}


class TestGetKeyMetrics:
    def test_selects_headline(self) -> None:
        out = _server().get_key_metrics(
            {"mean_reward": 0.5, "exact_match_rate": 0.25, "parse_rate": 0.75, "count": 4}
        )
        assert out == {"mean_reward": approx(0.5), "exact_match_rate": approx(0.25)}


# ── _response_text() ───────────────────────────────────────────────────────


class TestResponseText:
    def test_output_text_fast_path(self) -> None:
        assert _response_text(_make_response("hello")) == "hello"

    def test_fallback_joins_message_content(self) -> None:
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(text="a"), {"text": "b"}],
        )
        reasoning = SimpleNamespace(type="reasoning", content="ignored")
        response = SimpleNamespace(output_text=None, output=[reasoning, message])
        assert _response_text(response) == "ab"

    def test_fallback_string_content(self) -> None:
        message = SimpleNamespace(type="message", content="plain")
        response = SimpleNamespace(output_text="", output=[message])
        assert _response_text(response) == "plain"


# ── Acceptance / parity ────────────────────────────────────────────────────


class TestAcceptance:
    async def test_each_example_target_scores_one(self) -> None:
        server = _server()
        rows = [
            json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()
        ]
        assert len(rows) >= 5
        for row in rows:
            resp_text = f"Final answer: {row['target']}"
            result = await server.verify(
                _make_request(
                    resp_text, target=row["target"], all_targets=row.get("all_targets", [])
                )
            )
            assert result.reward == approx(1.0)

