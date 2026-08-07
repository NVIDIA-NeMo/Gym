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
"""Acceptance tests for the SpartQA gym benchmark.

These verify the approved user story's acceptance criteria against the built
implementation. They are independent of the builder's unit tests
(``tests/test_app.py``): fixtures are re-declared here and data files are loaded
by path relative to this test file. One ``test_ac_*`` per acceptance criterion,
one ``test_edge_*`` per story edge case.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest
import yaml
from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.spartqa import app as spartqa_app
from resources_servers.spartqa.app import (
    BOTH_LABEL,
    NONE_LABEL,
    PROMPT,
    SimpleResourcesServer,
    SpartqaResourcesServer,
    SpartqaResourcesServerConfig,
    SpartqaVerifyRequest,
    SpartqaVerifyResponse,
    _clean_candidate,
    _extract_answer,
    _label_key,
    _normalize,
    _strip_reasoning,
    candidate_labels,
    match_label,
)


# ── Paths (relative to this test file) ─────────────────────────────────────

_SERVER_DIR = Path(__file__).resolve().parent.parent
_EXAMPLE_JSONL = _SERVER_DIR / "data" / "example.jsonl"
_ROLLOUTS_JSONL = _SERVER_DIR / "data" / "example_rollouts.jsonl"
_CONFIG_YAML = _SERVER_DIR / "configs" / "spartqa.yaml"
_PREPARE_PY = _SERVER_DIR / "prepare_spartqa.py"


# ── Fixture builders (mirror tests/test_app.py idioms) ─────────────────────


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


def _make_request(text: str, *, target: str = "", options: list[str] | None = None) -> SpartqaVerifyRequest:
    return SpartqaVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        target=target,
        options=options or [],
    )


# A representative CO row: two story objects plus the two fixed labels.
_OPTIONS = ["a big blue square", "a small blue square"]


def _server() -> SpartqaResourcesServer:
    return SpartqaResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


def _read_jsonl(path: Path) -> List[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_prepare_module() -> Any:
    """Load ``prepare_spartqa.py`` (which does ``from app import PROMPT``).

    The prepare script imports the sibling ``app`` module by bare name, so the
    already-imported ``resources_servers.spartqa.app`` is registered under
    ``app`` for the duration of the load, then removed to avoid polluting
    ``sys.modules`` for other servers' prepare scripts.
    """
    saved = sys.modules.get("app")
    sys.modules["app"] = spartqa_app
    try:
        spec = importlib.util.spec_from_file_location("spartqa_prepare", _PREPARE_PY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is not None:
            sys.modules["app"] = saved
        else:
            sys.modules.pop("app", None)


# ── AC1: server subclasses SimpleResourcesServer, async verify -> reward ────


class TestAcServerContract:
    def test_ac_subclasses_simple_resources_server(self) -> None:
        assert issubclass(SpartqaResourcesServer, SimpleResourcesServer)

    async def test_ac_verify_is_async_and_returns_reward(self) -> None:
        result = await _server().verify(
            _make_request(f"Final answer: {BOTH_LABEL}", target=BOTH_LABEL, options=_OPTIONS)
        )
        assert isinstance(result, SpartqaVerifyResponse)
        assert hasattr(result, "reward")
        assert result.reward == approx(1.0)


# ── AC2: scoring correctness ─────────────────────────────────────────────────

# Shared fixtures: (response_text, gold_label, options, expected_correct).
_PARITY_FIXTURES: list[tuple[str, str, list[str], bool]] = [
    # Verbatim candidate.
    ("Final answer: a big blue square", _OPTIONS[0], _OPTIONS, True),
    # Candidate recovered from a longer sentence.
    ("Final answer: it is a big blue square.", _OPTIONS[0], _OPTIONS, True),
    # The other candidate.
    ("Final answer: a small blue square", _OPTIONS[0], _OPTIONS, False),
    # "both of them" gold is NOT earned by naming one object.
    (f"Final answer: {_OPTIONS[0]}", BOTH_LABEL, _OPTIONS, False),
    # Echoing the question's options is ambiguous.
    (f"Final answer: {_OPTIONS[0]} or {_OPTIONS[1]}", _OPTIONS[0], _OPTIONS, False),
    # Aliases of the two fixed labels.
    ("Final answer: Both", BOTH_LABEL, _OPTIONS, True),
    ("Final answer: neither", NONE_LABEL, _OPTIONS, True),
    ("   ", _OPTIONS[0], _OPTIONS, False),
    (f"<think>reasoning</think>Final answer: {BOTH_LABEL}", BOTH_LABEL, _OPTIONS, True),
]


class TestAcMetricParity:
    def test_ac_prompt_text_contract(self) -> None:
        assert "Final answer: <candidate answer>" in PROMPT
        assert PROMPT.startswith("Answer the spatial reasoning question below.")
        assert "{question}" in PROMPT
        # The four candidates must be rendered, else "both of them" / "none of
        # them" are unreachable answers.
        assert PROMPT.rstrip().endswith("{candidates}")

    def test_ac_normalize_logic(self) -> None:
        assert _normalize("  The  Big, BLACK Square!! ") == "the big black square"
        assert _normalize("") == ""

    def test_ac_strip_reasoning_logic(self) -> None:
        assert _strip_reasoning("<think>hidden</think>visible") == "visible"
        assert _strip_reasoning("plain") == "plain"

    def test_ac_clean_candidate_logic(self) -> None:
        assert _clean_candidate("- *green cup*") == "green cup"
        assert _clean_candidate('  "yes"  ') == "yes"

    def test_ac_extract_answer_logic(self) -> None:
        assert _extract_answer("Reasoning.\nFinal answer: below the circle") == "below the circle"
        assert _extract_answer("<think>x</think>Final answer: yes") == "yes"
        assert _extract_answer("   ") == ""

    @pytest.mark.parametrize("text,target,options,expected", _PARITY_FIXTURES)
    async def test_ac_scoring_correctness(self, text: str, target: str, options: list[str], expected: bool) -> None:
        result = await _server().verify(_make_request(text, target=target, options=options))
        assert bool(result.reward) is expected


# ── AC3: reward is strictly 1.0 or 0.0 ─────────────────────────────────────


class TestAcBinaryReward:
    @pytest.mark.parametrize(
        "text,target,options",
        [(f[0], f[1], f[2]) for f in _PARITY_FIXTURES]
        + [("no marker at all", _OPTIONS[0], _OPTIONS), ("", _OPTIONS[0], _OPTIONS)],
    )
    async def test_ac_reward_is_binary(self, text: str, target: str, options: list[str]) -> None:
        result = await _server().verify(_make_request(text, target=target, options=options))
        assert result.reward in {0.0, 1.0}


# ── AC4: options read from field / verifier_metadata; falls back to [target] ─


class TestAcCandidateSet:
    async def test_ac_uses_options_field(self) -> None:
        # The non-gold option must resolve to itself, not to the gold.
        result = await _server().verify(
            _make_request(f"Final answer: {_OPTIONS[1]}", target=_OPTIONS[0], options=_OPTIONS)
        )
        assert result.reward == approx(0.0)
        assert result.predicted_label == _OPTIONS[1]

    async def test_ac_reads_options_from_verifier_metadata(self) -> None:
        # The native driver drops list fields; verifier_metadata is the fallback.
        request = SpartqaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(f"Final answer: {BOTH_LABEL}"),
            target="",
            verifier_metadata={"target": BOTH_LABEL, "options": _OPTIONS},
        )
        result = await _server().verify(request)
        assert result.reward == approx(1.0)

    async def test_ac_falls_back_to_target_when_options_absent(self) -> None:
        request = SpartqaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(f"Final answer: {_OPTIONS[0]}"),
            target=_OPTIONS[0],
        )
        result = await _server().verify(request)
        assert result.options == []
        assert result.reward == approx(1.0)


# ── AC5: response carries exact, parsed, extracted extras ───────────────────


class TestAcExtraFields:
    async def test_ac_extra_fields_present(self) -> None:
        result = await _server().verify(
            _make_request(f"Final answer: {_OPTIONS[0]}", target=_OPTIONS[0], options=_OPTIONS)
        )
        assert result.exact is True
        assert result.parsed is True
        assert result.extracted == _OPTIONS[0]
        assert result.predicted_label == _OPTIONS[0]

    async def test_ac_exact_false_when_recovered_from_a_sentence(self) -> None:
        result = await _server().verify(
            _make_request(
                f"Final answer: it is {_OPTIONS[0]} on the left.",
                target=_OPTIONS[0],
                options=_OPTIONS,
            )
        )
        assert result.reward == approx(1.0)
        assert result.exact is False
        assert result.parsed is True


# ── AC6: empty / whitespace output -> reward 0.0, no exception ───────────────


class TestAcEmptyOutput:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    async def test_ac_empty_output_scores_zero(self, text: str) -> None:
        result = await _server().verify(_make_request(text, target=_OPTIONS[0], options=_OPTIONS))
        assert result.reward == approx(0.0)
        assert result.parsed is False
        assert result.exact is False


# ── AC7: example.jsonl row shape ────────────────────────────────────────────


class TestAcExampleDataset:
    def test_ac_example_rows_conform(self) -> None:
        rows = _read_jsonl(_EXAMPLE_JSONL)
        assert len(rows) >= 5
        for row in rows:
            params = row["responses_create_params"]
            messages = params["input"]
            assert isinstance(messages, list) and messages
            assert any(m.get("role") == "user" for m in messages)
            assert isinstance(row["target"], str) and row["target"]
            assert isinstance(row["options"], list) and len(row["options"]) == 2
            assert row["verifier_metadata"]["options"] == row["options"]
            assert row["agent_ref"]["name"] == "spartqa_simple_agent"
            # The four candidates must be visible to the model.
            content = messages[0]["content"]
            for label in [*row["options"], BOTH_LABEL, NONE_LABEL]:
                assert label in content
            # The gold must be one of the four candidates.
            assert _label_key(row["target"]) in {_label_key(c) for c in candidate_labels(row["options"])}


# ── AC8: config wires server + agent ────────────────────────────────────────


class TestAcConfig:
    def test_ac_config_parses_and_wires_server_and_agent(self) -> None:
        config = yaml.safe_load(_CONFIG_YAML.read_text())
        assert "spartqa" in config
        assert "spartqa" in config["spartqa"]["resources_servers"]
        assert "spartqa_simple_agent" in config
        agent = config["spartqa_simple_agent"]["responses_api_agents"]["simple_agent"]
        assert agent["resources_server"]["name"] == "spartqa"


# ── AC9: prepare_spartqa.py record-building logic ───────────────────────────


class TestAcPrepareScript:
    def test_ac_prepare_defines_build_helpers(self) -> None:
        prepare = _load_prepare_module()
        assert callable(prepare.build_records)
        assert callable(prepare._unique_preserve_order)
        assert callable(prepare.parse_options)
        assert callable(prepare.resolve_gold)

    def test_ac_parse_options_splits_the_question_tail(self) -> None:
        prepare = _load_prepare_module()
        question = "Block A is above block B. What is below the circle? a big square or a small square?"
        assert prepare.parse_options(question) == ["a big square", "a small square"]
        assert prepare.parse_options("A statement with no alternatives.") is None

    def test_ac_resolve_gold_unflattens_the_retrieval_qrels(self) -> None:
        prepare = _load_prepare_module()
        options = ["a big square", "a small square"]
        # qrels marks all three phrases -> the single label is "both of them".
        assert prepare.resolve_gold([BOTH_LABEL, *options], options) == BOTH_LABEL
        # Two objects and no sentinel is still "both of them".
        assert prepare.resolve_gold(options, options) == BOTH_LABEL
        assert prepare.resolve_gold([NONE_LABEL], options) == NONE_LABEL
        # A single object gold snaps to the option's own wording.
        assert prepare.resolve_gold(["big square"], options) == "a big square"
        assert prepare.resolve_gold(["a green cup"], options) is None

    def test_ac_unique_preserve_order_dedupes_casefold_preserving_order(self) -> None:
        prepare = _load_prepare_module()
        result = prepare._unique_preserve_order(
            ["Below the circle", "  below the circle ", "Under the Circle", "", "  "]
        )
        assert result == ["Below the circle", "Under the Circle"]


# ── AC10: round-trip over example.jsonl -> reward 1.0 ───────────────────────


class TestAcRoundTrip:
    async def test_ac_example_targets_score_one(self) -> None:
        server = _server()
        rows = _read_jsonl(_EXAMPLE_JSONL)
        assert rows
        for row in rows:
            result = await server.verify(
                _make_request(
                    f"Final answer: {row['target']}",
                    target=row["target"],
                    options=row["options"],
                )
            )
            assert result.reward == approx(1.0), row["target"]


# ── Committed rollouts are self-consistent with the scorer ──────────────────


class TestAcRolloutsSelfConsistent:
    async def test_ac_rollouts_reproduce_committed_fields(self) -> None:
        server = _server()
        rows = _read_jsonl(_ROLLOUTS_JSONL)
        assert rows
        for row in rows:
            text = row["response"]["output"][0]["content"][0]["text"]
            result = await server.verify(_make_request(text, target=row["target"], options=row["options"]))
            assert result.reward == approx(row["reward"]), text
            assert result.exact is row["exact"]
            assert result.parsed is row["parsed"]
            assert result.extracted == row["extracted"]
            assert result.predicted_label == row["predicted_label"]


# ── Story edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_edge_reasoning_wrapped_output_is_stripped_and_scores(self) -> None:
        result = await _server().verify(
            _make_request(
                f"<think>the star is north of the moon</think>Final answer: {BOTH_LABEL}",
                target=BOTH_LABEL,
                options=_OPTIONS,
            )
        )
        assert result.reward == approx(1.0)
        assert result.extracted == BOTH_LABEL

    async def test_edge_single_object_never_satisfies_a_both_gold(self) -> None:
        for option in _OPTIONS:
            result = await _server().verify(
                _make_request(f"Final answer: {option}", target=BOTH_LABEL, options=_OPTIONS)
            )
            assert result.reward == approx(0.0), option

    async def test_edge_ambiguous_answer_resolves_to_no_label(self) -> None:
        result = await _server().verify(
            _make_request(
                f"Final answer: {_OPTIONS[0]} or {_OPTIONS[1]}",
                target=BOTH_LABEL,
                options=_OPTIONS,
            )
        )
        assert result.reward == approx(0.0)
        assert result.predicted_label == ""

    async def test_edge_nested_options_pick_the_most_specific(self) -> None:
        options = ["a triangle in block B", "a big blue triangle in block B"]
        assert match_label(options[1], candidate_labels(options)) == options[1]

    async def test_edge_exact_true_only_on_verbatim_answer(self) -> None:
        strict = await _server().verify(
            _make_request(f"Final answer: {_OPTIONS[0]}", target=_OPTIONS[0], options=_OPTIONS)
        )
        assert strict.exact is True

        loose = await _server().verify(
            _make_request(
                f"Final answer: I believe it is {_OPTIONS[0]}.",
                target=_OPTIONS[0],
                options=_OPTIONS,
            )
        )
        assert loose.reward == approx(1.0)
        assert loose.exact is False
