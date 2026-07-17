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
    PROMPT,
    SpartqaResourcesServer,
    SpartqaResourcesServerConfig,
    SpartqaVerifyRequest,
    SpartqaVerifyResponse,
    SimpleResourcesServer,
    _clean_candidate,
    _extract_answer,
    _normalize,
    _strip_reasoning,
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


def _make_request(
    text: str, *, target: str = "", all_targets: list[str] | None = None
) -> SpartqaVerifyRequest:
    return SpartqaVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        target=target,
        all_targets=all_targets or [],
    )


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
        result = await _server().verify(_make_request("Final answer: yes", target="yes"))
        assert isinstance(result, SpartqaVerifyResponse)
        assert hasattr(result, "reward")
        assert result.reward == approx(1.0)


# ── AC2: scoring correctness ─────────────────────────────────────────────────

# Shared fixtures: (response_text, target, all_targets, expected_correct).
_PARITY_FIXTURES: list[tuple[str, str, list[str], bool]] = [
    ("Final answer: green cup", "green cup", ["green cup"], True),
    ("Final answer: the green cup on the table", "green cup", ["green cup"], True),
    ("Final answer: red ball", "green cup", ["green cup"], False),
    (
        "Final answer: under the circle",
        "below the circle",
        ["below the circle", "under the circle"],
        True,
    ),
    ("   ", "green cup", ["green cup"], False),
    ("<think>the star is north</think>Final answer: yes", "yes", ["yes"], True),
]


class TestAcMetricParity:
    def test_ac_prompt_text_contract(self) -> None:
        assert "Final answer: <answer phrase>" in PROMPT
        assert PROMPT.startswith("Answer the spatial reasoning query below.")
        assert PROMPT.rstrip().endswith("{question}")

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

    @pytest.mark.parametrize("text,target,all_targets,expected", _PARITY_FIXTURES)
    async def test_ac_scoring_correctness(
        self, text: str, target: str, all_targets: list[str], expected: bool
    ) -> None:
        result = await _server().verify(
            _make_request(text, target=target, all_targets=all_targets)
        )
        assert bool(result.reward) is expected


# ── AC3: reward is strictly 1.0 or 0.0 ─────────────────────────────────────


class TestAcBinaryReward:
    @pytest.mark.parametrize(
        "text,target,all_targets",
        [
            ("Final answer: green cup", "green cup", ["green cup"]),
            ("Final answer: the green cup on the table", "green cup", ["green cup"]),
            ("Final answer: red ball", "green cup", ["green cup"]),
            ("", "green cup", ["green cup"]),
            ("no marker at all", "green cup", ["green cup"]),
            ("Final answer: under the circle", "below the circle", ["below the circle", "under the circle"]),
        ],
    )
    async def test_ac_reward_is_binary(
        self, text: str, target: str, all_targets: list[str]
    ) -> None:
        result = await _server().verify(
            _make_request(text, target=target, all_targets=all_targets)
        )
        assert result.reward in {0.0, 1.0}


# ── AC4: all_targets read from field; falls back to [target] ────────────────


class TestAcAllTargets:
    async def test_ac_uses_all_targets_field(self) -> None:
        # target does not match; a non-first accepted phrase does.
        result = await _server().verify(
            _make_request(
                "Final answer: under the circle",
                target="something else",
                all_targets=["something else", "under the circle"],
            )
        )
        assert result.reward == approx(1.0)

    async def test_ac_falls_back_to_target_when_all_targets_empty(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: yes", target="yes", all_targets=[])
        )
        assert result.reward == approx(1.0)

    async def test_ac_falls_back_to_target_when_all_targets_absent(self) -> None:
        # all_targets defaults to [] when not supplied -> uses [target].
        request = SpartqaVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Final answer: green cup"),
            target="green cup",
        )
        result = await _server().verify(request)
        assert result.all_targets == []
        assert result.reward == approx(1.0)


# ── AC5: response carries exact, parsed, extracted extras ───────────────────


class TestAcExtraFields:
    async def test_ac_extra_fields_present(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: green cup", target="green cup")
        )
        assert result.exact is True
        assert result.parsed is True
        assert result.extracted == "green cup"

    async def test_ac_exact_false_on_contains_only(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: the green cup on the table", target="green cup")
        )
        assert result.reward == approx(1.0)
        assert result.exact is False
        assert result.parsed is True


# ── AC6: empty / whitespace output -> reward 0.0, no exception ───────────────


class TestAcEmptyOutput:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    async def test_ac_empty_output_scores_zero(self, text: str) -> None:
        result = await _server().verify(_make_request(text, target="green cup"))
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
            assert isinstance(row["all_targets"], list) and row["all_targets"]
            assert row["agent_ref"]["name"] == "spartqa_simple_agent"


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
                    all_targets=row["all_targets"],
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
            result = await server.verify(
                _make_request(text, target=row["target"], all_targets=row["all_targets"])
            )
            assert result.reward == approx(row["reward"]), text
            assert result.exact is row["exact"]
            assert result.parsed is row["parsed"]
            assert result.extracted == row["extracted"]


# ── Story edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_edge_reasoning_wrapped_output_is_stripped_and_scores(self) -> None:
        result = await _server().verify(
            _make_request(
                "<think>the star is north of the moon</think>Final answer: yes",
                target="yes",
            )
        )
        assert result.reward == approx(1.0)
        assert result.extracted == "yes"

    async def test_edge_multiple_answers_any_match_scores_one(self) -> None:
        result = await _server().verify(
            _make_request(
                "Final answer: above and to the right",
                target="upper right",
                all_targets=["upper right", "above and to the right"],
            )
        )
        assert result.reward == approx(1.0)

    async def test_edge_exact_true_only_on_strict_equality(self) -> None:
        strict = await _server().verify(
            _make_request(
                "Final answer: above and to the right",
                target="upper right",
                all_targets=["upper right", "above and to the right"],
            )
        )
        assert strict.exact is True

        loose = await _server().verify(
            _make_request(
                "Final answer: it is above and to the right of the cat",
                target="upper right",
                all_targets=["upper right", "above and to the right"],
            )
        )
        assert loose.reward == approx(1.0)
        assert loose.exact is False
