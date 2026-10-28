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
"""Caveman check on verify: scoring and logging always run; with caveman_penalty.enabled a flagged rollout gets
reward 0."""

import os
import time
from unittest.mock import MagicMock

import pytest
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness import app as app_module
from resources_servers.browsecomp_advanced_harness import caveman_detector
from resources_servers.browsecomp_advanced_harness.app import (
    CavemanPenaltyConfig,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    TavilySearchVerifyRequest,
)


_DUMMY_EXCLUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_exclude_domains_file.json")

CAVEMAN = (
    "We need solve. Need investigate web. Search not enough. Maybe candidate is novelist. Need verify dates. "
    "Found obituary. Seems match. Could be translator. Check source. Likely answer Smollett? Need confirm year."
)
NORMAL = (
    "Let me look at the search results more carefully. The first result mentions that the author was born in a city "
    "with a university, which matches the clue in the question. I should check whether the article was published in "
    "the right year before I commit to this candidate."
)


def _config(**penalty) -> TavilySearchResourcesServerConfig:
    return TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        tavily_api_key="test_api_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE,
        use_judge=False,
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        caveman_penalty=CavemanPenaltyConfig(**penalty),
    )


def _response(turn_texts, final="Answer: Paris Confidence: 95%") -> NeMoGymResponse:
    output = [
        NeMoGymResponseReasoningItem(id=f"rs_{i}", summary=[NeMoGymSummary(text=t, type="summary_text")])
        for i, t in enumerate(turn_texts)
    ]
    output.append(
        NeMoGymResponseOutputMessage(
            id="msg",
            content=[NeMoGymResponseOutputText(annotations=[], text=final, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    return NeMoGymResponse(
        id="resp", created_at=0.0, model="m", object="response", output=output, parallel_tool_calls=False,
        tool_choice="none", tools=[],
    )


async def _verify(cfg, turn_texts, final="Answer: Paris Confidence: 95%"):
    server = TavilySearchResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test_session_id"}
    body = TavilySearchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response(turn_texts, final),
        ground_truth="Paris",
        question="What is the capital of France?",
    )
    return await server.verify(request, body)


# ---- detector ----


def test_detector_scores_known_texts():
    assert caveman_detector.score_turn(CAVEMAN, "A") > 0.85
    assert caveman_detector.score_turn(NORMAL, "A") < 0.2
    assert caveman_detector.score_turn(CAVEMAN, "B") > caveman_detector.score_turn(NORMAL, "B")
    assert caveman_detector.score_turn("too short", "A") is None
    score, n = caveman_detector.score_rollout([CAVEMAN, NORMAL, "x"], "A")
    assert n == 2 and 0.0 < score < 1.0


def test_detector_is_fast_on_long_and_pathological_turns():
    turns = [(NORMAL + " ") * 6] * 20 + [(CAVEMAN + " ") * 6] * 20  # a 40-turn rollout of ~1.5k-char turns
    t0 = time.perf_counter()
    caveman_detector.score_rollout(turns, "A")
    caveman_detector.score_rollout(turns, "B")
    assert time.perf_counter() - t0 < 1.0
    t0 = time.perf_counter()
    caveman_detector.score_turn("QUJD" * 5000, "A")  # a 20k-char run without spaces (quadratic in the original)
    caveman_detector.score_turn("QUJD" * 5000, "B")
    assert time.perf_counter() - t0 < 0.5


# ---- verify hook ----


async def test_enabled_flagged_correct_answer_gets_zero():
    res = await _verify(_config(enabled=True), [CAVEMAN] * 5)
    assert res.caveman_flag is True
    assert res.caveman_score > 0.85 and res.caveman_scored_turns == 5
    assert res.reward_before_penalty == approx(1.0)
    assert res.reward == approx(0.0)


async def test_enabled_unflagged_answer_unchanged():
    res = await _verify(_config(enabled=True), [NORMAL] * 5)
    assert res.caveman_flag is False
    assert res.caveman_score < 0.5
    assert res.reward == approx(1.0) and res.reward_before_penalty == approx(1.0)


async def test_disabled_flagged_answer_unchanged_but_logged():
    res = await _verify(_config(enabled=False), [CAVEMAN] * 5)
    assert res.caveman_flag is True and res.caveman_score > 0.85
    assert res.reward == approx(1.0) and res.reward_before_penalty == approx(1.0)


async def test_default_config_is_disabled_and_logs():
    cfg = _config()
    assert cfg.caveman_penalty.enabled is False
    res = await _verify(cfg, [CAVEMAN] * 5)
    assert res.caveman_flag is True and res.reward == approx(1.0)


async def test_wrong_answer_stays_zero():
    res = await _verify(_config(enabled=True), [CAVEMAN] * 5, final="Answer: London Confidence: 80%")
    assert res.reward_before_penalty == approx(0.0) and res.reward == approx(0.0) and res.caveman_flag is True


async def test_min_scored_turns():
    res = await _verify(_config(enabled=True, min_scored_turns=3), [CAVEMAN] * 2)
    assert res.caveman_scored_turns == 2 and res.caveman_flag is False and res.reward == approx(1.0)
    res = await _verify(_config(enabled=True, min_scored_turns=2), [CAVEMAN] * 2)
    assert res.caveman_flag is True and res.reward == approx(0.0)


async def test_no_reasoning_is_never_flagged():
    res = await _verify(_config(enabled=True, min_scored_turns=0), [])
    assert res.caveman_score is None and res.caveman_scored_turns == 0 and res.caveman_flag is False
    assert res.reward == approx(1.0)


async def test_threshold_controls_flag():
    res = await _verify(_config(enabled=True, threshold=1.0), [CAVEMAN] * 5)
    assert res.caveman_flag is False and res.reward == approx(1.0)


async def test_detector_both_requires_both():
    res = await _verify(_config(enabled=True, detector="both"), [CAVEMAN] * 5)
    assert res.caveman_flag is True and res.reward == approx(0.0)
    res = await _verify(_config(enabled=True, detector="both"), [NORMAL] * 5)
    assert res.caveman_flag is False and res.reward == approx(1.0)


async def test_detector_both_needs_b_agreement(monkeypatch):
    real = caveman_detector.summarize_rollout

    def fake(turns):
        return dict(real(turns), score_b=0.1)

    monkeypatch.setattr(app_module.caveman_detector, "summarize_rollout", fake)
    res = await _verify(_config(enabled=True, detector="both"), [CAVEMAN] * 5)
    assert res.caveman_flag is False and res.reward == approx(1.0)


async def test_detector_b_uses_b_score():
    res = await _verify(_config(enabled=False, detector="B", threshold=0.0), [NORMAL] * 5)
    assert res.caveman_score == approx(caveman_detector.score_rollout([NORMAL] * 5, "B")[0])


async def test_scoring_failure_leaves_reward(monkeypatch):
    def boom(turns):
        raise RuntimeError("detector failure")

    monkeypatch.setattr(app_module.caveman_detector, "summarize_rollout", boom)
    res = await _verify(_config(enabled=True), [CAVEMAN] * 5)
    assert res.reward == approx(1.0) and res.reward_before_penalty == approx(1.0)
    assert res.caveman_score is None and res.caveman_flag is False


async def test_monitoring_fields_present_and_numeric():
    res = await _verify(_config(), [NORMAL] * 5)
    fields = ("caveman_score", "caveman_score_b", "caveman_article_rate", "caveman_subjectless_share",
              "reward_before_penalty")
    for name in fields:
        value = getattr(res, name)
        assert isinstance(value, float) and not isinstance(value, bool), (name, value)
    assert isinstance(res.caveman_flag, bool) and isinstance(res.caveman_scored_turns, int)
    dumped = res.model_dump()
    for name in fields + ("caveman_flag", "caveman_scored_turns"):
        assert name in dumped  # top-level, so NeMo RL's per-agent metrics pick them up
    assert 0.0 <= res.caveman_article_rate <= 1.0 and 0.0 <= res.caveman_subjectless_share <= 1.0
    cav = await _verify(_config(), [CAVEMAN] * 5)
    assert cav.caveman_score_b > res.caveman_score_b
    assert cav.caveman_article_rate < res.caveman_article_rate


async def test_unscorable_rollout_fields_are_none():
    res = await _verify(_config(), ["ok", "fine"])
    assert res.caveman_score is None and res.caveman_score_b is None and res.caveman_article_rate is None
    assert res.caveman_subjectless_share is None and res.caveman_flag is False


# ---- config validation ----


@pytest.mark.parametrize(
    "kwargs",
    [dict(threshold=1.5), dict(threshold=-0.1), dict(detector="C"), dict(min_scored_turns=-1), dict(mode="scale")],
)
def test_config_validation_rejects(kwargs):
    with pytest.raises(ValueError):
        CavemanPenaltyConfig(**kwargs)
