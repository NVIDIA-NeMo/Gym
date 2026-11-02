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

import json
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.verification import JudgeError
from resources_servers.lc_interleaved_qa.app import (
    GUESS_SYSTEM_PROMPT,
    JUDGE_CONTEXT_CHARS,
    LcInterleavedQaResourcesServer,
    LcInterleavedQaResourcesServerConfig,
    LcInterleavedQaVerifyRequest,
    _build_judge_params,
    _locate_evidence,
    _parse_judge_verdict,
)


PREFIX = (
    "Meridian Capital Group is a business development company incorporated in Delaware in 1997. "
    "Our asset management business is conducted through our wholly-owned portfolio company, "
    "Meridian Asset Management. Total assets under management were $4.1 billion at year end."
)
QUOTE = "Total assets under management were $4.1 billion at year end"


def _config(
    *,
    min_rubric: int = 3,
    guess: bool = False,
    min_evidence_distance_chars: int = 0,
) -> LcInterleavedQaResourcesServerConfig:
    return LcInterleavedQaResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        judge_responses_create_params_overrides={},
        guess_model_server=({"type": "responses_api_models", "name": "guess"} if guess else None),
        guess_responses_create_params_overrides={},
        min_rubric=min_rubric,
        min_evidence_distance_chars=min_evidence_distance_chars,
    )


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
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


def _request(answer: str, **overrides: Any) -> LcInterleavedQaVerifyRequest:
    payload: dict[str, Any] = {
        "responses_create_params": {"input": [{"role": "user", "content": PREFIX}]},
        "question": "What were total assets under management at the end of 2015?",
        "prefix": PREFIX,
        "family": "recall",
        "evidence": [{"quote": QUOTE}],
        "response": _response(answer),
    }
    payload.update(overrides)
    return LcInterleavedQaVerifyRequest(**payload)


def _verdict(
    *,
    grounded: bool = True,
    follows: bool = True,
    matches_type: Optional[bool] = True,
    rubric: int = 5,
) -> str:
    payload: dict[str, Any] = {
        "grounded": grounded,
        "follows_from_quote": follows,
        "rubric": rubric,
        "reason": "ok",
    }
    if matches_type is not None:
        payload["matches_type"] = matches_type
    return json.dumps(payload)


async def _verify(
    config: LcInterleavedQaResourcesServerConfig,
    request: LcInterleavedQaVerifyRequest,
    *,
    judge_text: str = "",
    guess_text: str = "",
    calls: Optional[list[str]] = None,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Run verify with call_judge stubbed.

    Pass ``calls`` to record, in order, which judge servers were actually reached — that
    is how the tests assert a gate short-circuited before spending a judge call.
    """
    import resources_servers.lc_interleaved_qa.app as module

    recorded = calls if calls is not None else []

    async def fake_call_judge(_client, *, server_name: str, url_path: str, json: Any, response_model: Any):
        recorded.append(server_name)
        return _response(guess_text if server_name == "guess" else judge_text)

    monkeypatch.setattr(module, "call_judge", fake_call_judge)
    server = LcInterleavedQaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return await server.verify(body=request)


# ---- hard filters (no judge call) ------------------------------------------------------


@pytest.mark.asyncio
async def test_grounded_answer_earns_normalised_rubric(monkeypatch):
    result = await _verify(
        _config(),
        _request("Total assets under management were $4.1 billion."),
        judge_text=_verdict(rubric=5),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 1.0
    assert result.rejected is False
    assert result.rubric == 5
    assert result.evidence_match_kinds == ["exact"]


@pytest.mark.asyncio
async def test_rubric_scales_the_reward(monkeypatch):
    result = await _verify(
        _config(),
        _request("$4.1 billion."),
        judge_text=_verdict(rubric=3),
        monkeypatch=monkeypatch,
    )
    # 1 -> 0.0 and 5 -> 1.0, so a bare pass at 3 is worth half.
    assert result.reward == 0.5


@pytest.mark.asyncio
async def test_rubric_below_threshold_scores_zero(monkeypatch):
    result = await _verify(
        _config(min_rubric=4),
        _request("$4.1 billion."),
        judge_text=_verdict(rubric=3),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 0.0
    assert result.rejected is True
    assert "rubric_below_4" in result.rejection_reasons


@pytest.mark.asyncio
async def test_ungrounded_verdict_scores_zero(monkeypatch):
    result = await _verify(
        _config(),
        _request("$9.9 billion."),
        judge_text=_verdict(grounded=False),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 0.0
    assert result.rejection_reasons == ["not_grounded"]


@pytest.mark.asyncio
async def test_meta_reference_in_question_is_rejected_without_a_judge_call(monkeypatch):
    calls: list[str] = []
    result = await _verify(
        _config(),
        _request("$4.1 billion.", question="What figure is given in the chunk above?"),
        judge_text=_verdict(),
        calls=calls,
        monkeypatch=monkeypatch,
    )
    assert result.reward == 0.0
    assert result.rejected is True
    assert any(r.startswith("meta_reference:") for r in result.rejection_reasons)
    assert calls == [], "hard filters must short-circuit before the judge"


@pytest.mark.asyncio
async def test_plain_document_referent_is_allowed(monkeypatch):
    result = await _verify(
        _config(),
        _request("$4.1 billion.", question="According to the text, what were total assets under management?"),
        judge_text=_verdict(rubric=5),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 1.0, "a bare referent to the one document is ordinary English"


@pytest.mark.asyncio
async def test_evidence_absent_from_prefix_is_rejected(monkeypatch):
    calls: list[str] = []
    result = await _verify(
        _config(),
        _request("$4.1 billion.", evidence=[{"quote": "a sentence that never appears in the source"}]),
        judge_text=_verdict(),
        calls=calls,
        monkeypatch=monkeypatch,
    )
    assert "evidence_not_found" in result.rejection_reasons
    assert calls == []


@pytest.mark.asyncio
async def test_quote_with_curly_punctuation_is_relocated_fuzzily(monkeypatch):
    # The generator emitted the quote with typographic punctuation and a line wrap; the
    # source has neither. An exact search fails and the fuzzy pass must recover it.
    mangled = "Total assets under management were $4.1 billion\nat year **end**"
    result = await _verify(
        _config(),
        _request("$4.1 billion.", evidence=[{"quote": mangled}]),
        judge_text=_verdict(rubric=5),
        monkeypatch=monkeypatch,
    )
    assert result.evidence_match_kinds == ["fuzzy"]
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_refusal_opening_is_rejected_but_an_aside_is_not(monkeypatch):
    refusing = await _verify(
        _config(),
        _request("The text does not specify that figure."),
        judge_text=_verdict(),
        monkeypatch=monkeypatch,
    )
    assert "refusal" in refusing.rejection_reasons

    # The same phrase past the opening window is an aside inside a grounded answer, not a
    # refusal, so it must survive.
    aside_answer = (
        "Total assets under management were $4.1 billion at year end, up from $3.6 billion a year earlier. "
        "That figure covers the three funds managed by Meridian Asset Management, the wholly-owned portfolio "
        "company through which the group conducts its asset management business. The report does not specify "
        "the breakdown by individual fund."
    )
    assert len(aside_answer) > 160 and "not specify" not in aside_answer[:160].lower()
    aside = await _verify(
        _config(),
        _request(aside_answer),
        judge_text=_verdict(rubric=4),
        monkeypatch=monkeypatch,
    )
    assert aside.reward == 0.75, "a refusal phrase late in a grounded answer is not a refusal"


@pytest.mark.asyncio
async def test_unanswerable_family_may_refuse(monkeypatch):
    result = await _verify(
        _config(),
        _request(
            "The text so far does not specify the breakdown by fund; it gives only the $4.1 billion total.",
            family="unanswerable",
        ),
        judge_text=_verdict(rubric=4),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 0.75


@pytest.mark.asyncio
async def test_violated_response_constraint_is_rejected(monkeypatch):
    calls: list[str] = []
    result = await _verify(
        _config(),
        _request("$4.1 Billion", constraint={"kind": "lowercase"}),
        judge_text=_verdict(),
        calls=calls,
        monkeypatch=monkeypatch,
    )
    assert "constraint_lowercase" in result.rejection_reasons
    assert calls == []


@pytest.mark.asyncio
async def test_satisfied_response_constraint_passes(monkeypatch):
    result = await _verify(
        _config(),
        _request("$4.1 billion", constraint={"kind": "lowercase"}),
        judge_text=_verdict(rubric=5),
        monkeypatch=monkeypatch,
    )
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_verbatim_answer_must_occur_in_the_prefix(monkeypatch):
    missing = await _verify(
        _config(),
        _request("Total assets were four point one billion dollars at year end.", verbatim=True),
        judge_text=_verdict(),
        monkeypatch=monkeypatch,
    )
    assert "verbatim_not_in_prefix" in missing.rejection_reasons

    present = await _verify(
        _config(),
        _request(QUOTE, verbatim=True),
        judge_text=_verdict(rubric=5),
        monkeypatch=monkeypatch,
    )
    assert present.reward == 1.0


@pytest.mark.asyncio
async def test_degenerate_answer_is_rejected(monkeypatch):
    result = await _verify(
        _config(),
        _request("billion billion billion billion billion billion billion"),
        judge_text=_verdict(),
        monkeypatch=monkeypatch,
    )
    assert "repetition_run" in result.rejection_reasons


@pytest.mark.asyncio
async def test_shallow_evidence_is_rejected_when_a_floor_is_set(monkeypatch):
    # The quote sits at the very end of the prefix, so the question reaches back almost
    # nowhere; a long-context floor must reject it.
    result = await _verify(
        _config(min_evidence_distance_chars=500),
        _request("$4.1 billion."),
        judge_text=_verdict(),
        monkeypatch=monkeypatch,
    )
    assert "evidence_too_shallow" in result.rejection_reasons
    assert result.evidence_distance_chars is not None and result.evidence_distance_chars < 500


# ---- guessability ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guessable_question_scores_zero(monkeypatch):
    calls: list[str] = []
    result = await _verify(
        _config(guess=True),
        _request("$4.1 billion.", answer_key="$4.1 billion"),
        judge_text=_verdict(rubric=5),
        guess_text="$4.1 billion",
        calls=calls,
        monkeypatch=monkeypatch,
    )
    assert result.guessable is True
    assert result.reward == 0.0
    assert result.rejection_reasons == ["guessable"]
    assert calls == ["guess"], "a guessable pair needs no grounding verdict"


@pytest.mark.asyncio
async def test_unknown_guess_lets_the_pair_through(monkeypatch):
    calls: list[str] = []
    result = await _verify(
        _config(guess=True),
        _request("$4.1 billion.", answer_key="$4.1 billion"),
        judge_text=_verdict(rubric=5),
        guess_text="UNKNOWN",
        calls=calls,
        monkeypatch=monkeypatch,
    )
    assert result.guessable is False
    assert result.reward == 1.0
    assert calls == ["guess", "judge"]


# ---- judge protocol --------------------------------------------------------------------


def test_parse_judge_verdict_accepts_a_fenced_object():
    verdict = _parse_judge_verdict('```json\n{"grounded": true, "follows_from_quote": true, "rubric": 4}\n```')
    assert verdict is not None
    assert verdict["grounded"] is True and verdict["rubric"] == 4


def test_parse_judge_verdict_returns_none_without_a_verdict():
    assert _parse_judge_verdict('{"reason": "I could not decide"}') is None


def test_parse_judge_verdict_raises_on_empty_text():
    with pytest.raises(JudgeError):
        _parse_judge_verdict("   ")


@pytest.mark.asyncio
async def test_missing_verdict_is_flagged_and_scores_zero(monkeypatch):
    result = await _verify(
        _config(),
        _request("$4.1 billion."),
        judge_text="I think it is probably fine.",
        monkeypatch=monkeypatch,
    )
    assert result.reward == 0.0
    assert result.invalid_judge_response is True
    assert result.rejection_reasons == ["no_verdict"]


def test_judge_prompt_carries_a_window_around_the_quote_not_the_whole_prefix():
    params = _build_judge_params(*_with_offsets(_request("$4.1 billion."), "$4.1 billion."))
    user_prompt = params["input"][1]["content"]
    assert QUOTE in user_prompt
    assert "REQUESTED QUESTION TYPE: recall" in user_prompt
    # The window is bounded, so a 100k-character prefix never reaches the judge.
    assert len(user_prompt) < 2 * JUDGE_CONTEXT_CHARS + len(QUOTE) + 2000


def test_judge_prompt_includes_every_located_quote():
    request = _request(
        "senior floating rate loans",
        family="multi_hop",
        evidence=[{"quote": "Meridian Asset Management"}, {"quote": QUOTE}],
    )
    user_prompt = _build_judge_params(*_with_offsets(request, "senior floating rate loans"))["input"][1]["content"]
    assert "QUOTE 1:" in user_prompt and "QUOTE 2:" in user_prompt


def test_guess_prompt_withholds_the_document():
    assert "NOT given any document" in GUESS_SYSTEM_PROMPT


def test_rubric_is_clamped_so_the_reward_stays_bounded():
    """The prompt asks for 1-5 but the judge is not bound by it; reward is linear in it."""
    high = _parse_judge_verdict('{"grounded": true, "follows_from_quote": true, "rubric": 99}')
    low = _parse_judge_verdict('{"grounded": true, "follows_from_quote": true, "rubric": -5}')
    assert high["rubric"] == 5, "an unclamped 99 would be a reward of 24.5"
    assert low["rubric"] == 1
    for verdict in (high, low):
        assert 0.0 <= (verdict["rubric"] - 1) / 4 <= 1.0


def test_verdict_parsing_survives_a_chain_of_thought_containing_braces():
    text = (
        'Let me check. Consider {"grounded": false} as a first read — no, that is wrong.\n'
        '{"grounded": true, "follows_from_quote": true, "matches_type": true, "rubric": 4, "reason": "ok"}'
    )
    verdict = _parse_judge_verdict(text)
    assert verdict["grounded"] is True and verdict["rubric"] == 4


def test_verdict_parsing_is_fast_on_a_malformed_reply():
    """Scanning forward from every brace is quadratic: 10k unclosed braces took 3.9 s."""
    import time

    malformed = "x" * 20_000 + "{ " * 10_000
    started = time.monotonic()
    assert _parse_judge_verdict(malformed) is None
    assert time.monotonic() - started < 0.5, "the judge is untrusted input; this blocks the event loop"


def test_a_verdict_is_found_after_a_huge_preamble():
    verdict = '{"grounded": true, "follows_from_quote": true, "rubric": 5, "reason": "ok"}'
    assert _parse_judge_verdict("filler " * 50_000 + verdict)["rubric"] == 5


def test_fuzzy_relocation_gives_up_on_a_runaway_quote():
    """One prefix scan per three-word window: 2,000 words over 585 kB took 2.7 s."""
    import time

    from resources_servers.lc_interleaved_qa.quality import fuzzy_locate

    words = [f"w{i % 900}" for i in range(120_000)]
    prefix = " ".join(words)
    runaway = " ".join(words[:2_000]).replace("w5", "Q5", 50)
    started = time.monotonic()
    assert fuzzy_locate(runaway, prefix) is None, "a 2,000-word quote is not usable evidence"
    assert time.monotonic() - started < 0.5

    # A normal quote still relocates after small edits.
    normal = " ".join(words[50_000:50_030]).replace("w1", "W1", 2)
    assert fuzzy_locate(normal, prefix) is not None


def _with_offsets(request, answer):
    """(request, answer, offsets) as verify() computes them."""
    from resources_servers.lc_interleaved_qa.app import _locate_evidence

    offsets, _ = _locate_evidence(request, answer)
    return request, answer, offsets


def test_the_judge_window_follows_the_cited_occurrence_not_the_first():
    """A quote occurring twice: the judge must see the occurrence the row cites."""
    sentence = "Revenue was $10 million."
    prefix = sentence + " filler. " * 300 + sentence + " tail."
    second = prefix.rindex(sentence)
    request = _request(
        "$10 million",
        prefix=prefix,
        evidence=[{"quote": sentence, "prefix_offset": second}],
    )
    offsets, kinds = _locate_evidence(request, "$10 million")
    assert offsets == [second] and kinds == ["given"]

    window = _build_judge_params(request, "$10 million", offsets)["input"][1]["content"]
    assert prefix[second - 50 : second] in window, "the judge got the window around the wrong occurrence"


def test_verdict_parsing_is_linear_on_every_brace_shape():
    """Scanning from each "{" costs 3.9 s on unclosed braces; trying every pair, 49 s."""
    import time

    for malformed in ("{ " * 10_000, "{" * 5_000 + "}" * 5_000, "{a " * 5_000 + "} " * 5_000):
        started = time.monotonic()
        assert _parse_judge_verdict(malformed) is None
        assert time.monotonic() - started < 0.5, f"pathological on {malformed[:4]!r}"


def test_a_verdict_is_found_before_a_long_trailing_commentary():
    verdict = '{"grounded": true, "follows_from_quote": true, "rubric": 5, "reason": "ok"}'
    assert _parse_judge_verdict(verdict + "\n" + "commentary " * 5_000)["rubric"] == 5


def test_verdict_parsing_ignores_braces_inside_json_strings():
    text = '{"grounded": true, "follows_from_quote": true, "rubric": 3, "reason": "a } and { brace"}'
    assert _parse_judge_verdict(text)["rubric"] == 3


def test_a_guess_does_not_count_as_recovered_inside_a_longer_number():
    from resources_servers.lc_interleaved_qa.quality import answer_guessed

    assert not answer_guessed("I would guess around 1993 or so", "3")
    assert not answer_guessed("the figure was 120 million", "12")
    assert answer_guessed("the answer is 3", "3")
    assert answer_guessed("about 12 funds", "12")


def test_content_words_are_not_meta_references():
    from resources_servers.lc_interleaved_qa.quality import meta_reference

    for question in (
        "Who signed the passage of the bill?",
        "What is the passage rate of the reform?",
        "What were the above-market returns reported for 2015?",
        "Which chunk of the budget went to defence?",
    ):
        assert meta_reference(question) is None, question
    for question in (
        "What does the chunk above say?",
        "Summarise the previous chunk.",
        "What is in the passage above?",
        "As stated above, what was the total?",
    ):
        assert meta_reference(question) is not None, question


def test_the_plain_referent_allowlist_covers_what_the_pattern_matches():
    from resources_servers.lc_interleaved_qa.quality import meta_reference

    for noun in ("text", "document", "excerpt", "material", "snippet"):
        for determiner in ("the", "this"):
            assert meta_reference(f"What does {determiner} {noun} say about revenue?") is None


def test_a_derived_family_without_a_quote_still_gets_real_evidence():
    """The exemption from evidence_not_found is pointless if the judge sees no text."""
    request = _request("42", family="multistep", evidence=[{"quote": "nowhere in the document at all"}])
    offsets, _ = _locate_evidence(request, "42")
    assert all(offset is None for offset in offsets)
    prompt = _build_judge_params(request, "42", offsets)["input"][1]["content"]
    assert "(quote not located)" not in prompt
    assert "No quote was cited" in prompt
    assert PREFIX[:200] in prompt, "the judge must see real document text"
