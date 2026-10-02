# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import VerifierFixture, exercise_verifier_fixture
from resources_servers.facts_grounding_v2.app import (
    PROMPT_SHA256,
    PROMPTS_DIR,
    FACTSGroundingV2Config,
    FACTSGroundingV2ResourcesServer,
    FACTSGroundingV2VerifyRequest,
    extract_instruction_rating,
    extract_text_from_response,
    is_eligible,
    load_prompts,
    parse_grounding_verdict,
)


TESTS_DIR = Path(__file__).parent
GEMINI = "facts_grounding_judge_gemini"
GPT5 = "facts_grounding_judge_gpt5"


def _policy_response(text: str, *, incomplete: bool = False, usage: bool = True) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=[
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        incomplete_details={"reason": "max_output_tokens"} if incomplete else None,
        usage=(
            {
                "input_tokens": 1234,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 56,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 1290,
            }
            if usage
            else None
        ),
    )


def _chat_completion(content, *, model: str) -> dict:
    return {
        "id": f"chatcmpl-{hashlib.sha256((content or '').encode()).hexdigest()[:8]}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class _FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _request(text: str, **fields) -> FACTSGroundingV2VerifyRequest:
    base = dict(
        id="facts_grounding_v2_public_0001",
        user_request="Summarize the doc.",
        context_document="Apples are red fruits. Bananas are yellow fruits.",
        full_prompt="Only use the context.\n\nSummarize the doc.\n\nApples are red fruits. Bananas are yellow fruits.",
        domain="Retail/Product",
        type="Summarize",
        high_level_type="Text Transformation",
        responses_create_params={"input": [{"role": "user", "content": "prompt"}]},
        response=_policy_response(text, incomplete=fields.pop("incomplete", False)),
    )
    base.update(fields)
    return FACTSGroundingV2VerifyRequest(**base)


def _config(**overrides) -> FACTSGroundingV2Config:
    fields = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="facts_grounding_v2",
        judge_model_servers=[
            {"type": "responses_api_models", "name": GEMINI},
            {"type": "responses_api_models", "name": GPT5},
        ],
        judge_labels=["gemini-2.5-flash", "gpt-5"],
    )
    fields.update(overrides)
    return FACTSGroundingV2Config(**fields)


GROUNDED = (
    '{"sentence": "Apples are red.", "label": "supported", "rationale": "stated", "excerpt": "Apples are red fruits."}'
)
UNSUPPORTED = (
    '{"sentence": "Apples are green.", "label": "not_supported", "rationale": "contradicted", "excerpt": null}'
)
MAJOR = 'Analysis...\n```json\n{\n    "Instruction Following": "Major Issue(s)"\n}\n```'
NO_ISSUES = 'Analysis...\n```json\n{\n    "Instruction Following": "No Issues"\n}\n```'
MINOR = '{"Instruction Following": "Minor Issue(s)"}'


def _server(script, **overrides):
    """``script(server_name, stage, messages) -> content`` decides every judge reply."""
    client = MagicMock(spec=ServerClient)
    calls = []

    async def _post(server_name, url_path, json=None, **kwargs):
        assert url_path == "/v1/chat/completions"
        messages = json.messages
        stage = (
            "baseline"
            if len(messages) == 1
            else ("eligibility" if "baseline" in messages[1]["content"] else "grounding")
        )
        calls.append((server_name, stage, json))
        model = "gemini-2.5-flash" if server_name == GEMINI else "gpt-5-2025-08-07"
        return _FakeHTTPResponse(_chat_completion(script(server_name, stage, messages), model=model))

    client.post = AsyncMock(side_effect=_post)
    server = FACTSGroundingV2ResourcesServer(config=_config(**overrides), server_client=client)
    return server, calls


def _script(*, gemini_rating=NO_ISSUES, gpt5_rating=NO_ISSUES, gemini_grounding=GROUNDED, gpt5_grounding=GROUNDED):
    def script(server_name, stage, messages):
        if stage == "baseline":
            return f"baseline answer from {server_name}"
        if stage == "eligibility":
            return gemini_rating if server_name == GEMINI else gpt5_rating
        return gemini_grounding if server_name == GEMINI else gpt5_grounding

    return script


def test_prompts_are_the_pinned_starter_prompts(monkeypatch):
    prompts = load_prompts()
    for name, expected in PROMPT_SHA256.items():
        text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected
    rendered = prompts["eligibility_prompt"].format(user_request="Q", response_a="A", response_b="B")
    assert '{\n    "Instruction Following": "Major Issue(s)"\n}' in rendered and "{{" not in rendered
    assert "CONTEXT DOCUMENT REDACTED" in rendered  # the v2 no-context eligibility variant
    grounding = prompts["grounding_prompt"].format(user_query="Q", context="C", response="R")
    assert grounding.endswith("**Response:**\nR") and "{{" not in grounding
    assert "not_supported" in grounding  # the v2 UFG_REV21 label set
    monkeypatch.setitem(PROMPT_SHA256, "grounding_prompt.txt", "0" * 64)
    with pytest.raises(RuntimeError, match="drifted"):
        load_prompts()


@pytest.mark.parametrize(
    "text, rating, eligible",
    [
        (NO_ISSUES, "No Issues", True),
        (MINOR, "Minor Issue(s)", True),
        (MAJOR, "Major Issue(s)", False),
        ("no verdict here", "Invalid", True),
        ("", "Invalid", True),
        (
            '{"Instruction Following": "Major Issue(s)"} then {"Instruction Following": "No Issues"}',
            "Major Issue(s)",
            False,
        ),
    ],
)
def test_eligibility_parse_matches_the_starter(text, rating, eligible):
    assert extract_instruction_rating(text) == rating
    assert is_eligible(rating) is eligible


def test_grounding_parse_matches_the_starter():
    grounded, parsed, count, bad = parse_grounding_verdict(f"```json\n{GROUNDED}\n{GROUNDED}\n```")
    assert grounded and count == 2 and bad == 0
    grounded, parsed, count, bad = parse_grounding_verdict(f"{GROUNDED}\n{UNSUPPORTED}")
    assert not grounded and [item["label"] for item in parsed] == ["supported", "not_supported"]
    # no_rad and unknown labels do not break groundedness; a missing label becomes "unknown"
    grounded, parsed, _, _ = parse_grounding_verdict(
        '{"sentence": "Here is the summary:", "label": "no_rad", "rationale": "filler", "excerpt": null}\n'
        '{"sentence": "Apples are red.", "rationale": "stated", "excerpt": "Apples are red fruits."}'
    )
    assert grounded and parsed[1]["label"] == "unknown"
    # unparseable line that still names not_supported counts as a not_supported sentence
    grounded, parsed, count, bad = parse_grounding_verdict('{"sentence": "x", "label": "not_supported", broken')
    assert not grounded and count == 1 and bad == 1 and parsed[0]["label"] == "not_supported"
    # stray quotes inside field values are cleaned like the starter does
    grounded, parsed, _, _ = parse_grounding_verdict(
        '{"sentence": "He said "hi" today.", "label": "supported", "rationale": "the "quote"", "excerpt": "He said hi"}'
    )
    assert grounded and parsed[0]["sentence"] == "He said hi today."
    assert parse_grounding_verdict("") == (False, [], 0, 1)
    assert parse_grounding_verdict("nothing parseable") == (False, [], 0, 1)


def test_extract_text():
    assert extract_text_from_response(_policy_response("  hello  ")) == "hello"


async def test_eligible_first_judge_stops_eligibility_and_both_judges_ground():
    server, calls = _server(_script())
    result = await server.verify(_request("Apples are red."))
    assert result.reward == 1.0 and result.eligible and result.eligibility_deciding_judge == "gemini-2.5-flash"
    assert result.eligibility_ratings == {"gemini-2.5-flash": "No Issues"}  # GPT-5 was not asked
    assert result.grounding_verdicts == {"gemini-2.5-flash": True, "gpt-5": True}
    assert result.grounding_score == 1.0 and result.unadjusted_score == 1.0 and result.judges_agree
    assert result.is_grounded_all_judges == 1.0 and result.policy_input_tokens == 1234
    stages = [(name, stage) for name, stage, _ in calls]
    assert stages == [(GEMINI, "baseline"), (GEMINI, "eligibility"), (GEMINI, "grounding"), (GPT5, "grounding")]
    # baseline: the judge answers full_prompt as the only user message; ratings carry the starter system prompts
    baseline_params = calls[0][2]
    assert baseline_params.messages == [{"role": "user", "content": _request("x").full_prompt}]
    eligibility_params = calls[1][2]
    assert eligibility_params.messages[0]["role"] == "system"
    assert "baseline answer from facts_grounding_judge_gemini" in eligibility_params.messages[1]["content"]
    assert "Apples are red." in eligibility_params.messages[1]["content"]
    grounding_params = calls[2][2]
    assert grounding_params.messages[1]["content"].endswith("**Response:**\nApples are red.")
    assert "Apples are red fruits. Bananas are yellow fruits." in grounding_params.messages[1]["content"]
    assert grounding_params.temperature is None and grounding_params.max_tokens is None
    assert [receipt["stage"] for receipt in result.judge_receipts] == [
        "baseline",
        "eligibility",
        "grounding",
        "grounding",
    ]
    assert result.judge_receipts[3]["judge_model"] == "gpt-5-2025-08-07"


async def test_second_judge_can_rescue_eligibility_with_its_own_baseline():
    server, calls = _server(_script(gemini_rating=MAJOR, gpt5_rating=MINOR))
    result = await server.verify(_request("Apples are red."))
    assert result.eligible and result.eligibility_deciding_judge == "gpt-5"
    assert result.eligibility_ratings == {"gemini-2.5-flash": "Major Issue(s)", "gpt-5": "Minor Issue(s)"}
    stages = [(name, stage) for name, stage, _ in calls]
    assert stages[:4] == [(GEMINI, "baseline"), (GEMINI, "eligibility"), (GPT5, "baseline"), (GPT5, "eligibility")]
    assert "baseline answer from facts_grounding_judge_gpt5" in calls[3][2].messages[1]["content"]
    assert result.reward == 1.0


async def test_ineligible_response_scores_zero_but_unadjusted_score_is_kept():
    server, calls = _server(_script(gemini_rating=MAJOR, gpt5_rating=MAJOR))
    result = await server.verify(_request("Fruit."))
    assert not result.eligible and result.reward == 0.0 and result.eligibility_deciding_judge is None
    assert result.unadjusted_score == 1.0 and result.grounding_score == 1.0
    assert result.is_eligible == 0.0 and result.eligibility_invalid_count == 0
    server, calls = _server(_script(gemini_rating=MAJOR, gpt5_rating=MAJOR), grade_ineligible_responses=False)
    result = await server.verify(_request("Fruit."))
    assert result.unadjusted_score is None and result.grounding_verdicts == {} and result.reward == 0.0
    assert [stage for _, stage, _ in calls] == ["baseline", "eligibility", "baseline", "eligibility"]


async def test_judges_disagree_gives_half_credit():
    server, _ = _server(_script(gpt5_grounding=UNSUPPORTED))
    result = await server.verify(_request("Apples are red."))
    assert result.reward == 0.5 and result.judges_agree is False
    assert result.grounding_label_counts["gpt-5"]["not_supported"] == 1
    assert result.is_grounded_all_judges == 0.0


async def test_invalid_eligibility_verdict_counts_as_eligible_like_the_starter():
    server, _ = _server(_script(gemini_rating="I cannot decide."))
    result = await server.verify(_request("Apples are red."))
    assert result.eligible and result.eligibility_ratings["gemini-2.5-flash"] == "Invalid"
    assert result.eligibility_invalid_count == 1


async def test_unparseable_grounding_reply_is_not_grounded_and_flagged():
    server, _ = _server(_script(gemini_grounding="garbage"))
    result = await server.verify(_request("Apples are red."))
    assert result.grounding_verdicts["gemini-2.5-flash"] is False
    assert result.grounding_parsed_sentences["gemini-2.5-flash"] == 0
    assert result.grounding_unparseable_lines["gemini-2.5-flash"] == 1
    assert result.reward == 0.5


async def test_empty_or_truncated_generation_is_still_judged_and_flagged():
    server, _ = _server(_script(gemini_rating=MAJOR, gpt5_rating=MAJOR, gemini_grounding="", gpt5_grounding=""))
    result = await server.verify(_request("", incomplete=True))
    assert result.generation_empty and result.generation_truncated and result.generation_chars == 0
    assert result.reward == 0.0 and not result.eligible


async def test_transport_failure_surfaces_as_judge_error():
    server, _ = _server(_script())
    server.server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(JudgeError):
        await server.verify(_request("Apples are red."))


def test_judge_labels_must_match_servers():
    with pytest.raises(ValueError, match="judge_labels"):
        _server(_script(), judge_labels=["only-one"])


def test_metrics_keep_eligibility_grounding_and_judge_validity_separate():
    server, _ = _server(_script())

    def row(
        reward,
        eligible,
        verdicts,
        *,
        deciding=None,
        ratings=None,
        domain="Medical",
        hl="Q&A",
        unadj=None,
        parsed=None,
        bad=None,
        truncated=False,
    ):
        return {
            "reward": reward,
            "eligible": eligible,
            "is_eligible": 1.0 if eligible else 0.0,
            "eligibility_deciding_judge": deciding,
            "eligibility_ratings": ratings or {},
            "eligibility_invalid_count": sum(1 for value in (ratings or {}).values() if value == "Invalid"),
            "grounding_verdicts": verdicts,
            "grounding_parsed_sentences": parsed or {label: 3 for label in verdicts},
            "grounding_unparseable_lines": bad or {label: 0 for label in verdicts},
            "grounding_score": (sum(verdicts.values()) / len(verdicts)) if verdicts else None,
            "unadjusted_score": unadj
            if unadj is not None
            else ((sum(verdicts.values()) / len(verdicts)) if verdicts else None),
            "judges_agree": (len(set(verdicts.values())) == 1) if verdicts else None,
            "is_grounded_all_judges": 1.0 if verdicts and all(verdicts.values()) else 0.0,
            "generation_empty": False,
            "generation_truncated": truncated,
            "policy_input_tokens": 1000,
            "domain": domain,
            "high_level_type": hl,
            "type": "Fact Finding",
        }

    both = {"gemini-2.5-flash": True, "gpt-5": True}
    split = {"gemini-2.5-flash": True, "gpt-5": False}
    tasks = [
        [row(1.0, True, both, deciding="gemini-2.5-flash", ratings={"gemini-2.5-flash": "No Issues"})],
        [
            row(
                0.5,
                True,
                split,
                deciding="gpt-5",
                ratings={"gemini-2.5-flash": "Major Issue(s)", "gpt-5": "Invalid"},
                domain="Legal",
            )
        ],
        [
            row(
                0.0,
                False,
                both,
                ratings={"gemini-2.5-flash": "Major Issue(s)", "gpt-5": "Major Issue(s)"},
                domain="Legal",
                truncated=True,
            )
        ],
        [
            row(
                0.0,
                True,
                {"gemini-2.5-flash": False, "gpt-5": False},
                deciding="gemini-2.5-flash",
                ratings={"gemini-2.5-flash": "Minor Issue(s)"},
                parsed={"gemini-2.5-flash": 0, "gpt-5": 2},
                bad={"gemini-2.5-flash": 1, "gpt-5": 0},
            )
        ],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["factuality_score"] == pytest.approx(1.5 / 4)
    assert metrics["eligibility_rate"] == pytest.approx(3 / 4) and metrics["num_eligible"] == 3
    assert metrics["unadjusted_factuality_score"] == pytest.approx((1.0 + 0.5 + 1.0 + 0.0) / 4)
    assert metrics["grounded_rate_eligible/gemini-2.5-flash"] == pytest.approx(2 / 3)
    assert metrics["grounded_rate_eligible/gpt-5"] == pytest.approx(1 / 3)
    assert metrics["grounded_all_judges_rate_eligible"] == pytest.approx(1 / 3)
    assert metrics["judge_disagreement_rate_eligible"] == pytest.approx(1 / 3)
    assert metrics["eligibility_decided_by/gemini-2.5-flash"] == pytest.approx(2 / 4)
    assert metrics["eligibility_decided_by/gpt-5"] == pytest.approx(1 / 4)
    assert metrics["eligibility_rating/gemini-2.5-flash/Major Issue(s)"] == pytest.approx(2 / 4)
    assert metrics["eligibility_rating/gpt-5/Invalid"] == pytest.approx(1 / 2)
    assert metrics["eligibility_invalid_rate"] == pytest.approx(1 / 4)
    assert metrics["grounding_parse_empty_rate/gemini-2.5-flash"] == pytest.approx(1 / 4)
    assert metrics["grounding_unparseable_line_rate/gemini-2.5-flash"] == pytest.approx(1 / 4)
    assert metrics["generation_truncated_rate"] == pytest.approx(1 / 4)
    assert metrics["policy_input_tokens_max"] == 1000
    assert metrics["factuality_score/domain/Legal"] == pytest.approx(0.25)
    assert metrics["num_rollouts/high_level_type/Q&A"] == 4
    assert (
        0.0
        <= metrics["factuality_score_ci95_low"]
        <= metrics["factuality_score"]
        <= metrics["factuality_score_ci95_high"]
    )
    assert server.compute_metrics([]) == {}
    key = server.get_key_metrics(metrics)
    assert {
        "factuality_score",
        "unadjusted_factuality_score",
        "eligibility_rate",
        "grounded_rate_eligible/gpt-5",
    } <= set(key)


def _fixture_script(server_name, stage, messages):
    if stage == "baseline":
        return "baseline"
    response_text = messages[1]["content"]
    if stage == "eligibility":
        return MAJOR if "[MAJOR]" in response_text else NO_ISSUES
    return UNSUPPORTED if "[UNSUPPORTED]" in response_text else GROUNDED


def test_verifier_fixture_contract():
    fixture = VerifierFixture(
        server_factory=lambda: _server(_fixture_script)[0],
        request_model=FACTSGroundingV2VerifyRequest,
        cases_path=TESTS_DIR / "verifier_cases.jsonl",
        reseed=lambda server, request: None,
    )
    results = asyncio.run(
        exercise_verifier_fixture(fixture, reward_range=(0.0, 1.0), higher_is_better=True, determinism="seeded")
    )
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed", "determinism"}
