# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import itertools
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.jeebench.app import (
    JeebenchResourcesServer,
    JeebenchResourcesServerConfig,
    JeebenchVerifyRequest,
    compute_score,
    extract_answer,
    normalize_answer,
    response_fingerprint,
    visible_answer,
)


def response(text="B", **updates):
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 1,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **updates,
        }
    )


def body(text="B", *, question_type="MCQ", gold="B", language="hi", **updates):
    return JeebenchVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "question"},
            "response": response(text),
            "verifier_metadata": {
                "task_id": "synthetic",
                "language": language,
                "subject": "math",
                "question_type": question_type,
                "gold": gold,
            },
            **updates,
        }
    )


@pytest.fixture
def server():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return JeebenchResourcesServer(
        config=JeebenchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="app.py", name="jeebench"),
        server_client=client,
    )


def test_score_all_multi_option_combinations():
    options = ["".join(combo) for count in range(5) for combo in itertools.combinations("ABCD", count)]
    for gold in options[1:]:
        for predicted in options:
            expected = 1 if predicted == gold else 0.25 * len(predicted) if set(predicted) < set(gold) else 0
            assert compute_score(gold, predicted or "None", "MCQ(multiple)") == expected
            assert compute_score(gold, predicted or "None", "MCQ") == float(predicted == gold)


@pytest.mark.parametrize(
    "gold,answer,kind,expected",
    [
        ("AB", "BAA", "MCQ(multiple)", 1),
        ("ABCD", "BCD", "MCQ(multiple)", 0.75),
        ("AB", "A", "MCQ(multiple)", 0.25),
        ("AB", "AC", "MCQ(multiple)", 0),
        ("3", "3.009", "Integer", 1),
        ("3", "3.011", "Integer", 0),
        ("1", "1.01", "Numeric", 0),  # Match upstream float behavior, not decimal rounding.
        ("0", "0.01", "Numeric", 1),
        ("3", "None", "Numeric", 0),
        ("3", "nan", "Numeric", 0),
    ],
)
def test_score_matches_upstream_formula(gold, answer, kind, expected):
    assert compute_score(gold, answer, kind) == expected


def test_unknown_type_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        compute_score("A", "A", "bogus")


@pytest.mark.parametrize(
    "text,kind,expected",
    [
        ("(b)", "MCQ", "B"),
        ("A, C and D.", "MCQ(multiple)", "ACD"),
        ("ABD", "MCQ(multiple)", "ABD"),
        ("a and b", "MCQ", None),
        ("Maybe B", "MCQ", None),
        ("E", "MCQ", None),
        ("()", "MCQ", None),
        ("$\\text{\\mathrm{B}}$", "MCQ", "B"),
        ("None", "Integer", "None"),
        ("१.२३", "Numeric", "१.२३"),
        ("१२", "Integer", "12"),
        ("−2.5", "Numeric", "-2.5"),
        ("1e3", "Integer", "1000"),
        ("3.0", "Integer", "3"),
        ("3.001", "Integer", None),
        ("-1", "Integer", None),
        ("1e999", "Numeric", None),
        ("1/2", "Numeric", None),
        ("nan", "Numeric", None),
        ("2 m/s", "Numeric", None),
    ],
)
def test_answer_normalization_is_type_aware(text, kind, expected):
    assert normalize_answer(text, kind) == expected


@pytest.mark.parametrize(
    "text,kind,answer,method",
    [
        (r"Intermediate \boxed{A}; final \boxed{\text{B}}", "MCQ", "B", "boxed"),
        (r"\boxed{A} then \boxed{BC}", "MCQ", "None", "unparsed"),
        (r"\boxed{A} then \boxed{B", "MCQ", "None", "unparsed"),
        ("\\boxed{A}\nFinal answer: B", "MCQ", "B", "answer_marker"),
        ("Answer: A\n\\boxed{B}", "MCQ", "B", "boxed"),
        ("Answer: A\nFinal answer: B", "MCQ", "B", "answer_marker"),
        ("**Answer**: B", "MCQ", "B", "answer_marker"),
        (r"Final Answer: (B) $6.4 \times 10^{-34}$", "MCQ", "B", "answer_marker"),
        ("Final Answer: B: option text", "MCQ", "None", "unparsed"),
        ("Final Answer: (B) or (C)", "MCQ", "None", "unparsed"),
        ("Final Answer: (B), (C)", "MCQ", "None", "unparsed"),
        ("Final Answer: Maybe B", "MCQ", "None", "unparsed"),
        ("Answer: B\nFinal answer: uncertain", "MCQ", "None", "unparsed"),
        ("A complex proof\n\nB", "MCQ", "B", "bare_final_line"),
        ("None", "MCQ", "None", "bare_final_line"),
        ("The answer might be B", "MCQ", "None", "unparsed"),
        ("", "Numeric", "None", "unparsed"),
        ("3.00", "Numeric", "3.00", "bare_final_line"),
    ],
)
def test_extraction_never_searches_solution_for_arbitrary_answers(text, kind, answer, method):
    assert extract_answer(text, kind) == (answer, method)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>A</think>B", "B"),
        ("<THINKING>A</THINKING>B", "B"),
        ("A</think>B", "B"),
        ("<think>B", ""),
        ("Answer: A<think>reconsider</think>Final answer: B", "Final answer: B"),
        ("<think>A</think>Answer: B<think>C</think>Answer: D", "Answer: D"),
        ("reasoning</think>more reasoning</think>Final answer: B", "Final answer: B"),
        ("B", "B"),
    ],
)
def test_visible_answer_omits_tagged_reasoning(text, expected):
    assert visible_answer(response(text)) == expected


def test_visible_answer_omits_structured_reasoning_refusals_and_empty_outputs():
    payload = response().model_dump()
    payload["output"].insert(0, {"id": "thinking", "type": "reasoning", "summary": []})
    payload["output"][1]["content"].append({"type": "refusal", "refusal": "Not B"})
    assert visible_answer(NeMoGymResponse.model_validate(payload)) == "B"
    assert visible_answer(response(output=[])) == ""


@pytest.mark.parametrize(
    "gold,kind", [("E", "MCQ"), ("b", "MCQ"), ("None", "Numeric"), ("-2", "Integer"), ("", "Numeric")]
)
def test_invalid_gold_rejected(gold, kind):
    with pytest.raises(ValidationError, match="canonical gold"):
        body(gold=gold, question_type=kind)


def test_required_identity_fields_nonempty():
    with pytest.raises(ValidationError, match="nonempty"):
        body(language=" ")


@pytest.mark.parametrize(
    "fields,pattern",
    [
        ({"reviewed_answer": "B"}, "requires an answer_review_id"),
        ({"reviewed_answer": "B", "answer_review_id": " "}, "requires an answer_review_id"),
        ({"reviewed_answer": "prose", "answer_review_id": "review-1"}, "type-compatible"),
        ({"answer_review_id": "review-1"}, "require reviewed_answer"),
        ({"reviewed_response_sha256": "0" * 64}, "require reviewed_answer"),
        ({"reviewed_answer": "B", "answer_review_id": "review-1"}, "requires reviewed_response_sha256"),
        (
            {"reviewed_answer": "B", "answer_review_id": "review-1", "reviewed_response_sha256": "0" * 64},
            "does not match the response",
        ),
    ],
)
def test_review_provenance_required(fields, pattern):
    with pytest.raises(ValidationError, match=pattern):
        body(**fields)


async def test_verify_correct_wrong_partial_and_unparsed(server):
    correct = await server.verify(body("<think>A</think>Final answer: B"))
    assert correct.reward == 1 and correct.exact_match
    assert correct.extraction_source == "automatic" and correct.provisional
    assert not correct.extraction_failed and not correct.mask_sample
    wrong = await server.verify(body("A"))
    assert wrong.reward == 0 and not wrong.exact_match
    partial = await server.verify(body("A", question_type="MCQ(multiple)", gold="ACD"))
    assert partial.reward == 0.25 and not partial.exact_match
    conclusion = await server.verify(body("After discussion I conclude B is right"))
    assert conclusion.reward == 1 and conclusion.extracted_answer == "B"
    assert conclusion.extraction_method == "concluding_choices"
    failed = await server.verify(body("After discussion I cannot determine the answer"))
    assert failed.reward == 0 and failed.extraction_failed and failed.failure_kind == "jeebench:unparsed_answer"
    assert not failed.mask_sample
    server.server_client.post.assert_not_called()


async def test_human_review_and_reverify_stale_fields(server):
    fingerprint = response_fingerprint(response("ambiguous prose"))
    result = await server.verify(
        body(
            "ambiguous prose",
            reviewed_answer="B",
            answer_review_id="review-1",
            reviewed_response_sha256=fingerprint,
        )
    )
    assert result.reward == 1 and not result.provisional and result.extraction_source == "human_review"
    assert result.answer_review_id == "review-1"
    assert result.reviewed_response_sha256 == fingerprint
    with pytest.raises(ValidationError, match="does not match the response"):
        JeebenchVerifyRequest.model_validate(result.model_dump() | {"response": response("new unreviewed answer")})
    rerun = JeebenchVerifyRequest.model_validate(result.model_dump() | {"reviewed_answer": "None"})
    result = await server.verify(rerun)
    assert result.reward == 0 and result.extracted_answer == "None" and not result.extraction_failed


@pytest.mark.parametrize("fingerprint", ["", "f" * 63, "f" * 65, "g" * 64, "A" * 64])
def test_review_hash_requires_lowercase_sha256(fingerprint):
    with pytest.raises(ValidationError, match="String should match pattern"):
        body(reviewed_answer="B", answer_review_id="review-1", reviewed_response_sha256=fingerprint)


def test_response_fingerprint_uses_canonical_native_json_including_reasoning():
    value = response("अंतिम उत्तर: B")
    expected = hashlib.sha256(
        json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert response_fingerprint(value) == expected
    reordered = NeMoGymResponse.model_validate(dict(reversed(list(value.model_dump().items()))))
    assert response_fingerprint(reordered) == expected
    changed = value.model_dump()
    changed["output"].insert(0, {"id": "thinking", "type": "reasoning", "summary": []})
    assert response_fingerprint(NeMoGymResponse.model_validate(changed)) != expected
    with pytest.raises(ValueError, match="Out of range float"):
        response_fingerprint(value.model_copy(update={"temperature": float("nan")}))


@pytest.mark.parametrize("scope", ["status", "details", "message"])
async def test_truncation_visible(server, scope):
    request = body()
    if scope == "status":
        request.response.status = "incomplete"
    elif scope == "details":
        request.response = response(incomplete_details={"reason": "max_output_tokens"})
    else:
        request.response.output[0].status = "incomplete"
    assert (await server.verify(request)).candidate_truncated


async def test_partial_credit_metrics_and_subsets(server):
    partial = (await server.verify(body("A", question_type="MCQ(multiple)", gold="AB"))).model_dump()
    correct = (await server.verify(body())).model_dump()
    missing = (await server.verify(body("Unparseable prose", language="ta"))).model_dump()
    metrics = server.compute_metrics([[partial], [correct], [missing]])
    assert metrics["pass@1[avg-of-1]/jeebench_score"] == pytest.approx(100 * 1.25 / 3)
    assert metrics["pass@1[avg-of-1]/full_credit"] == pytest.approx(100 / 3)
    assert metrics["language/hi/pass@1[avg-of-1]/jeebench_score"] == 62.5
    assert metrics["language/ta/pass@1[avg-of-1]/extraction_failure"] == 100
    assert metrics["question_type/MCQ(multiple)/pass@1[avg-of-1]/jeebench_score"] == 25
    assert metrics["subject/math/pass@1[avg-of-1]/jeebench_score"] == pytest.approx(100 * 1.25 / 3)
    assert server.compute_metrics([]) == {}
    assert not any("majority" in key for key in server.compute_metrics([[partial, correct]]))


async def test_native_aggregate_metrics_and_reverify_mode(server):
    row = (await server.verify(body())).model_dump()
    row.update({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0})
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[row]))
    assert result.agent_metrics["pass@1[avg-of-1]/jeebench_score"] == 100
    assert result.key_metrics["pass@1[avg-of-1]/jeebench_score"] == 100
    assert await server.get_reverify_mode() == ReverifyMode.STATELESS


def test_five_synthetic_examples_are_valid_requests():
    path = Path(__file__).resolve().parents[1] / "data/example.jsonl"
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    assert len(rows) == 5
    for row in rows:
        JeebenchVerifyRequest.model_validate(row | {"response": response()})


@pytest.mark.parametrize(
    "text,kind,gold",
    [
        ("Final Answer: (B) isn't correct.", "MCQ", "B"),
        ("Final Answer: (C) cannot be correct.", "MCQ", "C"),
        ("Final Answer: (B) isn’t correct.", "MCQ", "B"),
        ("Final Answer: (B) can't be correct.", "MCQ", "B"),
        ("Final Answer: (B) is ruled out.", "MCQ", "B"),
        ("Final Answer: (B) fails the test.", "MCQ", "B"),
        ("अंतिम उत्तर: A\nसही विकल्प B isn't correct.", "MCQ", "A"),
        ("अंतिम उत्तर: A\nसही विकल्प B एक असमर्थित विवरण", "MCQ", "A"),
        ("Final Answer: (A), (B) aren't correct.", "MCQ(multiple)", "AB"),
        ("Final Answer: (B) only if the assumption holds.", "MCQ", "B"),
        ("Final Answer: (B) as it yields a valid configuration only if the assumption holds.", "MCQ", "B"),
        ("Final answer: 2. This isn't correct.", "Numeric", "2"),
        ("Final answer: 2. This cannot be correct.", "Numeric", "2"),
        ("Final answer: 2. Only if the assumption holds.", "Numeric", "2"),
        ("Final answer: 2. This isn’t correct.", "Integer", "2"),
        ("Final answer: 2. This is ruled out.", "Numeric", "2"),
        ("Final answer: 2. This fails the test.", "Numeric", "2"),
        ("Final answer: 2. This follows from symmetry only if the assumption holds.", "Numeric", "2"),
        ("Final answer: 2, as required. This fails the test.", "Numeric", "2"),
        ("Assume Y = 2\n\nThis leads to a contradiction.", "Numeric", "2"),
        ("Suppose Y = 2\n\nThis leads to a contradiction.", "Numeric", "2"),
        ("Assume Y = 2\n\nThis is the coefficient.", "Numeric", "2"),
        ("Try Y = 2\n\nThis is the coefficient.", "Numeric", "2"),
        ("The tentative assignment is Y = 2\n\nThis is the coefficient.", "Numeric", "2"),
        ("Assume\nY = 2\n\nThis is the coefficient.", "Numeric", "2"),
        ("Y = 2\n\nThis leads to a contradiction.", "Numeric", "2"),
        ("Y = 2\n\nThis cannot be correct.", "Numeric", "2"),
        ("Y = 2\n\nOnly if the assumption holds.", "Numeric", "2"),
    ],
)
async def test_rejected_or_conditional_answers_receive_no_credit(server, text, kind, gold):
    result = await server.verify(body(text, question_type=kind, gold=gold, question="What is the value of $Y$?"))
    assert result.extracted_answer == "None"
    assert result.extraction_method == "unparsed"
    assert result.extraction_failed and result.failure_kind == "jeebench:unparsed_answer"
    assert result.reward == 0 and not result.exact_match
    assert result.provisional and not result.mask_sample


@pytest.mark.parametrize(
    "text,kind,gold",
    [
        ("Final Answer: (C), as it yields a valid configuration.", "MCQ", "C"),
        ("Final Answer: (A), (B), (D) are correct; (C) is not correct.", "MCQ(multiple)", "ABD"),
        ("Final answer: 2. This is the required coefficient.", "Numeric", "2"),
        ("Final answer: 2, as required.", "Integer", "2"),
        ("Final answer: 2. This follows from symmetry.", "Numeric", "2"),
        ("Therefore, Y = 2\n\nThe corresponding rate is 2e-5.", "Numeric", "2"),
        ("Y = 2\n\nThe corresponding rate is 2e-5.", "Numeric", "2"),
    ],
)
async def test_supported_affirmative_answers_retain_credit(server, text, kind, gold):
    result = await server.verify(body(text, question_type=kind, gold=gold, question="What is the value of $Y$?"))
    assert result.reward == 1 and result.exact_match
    assert not result.extraction_failed and result.extraction_method != "unparsed"


@pytest.mark.parametrize("separator", [" ", "\n", "\n\n"])
@pytest.mark.parametrize("suffix", ["This is ruled out.", "This selection is invalid.", "This fails the test."])
@pytest.mark.parametrize(
    "selection",
    [
        "Final Answer: (B) is correct.",
        "B is correct.",
        r"\boxed{B}",
        "(A) is false.\n(B) is true.\n(C) is false.\n(D) is false.",
    ],
)
async def test_option_retraction_preserves_all_concluding_context(server, separator, suffix, selection):
    result = await server.verify(body(selection + separator + suffix))
    assert result.extracted_answer == "None" and result.extraction_failed
    assert result.reward == 0 and result.extraction_method == "unparsed"


@pytest.mark.parametrize("separator", [" ", "\n", "\n\n"])
@pytest.mark.parametrize("selection,negation", [("চূড়ান্ত উত্তর: (B) সঠিক", "নয়।"), ("இறுதி விடை: (B) சரி", "அல்ல.")])
@pytest.mark.parametrize("question", ["", "Select one.\n[A] red\n[B] blue\n[C] green\n[D] yellow"])
async def test_localized_verdict_cannot_discard_unknown_negation(server, selection, negation, separator, question):
    result = await server.verify(body(selection + separator + negation, question=question))
    assert result.extracted_answer == "None" and result.extraction_failed and result.reward == 0


@pytest.mark.parametrize(
    "text",
    [
        "Try\nY = 2\n\nThis fails the test.",
        "Y = 2\n\nThis is ruled out.",
        "Y = 2\n\nThis selection is invalid.",
        "Consider this trial:\nY = 2\n\nThis is the required coefficient.",
        "Try\nY = 2\n\nThe corresponding rate is 2e-5.",
        "Try\n\nY = 2",
        "Suppose\n\nY = 2",
        "Y = 2\n\nAn unsupported qualification.",
        "Final answer: Y = 2\n\nThis is ruled out.",
        "Final answer: 2\n\nThis fails the test.",
        "Final answer: 2" + " " * 4096 + "This is ruled out.",
        "Try\n" + "\n" * 4096 + "Y = 2",
        "Y = 2\n\nThe corresponding rate is 2e-5. This is ruled out.",
    ],
)
@pytest.mark.parametrize("question", ["What is the value of $Y$?", "Find the answer."])
async def test_numeric_context_outside_assignment_is_not_discarded(server, text, question):
    result = await server.verify(body(text, question_type="Numeric", gold="2", question=question))
    assert result.extracted_answer == "None" and result.extraction_failed
    assert result.reward == 0 and result.extraction_method == "unparsed"


@pytest.mark.parametrize(
    "text,kind,gold",
    [
        ("Final answer: B is correct.\n\nThis is the required answer.", "MCQ", "B"),
        ("Final answer: B is correct.\nC is not correct.", "MCQ", "B"),
        ("(A) is false.\n(B) is true.\n(C) is false.\n(D) is false.", "MCQ", "B"),
        ("চূড়ান্ত উত্তর: (B) সঠিক।", "MCQ", "B"),
        ("இறுதி விடை: (B) சரி.", "MCQ", "B"),
        ("Y = 2\n\nThis is the required coefficient.", "Numeric", "2"),
        ("Therefore, Y = 2\n\nThe corresponding rate is 2e-5.", "Numeric", "2"),
        ("Try Y = 5. This fails the test.\n\nFinal answer: 2", "Numeric", "2"),
        ("Try B. This is ruled out.\n\nFinal answer: C", "MCQ", "C"),
    ],
)
async def test_explicit_conclusion_and_supported_context_retain_credit(server, text, kind, gold):
    result = await server.verify(body(text, question_type=kind, gold=gold, question="What is the value of $Y$?"))
    assert result.reward == 1 and not result.extraction_failed
