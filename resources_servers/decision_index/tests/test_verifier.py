# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config.config_types import AggregateMetricsRequest

from ..app import (
    DecisionIndexResourcesServer,
    DecisionIndexResourcesServerConfig,
    DecisionIndexVerifier,
    DecisionIndexVerifyRequest,
)
from ..verifier import (
    _last_json_object,
    engine_answers,
    flatten_scores,
    llm_answers,
    load_kit,
    row_reward,
    score_index,
    scoring_row,
    strip_thinking,
)


EXAMPLES = [
    json.loads(line) for line in (Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()
]
MMLU_PRO, GSM8K, BFCL, WHEN2CALL, HOME = EXAMPLES

CHOICE = {"type": "choice", "instructions": "Pick one.", "criteria": {"A": "first", "B": "second", "1": "third"}}
NOUL = {"type": "noul", "instructions": "Is it true?"}


def _response(text: str) -> dict:
    return {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "output": [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _request(row: dict, text: str) -> DecisionIndexVerifyRequest:
    return DecisionIndexVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": "x"},
            "response": _response(text),
            "verifier_metadata": row["verifier_metadata"],
        }
    )


def _gold_answers(di_row: dict) -> dict:
    answers = {}
    for key, q in di_row["questions"].items():
        gold = di_row["expected"][key]
        if q["type"] == "choice":
            choice = gold if gold is not None else next(iter(q["criteria"]))
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {o: float(o == choice) for o in q["criteria"]},
            }
        else:
            answers[key] = {"type": "noul", "noul": 1.0 if gold else 0.0}
    return answers


def _envelope(status: str, **fields) -> str:
    return json.dumps({"di_engine": {"status": status, **fields}})


def _server(mode: str) -> DecisionIndexResourcesServer:
    config = DecisionIndexResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="", answer_mode=mode)
    return DecisionIndexResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestParsing:
    def test_strip_thinking_removes_closed_and_orphaned_blocks(self) -> None:
        assert strip_thinking("<think>a</think> b") == "b"
        assert strip_thinking("reasoning only closes</thinking> answer") == "answer"
        assert strip_thinking(None) == ""

    def test_last_json_object_prefers_fences_and_takes_the_last_object(self) -> None:
        assert _last_json_object('```json\n{"q": "A"}\n``` then {"q": "B"}') == {"q": "A"}
        assert _last_json_object('{"q": "A"} and later {"q": "B"}') == {"q": "B"}
        assert _last_json_object("```json\nnot json\n``` {broken") is None
        assert _last_json_object('```\n[1, 2]\n``` {"q": "A"}') == {"q": "A"}

    def test_llm_choice_noul_and_wrapped_answers(self) -> None:
        questions = {"c": CHOICE, "n": NOUL}
        status, response, error = llm_answers('{"answers": {"c": "B", "n": "Yes"}}', questions)
        assert (status, error) == ("ok", None)
        assert response["answers"]["c"]["probabilities"] == {"A": 0.0, "B": 1.0, "1": 0.0}
        assert response["answers"]["n"] == {"type": "noul", "noul": 1.0}
        status, response, _ = llm_answers('{"c": 1, "n": false}', questions)
        assert status == "ok" and response["answers"]["c"]["choice"] == "1" and response["answers"]["n"]["noul"] == 0.0

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "empty reply"),
            ("no json here", "no JSON object"),
            ('{"c": "A"}', "'n' not answered"),
            ('{"c": "Z", "n": true}', "not one of its options"),
            ('{"c": true, "n": true}', "not one of its options"),
            ('{"c": "A", "n": "maybe"}', "not true or false"),
        ],
    )
    def test_llm_unparseable_replies_are_errors(self, text: str, message: str) -> None:
        status, response, error = llm_answers(text, {"c": CHOICE, "n": NOUL})
        assert status == "error" and response is None and message in error

    def test_llm_unknown_question_type_is_unsupported(self) -> None:
        status, _, _ = llm_answers('{"s": 3}', {"s": {"type": "score", "criteria": ["a", "b"]}})
        assert status == "unsupported"

    def test_engine_envelopes(self) -> None:
        questions = {"c": CHOICE}
        good = {"answers": {"c": {"type": "choice", "choice": "A", "probabilities": {"A": 0.7, "B": 0.2, "1": 0.1}}}}
        assert engine_answers(_envelope("ok", response=good, wall_ms=3.0), questions) == ("ok", good, None, 3.0, False)
        assert engine_answers("plain text", questions)[0] == "error"
        assert engine_answers(_envelope("unsupported", error="too many tokens"), questions)[:1] == ("unsupported",)
        status, _, error, _, failed = engine_answers(_envelope("error", error="HTTP 503"), questions)
        assert (status, error, failed) == ("error", "HTTP 503", True)
        bad = {"answers": {"c": {"type": "choice", "choice": "A", "probabilities": {"A": 1.0}}}}
        status, _, error, _, failed = engine_answers(_envelope("ok", response=bad), questions)
        assert status == "error" and "invalid engine answers" in error and failed is False


class TestRowReward:
    def test_reward_needs_every_scored_question(self) -> None:
        di_row = HOME["verifier_metadata"]["di_row"]
        answers = _gold_answers(di_row)
        assert row_reward(di_row, {"answers": answers}) == (
            1.0,
            len([g for g in di_row["expected"].values() if g is not None]),
        )
        key = next(
            k for k, q in di_row["questions"].items() if q["type"] == "choice" and di_row["expected"][k] is not None
        )
        other = next(o for o in di_row["questions"][key]["criteria"] if o != di_row["expected"][key])
        answers[key] = {**answers[key], "choice": other}
        assert row_reward(di_row, {"answers": answers})[0] == 0.0
        assert row_reward(di_row, None) == (0.0, 0)

    def test_accepted_answers_primary_fields_and_unscored_fields(self) -> None:
        chess = {
            "_evaluation": {"catalog_id": 31},
            "questions": {"move": CHOICE},
            "expected": {"move": "A"},
            "scoring": {"accepted": ["A", "B"]},
        }
        assert row_reward(chess, {"answers": {"move": {"choice": "B"}}})[0] == 1.0
        phish = {
            "_evaluation": {"catalog_id": 56},
            "questions": {"verdict": CHOICE, "is_phishing": NOUL, "sig": NOUL},
            "expected": {"verdict": "A", "is_phishing": True, "sig": None},
            "scoring": {"unscored_fields": ["sig"]},
        }
        answers = {"verdict": {"choice": "A"}, "is_phishing": {"noul": 0.1}, "sig": {"noul": 0.1}}
        assert row_reward(phish, {"answers": answers}) == (1.0, 1)

    def test_scoring_row_drops_state_and_instructions(self) -> None:
        di_row = {**BFCL["verifier_metadata"]["di_row"], "state": {"big": "context"}}
        row = scoring_row(di_row)
        assert "state" not in row and all("instructions" not in q for q in row["questions"].values())
        assert row["expected"] == di_row["expected"]


class TestVerify:
    def test_decision_mode_masks_engine_failures_only(self) -> None:
        server = _server("decision")
        failed = asyncio.run(server.verify(_request(HOME, _envelope("error", error="HTTP 503"))))
        assert failed.mask_sample is True and failed.failure_kind == "decision_index:engine_error"
        assert failed.di_result["status"] == "error" and failed.reward == 0.0
        unsupported = asyncio.run(server.verify(_request(HOME, _envelope("unsupported", error="too many tokens"))))
        assert not unsupported.mask_sample and unsupported.di_status == "unsupported"
        answers = {"model": "x", "answers": _gold_answers(HOME["verifier_metadata"]["di_row"])}
        ok = asyncio.run(server.verify(_request(HOME, _envelope("ok", response=answers, wall_ms=12.5))))
        assert (
            ok.reward == 1.0 and ok.di_result["total_wall_ms"] == 12.5 and "verifier_metadata" not in ok.model_dump()
        )

    def test_llm_mode_never_reads_envelopes(self) -> None:
        answers = {"model": "x", "answers": _gold_answers(MMLU_PRO["verifier_metadata"]["di_row"])}
        result = asyncio.run(_server("llm").verify(_request(MMLU_PRO, _envelope("ok", response=answers))))
        assert result.di_mode == "llm" and result.reward == 0.0 and result.di_status == "error"

    def test_auto_mode_detects_the_envelope(self) -> None:
        answers = {"model": "x", "answers": _gold_answers(GSM8K["verifier_metadata"]["di_row"])}
        result = asyncio.run(DecisionIndexVerifier().verify(_request(GSM8K, _envelope("ok", response=answers))))
        assert result.di_mode == "decision" and result.reward == 1.0


class TestRunScore:
    def _records(self, mode: str, text_for) -> list[dict]:
        server = _server(mode)
        records = []
        for index, row in enumerate(EXAMPLES):
            record = asyncio.run(server.verify(_request(row, text_for(row)))).model_dump()
            records.append({**record, "_ng_task_index": index, "_ng_rollout_index": 0})
        return records

    def test_aggregate_scores_every_rollout_with_the_kit(self) -> None:
        def text_for(row):
            di_row = row["verifier_metadata"]["di_row"]
            if di_row["_evaluation"]["catalog_id"] == 1:
                return _envelope("error", error="HTTP 503")
            return _envelope("ok", response={"answers": _gold_answers(di_row)})

        records = self._records("decision", text_for)
        server = _server("decision")
        result = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=records)))
        metrics = result.agent_metrics
        assert metrics["decision_index_requests"] == 5 and metrics["decision_index_full_suite"] == 0.0
        assert metrics["decision_index_status/error"] == 1 and metrics["decision_index_status/ok"] == 4
        # MMLU-Pro answered right: skill 100. BFCL's only request failed: skill 0 (unanswered counts as wrong).
        assert metrics["decision_index_skill/57:MMLU-Pro"] == pytest.approx(100.0)
        assert metrics["decision_index_skill/1:BFCL"] == pytest.approx(0.0)
        assert "decision_index" in result.key_metrics and any(
            k.startswith("decision_index_area/") for k in result.key_metrics
        )

    def test_score_index_keeps_one_answer_per_request(self) -> None:
        records = self._records("llm", lambda row: "{}")
        scores = score_index(records + records)
        assert scores["requests_in_run"] == 5 and scores["counts"] == {"error": 5}
        assert score_index([{"reward": 0.0}]) == {} and flatten_scores({}) == {}


def test_load_kit_rebinds_a_shadowed_package(monkeypatch) -> None:
    kit = load_kit()
    shadow = type(sys)("decision_index")
    shadow.__file__ = str(Path(__file__).parents[1] / "__init__.py")
    monkeypatch.setitem(sys.modules, "decision_index", shadow)
    rebound = load_kit()
    assert rebound is not shadow and (Path(rebound.__file__).parent / "scoring").is_dir()
    assert load_kit() is rebound and kit.__file__ == rebound.__file__
