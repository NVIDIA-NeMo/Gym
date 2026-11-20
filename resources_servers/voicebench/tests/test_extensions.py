# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All-subset runtime, message-text answers, unparsed-is-wrong accuracy, per-split metrics and SD-QA PEDANT."""

import asyncio
import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from nemo_gym.config import AggregateMetricsRequest
from resources_servers.voicebench.app import (
    VoiceBenchConfig,
    VoiceBenchServer,
    final_answer,
    message_text,
)
from resources_servers.voicebench.scorers import bbh, mcq
from resources_servers.voicebench.scorers.pedant import PortableLinearClassifier, load_pedant, pedant_correct
from resources_servers.voicebench.scorers.reference_qa import ReferenceQAScorer
from resources_servers.voicebench.tests.test_app import _global_config, candidate, request, server
from resources_servers.voicebench.tests.test_judge import client, judge_settings


def reasoning_only(text: str = "so the answer is C"):
    return candidate(
        status="incomplete",
        output=[{"id": "rs", "type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}],
    )


# ---- Message-text answers ----------------------------------------------------------------------------------------


def test_message_text_is_the_last_assistant_message_as_returned():
    assert message_text(candidate("\nThe answer is: C\n")) == "\nThe answer is: C\n"
    assert message_text(reasoning_only()) == ""
    two = candidate("Maybe A").output[0].model_dump(), candidate("The answer is B").output[0].model_dump()
    # An earlier (commentary) message is not fused into the answer: "Maybe AThe answer is B" would let the MCQ and BBH
    # parsers pick up the commentary's letter.
    response = candidate(output=list(two))
    assert message_text(response) == final_answer(response) == "The answer is B"


def test_message_mode_scores_a_response_without_final_message_as_empty(tmp_path):
    reasoned = reasoning_only("The answer is C.")
    assert final_answer(reasoned) == "The answer is C."
    legacy = asyncio.run(
        server(tmp_path, subset="mmsu", scoring_type="mcq").verify(
            request(subset="mmsu", response=reasoned, reference="C")
        )
    )
    assert legacy.scoring_answer == "The answer is C." and legacy.prediction == "C" and legacy.reward == 1

    strict = server(tmp_path, subset="mmsu", scoring_type="mcq", answer_extraction="message")
    result = asyncio.run(strict.verify(request(subset="mmsu", response=reasoned, reference="C")))
    assert result.candidate_answer == result.scoring_answer == "" and result.parse_failure
    assert not result.candidate_valid and result.scoring_metrics["correct_parsed"] == 0


def test_message_mode_keeps_an_empty_advbench_answer_safe(tmp_path):
    svc = server(tmp_path, subset="advbench", scoring_type="refusal", answer_extraction="message")
    assert asyncio.run(svc.verify(request(subset="advbench", response=reasoning_only()))).reward == 1
    legacy = server(tmp_path, subset="advbench", scoring_type="refusal")
    assert asyncio.run(legacy.verify(request(subset="advbench", response=reasoning_only("")))).scoring_answer == "nan"


# ---- Unparsed-is-wrong accuracy ------------------------------------------------------------------------------------


def test_correct_parsed_excludes_lucky_guesses():
    guessed = random.Random(41).choice(["A", "B", "C", "D"])
    assert mcq.score("no idea", {"reference": guessed}, rng=random.Random(41)).metrics["correct_parsed"] == 0
    assert (
        mcq.score(f"The answer is {guessed}", {"reference": guessed}, rng=random.Random(41)).metrics["correct_parsed"]
        == 1
    )
    expected = random.Random(23).choice([0, 1])
    lucky = bbh.score("", {"id": "navigate_0", "reference": "yes" if expected else "no"}, rng=random.Random(23))
    assert lucky.metrics["correct"] == 1 and lucky.metrics["correct_parsed"] == 0


def test_mcq_aggregate_reports_unparsed_wrong_and_parse_rate(tmp_path):
    svc = server(tmp_path, expected=4, subset="openbookqa", scoring_type="mcq")
    rows = [
        {
            "sample_id": str(i),
            "score_valid": True,
            "candidate_valid": True,
            "reward": float(correct),
            "scoring_metrics": m,
        }
        for i, (correct, m) in enumerate(
            [
                (1, {"correct": 1, "correct_parsed": 1, "total": 1, "parse_failure": 0}),
                (0, {"correct": 0, "correct_parsed": 0, "total": 1, "parse_failure": 0}),
                (1, {"correct": 1, "correct_parsed": 0, "total": 1, "parse_failure": 1}),
                (0, {"correct": 0, "correct_parsed": 0, "total": 1, "parse_failure": 1}),
            ]
        )
    ]
    metrics = svc.compute_metrics([[row] for row in rows])
    assert metrics["voicebench/legacy/acc"] == 50 and metrics["voicebench/legacy/acc_unparsed_wrong"] == 25
    assert metrics["voicebench/legacy/parse_rate"] == 50 and metrics["voicebench/legacy/fail"] == 50


def test_aggregate_without_correct_parsed_keeps_the_legacy_metrics_only(tmp_path):
    svc = server(tmp_path, subset="openbookqa", scoring_type="mcq")
    row = {
        "sample_id": "0",
        "score_valid": True,
        "candidate_valid": True,
        "scoring_metrics": {"correct": 1, "total": 1},
    }
    metrics = svc.compute_metrics([[row]])
    assert metrics["voicebench/legacy/acc"] == 100
    assert "voicebench/legacy/acc_unparsed_wrong" not in metrics and "voicebench/legacy/parse_rate" not in metrics


# ---- Per-split metrics -----------------------------------------------------------------------------------------------


def test_split_metrics_are_reported_for_two_or_more_splits(tmp_path):
    svc = server(tmp_path, expected=3, subset="mmsu", scoring_type="mcq")
    rows = [
        {
            "sample_id": str(i),
            "verifier_metadata": {"split": split},
            "score_valid": True,
            "candidate_valid": True,
            "reward": float(correct),
            "scoring_metrics": {"correct": correct, "correct_parsed": correct, "total": 1, "parse_failure": 0},
        }
        for i, (split, correct) in enumerate([("law", 1), ("law", 0), ("physics", 1)])
    ]
    metrics = svc.compute_metrics([[row] for row in rows])
    assert metrics["voicebench/split/law/count"] == 2 and metrics["voicebench/split/law/legacy/acc"] == 50
    assert metrics["voicebench/split/physics/legacy/acc_unparsed_wrong"] == 100
    only_one = svc.compute_metrics([[row] for row in rows if row["verifier_metadata"]["split"] == "law"])
    assert not any(key.startswith("voicebench/split/") for key in only_one)


# ---- PEDANT -------------------------------------------------------------------------------------------------------------


def test_portable_classifier_matches_a_linear_decision_rule():
    model = PortableLinearClassifier(
        coef=np.array([[1.0, -2.0]]), intercept=np.array([0.5]), classes=np.array(["correct", "incorrect"])
    )
    features = np.array([[1.0, 1.0], [0.0, 1.0], [2.0, 0.0]])
    assert model.decision_function(features).tolist() == [-0.5, -1.5, 2.5]
    assert model.predict(features).tolist() == ["correct", "correct", "incorrect"]


def test_vendored_pedant_passes_its_parity_check_and_scores_answers():
    pytest.importorskip("qa_metrics")
    pytest.importorskip("sklearn")
    pedant = load_pedant()
    question = "what is the capital of france"
    assert pedant_correct(pedant, reference="Paris", answer="The capital of France is Paris.", question=question)
    assert not pedant_correct(pedant, reference="Paris", answer="London", question=question)


def test_pedant_parity_failure_is_explicit(tmp_path):
    pytest.importorskip("sklearn")
    source = Path(__file__).resolve().parents[1] / "vendor" / "pedant"
    for path in source.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    parity = json.loads((tmp_path / "parity.json").read_text())
    parity["prediction"][0] = "incorrect"
    (tmp_path / "parity.json").write_text(json.dumps(parity))
    with pytest.raises(RuntimeError, match="parity"):
        load_pedant(pedant_module=MagicMock(), pedant_dir=tmp_path)


def test_pedant_weight_and_decision_mismatches_are_explicit(tmp_path):
    pytest.importorskip("sklearn")
    source = Path(__file__).resolve().parents[1] / "vendor" / "pedant"
    for path in source.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    meta = json.loads((tmp_path / "model.json").read_text())
    (tmp_path / "model.json").write_text(json.dumps({**meta, "classes": ["incorrect", "correct"]}))
    with pytest.raises(RuntimeError, match="model.json"):
        load_pedant(pedant_module=MagicMock(), pedant_dir=tmp_path)
    (tmp_path / "model.json").write_text(json.dumps(meta))
    parity = json.loads((tmp_path / "parity.json").read_text())
    parity["decision"][0] += 1.0
    (tmp_path / "parity.json").write_text(json.dumps(parity))
    with pytest.raises(RuntimeError, match="decision parity"):
        load_pedant(pedant_module=MagicMock(), pedant_dir=tmp_path)


def test_sd_qa_judge_content_failure_is_an_invalid_score_not_a_pedant_result(tmp_path):
    scorer = ReferenceQAScorer(judge_settings(tmp_path), client([None]))
    result = asyncio.run(scorer.score("Paris", {"prompt": "capital of France?", "reference": "Paris"}))
    assert not result.valid and "pedant_correct" not in result.metrics


def test_sd_qa_pedant_is_scored_next_to_the_judge_vote(tmp_path):
    pytest.importorskip("qa_metrics")
    pytest.importorskip("sklearn")
    scorer = ReferenceQAScorer(judge_settings(tmp_path, sdqa_pedant=True), client(["Yes", "Yes", "No"]))
    metadata = {"prompt": "What is the capital of France?", "reference": "Paris"}
    right = asyncio.run(scorer.score("The capital is Paris.", metadata))
    wrong = asyncio.run(scorer.score("It is London.", metadata))
    assert right.valid and right.metrics == {"correct": 1.0, "total": 1.0, "pedant_correct": 1.0}
    assert wrong.metrics["correct"] == 1.0 and wrong.metrics["pedant_correct"] == 0.0
    plain = ReferenceQAScorer(judge_settings(tmp_path), client(["Yes"]))
    assert "pedant_correct" not in asyncio.run(plain.score("Paris", metadata)).metrics


def test_sd_qa_panda_aggregate_uses_pedant_only_when_scored(tmp_path):
    svc = server(tmp_path, expected=2, subset="sd-qa", scoring_type="reference_qa")
    row = {"sample_id": "0", "score_valid": True, "candidate_valid": True}
    with_pedant = [
        row | {"scoring_metrics": {"correct": 1, "total": 1, "pedant_correct": 0}},
        row | {"sample_id": "1", "scoring_metrics": {"correct": 1, "total": 1, "pedant_correct": 1}},
    ]
    metrics = svc.compute_metrics([[value] for value in with_pedant])
    assert metrics["voicebench/legacy/gpt"] == 100 and metrics["voicebench/legacy/panda"] == 50
    legacy = svc.compute_metrics([[{**row, "scoring_metrics": {"correct": 1, "total": 1}}]])
    assert legacy["voicebench/legacy/panda"] == -100


# ---- Every subset in one runtime ----------------------------------------------------------------------------------------


def write_prompts(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for name, user in (("open", "{prompt}\n{response}"), ("reference_qa", "{prompt}\n{reference}\n{response}")):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"system": f"{name} judge", "user": user}))
        paths[name] = str(path)
    return paths


def all_server(
    tmp_path: Path, subsets: dict[str, int] | None = None, *, replies: tuple[str, ...] = ("4", "4", "4"), **overrides
) -> VoiceBenchServer:
    config = VoiceBenchConfig(
        name="voicebench_all",
        host="127.0.0.1",
        port=8104,
        entrypoint="app.py",
        subset="all",
        expected_rows=subsets or {"alpacaeval": 1, "mmsu": 1, "advbench": 1, "sd-qa": 1},
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_prompt_files=write_prompts(tmp_path),
        **overrides,
    )
    return VoiceBenchServer(config=config, server_client=client(list(replies)))


def test_all_mode_validates_its_configuration(tmp_path):
    base = dict(name="x", host="127.0.0.1", port=1, entrypoint="app.py", subset="all")
    with pytest.raises(ValidationError, match="expected_rows"):
        VoiceBenchConfig(**base)
    with pytest.raises(ValidationError, match="expected_rows"):
        VoiceBenchConfig(**base, expected_rows={"mmsu": 1}, scoring_type="mcq")
    with pytest.raises(ValidationError, match="positive expected_rows"):
        VoiceBenchConfig(**base, expected_rows={"mmsu": 0})
    with pytest.raises(ValidationError, match="judge_prompt_files"):
        VoiceBenchConfig(**base, expected_rows={"alpacaeval": 1})
    with pytest.raises(ValidationError, match="judge_prompt_files"):
        VoiceBenchConfig(
            **base,
            expected_rows={"sd-qa": 1},
            judge_model_server={"type": "responses_api_models", "name": "j"},
            judge_prompt_files={"open": "x"},
        )
    VoiceBenchConfig(**base, expected_rows={"mmsu": 1, "advbench": 1})
    with pytest.raises(ValidationError, match="scoring_type and expected_num_rows"):
        VoiceBenchConfig(**{**base, "subset": "mmsu"})


def test_remaining_configuration_checks(tmp_path):
    base = dict(name="x", host="127.0.0.1", port=1, entrypoint="app.py")
    with pytest.raises(ValidationError, match="legacy_v1"):
        VoiceBenchConfig(**base, subset="mmsu", scoring_type="mcq", expected_num_rows=1, scoring_version="v2")
    with pytest.raises(ValidationError, match="judge_prompt_file"):
        VoiceBenchConfig(**base, subset="alpacaeval", scoring_type="open", expected_num_rows=1)
    with pytest.raises(ValidationError, match="not scoring_type"):
        VoiceBenchConfig(**base, subset="all", expected_rows={"mmsu": 1}, expected_sample_ids_file="ids.json")


def test_all_mode_routes_each_row_to_its_scorer_and_shares_the_judge_budget(tmp_path):
    svc = all_server(tmp_path)
    assert svc._judges["open"].limiter is svc._judges["reference_qa"].limiter
    open_row = asyncio.run(svc.verify(request(subset="alpacaeval")))
    assert open_row.scoring_type == "open" and open_row.voicebench_score == 4
    assert open_row.canonical_sample_id == "alpacaeval/test_0"
    mcq_row = asyncio.run(svc.verify(request(subset="mmsu", response=candidate("The answer is C"), reference="C")))
    assert mcq_row.scoring_type == "mcq" and mcq_row.reward == 1
    refusal_row = asyncio.run(svc.verify(request(subset="advbench", response=candidate("I'm sorry, I can't"))))
    assert refusal_row.scoring_type == "refusal" and refusal_row.reward == 1
    svc.server_client.post.assert_called_once()
    assert svc._judges["open"].total_calls == svc._judges["reference_qa"].total_calls == 1
    with pytest.raises(ValueError, match="cannot score subset"):
        asyncio.run(svc.verify(request(subset="commoneval")))


def test_all_mode_reports_metrics_per_subset_with_their_own_coverage(tmp_path):
    svc = all_server(tmp_path, {"alpacaeval": 2, "mmsu": 1})
    rows = [
        asyncio.run(svc.verify(request("a0", subset="alpacaeval"))).model_dump(mode="json"),
        asyncio.run(
            svc.verify(request("m0", subset="mmsu", response=candidate("The answer is C"), reference="C"))
        ).model_dump(mode="json"),
    ]
    for index, row in enumerate(rows):
        row |= {"_ng_task_index": index, "_ng_rollout_index": 0}
    aggregated = asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    metrics = aggregated.key_metrics
    assert metrics["voicebench/alpacaeval/legacy/gpt"] == 4
    assert metrics["voicebench/alpacaeval/missing_rows"] == 1 and not metrics["voicebench/alpacaeval/complete"]
    assert metrics["voicebench/mmsu/legacy/acc_unparsed_wrong"] == 100 and metrics["voicebench/mmsu/complete"]
    assert not any(key.startswith("voicebench/legacy/") for key in metrics)
    with pytest.raises(ValueError, match="cannot aggregate a row of subset"):
        svc.compute_metrics([[{"subset": "commoneval"}]])


def test_judge_prompt_is_resolved_against_the_install_when_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = judge_settings(tmp_path).model_copy(
        update={"judge_prompt_file": "resources_servers/voicebench/tests/fixtures/judge_prompt.yaml"}
    )
    from resources_servers.voicebench.scorers.open_judge import OpenJudgeScorer

    scorer = OpenJudgeScorer(settings, client())
    assert scorer.prompt.system == "fixture judge"


# ---- Reviewed paths: subset failures, replay, startup check, PEDANT through the server, fixture --------------------------


def verified(svc: VoiceBenchServer, index: int, subset: str, **kwargs) -> dict:
    """A verify row as the aggregator receives it: JSON-shaped, with Gym's source task index."""
    row = asyncio.run(svc.verify(request(str(index), subset=subset, **kwargs))).model_dump(mode="json")
    return row | {"_ng_task_index": index, "_ng_rollout_index": 0}


def test_message_mode_scores_the_last_message_not_a_commentary_before_it(tmp_path):
    two = candidate("Maybe A").output[0].model_dump(), candidate("The answer is B").output[0].model_dump()
    svc = server(tmp_path, subset="mmsu", scoring_type="mcq", answer_extraction="message")
    result = asyncio.run(svc.verify(request(subset="mmsu", response=candidate(output=list(two)), reference="B")))
    assert result.scoring_answer == "The answer is B" and result.reward == 1


def test_one_unparseable_judge_grade_costs_only_its_subset(tmp_path):
    svc = all_server(tmp_path, {"alpacaeval": 1, "mmsu": 1}, replies=("Great answer",) * 3)
    rows = [
        verified(svc, 0, "alpacaeval"),
        verified(svc, 1, "mmsu", response=candidate("The answer is C"), reference="C"),
    ]
    assert not rows[0]["score_valid"] and rows[0]["legacy_scoring_error"]
    for aggregated in (
        svc.compute_metrics([rows]),
        asyncio.run(svc.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).agent_metrics,
    ):
        assert "legacy VoiceBench evaluation failed" in aggregated["voicebench/alpacaeval/aggregation_error"]
        assert not [key for key in aggregated if key.startswith("voicebench/alpacaeval/legacy/")]
        assert aggregated["voicebench/mmsu/legacy/acc"] == 100 and aggregated["voicebench/mmsu/complete"]
        assert "voicebench/mmsu/aggregation_error" not in aggregated
    # A single-subset runtime still refuses, as the reference evaluator does.
    single = server(tmp_path, expected=1)
    row = asyncio.run(single.verify(request(subset="alpacaeval_full"))).model_dump(mode="json")
    with pytest.raises(ValueError, match="legacy VoiceBench evaluation failed"):
        single.compute_metrics([[row | {"legacy_scoring_error": "unparseable grade"}]])


def test_all_mode_replays_random_fallback_guesses_like_a_single_subset_run(tmp_path):
    """Seed 2 guesses A, A, A, C for four unparsed answers; the reference is A, so each subset scores 75."""
    subsets = ("mmsu", "openbookqa")
    svc = all_server(tmp_path, {name: 4 for name in subsets}, seed=2)
    rows = [
        verified(svc, 2 * position + offset, name, response=candidate("I do not know"), reference="A")
        for position in range(4)
        for offset, name in enumerate(subsets)
    ]
    metrics = svc.compute_metrics([rows])
    single = server(tmp_path, subset="openbookqa", scoring_type="mcq", expected=4, seed=2)
    single_metrics = single.compute_metrics(
        [[verified(single, i, "openbookqa", response=candidate("I do not know"), reference="A") for i in range(4)]]
    )
    assert single_metrics["voicebench/legacy/acc"] == 75
    for name in subsets:
        assert metrics[f"voicebench/{name}/legacy/acc"] == 75
        assert metrics[f"voicebench/{name}/legacy/acc_unparsed_wrong"] == 0
        assert metrics[f"voicebench/{name}/legacy/acc"] == single_metrics["voicebench/legacy/acc"]
    # Arrival order does not matter: replay follows the source task index within each subset.
    assert svc.compute_metrics([list(reversed(rows))]) == metrics


def test_all_mode_startup_rejects_an_unset_judge_endpoint(tmp_path):
    unset = "resources_servers.voicebench.app.get_global_config_dict"
    with patch(unset, return_value=_global_config("http://unset.local/v1")):
        for subsets in ({"alpacaeval": 1}, {"sd-qa": 1}, {"mmsu": 1, "alpacaeval": 1, "sd-qa": 1}):
            with pytest.raises(ValueError, match="judge_base_url"):
                all_server(tmp_path, subsets)
        # Rule-scored subsets never read the judge endpoint.
        all_server(tmp_path, {"mmsu": 1, "advbench": 1})
    with patch(unset, return_value=_global_config("http://judge/v1")):
        all_server(tmp_path, {"mmsu": 1, "alpacaeval": 1, "sd-qa": 1})


def test_sd_qa_pedant_through_the_server_in_all_mode(tmp_path):
    pytest.importorskip("qa_metrics")
    pytest.importorskip("sklearn")
    svc = all_server(tmp_path, {"sd-qa": 2}, replies=("Yes", "Yes", "Yes"), sdqa_pedant=True)
    question = "What is the capital of France?"
    rows = [
        verified(
            svc,
            0,
            "sd-qa",
            response=candidate("The capital is Paris."),
            reference="Paris",
            prompt=question,
            split="aus",
        ),
        verified(
            svc, 1, "sd-qa", response=candidate("It is London."), reference="Paris", prompt=question, split="usa"
        ),
    ]
    metrics = svc.compute_metrics([rows])
    assert metrics["voicebench/sd-qa/legacy/gpt"] == 100 and metrics["voicebench/sd-qa/legacy/panda"] == 50
    assert metrics["voicebench/sd-qa/split/aus/legacy/panda"] == 100
    assert metrics["voicebench/sd-qa/split/usa/legacy/panda"] == 0


def test_pedant_verdict_that_only_the_classifier_can_give():
    pytest.importorskip("qa_metrics")
    pytest.importorskip("sklearn")
    # "quick fox" is not a substring of the answer, so qa-metrics' substring shortcut does not apply.
    pedant = load_pedant()
    assert pedant_correct(pedant, reference="quick fox", answer="the quick brown fox", question="summarize the phrase")


def test_the_verifier_fixture_runs_its_cases():
    from nemo_gym.verification import exercise_verifier_fixture
    from resources_servers.voicebench.app import VERIFIER_FIXTURE

    results = asyncio.run(exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="unknown"))
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}


def test_shipped_config_defaults_serve_all_mode_and_single_subsets_still_fail_fast():
    path = Path(__file__).resolve().parents[1] / "configs" / "voicebench.yaml"
    shipped = yaml.safe_load(path.read_text())["voicebench"]["resources_servers"]["voicebench"]
    assert shipped["scoring_type"] is None and shipped["expected_num_rows"] is None
    base = {key: value for key, value in shipped.items() if key in VoiceBenchConfig.model_fields}
    base |= {"name": "voicebench", "host": "127.0.0.1", "port": 1, "entrypoint": "app.py"}
    # `subset: all` works with the shipped defaults, as the YAML comment says.
    config = VoiceBenchConfig(**(base | {"subset": "all", "expected_rows": {"mmsu": 1, "advbench": 1}}))
    assert config.subsets == ["mmsu", "advbench"]
    # A single subset still reports the missing values.
    with pytest.raises(ValidationError, match="scoring_type and expected_num_rows"):
        VoiceBenchConfig(**(base | {"subset": "mmsu"}))
