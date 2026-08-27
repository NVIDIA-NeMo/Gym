# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.prompt import fill_prompt, load_prompt_config
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.primevul.app import (
    VERIFIER_FIXTURE,
    PrimeVulResourcesServer,
    PrimeVulResourcesServerConfig,
    PrimeVulVerifierMetadata,
    PrimeVulVerifyRequest,
    final_assistant_text,
)
from resources_servers.primevul.metrics import PAIRWISE_OUTCOME_KEYS, aggregate_paired, score_verdict
from resources_servers.primevul.primevul_data import (
    PRIMEVUL_HF_CONFIG,
    PRIMEVUL_HF_DATASET,
    PRIMEVUL_HF_REVISION,
    _assert_well_paired,
    _sample_pairs,
    load_pairs,
    raw_row,
)
from resources_servers.primevul.verdict import parse_verdict, strip_reasoning


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = REPOSITORY_ROOT / "benchmarks" / "primevul"
GOLD_VULNERABLE = {"id": "1", "pair_id": "p1", "gold_is_vulnerable": True}
GOLD_BENIGN = {"id": "2", "pair_id": "p1", "gold_is_vulnerable": False}


def tool_call_item(index: int = 0) -> dict:
    return {
        "id": f"call_{index}",
        "call_id": f"call_{index}",
        "name": "search",
        "arguments": "{}",
        "type": "function_call",
        "status": "completed",
    }


def make_response(*messages: str, trailing_tool_call: bool = False) -> NeMoGymResponse:
    output: list[dict] = []
    for index, text in enumerate(messages):
        output.append(
            {
                "id": f"msg_{index}",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        )
    if trailing_tool_call:
        output.append(tool_call_item(len(messages)))
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def make_server() -> PrimeVulResourcesServer:
    return PrimeVulResourcesServer(
        config=PrimeVulResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


def verify_row(rollout_index: int = 0, **updates) -> dict:
    row = {
        "correct": True,
        "parse_error": False,
        "pair_id": "p1",
        "pred_is_vulnerable": True,
        "gold_is_vulnerable": True,
        ROLLOUT_INDEX_KEY_NAME: rollout_index,
    }
    row.update(updates)
    return row


class TestVerdictParsing:
    def test_strips_reasoning_blocks(self) -> None:
        assert strip_reasoning("<think>NO</think> YES").strip() == "YES"
        assert strip_reasoning("<THINKING>NO</THINKING> YES").strip() == "YES"
        assert strip_reasoning(None) == ""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("YES", True),
            ("yes", True),
            ("(1) YES: A security vulnerability detected.", True),
            ("(1)", True),
            ("NO", False),
            ("(2) NO: No security vulnerability.", False),
            ("2", False),
        ],
    )
    def test_parses_published_options(self, text: str, expected: bool) -> None:
        assert parse_verdict(text) == {"is_vulnerable": expected, "parse_error": False}

    def test_last_explicit_option_wins(self) -> None:
        assert parse_verdict("I considered NO, but the final answer is YES.")["is_vulnerable"] is True

    def test_reasoning_draft_is_ignored(self) -> None:
        assert parse_verdict("<think>Maybe YES</think>NO")["is_vulnerable"] is False

    def test_non_answer_is_parse_error(self) -> None:
        verdict = parse_verdict("I cannot determine the answer.")
        assert verdict["parse_error"] is True
        assert verdict["is_vulnerable"] is None
        assert parse_verdict("Analysis section 1")["parse_error"] is True

    def test_error_raw_text_is_bounded(self) -> None:
        assert len(parse_verdict("x" * 900)["raw"]) == 500


class TestScoring:
    def test_correct_vulnerable(self) -> None:
        scored = score_verdict(parse_verdict("YES"), GOLD_VULNERABLE)
        assert scored == {
            "reward": 1.0,
            "correct": True,
            "parse_error": False,
            "pair_id": "p1",
            "pred_is_vulnerable": True,
            "gold_is_vulnerable": True,
        }

    def test_correct_benign(self) -> None:
        assert score_verdict(parse_verdict("NO"), GOLD_BENIGN)["reward"] == 1.0

    def test_wrong_label(self) -> None:
        assert score_verdict(parse_verdict("NO"), GOLD_VULNERABLE)["reward"] == 0.0

    def test_parse_error_scores_zero(self) -> None:
        scored = score_verdict(parse_verdict("unknown"), GOLD_BENIGN)
        assert scored["reward"] == 0.0
        assert scored["pred_is_vulnerable"] is None

    def test_missing_gold_fields_fail(self) -> None:
        with pytest.raises(KeyError, match="gold_is_vulnerable"):
            score_verdict(parse_verdict("NO"), {})


class TestPairedMetrics:
    def test_empty(self) -> None:
        assert aggregate_paired([]) == {}
        assert aggregate_paired([[]]) == {}

    @pytest.mark.parametrize(
        "vulnerable,benign,outcome",
        [
            (verify_row(), verify_row(gold_is_vulnerable=False, pred_is_vulnerable=False), "correct"),
            (
                verify_row(),
                verify_row(correct=False, gold_is_vulnerable=False, pred_is_vulnerable=True),
                "vulnerable",
            ),
            (
                verify_row(correct=False, pred_is_vulnerable=False),
                verify_row(gold_is_vulnerable=False, pred_is_vulnerable=False),
                "benign",
            ),
            (
                verify_row(correct=False, pred_is_vulnerable=False),
                verify_row(correct=False, gold_is_vulnerable=False, pred_is_vulnerable=True),
                "reversed",
            ),
        ],
    )
    def test_pairwise_outcomes(self, vulnerable: dict, benign: dict, outcome: str) -> None:
        metrics = aggregate_paired([[vulnerable], [benign]])
        assert metrics[PAIRWISE_OUTCOME_KEYS[outcome]] == 1.0
        assert sum(metrics[key] for key in PAIRWISE_OUTCOME_KEYS.values()) == 1.0

    def test_incomplete_pair_has_no_pair_metric(self) -> None:
        assert "mean/paired_accuracy" not in aggregate_paired([[verify_row()]])

    def test_pair_metrics_align_by_rollout_index_not_input_order(self) -> None:
        tasks = [
            [
                verify_row(correct=False, pred_is_vulnerable=False, rollout_index=1),
                verify_row(rollout_index=0),
            ],
            [
                verify_row(
                    correct=True,
                    gold_is_vulnerable=False,
                    pred_is_vulnerable=False,
                    rollout_index=0,
                ),
                verify_row(
                    correct=False,
                    gold_is_vulnerable=False,
                    pred_is_vulnerable=True,
                    rollout_index=1,
                ),
            ],
        ]
        assert aggregate_paired(tasks)["mean/paired_accuracy"] == 0.5

    def test_ragged_repeats_are_tolerated(self) -> None:
        tasks = [
            [verify_row(rollout_index=0), verify_row(rollout_index=1)],
            [verify_row(gold_is_vulnerable=False, pred_is_vulnerable=False, rollout_index=0)],
        ]
        assert aggregate_paired(tasks)["mean/paired_accuracy"] == 1.0

    def test_confusion_counts_binary_metrics_and_parse_errors(self) -> None:
        tasks = [
            [verify_row(pair_id="p1")],
            [
                verify_row(
                    pair_id="p1",
                    correct=False,
                    gold_is_vulnerable=False,
                    pred_is_vulnerable=True,
                )
            ],
            [verify_row(pair_id="p2", correct=False, pred_is_vulnerable=False)],
            [verify_row(pair_id="p2", gold_is_vulnerable=False, pred_is_vulnerable=False)],
            [verify_row(pair_id="incomplete", correct=False, parse_error=True, pred_is_vulnerable=None)],
        ]
        metrics = aggregate_paired(tasks)
        assert metrics["mean/binary_accuracy"] == 0.4
        assert metrics["mean/binary_precision"] == 0.5
        assert metrics["mean/binary_recall"] == 1 / 3
        assert metrics["mean/binary_f1"] == 0.4
        assert metrics["mean/parse_error_rate"] == 0.2
        assert (metrics["n_true_positives"], metrics["n_false_positives"]) == (1, 1)
        assert (metrics["n_false_negatives"], metrics["n_true_negatives"]) == (2, 1)
        assert metrics["n_pairs"] == 2
        assert metrics["n_rollouts"] == 5


class TestResourcesServer:
    def test_final_assistant_text(self) -> None:
        assert final_assistant_text(make_response("draft", "YES")) == "YES"
        assert final_assistant_text(make_response("YES", trailing_tool_call=True)) == "YES"
        assert final_assistant_text(make_response()) == ""
        assert final_assistant_text(make_response(trailing_tool_call=True)) == ""

    async def test_verify(self) -> None:
        result = await make_server().verify(
            PrimeVulVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "code"}]},
                response=make_response("YES"),
                verifier_metadata=GOLD_VULNERABLE,
            )
        )
        assert result.reward == 1.0
        assert result.correct is True
        assert result.pair_id == "p1"

    async def test_unparseable_response(self) -> None:
        result = await make_server().verify(
            PrimeVulVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "code"}]},
                response=make_response("unknown"),
                verifier_metadata=GOLD_VULNERABLE,
            )
        )
        assert result.reward == 0.0
        assert result.parse_error is True

    def test_metadata_is_required(self) -> None:
        with pytest.raises(ValidationError, match="verifier_metadata"):
            PrimeVulVerifyRequest.model_validate(
                {
                    "responses_create_params": {"input": [{"role": "user", "content": "code"}]},
                    "response": make_response("NO").model_dump(),
                }
            )

    def test_http_routes(self) -> None:
        client = TestClient(make_server().setup_webserver())
        response = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "code"}]},
                "response": make_response("YES").model_dump(),
                "verifier_metadata": GOLD_VULNERABLE,
            },
        )
        assert response.status_code == 200
        assert response.json()["reward"] == 1.0

        malformed = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "code"}]},
                "response": make_response("NO").model_dump(),
            },
        )
        assert malformed.status_code == 422

    def test_metric_hooks(self) -> None:
        tasks = [
            [verify_row()],
            [verify_row(gold_is_vulnerable=False, pred_is_vulnerable=False)],
        ]
        assert make_server().compute_metrics(tasks) == aggregate_paired(tasks)
        metrics = {
            "mean/paired_accuracy": 1.0,
            "mean/pairwise_reversed_rate": 0.0,
            "mean/reward": 1.0,
            "mean/unrelated": 1.0,
        }
        assert make_server().get_key_metrics(metrics) == {
            "mean/paired_accuracy": 1.0,
            "mean/pairwise_reversed_rate": 0.0,
            "mean/reward": 1.0,
        }
        assert make_server().get_key_metrics({}) == {}

    def test_aggregate_metrics_route(self) -> None:
        client = TestClient(make_server().setup_webserver())
        rows = [
            {TASK_INDEX_KEY_NAME: 0, "reward": 1.0, **verify_row()},
            {
                TASK_INDEX_KEY_NAME: 1,
                "reward": 0.0,
                **verify_row(correct=False, gold_is_vulnerable=False, pred_is_vulnerable=True),
            },
        ]
        response = client.post("/aggregate_metrics", json={"verify_responses": rows})
        assert response.status_code == 200
        assert response.json()["key_metrics"]["mean/paired_accuracy"] == 0.0

    async def test_verifier_fixture(self) -> None:
        results = await exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
        assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed"]


class TestData:
    RECORD = {
        "id": "1",
        "pair_id": "primevul-benchmark-0",
        "lang": "c",
        "code": "int main(void) { return 0; }",
        "gold_is_vulnerable": True,
    }

    def test_raw_row_and_prompt_contract(self) -> None:
        row = raw_row(self.RECORD)
        assert row["verifier_metadata"] == {
            "id": "1",
            "pair_id": "primevul-benchmark-0",
            "gold_is_vulnerable": True,
        }
        messages = fill_prompt(load_prompt_config(str(BENCHMARK_DIR / "prompt.yaml")), row)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "YES: A security vulnerability detected." in messages[1]["content"]
        assert self.RECORD["code"] in messages[1]["content"]

    def test_committed_sample_has_exactly_five_synthetic_rows(self) -> None:
        path = BENCHMARK_DIR / "data" / "primevul_benchmark.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(rows) == 5
        assert all(row["verifier_metadata"]["id"].startswith("synthetic-") for row in rows)
        assert len({row["verifier_metadata"]["pair_id"] for row in rows}) == 3

    def test_pair_sampling(self) -> None:
        records = [{"pair_id": f"p{index // 2}", "gold_is_vulnerable": index % 2 == 0} for index in range(10)]
        sampled = _sample_pairs(records, 2, seed=7)
        assert len(sampled) == 4
        assert len({row["pair_id"] for row in sampled}) == 2
        assert _sample_pairs(records, None, seed=0) == records
        assert _sample_pairs(records, 20, seed=0) == records
        with pytest.raises(ValueError, match="max_pairs must be positive"):
            _sample_pairs(records, 0, seed=0)

    def test_pairing_guard(self) -> None:
        _assert_well_paired(
            [
                {"pair_id": "p0", "gold_is_vulnerable": True},
                {"pair_id": "p0", "gold_is_vulnerable": False},
            ],
            "benchmark",
        )
        with pytest.raises(ValueError, match="row ordering"):
            _assert_well_paired(
                [
                    {"pair_id": "p0", "gold_is_vulnerable": True},
                    {"pair_id": "p0", "gold_is_vulnerable": True},
                ],
                "benchmark",
            )

    def test_unknown_split_is_rejected_without_downloading(self) -> None:
        with pytest.raises(ValueError, match="split must be one of"):
            load_pairs("validation")

    def test_loads_pinned_mirror_and_normalizes_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import datasets

        calls: dict = {}

        def fake_load_dataset(name, config, **kwargs):
            calls.update(name=name, config=config, **kwargs)
            return [
                {"idx": 10, "target": 1, "func": "vulnerable()"},
                {"idx": 11, "target": 0, "func": "fixed()"},
                {"idx": 12, "target": 1, "func": "vulnerable2()"},
                {"idx": 13, "target": 0, "func": "fixed2()"},
            ]

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        records = load_pairs("benchmark", max_pairs=1, seed=0, hf_token="token")

        assert calls == {
            "name": PRIMEVUL_HF_DATASET,
            "config": PRIMEVUL_HF_CONFIG,
            "split": "test",
            "revision": PRIMEVUL_HF_REVISION,
            "token": "token",
        }
        assert len(records) == 2
        assert records[0]["pair_id"] == records[1]["pair_id"]
        assert {record["gold_is_vulnerable"] for record in records} == {False, True}
        assert records[0]["id"] in {"10", "12"}
        assert records[0]["lang"] == "c"

    def test_load_rejects_broken_mirror_pairing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import datasets

        monkeypatch.setattr(
            datasets,
            "load_dataset",
            lambda *args, **kwargs: [
                {"idx": 1, "target": 1, "func": "a"},
                {"idx": 2, "target": 1, "func": "b"},
            ],
        )
        with pytest.raises(ValueError, match="row ordering"):
            load_pairs("benchmark")


def test_verifier_metadata_model() -> None:
    assert PrimeVulVerifierMetadata.model_validate(GOLD_BENIGN).gold_is_vulnerable is False
