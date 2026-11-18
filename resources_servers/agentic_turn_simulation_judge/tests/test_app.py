# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.agentic_turn_simulation_judge import app as app_module
from resources_servers.agentic_turn_simulation_judge.app import (
    AgenticTurnSimulationJudgeConfig,
    AgenticTurnSimulationJudgeResourcesServer,
    AgenticTurnSimulationJudgeVerifyRequest,
    FieldBonusConfig,
    assistant_text,
    classification_stats,
    score_output,
)


ENV_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_ROWS = [json.loads(line) for line in (ENV_DIR / "data" / "example.jsonl").read_text().splitlines()]
GOLD_YES = {
    "addresses_concern": "yes",
    "correct": "yes",
    "blocked_by": "none",
    "request_questionable": "ok",
    "discard": "no",
    "discard_reason": "none",
}


def verdict_text(**overrides) -> str:
    verdict = {"analysis": "Checked the tool results against the claim.", **GOLD_YES, "reason": "Fine."}
    verdict.update(overrides)
    return json.dumps(verdict)


YES = verdict_text()
NO = verdict_text(correct="no", blocked_by="model")


def make_response(text: str | None, reasoning: str | None = None) -> NeMoGymResponse:
    output = []
    if reasoning is not None:
        output.append(
            {
                "id": "rs_1",
                "summary": [],
                "type": "reasoning",
                "content": [{"text": reasoning, "type": "reasoning_text"}],
            }
        )
    if text is not None:
        output.append(
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        )
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def make_server(**overrides) -> AgenticTurnSimulationJudgeResourcesServer:
    config = AgenticTurnSimulationJudgeConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **overrides)
    return AgenticTurnSimulationJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_request(text: str | None, *, gold: str = "yes", **fields) -> AgenticTurnSimulationJudgeVerifyRequest:
    return AgenticTurnSimulationJudgeVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "<step_under_evaluation>x"}]},
        response=make_response(text),
        gold_ladder=gold,
        **fields,
    )


def score(text: str, gold: str = "yes", *, p_yes=None, gold_fields=None, **config) -> dict:
    cfg = AgenticTurnSimulationJudgeConfig(host="", port=0, entrypoint="", name="", **config)
    return score_output(text, gold_ladder=gold, gold_p_yes=p_yes, gold_fields=gold_fields, config=cfg)


class TestConfig:
    def test_yaml_defaults(self) -> None:
        raw = yaml.safe_load((ENV_DIR / "configs" / "agentic_turn_simulation_judge.yaml").read_text())
        server = raw["agentic_turn_simulation_judge_resources_server"]["resources_servers"][
            "agentic_turn_simulation_judge"
        ]
        config = AgenticTurnSimulationJudgeConfig(host="", port=0, name="", **server)
        assert (config.parse_mode, config.validate_schema, config.reward_mode) == ("strict", True, "hard")
        assert config.field_bonus is None and config.max_analysis_words is None
        assert server["domain"] == "agent" and server["verified"] is False
        agent = raw["agentic_turn_simulation_judge_simple_agent"]["responses_api_agents"]["simple_agent"]
        assert agent["resources_server"]["name"] == "agentic_turn_simulation_judge_resources_server"
        assert [d["type"] for d in agent["datasets"]] == ["example"]

    def test_field_bonus_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="not verdict fields"):
            FieldBonusConfig(weight=0.2, fields=["blocked_by", "reason"])
        with pytest.raises(ValidationError):
            FieldBonusConfig(weight=1.5, fields=["blocked_by"])


class TestAssistantText:
    def test_reads_only_assistant_messages(self) -> None:
        response = make_response(YES, reasoning=NO)
        assert assistant_text(response) == YES

    def test_joins_messages_and_string_content(self) -> None:
        fake = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", role="assistant", content="a"),
                SimpleNamespace(type="message", role="user", content="not mine"),
                SimpleNamespace(type="function_call", role=None, content=None),
                SimpleNamespace(type="message", role="assistant", content=[SimpleNamespace(text="b")]),
            ]
        )
        assert assistant_text(fake) == "a\nb"

    def test_empty(self) -> None:
        assert assistant_text(None) == ""
        assert assistant_text(make_response(None)) == ""


class TestScoreOutput:
    @pytest.mark.parametrize(
        "text,gold,reward,confusion",
        [
            (YES, "yes", 1.0, "tn"),
            (NO, "no", 1.0, "tp"),
            (NO, "yes", 0.0, "fp"),
            (YES, "no", 0.0, "fn"),
            (verdict_text(addresses_concern="no"), "no", 1.0, "tp"),
        ],
    )
    def test_hard_reward_and_confusion(self, text: str, gold: str, reward: float, confusion: str) -> None:
        out = score(text, gold)
        assert out["reward"] == out["ladder_match"] == reward
        assert out["parse_ok"] is True and out["parse_error"] is None
        assert {k: out[k] for k in ("tp", "fp", "fn", "tn")} == {
            k: float(k == confusion) for k in ("tp", "fp", "fn", "tn")
        }

    @pytest.mark.parametrize(
        "text,error",
        [("", "empty"), ("yes", "no_json"), (YES[:-10], "bad_json"), (f"{YES}{NO}", "multiple_objects")],
    )
    def test_unreadable_output_scores_zero(self, text: str, error: str) -> None:
        for reward_mode in ("hard", "soft"):
            out = score(text, "yes", p_yes=0.75, reward_mode=reward_mode)
            assert out["reward"] == 0.0 and out["pred_ladder"] is None
            assert (out["parse_ok"], out["parse_error"]) == (False, error)
            assert out["tp"] == out["fp"] == out["fn"] == out["tn"] == 0.0

    def test_soft_reward(self) -> None:
        assert score(YES, "yes", p_yes=0.75, reward_mode="soft")["reward"] == 0.75
        out = score(NO, "yes", p_yes=0.75, reward_mode="soft")
        assert out["reward"] == pytest.approx(0.25) and out["ladder_match"] == 0.0
        assert score(NO, "no", p_yes=0.0, reward_mode="soft")["reward"] == 1.0

    def test_soft_reward_falls_back_to_hard_without_p_yes(self) -> None:
        assert score(NO, "yes", reward_mode="soft")["reward"] == 0.0
        assert score(YES, "yes", reward_mode="soft")["reward"] == 1.0

    def test_field_matches_are_reported(self) -> None:
        gold = {**GOLD_YES, "blocked_by": "infra", "reason": "not a verdict field", "discard_reason": None}
        out = score(YES, "yes", gold_fields=gold)
        assert out["field_matches"] == {
            "addresses_concern": True,
            "correct": True,
            "blocked_by": False,
            "request_questionable": True,
            "discard": True,
        }
        assert out["field_match_share"] == pytest.approx(0.8)
        assert out["reward"] == 1.0  # no bonus configured
        assert out["pred_fields"]["blocked_by"] == "none" and "analysis" not in out["pred_fields"]

    def test_field_bonus(self) -> None:
        bonus = {"weight": 0.2, "fields": ["blocked_by", "request_questionable"]}
        gold = {**GOLD_YES, "request_questionable": "questionable"}
        assert score(YES, "yes", gold_fields=gold, field_bonus=bonus)["reward"] == pytest.approx(0.9)
        assert score(YES, "yes", gold_fields=GOLD_YES, field_bonus=bonus)["reward"] == 1.0
        # Fields never buy back a wrong ladder.
        wrong = score(YES, "no", gold_fields=GOLD_YES, field_bonus=bonus)
        assert wrong["reward"] == 0.0 and wrong["field_match_share"] == 1.0
        # Rows without gold for the listed fields keep the ladder score.
        assert score(YES, "yes", field_bonus=bonus)["reward"] == 1.0
        soft = score(YES, "yes", p_yes=0.5, gold_fields=gold, field_bonus=bonus, reward_mode="soft")
        assert soft["reward"] == pytest.approx(0.45)

    def test_unparsed_output_matches_no_field(self) -> None:
        out = score("not json", "yes", gold_fields=GOLD_YES)
        assert set(out["field_matches"].values()) == {False} and out["field_match_share"] == 0.0
        assert out["pred_fields"] is None

    def test_invalid_schema_and_too_long(self) -> None:
        out = score(verdict_text(blocked_by="user"), "yes")
        assert (out["reward"], out["parse_error"]) == (0.0, "bad_enum")
        assert out["pred_fields"]["blocked_by"] == "user"
        assert score(YES, "yes", max_analysis_words=3)["parse_error"] == "too_long"
        ladder_only = json.dumps({"addresses_concern": "yes", "correct": "yes"})
        assert score(ladder_only, "yes", validate_schema=False)["reward"] == 1.0

    def test_lenient_mode(self) -> None:
        text = f"Verdict:\n```json\n{YES}\n```"
        assert score(text, "yes")["parse_error"] == "bad_json"
        assert score(text, "yes", parse_mode="lenient")["reward"] == 1.0

    def test_counts(self) -> None:
        text = f"<think>hidden</think>{YES}"
        out = score(text, "yes")
        assert out["output_chars"] == len(text)
        assert out["analysis_words"] == len("Checked the tool results against the claim.".split())


class TestVerify:
    @pytest.mark.parametrize(
        ("text", "error"),
        [('{"correct": "yes"}', "missing_keys"), ('{"addresses_concern": "maybe", "correct": "yes"}', "bad_enum")],
    )
    async def test_schema_disabled_undecided_engagement_scores_zero(self, text: str, error: str) -> None:
        response = await make_server(validate_schema=False).verify(make_request(text, gold="yes"))
        assert (response.reward, response.pred_ladder, response.parse_error) == (0.0, None, error)
        assert not response.mask_sample

    async def test_verify(self) -> None:
        response = await make_server().verify(make_request(NO, gold="no", row_id="r1"))
        assert (response.reward, response.pred_ladder, response.gold_ladder, response.tp) == (1.0, "no", "no", 1.0)
        assert response.row_id == "r1"

    async def test_reasoning_items_are_ignored(self) -> None:
        request = make_request(None)
        request.response = make_response(YES, reasoning=f"Draft: {NO}")
        assert (await make_server().verify(request)).reward == 1.0

    async def test_json_schema_text_format_round_trips(self) -> None:
        # A row run with a JSON-schema response format: the echoed request must re-validate (aliased `schema`).
        schema = json.loads((ENV_DIR / "prompts" / "evaluation_schema.json").read_text())
        text_format = {"format": {"type": "json_schema", "name": "evaluation", "schema": schema, "strict": True}}
        request = AgenticTurnSimulationJudgeVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "x"}], "text": text_format},
            response=make_response(YES).model_copy(update={"text": text_format}),
            gold_ladder="yes",
        )
        response = await make_server().verify(request)
        assert response.reward == 1.0
        echoed = response.model_dump(by_alias=True)
        assert echoed["responses_create_params"]["text"]["format"]["schema"] == schema
        assert echoed["response"]["text"]["format"]["schema"] == schema

    async def test_lenient_gap_ignores_unparsed(self) -> None:
        pairs = [("yes", "yes"), ("yes", "no"), (None, "yes"), (None, "yes")]
        stats = classification_stats(pairs)
        # All rollouts: 2/4 predicted yes vs 3/4 gold yes; parsed: 2/2 yes vs 1/2 gold yes -> lenient.
        assert stats["pred_yes_rate"] == 0.5 and stats["leniency_gap"] == pytest.approx(0.5)

    async def test_internal_error_scores_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr(app_module, "score_output", boom)
        response = await make_server().verify(make_request(YES))
        assert (response.reward, response.parse_ok, response.parse_error) == (0.0, False, "internal")
        assert response.mask_sample is False


class TestServer:
    """Through the FastAPI app: /verify on the example rows, then /aggregate_metrics."""

    def test_verify_and_aggregate_over_http(self) -> None:
        client = TestClient(make_server().setup_webserver())
        outputs = {"yes": YES, "no": NO}
        flipped = {"yes": NO, "no": YES}
        verify_responses = []
        for task_index, row in enumerate(EXAMPLE_ROWS):
            # Rollout 0 agrees with the label, rollout 1 disagrees on the first two rows and is cut off on the third.
            texts = [outputs[row["gold_ladder"]]]
            texts.append(flipped[row["gold_ladder"]] if task_index < 2 else YES[:20] if task_index == 2 else texts[0])
            for rollout_index, text in enumerate(texts):
                body = {**row, "response": make_response(text).model_dump(mode="json")}
                reply = client.post("/verify", json=body)
                assert reply.status_code == 200, reply.text
                result = reply.json()
                verify_responses.append({**result, "_ng_task_index": task_index, "_ng_rollout_index": rollout_index})

        rewards = [r["reward"] for r in verify_responses]
        assert rewards == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        assert verify_responses[5]["parse_error"] == "bad_json"
        assert verify_responses[0]["meta"]["source"] == "synthetic"  # row fields pass through

        reply = client.post("/aggregate_metrics", json={"verify_responses": verify_responses})
        assert reply.status_code == 200, reply.text
        aggregate = reply.json()
        key = aggregate["key_metrics"]
        assert key["accuracy"] == pytest.approx(0.7)
        assert key["parse_fail"] == pytest.approx(0.1)
        assert key["mean/reward"] == pytest.approx(0.7)
        assert set(key) >= {"accuracy", "balanced_accuracy", "kappa", "parse_fail", "leniency_gap"}
        metrics = aggregate["agent_metrics"]
        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(70.0)
        assert metrics["family/obstacle/n"] == 1.0 and metrics["tier/B/accuracy"] == 0.5
        assert metrics["hand/accuracy"] == 1.0 and metrics["hand/n"] == 1.0

    def test_row_without_gold_is_rejected(self) -> None:
        client = TestClient(make_server().setup_webserver())
        row = {k: v for k, v in EXAMPLE_ROWS[0].items() if k != "gold_ladder"}
        reply = client.post("/verify", json={**row, "response": make_response(YES).model_dump(mode="json")})
        assert reply.status_code == 422

    def test_reverifying_the_committed_example_rollouts(self) -> None:
        """The example rollouts are hand-written verdicts (reasoning items included): /verify must reproduce them."""
        client = TestClient(make_server().setup_webserver())
        lines = [json.loads(line) for line in (ENV_DIR / "data" / "example_rollouts.jsonl").read_text().splitlines()]
        assert len(lines) == 5
        for line in lines:
            assert {o["type"] for o in line["response"]["output"]} == {"reasoning", "message"}
            row = {k: line[k] for k in EXAMPLE_ROWS[line["_ng_task_index"]]}
            reply = client.post("/verify", json={**row, "response": line["response"]}).json()
            for key in ("reward", "pred_ladder", "parse_error", "tp", "fp", "fn", "tn", "field_matches"):
                assert reply[key] == line[key], (line["row_id"], key)


def rollout(pred, gold, *, error=None, source=None, tier=None, hand=None, fields=None) -> dict:
    meta = {k: v for k, v in (("source", source), ("hand_ladder", hand)) if v is not None}
    return {
        "reward": float(pred == gold),
        "ladder_match": float(pred == gold),
        "pred_ladder": pred,
        "gold_ladder": gold,
        "parse_ok": pred is not None,
        "parse_error": error,
        "field_matches": fields or {},
        "meta": meta or None,
        "label": {"tier": tier} if tier else None,
    }


# Two rollouts per task (see the expected numbers in TestComputeMetrics).
TASKS = [
    [
        rollout("yes", "yes", source="src_a", tier="A", hand="yes", fields={"blocked_by": True, "discard": True}),
        rollout("yes", "yes", source="src_a", tier="A", hand="yes", fields={"blocked_by": True, "discard": True}),
    ],
    [
        rollout("no", "yes", source="src_a", tier="A", fields={"blocked_by": False, "discard": True}),
        rollout(
            None, "yes", error="bad_json", source="src_a", tier="A", fields={"blocked_by": False, "discard": False}
        ),
    ],
    [
        rollout("no", "no", source="src_b", tier="B", hand="no"),
        rollout("yes", "no", source="src_b", tier="B", hand="no"),
    ],
    [rollout("no", "no", source="src_b", tier="B"), rollout("no", "no", source="src_b", tier="B")],
]


class TestComputeMetrics:
    def test_hand_built_rollouts(self) -> None:
        m = make_server().compute_metrics(TASKS)
        # Pooled over 8 rollouts: 5 right; gold no caught 3/4, gold yes passed 2/4; one unparsed rollout.
        assert m["accuracy"] == pytest.approx(0.625)
        assert m["fail_recall"] == pytest.approx(0.75)
        assert m["pass_recall"] == pytest.approx(0.5)
        assert m["balanced_accuracy"] == pytest.approx(0.625)
        # Chance agreement (3 pred yes x 4 gold yes + 4 pred no x 4 gold no) / 64 = 0.4375.
        assert m["kappa"] == pytest.approx((0.625 - 0.4375) / (1 - 0.4375))
        assert (m["pred_yes_rate"], m["gold_yes_rate"]) == (pytest.approx(0.375), pytest.approx(0.5))
        # Leniency over the 7 parsed rollouts: 3 predicted yes vs 3 gold yes.
        assert m["parsed_pred_yes_rate"] == pytest.approx(3 / 7)
        assert m["leniency_gap"] == pytest.approx(0.0)
        assert m["parse_fail"] == pytest.approx(0.125)
        assert m["parse_fail/bad_json"] == pytest.approx(0.125) and m["parse_fail/empty"] == 0.0
        # Gym's pass@k / majority@k (percent): majority over pred_ladder, ties averaged, unparsed votes dropped.
        assert m["pass@1[avg-of-2]/accuracy"] == pytest.approx(62.5)
        assert m["majority@2/accuracy"] == pytest.approx(62.5)
        assert m["pass@2/accuracy"] == pytest.approx(75.0)
        assert m["pass@1[avg-of-2]/no_answer"] == pytest.approx(12.5)
        # Slices.
        assert (m["source/src_a/accuracy"], m["source/src_a/balanced_accuracy"], m["source/src_a/n"]) == (
            0.5,
            0.5,
            2.0,
        )
        assert (m["source/src_b/accuracy"], m["source/src_b/n"]) == (0.75, 2.0)
        assert (m["tier/A/accuracy"], m["tier/B/accuracy"]) == (0.5, 0.75)
        assert not any(k.startswith(("family/", "harness/")) for k in m)  # no row carries them
        # Hand labels (tasks 1 and 3).
        assert m["hand/accuracy"] == pytest.approx(0.75) and m["hand/n"] == 2.0
        assert m["hand/balanced_accuracy"] == pytest.approx(0.75)
        assert m["hand/kappa"] == pytest.approx(0.5)
        assert (m["hand/gold_accuracy"], m["hand/gold_kappa"]) == (1.0, 1.0)
        assert (m["field_acc/blocked_by"], m["field_acc/discard"]) == (0.5, 0.75)

    def test_field_acc_counts_rollouts_without_a_prediction(self) -> None:
        gold = {"correct": "yes"}
        parsed = {**rollout("yes", "yes", fields={"correct": True}), "gold_fields": gold}
        unscored = {"reward": 0.0, "gold_ladder": "yes", "gold_fields": gold}  # no verifier output
        m = make_server().compute_metrics([[parsed], [unscored]])
        assert m["field_acc/correct"] == pytest.approx(0.5)

    def test_field_acc_needs_a_reference_value(self) -> None:
        # No gold and no match data, a null gold value, or a non-verdict gold key: no denominator.
        tasks = [
            [rollout("yes", "yes", fields={"correct": True})],
            [rollout("no", "yes")],
            [{**rollout(None, "yes", error="bad_json"), "gold_fields": {"correct": None, "reason": "free text"}}],
        ]
        m = make_server().compute_metrics(tasks)
        assert m["field_acc/correct"] == 1.0
        assert "field_acc/reason" not in m

    def test_expect_labels(self) -> None:
        # A BAD stratum row (expect "no") the reference judge passed: the policy that fails it beats the label.
        bad = {"gold_ladder": "yes", "meta": {"expect": "no"}}
        good = {"gold_ladder": "yes", "meta": {"expect": "yes"}}
        tasks = [
            [{**rollout("no", "yes"), **bad}, {**rollout("yes", "yes"), **bad}],
            [{**rollout("yes", "yes"), **good}, {**rollout("yes", "yes"), **good}],
            [rollout("no", "no")],  # no expect label: not in the expect metrics
        ]
        m = make_server().compute_metrics(tasks)
        assert m["expect/n"] == 2.0
        # Policy vs expect: 1/2 BAD rollouts failed, 2/2 GOOD passed. Reference: never fails the BAD row.
        assert m["expect/accuracy"] == pytest.approx(0.75)
        assert (m["expect/fail_recall"], m["expect/pass_recall"]) == (0.5, 1.0)
        assert m["expect/gold_accuracy"] == pytest.approx(0.5)
        assert (m["expect/gold_fail_recall"], m["expect/gold_pass_recall"]) == (0.0, 1.0)
        assert "hand/n" not in m

    def test_rows_without_meta_or_label(self) -> None:
        tasks = [[rollout("yes", "yes")], [{"reward": 0.0, "gold_ladder": "no", "meta": "garbage", "label": 3}]]
        m = make_server().compute_metrics(tasks)
        assert m["accuracy"] == 0.5 and m["parse_fail"] == 0.5 and m["parse_fail/unscored"] == 0.5
        assert not any(k.startswith(("source/", "tier/", "hand/", "field_acc/")) for k in m)

    def test_partial_meta_counts_as_none(self) -> None:
        tasks = [[rollout("yes", "yes", source="src_a")], [rollout("no", "yes")]]
        m = make_server().compute_metrics(tasks)
        assert (m["source/src_a/accuracy"], m["source/none/accuracy"]) == (1.0, 0.0)
        assert m["source/none/n"] == 1.0

    def test_kappa_undefined_is_omitted(self) -> None:
        m = make_server().compute_metrics([[rollout("yes", "yes")], [rollout("yes", "yes")]])
        assert m["accuracy"] == 1.0 and "kappa" not in m and "fail_recall" not in m
        assert m["balanced_accuracy"] == 1.0

    def test_empty(self) -> None:
        assert make_server().compute_metrics([]) == {}
        assert classification_stats([]) == {}

    def test_key_metrics(self) -> None:
        agent_metrics = {**make_server().compute_metrics(TASKS), "mean/reward": 0.625}
        key = make_server().get_key_metrics(agent_metrics)
        assert list(key) == ["mean/reward", "accuracy", "balanced_accuracy", "kappa", "parse_fail", "leniency_gap"]
