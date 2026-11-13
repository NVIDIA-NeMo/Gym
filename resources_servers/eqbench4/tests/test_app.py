# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import JudgeError, exercise_verifier_fixture
from resources_servers.eqbench4 import app as app_module
from resources_servers.eqbench4.app import (
    VERIFIER_FIXTURE,
    EQBench4ResourcesServer,
    EQBench4ResourcesServerConfig,
    EQBench4VerifyRequest,
    response_text,
)
from resources_servers.eqbench4.metrics import POLICY, POLICY_STOP_REASONS
from resources_servers.eqbench4.task_data import TaskData
from resources_servers.eqbench4.verifier_fixture import fixture_verdict


SERVER_DIR = Path(__file__).resolve().parents[1]
MARKER = "POLICY-REPLY-7F3"


def _example_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]


def _reply(text: str, *, response_id: str = "judge", incomplete: str | None = None) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0,
        model="judge",
        object="response",
        status="incomplete" if incomplete else "completed",
        incomplete_details={"reason": incomplete} if incomplete else None,
        output=[
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _policy_response() -> dict[str, Any]:
    return _reply(f"{MARKER} bye").model_dump(mode="json")


def _transcript() -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "content": f"{MARKER} What happened?"},
        {
            "role": "user",
            "content": "They left me out.",
            "raw_response": "<internal>\nhurt\n</internal>\n<dialogue>\nThey left me out.\n</dialogue>",
        },
    ]


def make_server(tmp_path: Path, **overrides: Any) -> EQBench4ResourcesServer:
    reference = tmp_path / "reference.jsonl"
    reference.write_text(
        "".join(
            json.dumps({"model_a": a, "model_b": b, "fraction_a": f, "scenario_id": "s"}) + "\n"
            for a, b, f in [("ref-a", "ref-b", 0.8), ("ref-a", "ref-b", 0.7), ("ref-b", "ref-c", 0.65)]
        )
    )
    settings = {
        "host": "127.0.0.1",
        "port": 0,
        "entrypoint": "app.py",
        "name": "eqbench4",
        "judge_model_servers": [
            {"type": "responses_api_models", "name": "judge_one"},
            {"type": "responses_api_models", "name": "judge_two"},
        ],
        "judge_responses_create_params": {"input": [], "temperature": 0.3, "max_output_tokens": 4096},
        "trait_judge_model_server": None,
        "trait_judge_responses_create_params": {"input": [], "temperature": 0.7},
        "trait_judge_system_message": "Score traits.",
        "reference_comparisons_path": str(reference),
        "rating_bootstrap_resamples": 20,
    }
    config = EQBench4ResourcesServerConfig.model_validate(settings | overrides)
    return EQBench4ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_request(**overrides: Any) -> EQBench4VerifyRequest:
    row = _example_rows()[0]
    row["opponents"].append(row["opponents"][0] | {"model": "synthetic/second", "comparison_index": 7})
    fields = row | {"response": _policy_response(), "transcript": _transcript(), "conversation_status": "completed"}
    return EQBench4VerifyRequest.model_validate(fields | overrides)


class ScriptedJudge:
    """A stand-in judge. Pairwise calls take replies from ``script`` first, then a verdict for ``winner`` (five
    plus signs on rapport, attunement tied); calls to the ``traits`` server get ``trait_text``."""

    def __init__(self, script: list[str] | None = None, *, winner: str = "policy", trait_text: str = "") -> None:
        self.script = list(script or [])
        self.winner = winner
        self.trait_text = trait_text
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, server_client: Any, *, server_name: str, **kwargs: Any) -> NeMoGymResponse:
        params = kwargs["json"]
        prompt = params.input[-1].content
        self.calls.append({"server": server_name, "prompt": prompt, "input": params.input})
        response_id = f"r{len(self.calls)}"
        if server_name == "traits":
            return _reply(self.trait_text, response_id=response_id)
        if self.script:
            return _reply(self.script.pop(0), response_id=response_id)
        verdict = json.loads(fixture_verdict(prompt, winner=self.winner, policy_marker=MARKER, dimensions=["rapport"]))
        return _reply(json.dumps(verdict | {"attunement": "tie"}), response_id=response_id)


async def test_policy_wins_both_orders_against_every_opponent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge = ScriptedJudge(winner="policy")
    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path).verify(make_request())

    # rapport +++++ for the policy, attunement tied, out of a 15-point maximum.
    assert result.reward == pytest.approx(0.5 + 5 / 15 / 2)
    assert [comparison.opponent for comparison in result.comparisons] == [
        "synthetic/reference-model",
        "synthetic/second",
    ]
    assert [comparison.comparison_index for comparison in result.comparisons] == [0, 7]
    assert result.policy_turns == 1 and not result.mask_sample and result.evaluation_completed is True
    # The judge panel rotates by comparison index; both orders of one comparison use the same judge.
    assert sorted(call["server"] for call in judge.calls) == ["judge_one"] * 2 + ["judge_two"] * 2
    reading = result.comparisons[0].readings
    assert [item.direction for item in reading] == ["policy_first", "opponent_first"]
    assert all(item.policy_margin == 5.0 and item.opponent_margin == 0.0 for item in reading)
    policy_first = next(
        call["prompt"] for call in judge.calls if call["prompt"].index(MARKER) < call["prompt"].index("Second")
    )
    assert "User (internal monologue):\n<internal>\nhurt\n</internal>" in policy_first
    assert "Final persona state: (No final state available)" in policy_first
    assert len(judge.calls[0]["input"]) == 1 and judge.calls[0]["input"][0].role == "user"


async def test_opponent_wins_and_codes_are_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = []
    for _ in range(2):
        judge = ScriptedJudge(winner="opponent")
        monkeypatch.setattr(app_module, "call_judge", judge)
        result = await make_server(tmp_path).verify(make_request())
        prompts.append(sorted(call["prompt"] for call in judge.calls))
    assert result.reward == pytest.approx(0.5 - 5 / 15 / 2)
    assert prompts[0] == prompts[1]
    codes = set(re.findall(r"(?:First|Second) (\w+):", "".join(prompts[0])))
    assert len(codes) == 8


async def test_unparseable_reading_is_asked_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge = ScriptedJudge(["I prefer not to say."], winner="policy")
    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path).verify(make_request())
    assert len(judge.calls) == 5
    assert sorted(item.attempts for comparison in result.comparisons for item in comparison.readings) == [1, 1, 1, 2]


async def test_a_comparison_with_no_readable_order_fails_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def silent(*_: Any, **__: Any) -> NeMoGymResponse:
        return _reply("", incomplete="max_output_tokens")

    monkeypatch.setattr(app_module, "call_judge", silent)
    with pytest.raises(
        JudgeError,
        match=r"in either order after 2 attempt\(s\) each: policy_first: judge output incomplete: "
        r"max_output_tokens; opponent_first: judge output incomplete: max_output_tokens",
    ):
        await make_server(tmp_path, judge_max_attempts=2).verify(make_request())

    async def chatty(*_: Any, **__: Any) -> NeMoGymResponse:
        return _reply("no braces at all")

    monkeypatch.setattr(app_module, "call_judge", chatty)
    with pytest.raises(JudgeError, match="no JSON verdict"):
        await make_server(tmp_path, judge_max_attempts=1).verify(make_request())


async def test_persona_failure_is_masked_without_judging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    judge = ScriptedJudge()
    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path).verify(
        make_request(conversation_status="persona_failed", conversation_error="persona turn 2: no <internal> section")
    )
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, "eqbench4:persona_failed")
    assert result.failure_reason == "persona turn 2: no <internal> section"
    assert not judge.calls and not result.comparisons
    # Never judged, so no measurement was made.
    assert result.evaluation_completed is False


async def test_one_readable_order_decides_the_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = ScriptedJudge(winner="policy")

    async def judge(server_client: Any, *, server_name: str, **kwargs: Any) -> NeMoGymResponse:
        prompt = kwargs["json"].input[-1].content
        if prompt.index(MARKER) > prompt.index("Second"):  # every opponent-first reading is unreadable
            return _reply("I would rather not choose.")
        return await scripted(server_client, server_name=server_name, **kwargs)

    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path, judge_max_attempts=2).verify(make_request())
    assert result.reward == pytest.approx(0.5 + 5 / 15 / 2) and not result.mask_sample
    for comparison in result.comparisons:
        assert [reading.direction for reading in comparison.readings] == ["policy_first"]
        assert comparison.unreadable_orders == {"opponent_first": "no JSON verdict in the judge output"}


@pytest.mark.parametrize(
    ("reason", "failure_kind"),
    [
        ("context_limit", "model_context_limit"),
        ("output_budget", "eqbench4:policy_output_budget"),
        ("content_filter", "eqbench4:policy_content_filter"),
        ("empty_reply", "eqbench4:policy_empty_reply"),
        (None, "eqbench4:policy_empty_reply"),
    ],
)
async def test_a_conversation_the_policy_stopped_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str | None, failure_kind: str
) -> None:
    judge = ScriptedJudge()
    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path).verify(
        make_request(conversation_status="policy_failed", policy_stop_reason=reason, conversation_error="stopped")
    )
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, failure_kind)
    assert result.policy_stop_reason == (reason or "empty_reply")
    assert result.failure_reason == "stopped" and result.policy_turns == 1
    assert not result.comparisons and not judge.calls
    assert result.evaluation_completed is False


def test_every_stop_reason_has_a_failure_kind() -> None:
    assert set(app_module.POLICY_STOP_FAILURE_KINDS) == set(POLICY_STOP_REASONS)


async def test_aggregate_metrics_report_and_warn_about_left_out_conversations(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scored = {
        "reward": 0.8,
        "conversation_status": "completed",
        "policy_turns": 8,
        "comparisons": [{"opponent": "ref-a", "fraction": 0.8}],
    }
    dropped = [
        {"reward": 0.0, "mask_sample": True, "conversation_status": "policy_failed", "policy_stop_reason": reason}
        for reason in ("context_limit", "context_limit", "output_budget")
    ] + [{"reward": 0.0, "mask_sample": True, "conversation_status": "persona_failed"}]
    rows = [
        row | {"_ng_task_index": index, "_ng_rollout_index": 0}
        for index, row in enumerate([scored, scored | {"reward": 0.6}, *dropped])
    ]
    with caplog.at_level("WARNING"):
        result = await make_server(tmp_path).aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    metrics = result.agent_metrics
    assert metrics["eqbench4_conversations"] == 6 and metrics["eqbench4_judged_conversations"] == 2
    assert metrics["eqbench4_policy_stopped_conversations"] == 3
    assert metrics["eqbench4_policy_stopped_rate"] == pytest.approx(0.5)
    assert metrics["eqbench4_policy_stopped_rate/context_limit"] == pytest.approx(2 / 6)
    assert metrics["eqbench4_policy_stopped/output_budget"] == 1
    assert metrics["eqbench4_policy_stopped/content_filter"] == 0
    assert metrics["eqbench4_persona_failed_rate"] == pytest.approx(1 / 6)
    assert "eqbench4_elo" in metrics
    assert result.key_metrics["eqbench4_policy_stopped_rate"] == pytest.approx(0.5)
    assert result.key_metrics["eqbench4_persona_failed_rate"] == pytest.approx(1 / 6)
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any("stopped replying in 3 of 6 conversations (50.0%)" in w and "context_limit 2" in w for w in warnings)
    assert any("persona simulator gave no usable reply in 1 of 6" in w for w in warnings)


async def test_aggregate_metrics_stay_quiet_when_nothing_was_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [{"reward": 0.5, "conversation_status": "completed", "comparisons": [], "_ng_task_index": 0}]
    with caplog.at_level("WARNING"):
        result = await make_server(tmp_path).aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.key_metrics["eqbench4_policy_stopped_rate"] == 0.0
    assert not [record for record in caplog.records if "EQ-Bench 4:" in record.getMessage()]


async def test_trait_judge_scores_are_informational(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trait_server = {"trait_judge_model_server": {"type": "responses_api_models", "name": "traits"}}
    judge = ScriptedJudge(trait_text='[SCORES]\n```json\n{"validation_propensity": 7}\n```\n[/SCORES]')
    monkeypatch.setattr(app_module, "call_judge", judge)
    result = await make_server(tmp_path, **trait_server).verify(make_request())
    trait_call = next(call for call in judge.calls if call["server"] == "traits")
    assert [message.role for message in trait_call["input"]] == ["system", "user"]
    assert "Chatbot" not in trait_call["prompt"] and "Friend: " in trait_call["prompt"]
    assert result.trait_scores == {"validation_propensity": 7.0}
    assert result.trait_judge_error is None
    assert result.reward == pytest.approx(0.5 + 5 / 15 / 2)

    monkeypatch.setattr(app_module, "call_judge", ScriptedJudge(trait_text="no scores"))
    result = await make_server(tmp_path, **trait_server).verify(make_request())
    assert result.trait_scores is None and result.trait_judge_error == "no trait scores in the trait judge output"


async def test_trait_judge_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairwise = ScriptedJudge(winner="policy")

    async def judge(server_client: Any, *, server_name: str, **kwargs: Any) -> NeMoGymResponse:
        if server_name == "traits":
            raise JudgeError("trait judge unreachable")
        return await pairwise(server_client, server_name=server_name, **kwargs)

    monkeypatch.setattr(app_module, "call_judge", judge)
    server = make_server(tmp_path, trait_judge_model_server={"type": "responses_api_models", "name": "traits"})
    result = await server.verify(make_request())
    assert result.trait_judge_error == "trait judge unreachable"
    assert result.trait_scores is None and not result.mask_sample


async def test_a_pairwise_failure_cancels_the_trait_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    started = asyncio.Event()

    async def judge(server_client: Any, *, server_name: str, **_: Any) -> NeMoGymResponse:
        if server_name == "traits":
            started.set()
            await asyncio.sleep(3600)
        await started.wait()
        raise JudgeError("pairwise judge down")

    monkeypatch.setattr(app_module, "call_judge", judge)
    server = make_server(tmp_path, trait_judge_model_server={"type": "responses_api_models", "name": "traits"})
    with pytest.raises(JudgeError, match="pairwise judge down"):
        await asyncio.wait_for(server.verify(make_request()), timeout=10)


def test_response_text_skips_reasoning() -> None:
    reply = _reply("<think>hidden</think>Visible").model_dump(mode="json")
    reply["output"].insert(0, {"id": "r", "type": "reasoning", "summary": []})
    assert response_text(NeMoGymResponse.model_validate(reply)) == "Visible"


def test_metrics_place_the_policy_among_reference_models(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    rollouts = [
        {
            "reward": 0.8,
            "conversation_status": "completed",
            "policy_turns": 8,
            "comparisons": [
                {"opponent": "ref-a", "fraction": 0.8},
                {"opponent": "ref-c", "fraction": 0.9},
            ],
        }
    ]
    metrics = server.compute_metrics([rollouts])
    assert metrics["eqbench4_rank"] == 1 and metrics["eqbench4_models_ranked"] == 4
    assert metrics["eqbench4_elo_ci_low"] <= metrics["eqbench4_elo"] <= metrics["eqbench4_elo_ci_high"]
    assert (
        set(metrics) >= {"eqbench4_ladder/ref-a", "eqbench4_ladder/ref-c"}
        and f"eqbench4_ladder/{POLICY}" not in metrics
    )
    repeat = server.compute_repeat_metrics([rollouts])
    assert repeat["eqbench4_elo"] == pytest.approx(metrics["eqbench4_elo"])
    assert server.get_key_metrics(metrics | {"mean/reward": 0.8, "other": 1}) == {
        "eqbench4_elo": metrics["eqbench4_elo"],
        "eqbench4_elo_ci_low": metrics["eqbench4_elo_ci_low"],
        "eqbench4_elo_ci_high": metrics["eqbench4_elo_ci_high"],
        "eqbench4_rank": 1,
        "eqbench4_win_rate": pytest.approx(0.85),
        "mean/reward": 0.8,
    }


def test_missing_reference_comparisons_name_the_prepare_command(tmp_path: Path) -> None:
    server = make_server(tmp_path, reference_comparisons_path="benchmarks/eqbench4/data/absent.jsonl")
    with pytest.raises(FileNotFoundError, match="gym eval prepare --benchmark eqbench4"):
        server.compute_metrics([[{"comparisons": [{"opponent": "x", "fraction": 0.5}]}]])


def test_config_requires_a_judge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="judge_model_servers"):
        make_server(tmp_path, judge_model_servers=[])


def test_example_rows_are_valid_tasks() -> None:
    rows = _example_rows()
    assert len(rows) == 5
    for row in rows:
        TaskData.model_validate(row)
        assert row["responses_create_params"]["input"][0]["content"] == row["evaluated_system_prompts"][0]


async def test_verifier_fixture() -> None:
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=[0.0, 1.0], determinism="stochastic")
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}
