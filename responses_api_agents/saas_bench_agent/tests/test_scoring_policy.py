# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native grading, response policy and aggregation across both harnesses."""

import importlib.util
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest, ModelServerRef
from responses_api_agents.saas_bench_agent.app import SaasBenchAgent, SaasBenchConfig
from responses_api_agents.saas_bench_agent.runtime import BENCHMARK_ROOT
from responses_api_agents.saas_bench_agent.schemas import SaasBenchRunRequest


@pytest.fixture(autouse=True)
def benchmark_imports(monkeypatch):
    import sys

    monkeypatch.delenv("SAAS_VERIFIER_LAUNCHER", raising=False)
    monkeypatch.syspath_prepend(str(BENCHMARK_ROOT))
    # Other boundary fixtures deliberately install a temporary SaaS package.
    monkeypatch.delitem(sys.modules, "saas_bench", raising=False)


def verdict(score=0.25):
    return {
        "status": "PASS" if score == 1 else "FAIL",
        "score": score,
        "earned": int(score * 4),
        "total": 4,
        "all_pass": score == 1,
        "checks": [
            {"label": "saved artifact", "weight": 1, "passed": score > 0, "detail": ""},
            {"label": "required contents", "weight": 3, "passed": score == 1, "detail": ""},
        ],
    }


def config(tmp_path, harness="opencode", **kwargs):
    return SaasBenchConfig(
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        name="fixture",
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "episodes"),
        source_root=str(BENCHMARK_ROOT),
        harness=harness,
        model_name="policy",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **kwargs,
    )


def request():
    return SaasBenchRunRequest.model_validate(
        {
            "responses_create_params": {"input": "Fixture task"},
            "_ng_rollout_id": "fixture-0",
            "verifier_metadata": {"task_id": "fixture", "category_id": "fixture", "source_sha256": "fixture"},
        }
    )


def agent(tmp_path, harness):
    return SaasBenchAgent(config=config(tmp_path, harness), server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize("harness", ["opencode", "browser_use"])
@pytest.mark.parametrize(
    "ending",
    [
        None,
        "output_token_limit",
        "context_limit",
        "malformed_tool_call",
        "model_protocol_error",
        "timeout",
        "permission_abort",
    ],
)
@pytest.mark.parametrize("score", [0.0, 0.25, 1.0])
def test_every_model_ending_preserves_native_grade(tmp_path, harness, ending, score):
    scorer = agent(tmp_path, harness)
    native = {
        "status": "completed",
        "artifacts": "fixture",
        "verify": verdict(score),
        "reward": 0.0,
        "mask_sample": False,
        "agent_output": "Retained model output",
        "model_turn_observed": True,
        "failure_origin": "model",
        "failure_kind": "saas_bench:model_ending" if ending else None,
        "execution_health": {"termination": ending or "final_response"},
    }
    response = scorer._response(request(), native)
    assert response.reward == score and response.mask_sample is False
    assert response.native_metrics["weighted_score"] == score
    assert response.scoring_policy == "native_weighted_final_state_v1"
    if ending:
        assert response.failure_outcome_action == "verify"
        assert response.failure_outcome_attribution == "model"


@pytest.mark.parametrize("harness", ["opencode", "browser_use"])
@pytest.mark.parametrize(
    "bad",
    [
        "judge",
        "error",
        "nan",
        "boolean",
        "out_of_range",
        "wrong_total",
        "wrong_earned",
        "wrong_status",
        "invalid_checks",
        "malformed_judge",
        "malformed_result",
    ],
)
def test_invalid_grading_is_masked_even_when_worker_claims_available(tmp_path, harness, bad):
    v = verdict()
    if bad == "judge":
        v["judge_failures"] = [{"failure_kind": "judge_failed", "reason": "503"}]
    if bad == "error":
        v["error"] = "readback unavailable"
    if bad == "nan":
        v["score"] = float("nan")
    if bad == "boolean":
        v["score"] = True
    if bad == "out_of_range":
        v["score"] = 1.1
    if bad == "wrong_total":
        v["total"] = 5
    if bad == "wrong_earned":
        v["earned"] = 2
    if bad == "wrong_status":
        v["status"] = "PASS"
    if bad == "invalid_checks":
        v["checks"] = None
    if bad == "malformed_judge":
        v["judge_failures"] = [False]
    if bad == "malformed_result":
        v = ["not a verifier object"]
    native = {
        "status": "completed",
        "artifacts": "fixture",
        "verify": v,
        "mask_sample": False,
        "verifier_diagnostics": {"unavailable": False},
        "execution_health": {"termination": "output_token_limit"},
        "model_turn_observed": True,
    }
    response = agent(tmp_path, harness)._response(request(), native)
    assert response.mask_sample and response.reward == 0 and not response.native_metrics
    assert response.failure_outcome_attribution == "evaluation"


@pytest.mark.parametrize(
    "stage,kind,action",
    [
        ("agent", "agent_limit", "count_as_zero"),
        ("agent", "agent_timeout", "retry"),
        ("agent", "agent_run_error", "count_as_zero"),
        ("verification", "judge_failed", "count_as_zero"),
        ("verification", "verifier_error", "verify"),
    ],
)
def test_conflicting_outcome_policy_cannot_start(tmp_path, stage, kind, action):
    with pytest.raises(ValidationError):
        config(tmp_path, outcome_policy={"by_stage": {stage: {"by_failure_kind": {kind: action}}}})


@pytest.mark.asyncio
@pytest.mark.parametrize("reward,status", [(float("nan"), "FAIL"), (True, "FAIL"), (0.5, "ERROR"), (1.2, "PASS")])
async def test_invalid_unmasked_aggregate_rows_are_rejected(tmp_path, reward, status):
    scorer = agent(tmp_path, "opencode")
    with pytest.raises(ValueError):
        await scorer.aggregate_metrics(
            AggregateMetricsRequest(
                verify_responses=[
                    {
                        "reward": reward,
                        "mask_sample": False,
                        "verify_status": status,
                        "category_id": "fixture",
                    }
                ]
            )
        )


@pytest.mark.asyncio
async def test_aggregation_retains_zero_and_masks_faults_and_detects_tampering(tmp_path):
    scorer = agent(tmp_path, "opencode")
    rows = [
        {
            "reward": value,
            "mask_sample": False,
            "verify_status": "PASS" if value == 1 else "FAIL",
            "category_id": "fixture",
        }
        for value in [0.0, 0.25, 1.0]
    ]
    rows += [{"reward": 0, "mask_sample": True, "verify_status": "ERROR", "category_id": "fixture"}]
    metrics = (await scorer.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).agent_metrics
    assert metrics["saas_bench/num_scored"] == 3 and metrics["saas_bench/num_masked"] == 1
    assert metrics["saas_bench/mean_score"] == pytest.approx(1.25 / 3)
    assert metrics["saas_bench/pass_rate"] == pytest.approx(1 / 3)
    rows[1]["native_diagnostics"] = {"verifier": verdict(0.0)}
    with pytest.raises(ValueError, match="differs"):
        await scorer.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))


def runner():
    spec = importlib.util.spec_from_file_location("scoring_runner", BENCHMARK_ROOT / "saas_bench/verify_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "output",
    [
        "[PASS] (1pt) first\nSCORE: 1.000 PASS: True (1/1)\nSCORE: 1.000 PASS: True (1/1)",
        "[PASS] (1pt) first\nSCORE: 0.500 PASS: False (1/2)",
        "[FAIL] (1pt) missing\nSCORE: 1.000 PASS: True (1/1)",
        "[PASS] (1pt) first\nSCORE: 1.000 PASS: True (1/1) trailing garbage",
        "[PASS] (1pt) first\nSCORE: nan PASS: True (1/1)",
        "[PASS] (1pt) first\nSCORE: 1.000 PASS: True (0/0)",
        "[PASS] (1pt) first",
        "SCORE: 1.000 PASS: True (1/1)",
    ],
)
def test_malformed_verifier_subprocess_is_unscorable(tmp_path, output):
    script = tmp_path / "verify.py"
    script.write_text("import sys\nprint(" + repr(output) + ",file=sys.stderr)\n")
    result = runner().run_verify(
        {"task_id": "fixture", "verify_py_path": str(script)}, 0, {}, "localhost", str(tmp_path)
    )
    assert result["status"] == "ERROR" and result["error"]
    from saas_bench.opencode.verifier import native_verification_outcome

    assert native_verification_outcome(result)["mask_sample"] is True


def test_caught_readback_fault_retries_same_state_and_preserves_real_zero(tmp_path):
    script = tmp_path / "verify.py"
    marker = tmp_path / "attempts"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        + f"p=Path({str(marker)!r});n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))\n"
        + "fault='  (connection refused)' if n==1 else ''\n"
        + "print('[FAIL] (1pt) missing expected artifact'+fault+'\\nSCORE: 0.000 PASS: False (0/1)',file=sys.stderr)\n"
    )
    result = runner().run_verify(
        {"task_id": "fixture", "verify_py_path": str(script)},
        0,
        {},
        "localhost",
        str(tmp_path),
        inconclusive_verification_retries=1,
    )
    assert marker.read_text() == "2" and result["status"] == "FAIL" and result["score"] == 0
    assert len(result["verification_attempts"]) == 2
    from saas_bench.opencode.verifier import native_verification_outcome

    assert native_verification_outcome(result)["mask_sample"] is False


def test_legacy_pass_cannot_claim_partial_credit():
    from saas_bench.opencode.verifier import native_verification_outcome

    assert native_verification_outcome({"status": "PASS", "score": 0.25})["mask_sample"] is True


def test_legacy_cached_response_is_not_relabelled_with_corrected_policy(tmp_path):
    from responses_api_agents.saas_bench_agent.schemas import SaasBenchVerifyResponse

    native = {"artifacts": "fixture", "verify": verdict(), "mask_sample": False}
    response = agent(tmp_path, "opencode")._response(request(), native)
    data = response.model_dump()
    data.pop("scoring_policy")
    assert SaasBenchVerifyResponse.model_validate(data).scoring_policy is None
    assert response.scoring_policy == "native_weighted_final_state_v1"
