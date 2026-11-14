# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A strict run re-judges its saved judge failures before its completion check can reject it.

``require_complete`` (on by default for ``gym eval submit``) makes ``run_from_config`` raise
``IncompleteEvaluationError`` when completed is below expected, and a masked judge failure is not completed. The
re-judge pass used to run only after ``run_from_config`` returned, so a strict run that ended with a due judge failure
was rejected before its judge was ever asked again. Both built-in collection entry points now run the pass first.

Not covered: a benchmark with a custom ``rollout_collection_driver`` runs neither the pass nor this ordering; its
driver owns collection and completion. These tests make no claim about such drivers.
"""

import pytest

from nemo_gym import rollout_reverification
from nemo_gym.cli import eval as eval_cli
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.eval.test_rejudge_saved_judge_failures import ANSWER, _run, _settle


@pytest.fixture
def strict_cli(tmp_path, monkeypatch):
    """``gym eval collect`` around a collection that saves one judge failure and is rejected by its real check."""
    output = tmp_path / "rollouts.jsonl"
    state = {"failures": None, "rejection": None, "failed_marks": 0, "reverify_passes": 0}

    async def collect(self, config):
        _, state["failures"] = _run(tmp_path, judge_failures=1)
        coverage = RolloutStore.read(output, import_legacy=False).coverage()
        try:
            config.check_completion(
                expected=coverage["expected"],
                results=[],
                completed=coverage["successful"] + coverage["final_zeros"],
                intentionally_omitted=coverage["intentionally_omitted"],
                retryable=coverage["retryable"] > 0,
                made_progress=True,
            )
        except IncompleteEvaluationError as error:
            state["rejection"] = error
            raise
        return []

    def configure(*, strict: bool, judge_answers: bool) -> None:
        async def reverify(self, config):
            state["reverify_passes"] += 1
            if judge_answers:
                _settle(output, state["failures"][0], 1.0)

        monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", collect)
        monkeypatch.setattr(rollout_reverification.RolloutReverificationHelper, "run_from_config", reverify)
        monkeypatch.setattr(
            eval_cli,
            "get_global_config_dict",
            lambda: {
                "input_jsonl_fpath": str(tmp_path / "input.jsonl"),
                "output_jsonl_fpath": str(output),
                "require_complete": strict,
                "disable_aggregation": True,
            },
        )

    def mark_failed():
        state["failed_marks"] += 1

    monkeypatch.setattr(eval_cli, "mark_run_failed", mark_failed)
    monkeypatch.setattr(eval_cli, "_interrupt_on_sigterm", lambda: None)
    monkeypatch.setattr(eval_cli, "_finalize_live_exporters", lambda output: None)
    monkeypatch.setattr("nemo_gym.artifacts.resolved_config.write_resolved_config", lambda *args, **kwargs: None)
    for name in ("configure_telemetry_env", "init_telemetry", "shutdown_telemetry"):
        monkeypatch.setattr(f"nemo_gym.telemetry.setup.{name}", lambda *args, **kwargs: None)
    return output, state, configure


def test_a_strict_run_whose_judge_answers_on_the_pass_completes(strict_cli) -> None:
    output, state, configure = strict_cli
    configure(strict=True, judge_answers=True)

    eval_cli.collect_rollouts()

    assert state["rejection"] is not None  # the collection's own check did reject it first
    assert state["reverify_passes"] == 1 and state["failed_marks"] == 0
    coverage = RolloutStore.read(output, import_legacy=False).coverage()
    assert (coverage["successful"], coverage["expected"]) == (3, 3)


def test_a_strict_run_whose_judge_stays_down_keeps_its_original_rejection(strict_cli) -> None:
    output, state, configure = strict_cli
    configure(strict=True, judge_answers=False)

    with pytest.raises(SystemExit) as raised:
        eval_cli.collect_rollouts()

    assert state["reverify_passes"] == 1  # the pass ran before the rejection
    assert raised.value.code == state["rejection"].exit_code and state["failed_marks"] == 1
    coverage = RolloutStore.read(output, import_legacy=False).coverage()
    assert (coverage["successful"], coverage["expected"]) == (2, 3)  # the judge failure stays masked


def test_a_non_strict_rejection_is_raised_as_before_without_a_pass(strict_cli, monkeypatch) -> None:
    output, state, configure = strict_cli
    configure(strict=False, judge_answers=True)

    async def reject(self, config):
        _, state["failures"] = _run(output.parent, judge_failures=3)
        state["rejection"] = IncompleteEvaluationError("none of the dispatched rollouts produced a result")
        raise state["rejection"]

    monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", reject)
    with pytest.raises(SystemExit) as raised:
        eval_cli.collect_rollouts()

    assert raised.value.code == state["rejection"].exit_code
    assert state["reverify_passes"] == 0 and state["failed_marks"] == 1


def test_the_e2e_entry_point_uses_the_same_ordering_and_the_custom_driver_branch_does_not() -> None:
    import inspect

    # The e2e entry point (what a submitted run executes) cannot run in a unit test: it starts the run's servers.
    # Its built-in branch calls the same ordered helper as `gym eval collect`; its custom-driver branch calls neither.
    e2e = inspect.getsource(eval_cli.e2e_rollout_collection)
    driver_call = e2e.index("asyncio.run(driver_fn(")
    driver = e2e[e2e.index("module_name, _, fn_name = driver_path") : driver_call]
    assert "_rejudge" not in driver
    built_in = e2e[driver_call : e2e.index("collection_completed = True", driver_call)]
    assert "_collect_and_rejudge(rch, rollout_collection_config, global_config_dict, output_fpath)" in built_in
    assert "_collect_and_rejudge(rch, config, " in inspect.getsource(eval_cli.collect_rollouts)


def test_a_rejected_run_that_the_pass_completes_ends_with_its_aggregate_and_a_settled_batch_status(
    tmp_path, monkeypatch
) -> None:
    """The real collector rejects a strict run and removes its aggregate; the pass completes it and the aggregate returns."""
    import orjson

    import nemo_gym.rollouts.rollout_collection as collection
    from nemo_gym.rollouts.batch_status import observe_materialized_rows
    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig
    from nemo_gym.utils.path_utils import aggregate_metrics_path_for
    from tests.unit_tests.rollouts.test_rollout_collection import (
        AsyncMock,
        FakeResponse,
        compute_aggregate_metrics,
        install_fake_server_client,
    )

    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    input_fpath = tmp_path / "input.jsonl"
    input_fpath.write_text(
        "".join(
            orjson.dumps(
                {
                    "responses_create_params": {"input": []},
                    "agent_ref": {"name": "my_agent"},
                    "x": i,
                    "task_source": "src",
                }
            ).decode()
            + "\n"
            for i in range(2)
        )
    )
    output = tmp_path / "rollouts.jsonl"
    manifest_fpath = tmp_path / "batch_manifest.json"
    run_config = {
        "input_jsonl_fpath": str(input_fpath),
        "output_jsonl_fpath": str(output),
        "batch_manifest_fpath": str(manifest_fpath),
        "route_failures_to_sidecar": True,
        "disable_health_check": True,
        "require_complete": True,
    }
    config = RolloutCollectionConfig(**run_config)
    rows = collection.RolloutCollectionHelper._preprocess_rows_from_config(None, config)
    [(agent, observed)] = observe_materialized_rows(rows).items()
    manifest_fpath.write_bytes(
        orjson.dumps(
            {
                "schema_version": "1",
                "members": {
                    "bench": {
                        "agent_name": agent,
                        "task_sources": observed.task_sources,
                        "dataset_sha256": observed.dataset_sha256,
                        "expected_task_count": observed.task_count,
                        "expected_rollout_count": observed.rollout_count,
                        "repeat_policy": observed.repeat_policy.model_dump(),
                        "resolved_recipe_sha256": "a" * 64,
                        "metric_keys": ["mean/reward"],
                    }
                },
            }
        )
    )
    judge_down = {
        "_ng_failure_id": "judge-0",
        "reward": 0.0,
        "mask_sample": True,
        "failure_kind": "judge_failed",
        "_ng_failure_class": "judge_failed",
        "response": ANSWER,
    }

    async def post(server_name, url_path, json, **kwargs):
        if url_path == "/run":
            return FakeResponse(200, judge_down if json["x"] == 0 else {"reward": 1.0, "response": {}})
        return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    aggregate = aggregate_metrics_path_for(output)
    aggregate.write_text("[]")  # an earlier segment's aggregate, which the rejected run removes

    async def judge_answers(self, config):
        failure = RolloutStore.read(output, import_legacy=False).failures()[0]
        _settle(output, failure, 1.0)
        # The pass starts from the rejected run: no score published, aggregation still pending.
        assert not aggregate.exists()
        pending = orjson.loads((tmp_path / "batch_status.json").read_bytes())
        assert pending["members"][agent]["aggregation_status"] == "pending"

    monkeypatch.setattr(rollout_reverification.RolloutReverificationHelper, "run_from_config", judge_answers)

    eval_cli._collect_and_rejudge(collection.RolloutCollectionHelper(), config, run_config, output)

    metrics = orjson.loads(aggregate.read_bytes())
    assert metrics[0]["key_metrics"]["mean/reward"] == 1.0
    status = orjson.loads((tmp_path / "batch_status.json").read_bytes())
    assert status["members"][agent]["aggregation_status"] == "complete"
    assert status["members"][agent]["completed_rollout_count"] == 2
