# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gym.rollout.completed_total``: the driver-side attempt counter, against a real in-memory reader."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym import rollout_collection
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client
from tests.unit_tests.test_rollout_recovery import runner_config  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

COMPLETED = gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT
OUTCOME = gym_metrics.ROLLOUT_OUTCOME_ATTRIBUTE
DISPATCH = gym_metrics.DISPATCH_NAME_ATTRIBUTE
CLASS = gym_metrics.FAILURE_CLASS_ATTRIBUTE
KIND = gym_metrics.FAILURE_KIND_ATTRIBUTE
TYPE = gym_metrics.FAILURE_TYPE_ATTRIBUTE


def _by_attrs(collected):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(COMPLETED, [])}


def _key(**attrs):
    return tuple(sorted(attrs.items()))


def _rollout_group_only(monkeypatch):
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: group == GymSpanGroup.ROLLOUT)


def test_outcomes_are_counted_with_bounded_failure_attributes(collected_metrics):  # noqa: F811
    gym_metrics.record_rollout_completed("scored", dispatch_name="env")
    gym_metrics.record_rollout_completed("scored", dispatch_name="env")
    gym_metrics.record_rollout_completed(
        "failed",
        dispatch_name="env",
        failure_class="agent_run_error",
        failure_kind="agent_run_error",
        failure_type="ClientResponseError",
    )
    gym_metrics.record_rollout_completed("omitted", dispatch_name="env", failure_class="kill_shaped")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored", DISPATCH: "env"}): 2,
        _key(
            **{
                OUTCOME: "failed",
                DISPATCH: "env",
                CLASS: "agent_run_error",
                KIND: "agent_run_error",
                TYPE: "ClientResponseError",
            }
        ): 1,
        _key(**{OUTCOME: "omitted", DISPATCH: "env", CLASS: "kill_shaped"}): 1,
    }


def test_driver_helper_reads_the_record_and_respects_the_span_group(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    rollout_collection._record_rollout_outcome(
        {
            rollout_collection.NG_FAILURE_CLASS_KEY: "agent_request_failed",
            "failure_kind": "transport_timeout",
            "_ng_failure_type": "ServerTimeoutError",
            "_ng_failure_message": "free text that must not become an attribute",
        },
        "failed",
        "env",
    )
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: False)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored", DISPATCH: "env"}): 1,
        _key(
            **{
                OUTCOME: "failed",
                DISPATCH: "env",
                CLASS: "agent_request_failed",
                KIND: "transport_timeout",
                TYPE: "ServerTimeoutError",
            }
        ): 1,
    }


def test_an_unregistered_kind_is_bucketed(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    for kind in ("connect to 10.0.0.7:8000 refused after 3 retries", "session_lost", "tb4:sandbox_gone"):
        rollout_collection._record_rollout_outcome(
            {rollout_collection.NG_FAILURE_CLASS_KEY: "infrastructure_error", "failure_kind": kind}, "failed", "env"
        )
    assert {dict(key)[KIND] for key in _by_attrs(collected_metrics)} == {
        gym_metrics.UNREGISTERED_FAILURE_KIND,
        "session_lost",
        "tb4:sandbox_gone",
    }


def test_failure_fields_on_a_scored_row_are_not_labels(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, "failure_kind": "verifier_error", "_ng_failure_type": "StaleEcho"}, "scored", "env"
    )
    assert _by_attrs(collected_metrics) == {_key(**{OUTCOME: "scored", DISPATCH: "env"}): 1}


@pytest.mark.parametrize("kind", [None, "verifier_error", "tb4:sandbox_gone", "unbounded text"])
def test_original_routing_labels_are_not_replaced_by_saved_record(collected_metrics, monkeypatch, kind):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome(
        {
            "_ng_failure_class": "agent_run_error",
            "failure_kind": kind,
            "_ng_failure_type": "TimeoutError",
            "_ng_failure_record": {"failure": {"failure_kind": "judge_failed"}, "exception_type": "StaleEcho"},
        },
        "failed",
        "env",
    )
    attrs = {OUTCOME: "failed", DISPATCH: "env", CLASS: "agent_run_error", TYPE: "TimeoutError"}
    if kind is not None:
        attrs[KIND] = gym_metrics.UNREGISTERED_FAILURE_KIND if kind == "unbounded text" else kind
    assert _by_attrs(collected_metrics) == {_key(**attrs): 1}


async def test_persisted_outcomes_count_once_and_resumed_cache_is_not_counted(
    runner_config,
    collected_metrics,
    monkeypatch,  # noqa: F811
):
    from nemo_gym.rollout_store import RolloutStore

    _rollout_group_only(monkeypatch)
    runner_config.route_failures_to_sidecar = True

    async def post(**kwargs):
        row = kwargs["json"]
        if row["_ng_task_index"] == 1 and row["_ng_attempt_index"] == 0:
            return FakeResponse(200, {"_ng_failure_class": "kill_shaped", "_ng_no_persist": True})
        if row["_ng_task_index"] == 2:
            return FakeResponse(200, {"_ng_failure_class": "skipped", "_ng_failure_terminal": True})
        return FakeResponse(200, {"reward": 1.0, "response": {}})

    post_mock = AsyncMock(side_effect=post)
    install_fake_server_client(monkeypatch, post_mock)
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    expected = {
        _key(**{OUTCOME: "scored", DISPATCH: "my_agent"}): 1,
        _key(**{OUTCOME: "failed", DISPATCH: "my_agent", CLASS: "kill_shaped"}): 1,
        _key(**{OUTCOME: "omitted", DISPATCH: "my_agent", CLASS: "skipped"}): 1,
    }
    assert _by_attrs(collected_metrics) == expected
    output = Path(runner_config.output_jsonl_fpath)
    coverage = RolloutStore.read(output).coverage()
    assert (coverage["successful"], coverage["failed"], coverage["intentionally_omitted"]) == (1, 1, 1)
    runner_config.resume_from_cache = True
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    expected[_key(**{OUTCOME: "scored", DISPATCH: "my_agent"})] = 2
    assert _by_attrs(collected_metrics) == expected
    assert post_mock.await_count == 4
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert _by_attrs(collected_metrics) == expected
    assert post_mock.await_count == 4


async def test_outcome_rejected_by_storage_is_not_counted(runner_config, collected_metrics, monkeypatch):  # noqa: F811
    from nemo_gym.rollout_store import RolloutStore

    _rollout_group_only(monkeypatch)
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, {"reward": 1.0, "response": {}})))

    def reject(self, result, **kwargs):
        raise OSError("Cannot save outcome")

    monkeypatch.setattr(RolloutStore, "record_outcome", reject)
    with pytest.raises(OSError, match="Cannot save outcome"):
        await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert _by_attrs(collected_metrics) == {}


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize(
    ("failure_class", "kind"),
    [
        ("agent_run_error", None),
        ("infrastructure_error", None),
        ("skipped", None),
        ("agent_run_error", "verifier_error"),
        ("infrastructure_error", "verifier_error"),
        ("agent_run_error", "private task 123 failed at host 10.0.0.1"),
        ("infrastructure_error", "private task 123 failed at host 10.0.0.1"),
    ],
)
async def test_legacy_counter_keeps_routing_class_and_only_explicit_kind(
    runner_config,
    collected_metrics,
    monkeypatch,
    routing,
    failure_class,
    kind,  # noqa: F811
):
    _rollout_group_only(monkeypatch)
    runner_config.route_failures_to_sidecar = routing

    async def post(**kwargs):
        if kwargs["json"]["_ng_task_index"] != 0:
            return FakeResponse(200, {"reward": 1.0, "response": {}})
        failure = {
            "_ng_failure_class": failure_class,
            "_ng_failure_type": "TimeoutError",
            "_ng_failure_terminal": failure_class == "skipped",
        }
        if kind is not None:
            failure["failure_kind"] = kind
        return FakeResponse(200, failure)

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    attrs = {
        OUTCOME: "omitted" if failure_class == "skipped" else "failed",
        DISPATCH: "my_agent",
        CLASS: failure_class,
        TYPE: "TimeoutError",
    }
    if kind is not None:
        attrs[KIND] = "verifier_error" if kind == "verifier_error" else gym_metrics.UNREGISTERED_FAILURE_KIND
    assert _by_attrs(collected_metrics) == {
        _key(**attrs): 1,
        _key(**{OUTCOME: "scored", DISPATCH: "my_agent"}): 2,
    }


@pytest.mark.parametrize(("status", "failure_class"), [(500, "agent_run_error"), (502, "agent_request_failed")])
async def test_transport_counter_does_not_invent_a_kind(
    runner_config, collected_metrics, monkeypatch, status, failure_class
):  # noqa: F811
    from tests.unit_tests.test_rollout_collection import http_error

    _rollout_group_only(monkeypatch)

    async def post(**kwargs):
        if kwargs["json"]["_ng_task_index"] == 0:
            raise http_error(status, "private error text")
        return FakeResponse(200, {"reward": 1.0, "response": {}})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "failed", DISPATCH: "my_agent", CLASS: failure_class, TYPE: "ClientResponseError"}): 1,
        _key(**{OUTCOME: "scored", DISPATCH: "my_agent"}): 2,
    }


@pytest.mark.parametrize("kind", [None, "verifier_error", "private task 123 failed"])
async def test_native_failure_kind_cannot_become_free_text_counter_class(
    runner_config, collected_metrics, monkeypatch, kind
):  # noqa: F811
    import orjson
    from omegaconf import OmegaConf

    _rollout_group_only(monkeypatch)
    Path(runner_config.input_jsonl_fpath).write_bytes(
        b"".join(
            orjson.dumps({"task_id": {"taskset": "native", "task_id": str(i)}, "task_input": {}}) + b"\n"
            for i in range(2)
        )
    )
    runner_config.environment_server_routes = {"native": "environment"}

    async def post(**kwargs):
        body = kwargs["json"]
        reply = {"episode_id": body["episode_id"], "task_id": body["task"]["task_id"]}
        if body["task"]["task_id"]["task_id"] == "0":
            reply["failure"] = {"failure_reason": "Unavailable", "terminal": False, "failure_kind": kind}
        else:
            reply["result"] = {"reward": 1.0}
        return FakeResponse(200, reply)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
    await rollout_collection.RolloutCollectionHelper().run_from_config(runner_config)
    label = (
        "environment_server_failed"
        if kind is None
        else ("verifier_error" if kind == "verifier_error" else "unregistered")
    )
    attrs = {OUTCOME: "failed", DISPATCH: "environment", CLASS: label}
    if kind is not None:
        attrs[KIND] = label
    assert _by_attrs(collected_metrics) == {
        _key(**attrs): 1,
        _key(**{OUTCOME: "scored", DISPATCH: "environment"}): 1,
    }


async def _failed_points(tmp_path, monkeypatch, collected_metrics, input_extra, reply):  # noqa: F811
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: group == GymSpanGroup.ROLLOUT)
    monkeypatch.setattr(rollout_collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(failing_row(0) | input_extra) + "\n" + json.dumps(failing_row(1)) + "\n")
    fields = rollout_collection.RolloutCollectionConfig.model_fields
    values = {
        "input_jsonl_fpath": str(source),
        "output_jsonl_fpath": str(tmp_path / "out.jsonl"),
        "route_failures_to_sidecar": True,
        "disable_aggregation": True,
        "disable_health_check": True,
        "num_samples_in_parallel": 1,
    }
    config = rollout_collection.RolloutCollectionConfig(**{k: v for k, v in values.items() if k in fields})

    async def post(**kwargs):
        body = dict(kwargs["json"])
        if body["_ng_task_index"] != 0:
            return FakeResponse(200, body | {"reward": 1.0, "response": {}})
        # Real agents build their reply from the request body, so input keys are echoed.
        return FakeResponse(200, body | reply)

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await rollout_collection.RolloutCollectionHelper().run_from_config(config)
    return [
        dict(point.attributes)
        for point in collected_metrics().get(gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT, [])
        if point.attributes.get("nemo.gym.rollout.outcome") == "failed"
    ]


@pytest.mark.parametrize(
    ("input_extra", "reply", "expected_class", "expected_kind"),
    [
        # A stale kind in the input row, echoed back by the agent.
        (
            {"_ng_failure_kind": None},
            {"_ng_failure_class": "timeout_exceeded", "reward": 0.0, "response": {}},
            "timeout_exceeded",
            None,
        ),
        (
            {},
            {"_ng_failure_class": "infrastructure_error", "_ng_failure_kind": "verifier_error"},
            "infrastructure_error",
            "verifier_error",
        ),
    ],
)
async def test_legacy_routing_class_survives_a_kind_key(
    tmp_path,
    monkeypatch,
    collected_metrics,
    input_extra,
    reply,
    expected_class,
    expected_kind,  # noqa: F811
):
    [point] = await _failed_points(tmp_path, monkeypatch, collected_metrics, input_extra, reply)
    assert point["nemo.gym.failure_class"] == expected_class
    assert point.get("nemo.gym.failure_kind") == expected_kind


@pytest.mark.parametrize("diagnostics", [None, {"error": "custom dispatcher detail"}])
async def test_custom_typed_dispatcher_keeps_labels_without_routing_diagnostics(
    runner_config, collected_metrics, monkeypatch, diagnostics
):  # noqa: F811
    import asyncio

    from nemo_gym.core.episode import EpisodeId
    from nemo_gym.rollout_outcomes import RolloutFailureRecord
    from nemo_gym.rollout_records import logical_rollout_id
    from nemo_gym.rollout_store import RolloutStore
    from nemo_gym.server_runtime.episode_types import EpisodeFailure

    _rollout_group_only(monkeypatch)
    install_fake_server_client(monkeypatch, AsyncMock())

    class TypedDispatcher(rollout_collection.RolloutCollectionHelper):
        def _run_examples_with_metadata(self, examples, **kwargs):
            kwargs["on_dispatch_batch"](examples)
            futures = []
            for row in examples:
                result = {"reward": 1.0, "response": {}}
                if row["_ng_task_index"] == 0:
                    result = RolloutFailureRecord(
                        episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=row["_ng_attempt_index"]),
                        run_id=row["_ng_run_id"],
                        source="environment",
                        delivery="delivered",
                        failure=EpisodeFailure(
                            failure_reason="Judge unavailable", terminal=False, failure_kind="verifier_error"
                        ),
                        exception_type="TimeoutError",
                    )
                future = asyncio.get_running_loop().create_future()
                future.set_result(
                    rollout_collection._CompletedRollout(
                        row=row,
                        result=result,
                        rollout_latency_ms=None,
                        diagnostics=diagnostics if isinstance(result, RolloutFailureRecord) else None,
                    )
                )
                futures.append(future)
            return futures

    await TypedDispatcher().run_from_config(runner_config)
    assert _by_attrs(collected_metrics) == {
        _key(**{OUTCOME: "scored", DISPATCH: "my_agent"}): 2,
        _key(
            **{
                OUTCOME: "failed",
                DISPATCH: "my_agent",
                CLASS: "verifier_error",
                KIND: "verifier_error",
                TYPE: "TimeoutError",
            }
        ): 1,
    }
    [failure] = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).failures()
    assert failure["_ng_failure_record"]["failure"]["failure_kind"] == "verifier_error"
    if diagnostics:
        assert failure["error"] == diagnostics["error"]
