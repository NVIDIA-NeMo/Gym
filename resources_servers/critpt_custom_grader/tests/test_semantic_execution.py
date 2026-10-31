# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic-validator control-plane tests. Operator source remains inert in this file."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.client import ServerClient
from resources_servers.critpt_custom_grader.app import CritPtCustomGraderConfig, CritPtCustomGraderServer
from resources_servers.critpt_custom_grader.execution import Grader, build_compare_job, build_run_job
from resources_servers.critpt_custom_grader.reference_cache import ReferenceCache
from resources_servers.critpt_custom_grader.runner import validate_job
from resources_servers.critpt_custom_grader.task_data import TaskData
from resources_servers.critpt_custom_grader.tests.support import (
    CANDIDATE,
    VALUE,
    ScriptedProvider,
    compare_result,
    exited,
    happy_script,
    make_policy,
    make_spec,
    make_task,
)


SOURCE = "raise AssertionError('operator source must not execute on the host')\n"
REGISTRY = {"synthetic": {"source": SOURCE, "entrypoint": "validate"}}


def semantic_task(**selection):
    fields = make_task().model_dump(mode="json")
    fields["semantic_validator"] = {"id": "synthetic", "config": {"rank": 2}, **selection}
    return TaskData.model_validate(fields)


def grader(script=None, *, registry=None, cache=None):
    provider = ScriptedProvider(happy_script() if script is None else script)
    instance = Grader(
        make_policy(),
        provider=provider,
        spec=make_spec(),
        reference_cache=cache,
        semantic_validators=REGISTRY if registry is None else registry,
    )
    return instance, provider


def config(registry, **overrides):
    return CritPtCustomGraderConfig(
        host="127.0.0.1",
        port=8000,
        name="synthetic-grader",
        entrypoint="app.py",
        semantic_validators=registry,
        **overrides,
    )


def test_task_selects_only_a_bounded_validator_id_and_inert_config():
    task = semantic_task()
    assert task.semantic_validator.id == "synthetic"
    assert task.semantic_validator.config == {"rank": 2}
    for field in ("source", "source_path", "entrypoint", "timeout_s"):
        with pytest.raises(ValidationError):
            semantic_task(**{field: "not-task-owned"})


@pytest.mark.parametrize("identifier", ["../escape", "", "with.dot", "x" * 129, "validate\n"])
def test_task_validator_id_rejects_paths_and_unbounded_names(identifier):
    with pytest.raises(ValidationError):
        semantic_task(id=identifier)


@pytest.mark.parametrize("parameters", [{"value": "x" * 16385}, {"value": float("nan")}, {"value": object()}])
def test_task_validator_config_is_bounded_exact_json(parameters):
    with pytest.raises(ValidationError):
        semantic_task(config=parameters)


def test_semantic_validator_rejects_expected_exception_cases():
    fields = make_task().model_dump(mode="json")
    fields["semantic_validator"] = {"id": "synthetic"}
    fields["test_cases"][0]["expected"] = {"kind": "exception"}
    with pytest.raises(ValidationError):
        TaskData.model_validate(fields)


def test_operator_registry_accepts_source_without_executing_it():
    configured = config(REGISTRY)
    assert configured.semantic_validators["synthetic"].source == SOURCE


async def test_server_startup_propagates_its_registry_to_the_grader():
    configured = config(
        REGISTRY,
        sandbox_provider={"critpt_fake": {}},
        sandbox_spec={"image": "example/grader:test"},
        startup_probe=False,
    )
    provider = ScriptedProvider(happy_script())
    server = CritPtCustomGraderServer(config=configured, server_client=MagicMock(spec=ServerClient))
    server._provider = provider
    try:
        await server.startup()
        result = await server._grader.grade(semantic_task(), CANDIDATE)
        assert result.outcome == "passed"
    finally:
        await server.shutdown()
    assert provider.aclosed == 1


@pytest.mark.parametrize(
    "registry",
    [
        {"../escape": {"source": SOURCE}},
        {"synthetic": {"source": "x" * 262145}},
        {"synthetic": {"source": "é" * 131073}},
        {"synthetic": {"source": "\x00"}},
        {"synthetic": {"source": SOURCE, "entrypoint": "../load"}},
        {f"v{index}": {"source": SOURCE} for index in range(65)},
        {f"v{index}": {"source": "#" * 262144} for index in range(9)},
    ],
)
def test_operator_registry_rejects_unsafe_or_oversized_definitions(registry):
    with pytest.raises((ValidationError, ValueError)):
        config(registry)


async def test_unknown_validator_refuses_before_any_sandbox_allocation():
    instance, provider = grader(registry={})
    result = await instance.grade(semantic_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == ("unscorable", "task_invalid", None)
    assert provider.created == []


async def test_validator_source_and_inputs_are_delivered_only_to_comparator():
    instance, provider = grader()
    task = semantic_task()
    result = await instance.grade(task, CANDIDATE, job_id="semantic123")
    assert result.outcome == "passed"
    assert result.cleanup == {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
    for box in provider.sandboxes.values():
        files = box["files"]
        source = [value for name, value in files.items() if name.endswith("/validator_source.py")]
        if box["domain"] == "comparator":
            assert source == [SOURCE.encode()]
            (payload,) = [json.loads(value) for name, value in files.items() if name.endswith("/job.json")]
            assert payload["semantic_validator"]["entrypoint"] == "validate"
            assert payload["cases"][0]["inputs"] == {
                "args": [{"format": "critpt-value-v1", "value": ["int", "1"]}],
                "kwargs": {},
            }
            assert payload["cases"][0]["validator_config"] == {"rank": 2}
        else:
            assert source == []
            assert not any(name.endswith("/semantic_validation.py") for name in files)
    run_job = build_run_job(task, "f", "nonce123", make_policy())
    assert "semantic_validator" not in run_job
    assert "validator_config" not in run_job["cases"][0]


async def test_reference_validator_false_prevents_candidate_allocation():
    script = happy_script()
    script[("result", "comparator")] = [compare_result(["invalid_reference", "equal"], side="reference")]
    instance, provider = grader(script)
    result = await instance.grade(semantic_task(), CANDIDATE)
    assert (result.outcome, result.category) == ("unscorable", "reference_mismatch")
    assert all(spec.metadata["ng-grader-domain"] != "candidate" for spec in provider.created)


@pytest.mark.parametrize("statuses", [["mismatch", "uncertain"], ["uncertain", "mismatch"]])
async def test_semantic_uncertainty_outranks_a_different_case_rejection(statuses):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(statuses)
    instance, _ = grader(script)
    result = await instance.grade(semantic_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == ("unscorable", "comparator_uncertain", None)


async def test_semantic_batch_timeout_is_unscorable_after_good_preflight():
    script = happy_script()
    timeout = compare_result([])
    timeout.update(status="timeout", timed_out_case=0)
    script[("result", "comparator")][1] = timeout
    script[("exec", "comparator")] = [exited(0), exited(3)]
    instance, _ = grader(script)
    result = await instance.grade(semantic_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == ("unscorable", "comparator_uncertain", None)


async def test_semantic_false_is_a_candidate_failure_after_good_preflight():
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["mismatch", "equal"])
    instance, _ = grader(script)
    result = await instance.grade(semantic_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == ("failed", "candidate_mismatch", 0.0)


@pytest.mark.parametrize("change", ["source", "entrypoint", "config"])
async def test_reference_cache_does_not_reuse_acceptance_after_validator_change(change):
    cache = ReferenceCache()
    instance, _ = grader(cache=cache)
    assert (await instance.grade(semantic_task(), CANDIDATE)).outcome == "passed"
    registry = {"synthetic": dict(REGISTRY["synthetic"])}
    selection = {}
    if change == "config":
        selection["config"] = {"rank": 3}
    else:
        registry["synthetic"][change] = SOURCE + "# changed\n" if change == "source" else "other_validator"
    second, provider = grader(registry=registry, cache=cache)
    assert (await second.grade(semantic_task(**selection), CANDIDATE)).outcome == "passed"
    assert any(spec.metadata["ng-grader-domain"] == "reference" for spec in provider.created)


async def test_cached_semantic_preflight_still_uploads_validator_for_candidate_comparison():
    cache = ReferenceCache()
    initial, _ = grader(cache=cache)
    assert (await initial.grade(semantic_task(), CANDIDATE)).outcome == "passed"
    script = happy_script()
    del script[("result", "reference")]
    script[("result", "comparator")] = [compare_result(["equal", "equal"])]
    repeated, provider = grader(script, cache=cache)
    result = await repeated.grade(semantic_task(), CANDIDATE)
    assert result.outcome == "passed"
    assert result.cleanup == {"reference": "not_created", "comparator": "deleted", "candidate": "deleted"}
    (comparator,) = [box for box in provider.sandboxes.values() if box["domain"] == "comparator"]
    assert [data for path, data in comparator["files"].items() if path.endswith("/validator_source.py")] == [
        SOURCE.encode()
    ]


def test_default_compare_job_keeps_its_existing_shape():
    job = build_compare_job(make_task(), "candidate", [VALUE, VALUE], "nonce123", make_policy())
    assert "semantic_validator" not in job
    assert set(job["cases"][0]) == {"observed", "expected", "policy"}
    assert validate_job(job, "compare") is None
