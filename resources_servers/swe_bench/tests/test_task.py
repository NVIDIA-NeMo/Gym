# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.swe_bench.task import (
    ENVIRONMENT_NAME,
    SweTask,
    TaskSubmission,
    build_task,
    harness_family_key,
    parse_submission,
    parse_task_from_request,
)


def _sample_row() -> dict:
    inst = {
        "instance_id": "astropy__astropy-12907",
        "base_commit": "abc123",
        "test_patch": "",
        "FAIL_TO_PASS": '["tests/test_x.py::a"]',
        "PASS_TO_PASS": '["tests/test_x.py::b"]',
    }
    return {
        "instance_id": "astropy__astropy-12907",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "problem_statement": "Fix the bug.",
        "instance_dict": json.dumps(inst),
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Fix the bug."}],
        ),
    }


def test_harness_family_key_from_dataset_name() -> None:
    assert harness_family_key("princeton-nlp/SWE-bench_Verified") == "swe-bench"
    assert harness_family_key("something/R2E-Gym/foo") == "r2e-gym"


def test_build_task_sets_benchmark_fields() -> None:
    task = build_task(_sample_row(), container_formatter="swebench/sweb.eval.x86_64.{instance_id}")
    assert task.task_id == "astropy__astropy-12907"
    assert task.harness_family == "swe-bench"
    assert task.dataset_name == "princeton-nlp/SWE-bench_Verified"
    assert task.problem_statement == "Fix the bug."
    assert task.metadata["instance_dict"]["base_commit"] == "abc123"


def test_public_view_excludes_privileged_metadata() -> None:
    task = build_task(_sample_row(), container_formatter="x.{instance_id}")
    public = task.public_view()
    assert public.task_id == task.task_id
    assert public.environment == ENVIRONMENT_NAME
    assert public.harness_family == "swe-bench"
    assert not hasattr(public, "instance_dict")


def test_parse_task_from_request_requires_instance_id() -> None:
    class Body:
        responses_create_params = None
        verifier_metadata = {}

    with pytest.raises(ValueError, match="instance_id"):
        parse_task_from_request(Body(), container_formatter="x.{instance_id}")


def test_with_submission() -> None:
    task = SweTask(instance_id="x", benchmark="swe-bench")
    updated = task.with_submission(TaskSubmission(model_patch="diff"))
    assert updated.model_patch == "diff"


def test_parse_submission_accepts_git_patch_alias() -> None:
    assert parse_submission({"git_patch": "p"}).model_patch == "p"
