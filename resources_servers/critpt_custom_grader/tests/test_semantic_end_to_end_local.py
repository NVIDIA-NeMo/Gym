# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real grade of a task that selects a semantic validator, through Gym's `local` provider, on synthetic values.

Self-skips where the runner cannot run (not Linux, root, no procfs children, no sympy)."""

import os
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox import create_provider
from resources_servers.critpt_custom_grader.execution import ExecutionPolicy, Grader
from resources_servers.critpt_custom_grader.reference_cache import ReferenceCache
from resources_servers.critpt_custom_grader.task_data import TaskData
from resources_servers.critpt_custom_grader.tests.support import make_spec, value_outcome, wire


pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="the runner needs Linux"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="the runner refuses root"),
    pytest.mark.skipif(not Path(f"/proc/self/task/{os.getpid()}/children").exists(), reason="no procfs children"),
]
pytest.importorskip("sympy")

# Synthetic: any two positive integers whose product is the input are accepted, in either order.
VALIDATOR = """
def validate(*, observed, expected, args, kwargs, config):
    if type(observed) is not list or len(observed) != 2 or any(type(item) is not int for item in observed):
        return False
    return observed[0] > 0 and observed[1] > 0 and observed[0] * observed[1] == args[0]
"""
TASK = {
    "problem_id": "synthetic-factor-pair",
    "reference_source": "def f(n):\n    return [1, n]\n",
    "entrypoint": "f",
    "semantic_validator": {"id": "factor_pair"},
    "test_cases": [{"args": [wire(["int", "6"])], "expected": value_outcome(["list", [["int", "1"], ["int", "6"]]])}],
}


@pytest.fixture
def grader(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    provider = create_provider({"local": {"workspace_root": str(tmp_path / "ws")}})
    policy = ExecutionPolicy(
        interpreter=sys.executable,
        workdir=str(work),
        exec_user=None,
        suite_timeout_s=120,
        compare_timeout_s=120,
        compare_per_case_s=20,
        cpu_time_limit_s=120,  # the policy refuses a CPU limit above the suite deadline plus the exec margin
        memory_limit_mib=2048,
        max_processes=4096,  # the local provider shares the host uid; RLIMIT_NPROC counts every process of that uid
        create_timeout_s=30.0,
        transfer_timeout_s=30.0,
        exec_timeout_margin_s=20.0,
        cleanup_timeout_s=30.0,
        max_concurrent_jobs=2,
    )
    return Grader(
        policy,
        provider=provider,
        spec=make_spec(image=None, workdir=str(work)),
        reference_cache=ReferenceCache(),
        semantic_validators={"factor_pair": {"source": VALIDATOR}},
    )


@pytest.mark.asyncio
async def test_a_validator_accepts_another_valid_answer_and_rejects_a_wrong_one(grader):
    task = TaskData.model_validate(TASK)
    other_valid = await grader.grade(task, "def f(n):\n    return [3, n // 3]\n")
    assert other_valid.category == "passed", other_valid.category  # [3, 2] differs from the stored [1, 6]
    assert other_valid.cleanup == {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
    wrong = await grader.grade(task, "def f(n):\n    return [n, n]\n")
    assert (wrong.outcome, wrong.category) == ("failed", "candidate_mismatch")
    assert wrong.cleanup["reference"] == "not_created"  # the cached preflight skipped the reference run
