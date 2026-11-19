# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real grade through Gym's `local` provider: each example row passes with its own reference as the candidate and
fails with a wrong candidate. Self-skips where the runner cannot run (not Linux, root, no procfs children, no sympy)."""

import json
import os
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox import create_provider
from resources_servers.critpt_custom_grader.execution import ExecutionPolicy, Grader
from resources_servers.critpt_custom_grader.reference_cache import ReferenceCache
from resources_servers.critpt_custom_grader.task_data import TaskData
from resources_servers.critpt_custom_grader.tests.support import make_spec


EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"
pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="the runner needs Linux"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="the runner refuses root"),
    pytest.mark.skipif(not Path(f"/proc/self/task/{os.getpid()}/children").exists(), reason="no procfs children"),
]
pytest.importorskip("sympy")


def rows():
    return [json.loads(line) for line in EXAMPLES.read_text().splitlines() if line.strip()]


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
        policy, provider=provider, spec=make_spec(image=None, workdir=str(work)), reference_cache=ReferenceCache()
    )


@pytest.mark.asyncio
async def test_example_rows_pass_with_their_reference_and_fail_with_a_wrong_candidate(grader):
    for row in rows():
        task = TaskData.model_validate(row["task_data"])
        good = await grader.grade(task, task.reference_source)
        assert good.category == "passed", (task.problem_id, good.category)
        assert good.cleanup == {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
        wrong = await grader.grade(task, f"def {task.entrypoint}(*args, **kwargs):\n    return 10**9\n")
        assert wrong.outcome == "failed" and wrong.category in ("candidate_mismatch", "candidate_invalid_output"), (
            task.problem_id,
            wrong.category,
        )
        assert wrong.cleanup["reference"] == "not_created"  # the cached preflight skipped the reference run
