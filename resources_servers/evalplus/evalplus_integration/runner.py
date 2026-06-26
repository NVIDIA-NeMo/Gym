# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray-remote wrapper for EvalPlus per-task verification.

Wraps `evalplus.evaluate.check_correctness` so that each verify() call
in the resource server can dispatch a single-task check to a Ray worker
and await the result without blocking the asyncio loop.

The Ray runtime_env mirrors the pattern in
`resources_servers/code_gen/lcb_integration/compute_code_generation_metrics.py`:
the worker uses the server's venv Python and adds the server directory
to PYTHONPATH so the package-less `evalplus_integration` module resolves.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import ray


# evalplus_integration/ has no pyproject.toml so the worker needs the
# server dir on PYTHONPATH for `import evalplus_integration` to resolve.
_EVALPLUS_SERVER_DIR = str(Path(__file__).parent.parent)


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": _EVALPLUS_SERVER_DIR},
    },
)
def check_correctness_remote(
    dataset: str,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    min_time_limit: float,
    gt_time_limit_factor: float,
) -> Dict[str, Any]:
    """Run base + plus checks for a single (task, completion) pair.

    Args mirror `evalplus.evaluate.check_correctness` with `base_only=False`.
    Returns a dict with `base_status`, `plus_status` (values: "pass" |
    "fail" | "timeout" | other strings from evalplus.eval).
    """
    from evalplus.evaluate import check_correctness

    result = check_correctness(
        dataset=dataset,
        completion_id=0,
        problem=problem,
        solution=solution,
        expected_output=expected_output,
        base_only=False,
        fast_check=False,
        identifier=problem.get("task_id"),
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )
    base_outcome = result.get("base", [None, None])
    plus_outcome = result.get("plus", [None, None])
    return {
        "base_status": base_outcome[0] if base_outcome else None,
        "plus_status": plus_outcome[0] if plus_outcome else None,
        "base_details": base_outcome[1] if len(base_outcome) > 1 else None,
        "plus_details": plus_outcome[1] if len(plus_outcome) > 1 else None,
    }
