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

"""Grade one candidate against its unit tests in a child process.

Kept free of server dependencies for execution inside a verification sandbox
(``sandbox/sandbox_runner.py``).
"""

import json
import multiprocessing
import sys

from sandbox_limits import address_space_bytes, limit_address_space

from lcb_integration.testing_util import run_test


# Process-wide, so it must be set wherever grading runs, not only in the server process.
sys.set_int_max_str_digits(50000)


def _temp_run(in_outs, generation, debug, result, timeout, memory_limit_mb):
    memory = limit_address_space(memory_limit_mb) if memory_limit_mb is not None else None
    res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    if memory is not None:
        metadata = {**(metadata or {}), "memory": memory}
    # Publish together so a deadline kill cannot leave a result without its metadata.
    result.append((res, metadata))


def check_correctness(sample, generation, timeout, debug=True, *, memory_limit_mb=None, deadline_s=None):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`; *deadline_s* replaces it. *memory_limit_mb* adds an address-space cap above the
    grading process's baseline."""

    # Parse JSON once at the beginning to avoid multiple parsing
    try:
        in_outs = json.loads(sample["input_output"])
    except (ValueError, MemoryError):
        return [-1], None
    if memory_limit_mb is not None:
        # Fails here, in the runner, so a sandbox that cannot measure memory masks the sample.
        address_space_bytes()

    manager = multiprocessing.Manager()
    p: multiprocessing.Process | None = None
    try:
        result = manager.list()
        p = multiprocessing.Process(
            target=_temp_run,
            args=(in_outs, generation, debug, result, timeout, memory_limit_mb),
        )
        p.start()
        p.join(timeout=deadline_s if deadline_s is not None else (timeout + 1) * len(in_outs["inputs"]) + 5)
        timed_out = p.is_alive()
        if timed_out:
            p.kill()
            # Reap the worker after SIGKILL to release joinable resources.
            p.join(timeout=5)

        # Read the published pair before Manager shutdown; proxy access fails after shutdown.
        if result:
            return result[0]

        if timed_out:
            return [-3 for _ in in_outs["inputs"]], {
                "error_code": -3,
                "error_message": "Time Limit Exceeded",
                "stage": "grading",
            }
        # A worker that exits before the deadline without publishing a result is not a timeout.
        return [-1 for _ in in_outs["inputs"]], {
            "error_code": -1,
            "error_message": "Worker exited without a result",
            "stage": "worker",
            "exit_code": p.exitcode,
        }
    finally:
        if p is not None and p.is_alive():
            # Defensive: reap the worker if an exception bypassed the join above.
            p.kill()
            p.join(timeout=5)
        # Always shut down the Manager so its helper process doesn't leak under stress.
        manager.shutdown()
