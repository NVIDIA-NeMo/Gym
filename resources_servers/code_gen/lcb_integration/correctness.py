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

Kept free of Ray and other server dependencies so the identical code runs on a Ray worker
(``compute_code_generation_metrics.check_correctness_remote``) and inside a verification
sandbox (``sandbox/sandbox_runner.py``), which is what keeps the two execution paths at parity.
"""

import json
import multiprocessing
import sys

from lcb_integration.testing_util import run_test


# Process-wide, so it must be set wherever grading runs, not only in the server process.
sys.set_int_max_str_digits(50000)


def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    # Parse JSON once at the beginning to avoid multiple parsing
    try:
        in_outs = json.loads(sample["input_output"])
    except (ValueError, MemoryError):
        return [-1], None

    manager = multiprocessing.Manager()
    p: multiprocessing.Process | None = None
    try:
        result = manager.list()
        metadata_list = manager.list()
        p = multiprocessing.Process(
            target=_temp_run,
            args=(in_outs, generation, debug, result, metadata_list, timeout),
        )
        p.start()
        p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
        if p.is_alive():
            p.kill()
            # Reap the worker after SIGKILL to release joinable resources.
            p.join(timeout=5)

        # Drain ListProxy values into plain lists before Manager shutdown, since access
        # raises once the Manager helper process exits.
        if result:
            result_local: list = list(result)
            metadata_local: list = list(metadata_list)
            return result_local[0], metadata_local[0]

        if debug:
            print("global timeout")
        # consider that all tests failed
        return [-1 for _ in range(len(in_outs["inputs"]))], None
    finally:
        if p is not None and p.is_alive():
            # Defensive: reap the worker if an exception bypassed the join above.
            p.kill()
            p.join(timeout=5)
        # Always shut down the Manager so its helper process doesn't leak under stress.
        manager.shutdown()
