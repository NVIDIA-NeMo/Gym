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

"""Entry point that runs inside the verification sandbox.

Usage: ``python sandbox_runner.py REQUEST_JSON RESULT_JSON``

Grades one candidate with the same ``check_correctness`` the Ray path uses, so the time,
memory and process limits come from the harness itself rather than from the provider. A
candidate failure is a normal result; the runner exits nonzero only when grading itself
could not run, which the server treats as an infrastructure failure.
"""

import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
from lcb_integration.correctness import check_correctness


def _normalize_result_entry(entry: object) -> bool | int:
    # run_test can append numpy scalars or one-element arrays (call-based `prediction == gt_out`).
    if isinstance(entry, np.ndarray):
        entry = entry.item(0)
    if isinstance(entry, np.generic):
        entry = entry.item()
    # Anything else cannot equal True for reward purposes, so it scores as a failed test.
    return entry if isinstance(entry, (bool, int)) else False


def _jsonable_metadata(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else repr(value)


def _disable_core_dumps() -> None:
    # Hosts that pipe core dumps to a handler such as systemd-coredump ignore a zero RLIMIT_CORE, and a dump can
    # take seconds: long enough to turn a crashing candidate into a timeout. The kernel skips piped dumps when the
    # limit is exactly one byte. Candidates inherit it.
    if resource.getrlimit(resource.RLIMIT_CORE)[1] != 0:
        resource.setrlimit(resource.RLIMIT_CORE, (1, 1))


def main(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # The request holds every expected output, and candidates run in this directory with a real open().
    request_path.unlink()
    result, metadata = check_correctness(
        request["sample"], request["generation"], timeout=request["timeout"], debug=request["debug"]
    )
    payload = {"result": [_normalize_result_entry(entry) for entry in result], "metadata": metadata}
    # Publish atomically so the server never reads a partially written result.
    pending = result_path.with_name(result_path.name + ".pending")
    pending.write_text(json.dumps(payload, default=_jsonable_metadata), encoding="utf-8")
    os.replace(pending, result_path)


if __name__ == "__main__":
    _disable_core_dumps()
    main(Path(sys.argv[1]), Path(sys.argv[2]))
