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
"""Runs an isolated exporter's worker inside that exporter's own virtual environment.

Invoked as `python _isolated_worker.py RESULT_FPATH WORKER_FPATH` with the JSON payload on stdin. It imports
only the standard library, because the environment it runs in has none of Gym's dependencies.

The worker file defines `run(payload: dict) -> dict`. The returned dict may hold `state` (`succeeded` by default,
`skipped` or `failed`), `error`, `exit_code` (for a failure; 1 by default) and `detail`. The outcome is written as JSON to RESULT_FPATH rather than stdout, so
whatever the exporter's SDK prints cannot corrupt it.
"""

import os
import sys


# Running this file puts its directory first on sys.path, where the modules that sit beside it (`evalrdl.py`,
# `mlflow.py`, `wandb.py`) would shadow the SDKs of the same name installed in this environment.
sys.path[:] = [path for path in sys.path if path != os.path.dirname(os.path.abspath(__file__))]

import importlib.util  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402


def _load_worker(worker_fpath: str):
    spec = importlib.util.spec_from_file_location("gym_exporter_worker", worker_fpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    result_fpath, worker_fpath = argv[1], argv[2]
    try:
        outcome = _load_worker(worker_fpath).run(json.load(sys.stdin))
        result = {
            "state": outcome.get("state", "succeeded"),
            "error": outcome.get("error"),
            "exit_code": outcome.get("exit_code", 1),
            "detail": outcome.get("detail", {}),
        }
    except Exception as exc:
        traceback.print_exc()
        result = {"state": "failed", "error": f"{type(exc).__name__}: {exc}", "exit_code": 1, "detail": {}}
    temporary = f"{result_fpath}.tmp"
    with open(temporary, "w") as handle:
        json.dump(result, handle)
    os.replace(temporary, result_fpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
