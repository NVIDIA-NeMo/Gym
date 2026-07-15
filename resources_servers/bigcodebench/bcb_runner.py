# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""stdin/stdout shim around ``bigcodebench.eval.untrusted_check``.

Runs in the isolated Python 3.10 venv built by ``setup_bcb_venv.py``.
The Gym resource server (Python 3.12) spawns this script per ``verify()``.

Wire format:
    stdin:  JSON {"code", "test_code", "entry_point",
                  "max_as_limit", "max_data_limit", "max_stack_limit",
                  "min_time_limit", "gt_time_limit"}
    stdout: JSON {"status": "pass"|"fail"|"timeout"|None, "details": {...}}

Errors are funneled into ``status="error"`` so the parent never raises on
subprocess parsing.
"""

import json
import sys
import traceback


def main():
    try:
        from bigcodebench.eval import untrusted_check

        req = json.loads(sys.stdin.read())
        status, details = untrusted_check(
            code=req["code"],
            test_code=req["test_code"],
            entry_point=req["entry_point"],
            max_as_limit=req["max_as_limit"],
            max_data_limit=req["max_data_limit"],
            max_stack_limit=req["max_stack_limit"],
            min_time_limit=req["min_time_limit"],
            gt_time_limit=req["gt_time_limit"],
        )
        # untrusted_check may return status=None on TIMEOUT path before the
        # explicit ``stat = TIMEOUT`` branch; coerce to a JSON-stable value.
        json.dump({"status": status, "details": dict(details)}, sys.stdout)
    except Exception:
        json.dump(
            {"status": "error", "details": {"traceback": traceback.format_exc()}},
            sys.stdout,
        )


if __name__ == "__main__":
    main()
