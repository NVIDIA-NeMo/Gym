# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assertion grading must not inherit the sandbox interpreter's optimization level."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_CODE_GEN = Path(__file__).resolve().parents[1]
_DRIVER = """
import json
import sys
from assertion_harness import check

# This test isolates compilation and execution; Linux memory enforcement has separate provider tests.
check.limit_address_space = lambda _: {}
request = json.loads(sys.argv[1])
results, metadata = [], []
check._run_assertions(request['generation'], request['tests'], 5, 256, results, metadata)
print(json.dumps({'optimization': sys.flags.optimize, 'result': results[0], 'metadata': metadata[0]}))
"""


@pytest.mark.parametrize(
    "flags, optimization_env, level",
    [([], None, 0), (["-O"], None, 1), (["-OO"], None, 2), ([], "1", 1), ([], "2", 2)],
    ids=["normal", "O", "OO", "env-1", "env-2"],
)
@pytest.mark.parametrize(
    "generation, tests, expected",
    [
        ("answer = 0", ["assert answer == 42"], [-2]),
        ("import sys\nsys._getframe(1).f_locals['tests'][:] = ['pass']", ["assert False"], [-2]),
        ("import sys\nsys._getframe(1).f_locals['tests'].clear()", ["assert False"], [-2]),
        ("def broken(:", ["assert True"], [-1]),
        (
            "seen = []\ndef record():\n    seen.append(42)\n    return True",
            ["assert record()", "if seen != [42]:\n    raise RuntimeError('assertion did not execute')"],
            [True, True],
        ),
    ],
    ids=["failing-assertion", "replace-source", "clear-source", "candidate-syntax", "assertion-side-effect"],
)
def test_assertions_survive_optimization(
    flags: list[str],
    optimization_env: str | None,
    level: int,
    generation: str,
    tests: list[str],
    expected: list[bool | int],
) -> None:
    env = {**os.environ, "PYTHONPATH": str(_CODE_GEN)}
    env.pop("PYTHONOPTIMIZE", None)
    if optimization_env is not None:
        env["PYTHONOPTIMIZE"] = optimization_env
    run = subprocess.run(
        [sys.executable, *flags, "-c", _DRIVER, json.dumps({"generation": generation, "tests": tests})],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["optimization"] == level
    assert payload["result"] == expected, payload
    if expected == [-2]:
        assert payload["metadata"]["error_code"] == -2
