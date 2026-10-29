# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-validator execution tests; run only in a disposable, non-root Linux sandbox."""

import os
from pathlib import Path

import pytest


if os.environ.get("NG_CRITPT_RUNNER_REMOTE") != "1":
    pytest.skip("remote-only: needs a disposable sandbox", allow_module_level=True)

from resources_servers.critpt_custom_grader import execution, runner
from resources_servers.critpt_custom_grader.codec import LegacyText, SymbolicText, encode_value
from resources_servers.critpt_custom_grader.tests.test_runner_remote import compare_job, invoke, lay_out


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/stat").exists(), reason="needs POSIX and /proc"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="the runner refuses root"),
]


PROTOCOL_VALIDATOR = """
def validate(*, observed, expected, args, kwargs, config):
    if type(observed) is not list or not observed:
        return False
    if any(type(step) is not list or len(step) != 2 for step in observed):
        return False
    if any(type(duration) is not int or duration <= 0 or type(control) is not int or control not in (-1, 1)
           for duration, control in observed):
        return False
    distance = sum(duration * control for duration, control in observed)
    duration = sum(duration for duration, control in observed)
    return distance == args[0] and duration == expected
"""
BASIS_VALIDATOR = """
def validate(*, observed, expected, args, kwargs, config):
    if type(observed) is not list or len(observed) != 2:
        return False
    if any(type(row) is not list or len(row) != 3 for row in observed):
        return False
    if any(type(value) is not int for row in observed for value in row):
        return False
    first, second = observed
    return first[2] == second[2] == 0 and first[0] * second[1] - first[1] * second[0] != 0
"""


def case(observed, expected, *, args=(), config=None):
    return {
        "observed": {"kind": "value", "value": encode_value(observed)},
        "expected": {"kind": "value", "value": encode_value(expected)},
        "policy": {},
        "inputs": {"args": [encode_value(value) for value in args], "kwargs": {}},
        "validator_config": {} if config is None else config,
    }


def compare(tmp_path, source, cases, *, role="candidate", **limits):
    files = execution.COMPARATOR_FILES
    lay_out(tmp_path, files=files)
    source_path = tmp_path / "validator_source.py"
    source_path.write_text(source)
    job = compare_job(cases, role=role, per_case_s=2, **limits)
    job["semantic_validator"] = {"source_path": str(source_path), "entrypoint": "validate"}
    return invoke(tmp_path, "compare", job)


def statuses(result):
    assert result.returncode == runner.EXIT_COMPLETED
    assert result.result["status"] == "completed"
    return [item["status"] for item in result.result["results"]]


def test_split_protocol_is_accepted_but_wrong_target_and_longer_protocol_are_rejected(tmp_path):
    result = compare(
        tmp_path,
        PROTOCOL_VALIDATOR,
        [
            case([[1, 1], [1, 1]], 2, args=(2,)),
            case([[1, 1]], 2, args=(2,)),
            case([[3, 1], [1, -1]], 2, args=(2,)),
            case([[2, True]], 2, args=(2,)),
        ],
    )
    assert statuses(result) == ["equal", "mismatch", "mismatch", "mismatch"]


def test_invertible_basis_transform_is_accepted_but_dependent_or_wrong_subspace_is_rejected(tmp_path):
    canonical = [[1, 0, 0], [0, 1, 0]]
    result = compare(
        tmp_path,
        BASIS_VALIDATOR,
        [
            case([[1, 1, 0], [1, -1, 0]], canonical),
            case([[1, 1, 0], [2, 2, 0]], canonical),
            case([[1, 0, 0], [0, 1, 1]], canonical),
        ],
    )
    assert statuses(result) == ["equal", "mismatch", "mismatch"]


def test_same_semantic_predicate_checks_reference_acceptance(tmp_path):
    result = compare(tmp_path, PROTOCOL_VALIDATOR, [case([[1, 1]], 2, args=(2,))], role="reference")
    assert statuses(result) == ["invalid_reference"]


@pytest.mark.parametrize(
    "source",
    [
        "def validate(**kwargs):\n    raise RuntimeError('never expose this')\n",
        "raise RuntimeError('initialization failed')\n",
        "def validate(**kwargs):\n    return 1\n",
        "def validate(**kwargs):\n    return {'accepted': True}\n",
        "def validate(**kwargs):\n    return None\n",
        "def validate(**kwargs):\n    raise MemoryError()\n",
        "def validate(**kwargs):\n    while True: pass\n",
        "while True: pass\n",
    ],
)
def test_validator_errors_invalid_returns_and_initialization_or_call_timeouts_are_uncertain(tmp_path, source):
    result = compare(tmp_path, source, [case(1, 1)])
    assert statuses(result) == ["uncertain"]
    assert "never expose" not in str(result.result)


def test_candidate_text_is_inert_data_and_cannot_select_or_replace_validator(tmp_path):
    marker = tmp_path / "candidate_text_executed"
    candidate = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    source = "def validate(*, observed, expected, args, kwargs, config):\n    return observed == expected\n"
    result = compare(tmp_path, source, [case(candidate, candidate)])
    assert statuses(result) == ["equal"]
    assert not marker.exists()


def test_validator_namespace_is_fresh_for_every_case(tmp_path):
    source = "count = 0\ndef validate(**kwargs):\n    global count\n    count += 1\n    return count == 1\n"
    result = compare(tmp_path, source, [case(1, 1), case(1, 1)])
    assert statuses(result) == ["equal", "equal"]


def test_semantic_validator_receives_inert_symbolic_and_legacy_carriers(tmp_path):
    source = """
from critpt_grader_rt.codec import LegacyText, SymbolicText

def validate(*, observed, expected, args, kwargs, config):
    return (type(observed) is SymbolicText and observed.text == "x + 1"
            and type(expected) is SymbolicText and expected.text == "1 + x"
            and type(args[0]) is LegacyText and args[0].text == "lambda: forbidden()")
"""
    result = compare(
        tmp_path,
        source,
        [case(SymbolicText("x + 1"), SymbolicText("1 + x"), args=(LegacyText("lambda: forbidden()"),))],
    )
    assert statuses(result) == ["equal"]


def test_validator_process_exit_does_not_forge_a_completed_false_verdict(tmp_path):
    source = "import os\ndef validate(**kwargs):\n    os._exit(17)\n"
    result = compare(tmp_path, source, [case(1, 1)])
    assert result.returncode == runner.EXIT_WORKER_ABORTED
    assert result.result is None
