# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote-only runner tests: the runner runs as a subprocess with synthetic, deliberately hostile sources.

Skipped unless NG_CRITPT_RUNNER_REMOTE=1, which must only be set inside a disposable non-root sandbox with
/proc mounted. Source, codecs, parsers and comparisons run only inside opted-in tests, never at collection.
These mechanism tests do not qualify production runtime integrity or authenticate result files. Fixture sizes
are not measured RSS limits or proof of supervisor resource isolation.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


REMOTE = os.environ.get("NG_CRITPT_RUNNER_REMOTE") == "1"
if not REMOTE:
    pytest.skip("remote-only: set NG_CRITPT_RUNNER_REMOTE=1 inside a disposable sandbox", allow_module_level=True)

from decimal import Decimal
from fractions import Fraction

from pydantic import ValidationError

from resources_servers.critpt_custom_grader import execution, runner
from resources_servers.critpt_custom_grader.codec import (
    LegacyText,
    RepresentationError,
    SymbolicText,
    decode_value,
    encode_value,
)
from resources_servers.critpt_custom_grader.task_data import (
    MAX_CASES,
    MAX_DEPTH,
    MAX_ELEMENTS,
    MAX_INTEGER_DIGITS,
    MAX_POLICY_BYTES,
    MAX_STATEMENT_BYTES,
    MAX_STATEMENT_CHARS,
    MAX_TASK_BYTES,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    TaskData,
    bounded_json,
)
from resources_servers.critpt_custom_grader.task_data import TestCase as CaseData


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(not REMOTE, reason="remote-only: set NG_CRITPT_RUNNER_REMOTE=1 inside a disposable sandbox"),
    pytest.mark.skipif(os.name != "posix" or not Path("/proc/self/stat").exists(), reason="needs POSIX and /proc"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="the runner refuses to run as root"),
]

PACKAGE_DIR = Path(runner.__file__).resolve().parent
RUNTIME_FILES = ("task_data.py", "codec.py", "symbolic.py", "runner.py")
COMPARATOR_FILES = RUNTIME_FILES + ("comparator.py",)
NONCE = "abc123"
SLACK_S = runner._REAP_BUDGET_S + runner._SWEEP_BUDGET_S + 5.0  # cleanup budgets plus startup/write slack
LIMITS = {"suite_deadline_s": 60, "memory_mib": 2048, "cpu_time_s": 60, "processes": 32}

PLAIN = "def f(x):\n    if x < 0:\n        raise ValueError('negative')\n    return x + 1\n"

# Task-side helpers pasted into synthetic sources: locate the worker channel and forge files.
CHANNEL_HELPER = """
import os, stat

def _channel():
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        try:
            if stat.S_ISFIFO(os.fstat(fd).st_mode):
                return fd
        except OSError:
            pass
    raise RuntimeError("no channel")
"""


def forge_helper():
    value = repr(encode_value(999))
    return (
        "import json, os\n"
        "def _forge(path, status='completed', outcomes=None):\n"
        f"    payload = {{'version': 1, 'mode': 'run', 'nonce': {NONCE!r}, 'status': status,\n"
        "               'code': None, 'timed_out_case': None, 'forged': True,\n"
        f"               'outcomes': outcomes if outcomes is not None else [{{'kind': 'value', 'value': {value}}}]}}\n"
        "    with open(path, 'w') as handle:\n"
        "        json.dump(payload, handle)\n"
    )


@dataclass
class Run:
    returncode: int
    result: dict | None
    elapsed: float
    result_path: Path


def wire(value):
    return encode_value(value)


def run_job(source_path, cases, entrypoint="f", **limits):
    return {
        "version": 1,
        "mode": "run",
        "nonce": NONCE,
        "entrypoint": entrypoint,
        "source_path": str(source_path),
        "cases": [{"args": [wire(arg) for arg in args], "kwargs": {}} for args in cases],
        "input_conversions": [],
        "limits": {**LIMITS, **limits},
    }


def compare_job(cases, role="candidate", **limits):
    return {
        "version": 1,
        "mode": "compare",
        "nonce": NONCE,
        "observed_role": role,
        "cases": cases,
        "limits": {**LIMITS, **limits},
    }


def lay_out(tmp_path, files=RUNTIME_FILES):
    package = tmp_path / runner.RUNTIME_PACKAGE
    package.mkdir()
    for name in files:
        shutil.copy(PACKAGE_DIR / name, package / name)
    return package / "runner.py"


def invoke(tmp_path, mode, job, *, timeout_s=120.0, setup=None):
    job_path, result_path = tmp_path / "job.json", tmp_path / "result.json"
    if job is not None:  # None keeps a byte-boundary fixture already written to job.json.
        job_path.write_bytes(execution._dump(job))
    runner_path = tmp_path / runner.RUNTIME_PACKAGE / "runner.py"
    if setup is not None:
        bootstrap = tmp_path / "bootstrap.py"
        bootstrap.write_text(
            f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n"
            f"from {runner.RUNTIME_PACKAGE} import runner\n{setup}\n"
            "import os\nos._exit(runner._entrypoint(sys.argv[1:]))\n"
        )
        runner_path = bootstrap  # operator-controlled fault injection, only in the remote subprocess
    command = [sys.executable, "-I", "-B", str(runner_path), mode, str(job_path), str(result_path)]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
        cwd=tmp_path,
    )
    elapsed = time.monotonic() - started
    if result_path.is_file():
        with result_path.open("rb") as handle:
            data = handle.read(runner.RESULT_BYTES_LIMIT + 1)
        assert len(data) <= runner.RESULT_BYTES_LIMIT
        result = json.loads(data)
    else:
        result = None
    return Run(completed.returncode, result, elapsed, result_path)


def run_source(tmp_path, source, cases, **limits):
    lay_out(tmp_path)
    source_path = tmp_path / "source.py"
    source_path.write_text(source)
    return invoke(tmp_path, "run", run_job(source_path, cases, **limits))


def outcome_value(outcome):
    assert outcome["kind"] == "value"
    return outcome["value"]


def _subreaper() -> bool:
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        return prctl(36, 1, 0, 0, 0) == 0
    except (ImportError, OSError, AttributeError):
        return False


def _reap_descendants(grace_s: float) -> list[int]:
    """Observe leaks after a grace period, then kill/reap adopted children within a separate cleanup budget."""
    deadline = time.monotonic() + grace_s
    while True:
        runner._reap_orphans(deadline)
        remaining = runner._child_pids()
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    leaked = list(remaining)
    deadline = time.monotonic() + SLACK_S
    while remaining and time.monotonic() < deadline:
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        runner._reap_orphans(deadline)
        remaining = runner._child_pids()
        time.sleep(0.02)
    assert not remaining, f"test sandbox cleanup failed: {remaining}"
    return leaked


@pytest.fixture(autouse=True)
def no_leaked_processes():
    """The test process becomes a subreaper, so anything the runner leaves behind re-parents here and is caught."""
    assert _subreaper(), "remote cleanup requires Linux child-subreaper support"
    yield
    leaked = _reap_descendants(3.0)
    assert not leaked, f"processes outlived the runner: {leaked}"


# --- ordinary synthetic cases ------------------------------------------------------------------- #


def test_completed_relays_values_and_exceptions_in_case_order(tmp_path):
    run = run_source(tmp_path, PLAIN, [(1,), (-1,), (41,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert set(run.result) == {"version", "mode", "nonce", "status", "code", "timed_out_case", "outcomes"}
    envelope = (run.result["status"], run.result["code"], run.result["timed_out_case"], run.result["nonce"])
    assert envelope == ("completed", None, None, NONCE)
    outcomes = run.result["outcomes"]
    assert outcome_value(outcomes[0]) == wire(2) and outcome_value(outcomes[2]) == wire(42)
    assert outcomes[1] == {"kind": "exception", "type": "ValueError"}


def test_unrepresentable_return_value_is_an_encoding_error(tmp_path):
    # A set is a representable carrier, so return a value with no carrier: bytes are invalid_type.
    run = run_source(tmp_path, "def f(x):\n    return b'raw bytes'\n", [(1,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "encoding_error", "code": "invalid_type"}]


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("def f(x:\n    return x\n", "syntax_error"),
        ("def g(x):\n    return x\n", "entrypoint_missing"),
        ("def f(x):\n    return x\nf = 1\n", "entrypoint_conflict"),
        ("raise RuntimeError('at import')\n\ndef f(x):\n    return x\n", "import_error"),
        ("def deco(fn):\n    return 3\n\n@deco\ndef f(x):\n    return x\n", "entrypoint_not_callable"),
    ],
)
def test_source_errors_are_reported_with_their_code(tmp_path, source, code):
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == runner.EXIT_SOURCE_ERROR
    assert (run.result["status"], run.result["code"], run.result["outcomes"]) == ("source_error", code, [])


def test_invalid_job_is_refused_before_any_execution(tmp_path):
    lay_out(tmp_path)
    sentinel = tmp_path / "executed"
    (tmp_path / "source.py").write_text(f"open({str(sentinel)!r}, 'w').close()\ndef f(x):\n    return x\n")
    job = run_job(tmp_path / "source.py", [(1,)])
    del job["entrypoint"]
    run = invoke(tmp_path, "run", job)
    assert run.returncode == runner.EXIT_RUNNER_ERROR
    assert (run.result["status"], run.result["code"]) == ("runner_error", "invalid_job")
    assert not sentinel.exists()


def test_compare_mode_relays_verdicts_from_the_protected_comparator(tmp_path):
    lay_out(tmp_path, COMPARATOR_FILES)
    cases = [
        {"observed": {"kind": "value", "value": wire(2)}, "expected": {"kind": "value", "value": wire(2)}},
        {
            "observed": {"kind": "value", "value": wire(3)},
            "expected": {"kind": "value", "value": wire(2)},
            "policy": {},
        },
        {"observed": {"kind": "exception", "type": "ValueError"}, "expected": {"kind": "exception"}},
    ]
    run = invoke(tmp_path, "compare", compare_job(cases))
    assert run.returncode == runner.EXIT_COMPLETED
    assert set(run.result) == {"version", "mode", "nonce", "status", "code", "timed_out_case", "results"}
    assert [verdict["status"] for verdict in run.result["results"]] == ["equal", "mismatch", "equal"]
    assert all(
        set(verdict) == {"version", "status", "equal", "side", "code", "path"} for verdict in run.result["results"]
    )


def test_run_domain_does_not_need_the_comparator_module(tmp_path):
    run = run_source(tmp_path, PLAIN, [(1,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert not (tmp_path / runner.RUNTIME_PACKAGE / "comparator.py").exists()


# --- input preparation (synthetic, remote-only) -------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("-17", -17),
        ("+17", 17),
        ("9007199254740993", 9007199254740993),
        (" 1.25\n", 1.25),
        ("0.1", 0.1),
        ("-0.0", -0.0),
        ("+0.0", 0.0),
        ("5e-324", 5e-324),
        ("1.7976931348623157e308", 1.7976931348623157e308),
        ("-0e-999", -0.0),
        ("True", True),
        ("False", False),
        ("None", None),
        ("[True, False]", [True, False]),
        ("(2, 3)", (2, 3)),
        ("[]", []),
        ("()", ()),
        ("{}", {}),
        ("{'z': (2, 3), 'a': [True, None]}", {"z": (2, 3), "a": [True, None]}),
        ("'green'", "green"),
        ('"1/7"', "1/7"),
        ("'True'", "True"),
        ("['1e999', '[1]', 'sqrt(4)']", ["1e999", "[1]", "sqrt(4)"]),
        ("", ""),
        ("  pi\n", "  pi\n"),
        ("sqrt(4)", "sqrt(4)"),
        ("x+1", "x+1"),
    ],
)
def test_root_legacy_input_literal_semantics(text, expected):
    tagged = wire(LegacyText(text))
    assert decode_value(tagged) == LegacyText(text)  # The codec remains inert, including for malformed notation.
    [(args, kwargs)] = runner._decode_cases([{"args": [tagged], "kwargs": {"x": tagged}}])
    for value in (args[0], kwargs["x"]):
        assert type(value) is type(expected)
        assert wire(value) == wire(expected)  # Includes exact binary64 bits, tuple/list tags and nested bool types.
    if type(expected) is dict:
        assert list(args[0]) == list(expected)


def test_input_literal_exact_integer_and_structure_boundaries():
    for number in (-(10**400 + 73), 10 ** (MAX_INTEGER_DIGITS - 1)):
        value = runner._decode_input(wire(LegacyText(str(number))))
        assert type(value) is int and value == number
    # A legacy list literal is bounded by MAX_TEXT_BYTES, not MAX_ELEMENTS.
    count = (MAX_TEXT_BYTES - 1) // 2
    text = "[" + ",".join(["0"] * count) + "]"
    assert len(text) <= MAX_TEXT_BYTES and count < MAX_ELEMENTS
    values = runner._decode_input(wire(LegacyText(text)))
    assert len(values) == count and all(type(value) is int and value == 0 for value in values)
    text = "[" * MAX_DEPTH + "0" + "]" * MAX_DEPTH
    value = runner._decode_input(wire(LegacyText(text)))
    for _ in range(MAX_DEPTH):
        assert type(value) is list and len(value) == 1
        value = value[0]
    assert type(value) is int and value == 0
    for text in ("z" * MAX_TEXT_BYTES, "\U0001f642" * (MAX_TEXT_BYTES // 4), "'" + "z" * (MAX_TEXT_BYTES - 2) + "'"):
        value = runner._decode_input(wire(LegacyText(text)))
        assert value == (text[1:-1] if text.startswith("'") else text)


@pytest.mark.parametrize(
    "text",
    [
        "[",
        "'unfinished",
        "[x for x in (1, 2)]",
        "(lambda: 1)()",
        "(sqrt(4))",
        "[sqrt(4)]",
        "{'x': 1, 'x': 2}",
        "{1: 2}",
        "{**{}}",
        "{1, 2}",
        "[0] * 999",
        "[x]",
        "-True",
        "1+2j",
        "1/7",
        "1_000",
        "0xff",
        "1e99999",
        "1e309",
        "-1e309",
        "1e-999",
        "-1e-999",
        "'\\ud800'",
    ],
)
def test_malformed_root_legacy_inputs_are_invalid_jobs(text):
    tagged = wire(LegacyText(text))
    assert decode_value(tagged) == LegacyText(text)
    with pytest.raises(runner.RunnerError) as caught:
        runner._decode_cases([{"args": [], "kwargs": {"x": tagged}}])
    assert caught.value.code == "invalid_job"


def test_input_limits_cover_text_ast_and_parsed_representation():
    texts = [
        "z" * (MAX_TEXT_BYTES + 1),
        "\U0001f642" * (MAX_TEXT_BYTES // 4 + 1),
        "9" * (MAX_INTEGER_DIGITS + 1),
        "[" * (MAX_DEPTH + 1) + "0" + "]" * (MAX_DEPTH + 1),
        "[" + ",".join(["0"] * MAX_ELEMENTS) + "]",
    ]
    for text in texts:
        # Bypass the encoder so the remote decoder, not the fixture builder, rejects oversized text.
        tagged = {"format": WIRE_FORMAT, "value": ["legacy", text]}
        with pytest.raises(runner.RunnerError) as caught:
            runner._decode_cases([{"args": [tagged], "kwargs": {}}])
        assert caught.value.code == "invalid_job"


@pytest.mark.parametrize("bound", ["text", "integer", "depth", "elements", "wire_bytes"])
def test_root_legacy_parsed_results_use_codec_bounds(monkeypatch, bound):
    values = {
        "text": "z" * (MAX_TEXT_BYTES + 1),
        "integer": 10**MAX_INTEGER_DIGITS,
        "elements": [None] * MAX_ELEMENTS,
        "wire_bytes": ["z" * MAX_TEXT_BYTES] * (MAX_WIRE_BYTES // MAX_TEXT_BYTES + 1),
    }
    deep = None
    for _ in range(MAX_DEPTH + 1):
        deep = [deep]
    values["depth"] = deep
    monkeypatch.setattr(runner, "_input_literal", lambda text: values[bound])

    def unexpected_materialization(value):
        pytest.fail("oversized parsed representation reached materialization")

    monkeypatch.setattr(runner, "materialize_input", unexpected_materialization)
    with pytest.raises(runner.RunnerError) as caught:
        runner._decode_input(wire(LegacyText("0")))
    assert caught.value.code == "invalid_job"


@pytest.mark.parametrize(
    "node",
    [
        ["unknown", "bounded"],
        ["int", "1", "extra"],
        ["fraction", "1"],
        ["legacy"],
        ["legacy", 1],
        ["int", "not-an-integer"],
        ["bool", 1],
        ["fraction", "1", "0"],
        ["float", "1.0"],
        ["map", [["x", ["null"]], ["x", ["null"]]]],
        ["list", {}],
        ["list", [["legacy", "1"]]],
        ["tuple", [["legacy", "1"]]],
        ["map", [["x", ["legacy", "1"]]]],
    ],
)
def test_invalid_argument_wire_and_nested_legacy_are_rejected(node):
    tagged = {"format": WIRE_FORMAT, "value": node}
    for case in ({"args": [tagged], "kwargs": {}}, {"args": [], "kwargs": {"x": tagged}}):
        with pytest.raises(runner.RunnerError) as caught:
            runner._decode_cases([case])
        assert caught.value.code == "invalid_job"


def test_positional_nested_and_canonical_strings_remain_opaque():
    text = "[x for x in (1, 2)]"
    cases = [
        CaseData.model_validate({"input": [text, {"x": "1/7"}], "output": "unchanged"}),
        CaseData.model_validate({"inputs": {"x": [text, ("True", "[1]")]}, "expected_output": "unchanged"}),
        CaseData.model_validate({"kwargs": {"x": wire(text)}, "expected": {"kind": "exception"}}),
    ]
    inputs = [
        {
            "args": [item.model_dump() for item in case.args],
            "kwargs": {key: item.model_dump() for key, item in case.kwargs.items()},
        }
        for case in cases
    ]
    assert runner._decode_cases(inputs) == [
        ([text, {"x": "1/7"}], {}),
        ([], {"x": [text, ("True", "[1]")]}),
        ([], {"x": text}),
    ]


def test_canonical_fraction_decimal_and_symbolic_inputs_are_unchanged():
    import sympy

    nodes = [["fraction", "7", "19"], ["decimal", "-0.000"], ["decimal", "1e-800"], ["symbolic", "x+1"]]
    args, kwargs = runner._decode_cases(
        [{"args": [{"format": WIRE_FORMAT, "value": node} for node in nodes], "kwargs": {}}]
    )[0]
    assert kwargs == {}
    assert type(args[0]) is Fraction and args[0] == Fraction(7, 19)
    assert type(args[1]) is Decimal and args[1].as_tuple() == Decimal("-0.000").as_tuple()
    assert type(args[2]) is Decimal and args[2] == Decimal("1e-800")
    assert args[3] == sympy.Symbol("x") + 1
    for case in ({"input": [Fraction(7, 19)], "output": 0}, {"inputs": {"x": Fraction(7, 19)}, "expected_output": 0}):
        with pytest.raises(ValidationError, match="plain values"):
            CaseData.model_validate(case)


def test_bare_symbolic_constant_input_decodes_as_the_constant():
    # A bare reserved name as a symbolic input materializes as the constant, not a free symbol.
    import sympy

    nodes = [["symbolic", "pi"], ["symbolic", "(pi)"], ["symbolic", "x"]]
    args, kwargs = runner._decode_cases(
        [{"args": [{"format": WIRE_FORMAT, "value": node} for node in nodes], "kwargs": {}}]
    )[0]
    assert kwargs == {}
    assert args[0] == sympy.pi and not args[0].free_symbols
    assert args[1] == sympy.pi  # bare and parenthesized agree
    assert args[2] == sympy.Symbol("x")  # an ordinary name is still a free symbol


def test_bare_symbolic_constant_input_reaches_reference_execution_as_the_constant(tmp_path):
    # A task that floats a bare "pi" input succeeds because the input is the numeric constant.
    import sympy

    source = "def f(x):\n    return float(x)\n"
    run = run_source(tmp_path, source, [(SymbolicText("pi"),), (SymbolicText("(pi)"),)])
    assert run.returncode == runner.EXIT_COMPLETED
    outcomes = run.result["outcomes"]
    assert outcomes[0]["kind"] == "value" and outcomes[1]["kind"] == "value"
    assert outcome_value(outcomes[0]) == outcome_value(outcomes[1]) == wire(float(sympy.pi))


def test_delivered_keyword_inputs_are_parsed_in_limited_worker(tmp_path):
    lay_out(tmp_path)
    source = "def f(amount, flags, label, pair):\n    return (amount, flags, label, pair)\n"
    (tmp_path / "source.py").write_text(source)
    expected = "synthetic_expected_input_privacy_marker [1, 2]"
    task = TaskData.model_validate(
        {
            "problem_id": "synthetic-input-preparation",
            "reference_source": source,
            "entrypoint": "f",
            "test_cases": [
                {
                    "inputs": {"amount": "1.25", "flags": "[True, False]", "label": "'green'", "pair": "(2, 3)"},
                    "expected_output": expected,
                }
            ],
        }
    )
    policy = execution.ExecutionPolicy(
        snapshot="synthetic-input",
        os_user="sandbox",
        owner_id="input",
        journal_dir=str(tmp_path / "journal"),
        workdir=str(tmp_path),
        suite_timeout_s=60,
        cpu_time_limit_s=60,
        max_processes=32,
    )
    job = execution.build_run_job(task, task.entrypoint, NONCE, policy)
    assert all(set(case) == {"args", "kwargs"} for case in job["cases"])
    assert expected.encode() not in execution._dump(job)
    assert task.test_cases[0].expected.value.value == ["legacy", expected]
    setup = (
        "import os, resource\nsupervisor_pid = os.getpid()\noriginal = runner._input_literal\n"
        "def checked_literal(value):\n"
        "    assert os.getpid() != supervisor_pid\n"
        "    assert resource.getrlimit(resource.RLIMIT_AS) == (2147483648, 2147483648)\n"
        "    assert resource.getrlimit(resource.RLIMIT_CPU) == (60, 90)\n"
        "    return original(value)\nrunner._input_literal = checked_literal\n"
    )
    run = invoke(tmp_path, "run", job, setup=setup)
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "value", "value": wire((1.25, [True, False], "green", (2, 3)))}]
    assert task.test_cases[0].expected.value.value == ["legacy", expected]


@pytest.mark.parametrize(
    "node",
    [["legacy", "["], ["legacy", "'\\ud800'"], ["unknown", "bounded"], ["fraction", "1"], ["list", [["legacy", "1"]]]],
)
def test_invalid_input_is_rejected_before_source_import(tmp_path, node):
    lay_out(tmp_path)
    sentinel = tmp_path / "imported"
    source_path = tmp_path / "source.py"
    source_path.write_text(f"open({str(sentinel)!r}, 'w').close()\ndef f(x):\n    return x\n")
    job = run_job(source_path, [()])
    job["cases"] = [{"args": [], "kwargs": {"x": {"format": WIRE_FORMAT, "value": node}}}]
    assert runner.validate_job(job, "run") is None  # Valid envelope and source, only input preparation rejects.
    run = invoke(tmp_path, "run", job)
    assert run.returncode == runner.EXIT_RUNNER_ERROR
    assert (run.result["status"], run.result["code"], run.result["timed_out_case"], run.result["outcomes"]) == (
        "runner_error",
        "invalid_job",
        None,
        [],
    )
    assert not sentinel.exists()


@pytest.mark.parametrize("error", ["MemoryError", "RecursionError", "OverflowError"])
def test_input_parser_resource_failures_remain_limits(tmp_path, error):
    lay_out(tmp_path)
    sentinel = tmp_path / "imported"
    source_path = tmp_path / "source.py"
    source_path.write_text(f"open({str(sentinel)!r}, 'w').close()\ndef f(x):\n    return x\n")
    job = run_job(source_path, [(LegacyText("1"),)])
    setup = (
        f"def failed_parse(*args, **kwargs):\n    raise {error}\n"
        "original = runner.check_entrypoint_binding\n"
        "def checked_binding(*args):\n"
        "    result = original(*args)\n"
        "    runner.ast.parse = failed_parse\n"
        "    return result\nrunner.check_entrypoint_binding = checked_binding\n"
    )
    run = invoke(tmp_path, "run", job, setup=setup)
    assert run.returncode == runner.EXIT_RUNNER_ERROR
    assert (run.result["status"], run.result["code"], run.result["outcomes"]) == ("runner_error", "limits", [])
    assert not sentinel.exists()


def test_input_preparation_is_inside_supervised_deadline(tmp_path):
    lay_out(tmp_path)
    sentinel = tmp_path / "imported"
    source_path = tmp_path / "source.py"
    source_path.write_text(f"open({str(sentinel)!r}, 'w').close()\ndef f(x):\n    return x\n")
    job = run_job(source_path, [(LegacyText("1"),)], suite_deadline_s=2)
    setup = "import time\ndef stalled_input(value):\n    time.sleep(60)\nrunner._input_literal = stalled_input\n"
    run = invoke(tmp_path, "run", job, setup=setup, timeout_s=2 + SLACK_S)
    assert run.returncode == runner.EXIT_TIMEOUT and run.elapsed < 2 + SLACK_S
    assert (run.result["status"], run.result["timed_out_case"], run.result["outcomes"]) == ("timeout", None, [])
    assert not sentinel.exists()


# --- serialized job budgets (synthetic, remote-only) --------------------------------------------- #


def budget_job(tmp_path, statement, cases, outcomes, **settings):
    task = TaskData.model_validate(
        {
            "problem_id": "synthetic-job-budget",
            "reference_source": "def f():\n    return 0\n",
            "entrypoint": "f",
            "problem": statement,
            "test_cases": cases,
            **settings,
        }
    )
    policy = execution.ExecutionPolicy(
        snapshot="synthetic-budget",
        os_user="sandbox",
        owner_id="budget",
        journal_dir=str(tmp_path / "journal"),
        suite_timeout_s=60,
        compare_timeout_s=60,
        cpu_time_limit_s=60,
        max_processes=32,
    )
    return task, execution.build_compare_job(task, "candidate", outcomes, NONCE, policy)


@pytest.mark.parametrize(
    ("character", "bytes_per_character"),
    [("\x00", 6), ('"', 2), ("\\", 2), ("\U0001f642", 4)],
    ids=["six-byte-control", "quote", "backslash", "utf8-scalar"],
)
def test_maximum_statement_serialization_keeps_every_character(tmp_path, character, bytes_per_character):
    # One 128-Kcharacter statement, each serialized policy is at most 1.25 MiB.
    assert MAX_STATEMENT_CHARS == 131_072
    assert MAX_STATEMENT_BYTES == 786_432
    assert MAX_POLICY_BYTES == 1_310_720
    statement = character * MAX_STATEMENT_CHARS
    outcome = {"kind": "value", "value": wire(0)}
    task, job = budget_job(tmp_path, statement, [{"expected": outcome}], [outcome])
    policy = job["cases"][0]["policy"]
    assert task.problem == policy["statement"] == statement
    serialized = execution._dump(policy)
    empty = execution._dump(dict(policy, statement=""))
    assert len(serialized) - len(empty) == bytes_per_character * MAX_STATEMENT_CHARS
    assert len(serialized) <= MAX_POLICY_BYTES
    bounded_json(policy, max_bytes=MAX_POLICY_BYTES)
    assert runner.validate_job(job, runner.COMPARE_MODE) is None


def test_maximum_statement_expands_across_all_64_cases_without_flattening_tolerances(tmp_path):
    # The builder shares one 128-KiB string. Serialized statements total exactly 48 MiB, not 64 task blobs.
    assert MAX_CASES == 64 and MAX_STATEMENT_CHARS == 131_072
    statement = "\x00" * MAX_STATEMENT_CHARS
    expected = {"kind": "value", "value": wire([1, 2])}
    cases = [
        {"expected": expected, "tolerances": [{"path": "/1", "rtol": "0", "atol": "0.001"}]} for _ in range(MAX_CASES)
    ]
    outcomes = [
        {"kind": "value", "value": wire([1.05, 2.0005 if index % 2 == 0 else 2.005])} for index in range(MAX_CASES)
    ]
    task, job = budget_job(
        tmp_path,
        statement,
        cases,
        outcomes,
        rtol="0",
        atol="0",
        tolerances=[{"path": "/0", "rtol": "0", "atol": "0.1"}, {"path": "/1", "rtol": "0", "atol": "0.01"}],
    )
    assert len(execution._dump(task.model_dump(mode="json"))) <= MAX_TASK_BYTES
    assert len(job["cases"]) == MAX_CASES
    for case in job["cases"]:
        assert case["policy"]["statement"] == statement
        assert case["policy"]["tolerances"] == [
            {"path": "/0", "quantity": None, "rtol": "0", "atol": "0.1"},
            {"path": "/1", "quantity": None, "rtol": "0", "atol": "0.001"},
        ]
    serialized = execution._dump(job)
    # Every case carries its own full statement, 64 copies, still inside JOB_BYTES_LIMIT.
    assert MAX_CASES * MAX_STATEMENT_BYTES < len(serialized) < 49 * 1024**2
    assert len(serialized) <= runner.JOB_BYTES_LIMIT
    assert runner.validate_job(job, runner.COMPARE_MODE) is None
    (tmp_path / "job.json").write_bytes(serialized)
    del serialized  # Do not retain a full serialized copy while the child reads and decodes it.
    lay_out(tmp_path, COMPARATOR_FILES)
    run = invoke(tmp_path, "compare", None)
    assert run.returncode == runner.EXIT_COMPLETED
    assert (run.result["status"], run.result["nonce"], run.result["code"]) == ("completed", NONCE, None)
    assert [item["status"] for item in run.result["results"]] == ["equal", "mismatch"] * (MAX_CASES // 2)
    assert [item["equal"] for item in run.result["results"]] == [True, False] * (MAX_CASES // 2)
    assert all(item["path"] == "/1" for item in run.result["results"][1::2])


def test_compare_job_envelopes_and_actual_utf8_reader_boundary(tmp_path):
    # Bounded text leaves fill one 512-KiB wire exactly. encode_value caps the wire at MAX_WIRE_BYTES.
    assert MAX_WIRE_BYTES == 524_288 and MAX_TEXT_BYTES == 16_384
    leaf = "a" * MAX_TEXT_BYTES
    count = 0
    while True:
        try:
            filled = len(execution._dump(wire([leaf] * (count + 1) + [""])))
        except RepresentationError:
            break
        if filled > MAX_WIRE_BYTES:
            break
        count += 1
    parts = [leaf] * count + [""]
    remaining = MAX_WIRE_BYTES - len(execution._dump(wire(parts)))
    assert 0 < remaining <= MAX_TEXT_BYTES
    parts[-1] = "b" * remaining
    value = wire(parts)
    assert len(execution._dump(value)) == MAX_WIRE_BYTES
    value_outcome = {"kind": "value", "value": value}
    provenance = "\x00" * MAX_TEXT_BYTES
    exception_expected = {"kind": "exception", "message": provenance}
    exception_observed = {"kind": "exception", "type": provenance, "message": provenance}
    _, job = budget_job(
        tmp_path,
        "\U0001f642" * MAX_STATEMENT_CHARS,
        [{"expected": value_outcome}, {"expected": exception_expected}],
        [value_outcome, exception_observed],
    )
    for case in job["cases"]:
        assert execution._check_outcome(case["observed"])
        assert len(execution._dump(case["expected"])) <= MAX_WIRE_BYTES + 512
        assert len(execution._dump(case["observed"])) <= MAX_WIRE_BYTES + 512
        assert len(execution._dump(case["policy"])) <= MAX_POLICY_BYTES
        assert len(execution._dump(case)) <= 2 * (MAX_WIRE_BYTES + 512) + MAX_POLICY_BYTES
    serialized = execution._dump(job)
    assert len(serialized.decode("utf-8")) < len(serialized) < 3 * 1024**2
    path = tmp_path / "job.json"
    path.write_bytes(serialized)
    assert runner._read_json(str(path), len(serialized)) == job
    with pytest.raises(runner.RunnerError) as caught:
        runner._read_json(str(path), len(serialized) - 1)
    assert caught.value.code == "invalid_job"


@pytest.mark.parametrize("extra_byte", [0, 1], ids=["at-cap", "one-byte-over-cap"])
def test_serialized_compare_job_cap_is_inclusive_and_strict(tmp_path, extra_byte):
    intended_cap = MAX_TASK_BYTES + MAX_CASES * (2 * (MAX_WIRE_BYTES + 512) + MAX_POLICY_BYTES) + 1_048_576
    assert runner.JOB_BYTES_LIMIT == intended_cap == 156_303_360
    assert runner.FILE_LIMIT_BYTES == max(intended_cap, 4 * runner.RESULT_BYTES_LIMIT)
    outcome = {"kind": "value", "value": wire(0)}
    _, job = budget_job(tmp_path, "synthetic boundary", [{"expected": outcome}] * MAX_CASES, [outcome] * MAX_CASES)
    serialized = execution._dump(job)
    size = intended_cap + extra_byte
    assert len(serialized) < size <= 160 * 1024**2
    path = tmp_path / "job.json"
    # Valid JSON whitespace reaches the real cap. Only 64 KiB of padding is held at once.
    # The file is at most 149.0625 MiB + 1 byte, and the runner reads at most cap + 1 bytes.
    chunk = b" " * 65_536
    full_chunks, tail = divmod(size - len(serialized), len(chunk))
    with path.open("wb") as handle:
        handle.write(serialized)
        for _ in range(full_chunks):
            handle.write(chunk)
        handle.write(chunk[:tail])
    assert path.stat().st_size == size
    lay_out(tmp_path, COMPARATOR_FILES)
    run = invoke(tmp_path, "compare", None)
    if extra_byte:
        assert run.returncode == runner.EXIT_RUNNER_ERROR
        assert run.result == {
            "version": 1,
            "mode": "compare",
            "nonce": "",  # The oversized file is rejected before parsing or nonce extraction.
            "status": "runner_error",
            "code": "invalid_job",
            "timed_out_case": None,
            "results": [],
        }
    else:
        assert run.returncode == runner.EXIT_COMPLETED
        assert (run.result["status"], run.result["nonce"], run.result["code"]) == ("completed", NONCE, None)
        assert len(run.result["results"]) == MAX_CASES
        assert all(item["status"] == "equal" and item["equal"] is True for item in run.result["results"])


def test_worker_file_limit_covers_job_and_result_budgets(tmp_path):
    source = "import resource\ndef f():\n    return resource.getrlimit(resource.RLIMIT_FSIZE)\n"
    run = run_source(tmp_path, source, [()])
    assert run.returncode == runner.EXIT_COMPLETED
    assert runner.FILE_LIMIT_BYTES >= max(runner.JOB_BYTES_LIMIT, runner.RESULT_BYTES_LIMIT)
    assert outcome_value(run.result["outcomes"][0]) == wire((runner.FILE_LIMIT_BYTES, runner.FILE_LIMIT_BYTES))


# --- GIL-holding workloads and CPU budget ------------------------------------------------------- #


@pytest.mark.parametrize(
    "hang",
    [
        "import sys\n    sys.setswitchinterval(10**6)\n    while True:\n        pass",
        "import re\n    re.match(r'(a+)+$', 'a' * 64 + 'b')",
        (
            "import signal\n    for number in range(1, signal.NSIG):\n        try:\n"
            "            signal.signal(number, lambda *_: None)\n        except (OSError, ValueError, RuntimeError):\n"
            "            pass\n    while True:\n        pass"
        ),
    ],
    ids=["no-thread-switch", "regex-in-c", "all-signals-caught"],
)
def test_gil_holding_workload_is_killed_at_the_deadline(tmp_path, hang):
    source = f"def f(x):\n    if x == 1:\n        return x\n    {hang}\n"
    run = run_source(tmp_path, source, [(1,), (2,), (3,)], suite_deadline_s=2)
    assert run.returncode == runner.EXIT_TIMEOUT
    assert run.elapsed < 2 + SLACK_S
    assert (run.result["status"], run.result["timed_out_case"]) == ("timeout", 1)
    assert [outcome_value(outcome) for outcome in run.result["outcomes"]] == [wire(1)]


def test_timeout_during_import_has_no_case_index(tmp_path):
    source = "import sys\nsys.setswitchinterval(10**6)\nwhile True:\n    pass\n\ndef f(x):\n    return x\n"
    run = run_source(tmp_path, source, [(1,)], suite_deadline_s=2)
    assert run.returncode == runner.EXIT_TIMEOUT
    assert run.elapsed < 2 + SLACK_S
    assert (run.result["status"], run.result["timed_out_case"], run.result["outcomes"]) == ("timeout", None, [])


def test_measured_cpu_budget_exhaustion_is_reported_as_a_timeout(tmp_path):
    # Ignore the soft-limit signal and exit only after a measurable overshoot: no scheduler rounding assumption.
    source = (
        "import os, signal, time\n"
        "def f(x):\n"
        "    signal.signal(signal.SIGXCPU, signal.SIG_IGN)\n"
        "    while time.process_time() < 1.2:\n"
        "        pass\n"
        "    os._exit(0)\n"
    )
    run = run_source(tmp_path, source, [(1,)], suite_deadline_s=60, cpu_time_s=1)
    assert run.returncode == runner.EXIT_TIMEOUT
    assert run.elapsed < 2 + SLACK_S
    assert (run.result["status"], run.result["timed_out_case"]) == ("timeout", 0)


@pytest.mark.parametrize("lower_limit", [False, True], ids=["self-signal", "lowered-soft-limit"])
def test_sigxcpu_without_configured_budget_usage_is_not_timeout_evidence(tmp_path, lower_limit):
    action = (
        "resource.setrlimit(resource.RLIMIT_CPU, (1, 90))\n    while True:\n        pass"
        if lower_limit
        else "os.kill(os.getpid(), signal.SIGXCPU)"
    )
    source = f"import os, resource, signal\ndef f(x):\n    {action}\n"
    run = run_source(tmp_path, source, [(1,)], suite_deadline_s=60, cpu_time_s=60)
    assert run.returncode == runner.EXIT_WORKER_ABORTED
    assert run.result is None and run.elapsed < SLACK_S


def test_memory_limit_surfaces_as_the_task_exception(tmp_path):
    source = "def f(x):\n    return len(bytearray(8 * 1024 ** 3))\n"
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "exception", "type": "MemoryError"}]


# --- worker ends without output ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(x):\n    os._exit(0)\n",
        "import os, signal\ndef f(x):\n    os.kill(os.getpid(), signal.SIGSEGV)\n",
        "import os\nos._exit(0)\ndef f(x):\n    return x\n",
        "import os\ndef f(x):\n    if x == 2:\n        os._exit(3)\n    return x\n",
    ],
    ids=["exit-in-call", "segfault-in-call", "exit-at-import", "exit-after-one-case"],
)
def test_worker_ending_early_leaves_no_result_and_a_distinct_status(tmp_path, source):
    # The supervisor reports EXIT_WORKER_ABORTED with no result file. execution.py attributes this abort to
    # the candidate and scores it 0 (test_execution.test_candidate_worker_abort_scores_zero_and_is_not_recoverable).
    run = run_source(tmp_path, source, [(1,), (2,)])
    assert run.returncode == runner.EXIT_WORKER_ABORTED
    assert run.result is None and not run.result_path.exists()
    assert run.elapsed < SLACK_S


# --- forged results and interference ------------------------------------------------------------ #


def test_forged_result_file_is_replaced_by_the_supervisor(tmp_path):
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    source = forge_helper() + f"\ndef f(x):\n    _forge({str(result_path)!r}, 'timeout')\n    return x + 1\n"
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == runner.EXIT_COMPLETED
    assert "forged" not in run.result
    assert (run.result["status"], [outcome_value(outcome) for outcome in run.result["outcomes"]]) == (
        "completed",
        [wire(2)],
    )


def test_forged_result_then_worker_exit_leaves_nothing(tmp_path):
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    source = forge_helper() + f"\ndef f(x):\n    _forge({str(result_path)!r})\n    os._exit(0)\n"
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == runner.EXIT_WORKER_ABORTED
    assert not result_path.exists()


def test_forged_result_then_catchable_signal_to_the_supervisor_leaves_nothing(tmp_path):
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    source = forge_helper() + (
        "\nimport signal, time\ndef f(x):\n"
        f"    _forge({str(result_path)!r})\n"
        "    if os.fork() == 0:\n"
        "        os.setsid()\n"
        "        time.sleep(60)\n"
        "        os._exit(0)\n"
        "    os.kill(os.getppid(), signal.SIGTERM)\n"
        "    time.sleep(60)\n"
        "    return x\n"
    )
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == runner.EXIT_CONTAINMENT_FAILED
    assert run.returncode not in runner.RESULT_EXIT_CODES and not result_path.exists()
    assert run.elapsed < SLACK_S


def test_forged_result_then_kill_of_the_supervisor_requires_rejecting_the_exit_status(tmp_path):
    """A surviving forgery and signal exit must be rejected. This does not prove accepted exits authenticate files."""
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    source = forge_helper() + (
        f"\nimport signal\ndef f(x):\n    _forge({str(result_path)!r})\n"
        "    os.kill(os.getppid(), signal.SIGKILL)\n    return x\n"
    )
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == -signal.SIGKILL
    assert run.returncode not in runner.RESULT_EXIT_CODES
    assert run.result is not None and run.result.get("forged") is True  # what a controller without the gate would read


def test_escaped_descendant_is_killed_before_the_result_exists(tmp_path):
    """A daemonized grandchild that holds the channel open and plans to overwrite the result later is swept."""
    lay_out(tmp_path)
    result_path, pid_path = tmp_path / "result.json", tmp_path / "escapee.pid"
    source = forge_helper() + (
        "\nimport time\n"
        "def f(x):\n"
        "    reader, writer = os.pipe()\n"
        "    if os.fork() == 0:\n"
        "        os.close(reader)\n"
        "        os.setsid()\n"
        "        if os.fork() == 0:\n"
        f"            with open({str(pid_path)!r}, 'w') as handle:\n"
        "                handle.write(str(os.getpid()))\n"
        "            os.write(writer, b'r')\n"
        "            time.sleep(1.5)\n"
        f"            _forge({str(result_path)!r})\n"
        "        os._exit(0)\n"
        "    os.close(writer)\n"
        "    assert os.read(reader, 1) == b'r'\n"
        "    os.close(reader)\n"
        "    return x + 1\n"
    )
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == runner.EXIT_COMPLETED and run.elapsed < SLACK_S
    assert "forged" not in run.result and outcome_value(run.result["outcomes"][0]) == wire(2)
    time.sleep(2.5)
    assert json.loads(result_path.read_bytes()) == run.result  # nothing overwrote it after the runner exited
    escapee = int(pid_path.read_text())  # the handshake proves this descendant existed, not an optional assertion
    with pytest.raises(ProcessLookupError):
        os.kill(escapee, 0)


def test_fork_bomb_is_bounded_and_swept(tmp_path):
    source = (
        "import os, time\n"
        "def f(x):\n"
        "    count = 0\n"
        "    for _ in range(200):\n"
        "        try:\n"
        "            pid = os.fork()\n"
        "        except OSError:\n"
        "            break\n"
        "        if pid == 0:\n"
        "            time.sleep(30)\n"
        "            os._exit(0)\n"
        "        count += 1\n"
        "    return count\n"
    )
    run = run_source(tmp_path, source, [(1,)], processes=16)
    assert run.returncode == runner.EXIT_COMPLETED and run.elapsed < SLACK_S
    forks = int(outcome_value(run.result["outcomes"][0])["value"][1])  # wire node ["int", "<count>"]
    assert 0 < forks < 16


def test_result_path_squatted_by_a_directory_is_reclaimed(tmp_path):
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    source = (
        f"import os\ndef f(x):\n    os.makedirs({str(result_path)!r})\n"
        f"    open({str(result_path / 'x')!r}, 'w').close()\n    return x\n"
    )
    (tmp_path / "source.py").write_text(source)
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]))
    assert run.returncode == runner.EXIT_COMPLETED and result_path.is_file()
    assert outcome_value(run.result["outcomes"][0]) == wire(1)


# --- bounded channel ---------------------------------------------------------------------------- #


def test_oversized_channel_record_ends_the_run_without_a_result(tmp_path):
    source = CHANNEL_HELPER + (
        f"\ndef f(x):\n    os.write(_channel(), b'x' * {runner._RECORD_LIMIT + 1})\n    return x\n"
    )
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == runner.EXIT_PROTOCOL_VIOLATION
    assert run.result is None and run.elapsed < SLACK_S


@pytest.mark.parametrize(
    "record",
    [
        '{"event": "done"}',
        '{"event": "runner_error", "code": "limits"}',
        '{"event": "source_error", "code": "import_error"}',
        '{"event": "item", "item": {"kind": "value", "value": NaN}}',
        "not json at all",
    ],
    ids=["early-done", "late-runner-error", "late-source-error", "nonfinite-item", "garbage"],
)
def test_forged_or_malformed_control_records_are_protocol_violations(tmp_path, record):
    source = CHANNEL_HELPER + f"\ndef f(x):\n    os.write(_channel(), {record!r}.encode() + b'\\n')\n    return x\n"
    run = run_source(tmp_path, source, [(1,), (2,)])
    assert run.returncode == runner.EXIT_PROTOCOL_VIOLATION
    assert run.result is None and not run.result_path.exists()


def test_relayed_items_are_observations_of_the_task_process(tmp_path):
    """An item the task process writes itself is indistinguishable from a real call outcome, so the result
    file never counts as authenticated invocation evidence."""
    forged = json.dumps({"event": "item", "item": {"kind": "exception", "type": "ValueError"}})
    source = CHANNEL_HELPER + (
        "\ndef f(x):\n"
        f"    os.write(_channel(), {forged!r}.encode() + b'\\n')\n"
        '    os.write(_channel(), b\'{"event": "done"}\\n\')\n'
        "    os._exit(0)\n"
    )
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "exception", "type": "ValueError"}]


def test_channel_closed_by_the_task_still_ends_at_the_deadline(tmp_path):
    source = CHANNEL_HELPER + "\nimport time\ndef f(x):\n    os.close(_channel())\n    time.sleep(30)\n    return x\n"
    run = run_source(tmp_path, source, [(1,)], suite_deadline_s=2)
    assert run.returncode == runner.EXIT_TIMEOUT and run.elapsed < 2 + SLACK_S
    assert (run.result["status"], run.result["timed_out_case"], run.result["outcomes"]) == ("timeout", 0, [])


# --- repair regressions, including operator-controlled fault injection -------------------------- #


@pytest.mark.parametrize("path_kind", ["missing", "fifo", "symlink", "directory"])
def test_unreadable_or_nonregular_source_is_not_a_source_defect(tmp_path, path_kind):
    lay_out(tmp_path)
    source_path = tmp_path / "source.py"
    if path_kind == "fifo":
        os.mkfifo(source_path)
    elif path_kind == "symlink":
        target = tmp_path / "target.py"
        target.write_text(PLAIN)
        source_path.symlink_to(target)
    elif path_kind == "directory":
        source_path.mkdir()
    run = invoke(tmp_path, "run", run_job(source_path, [(1,)]), timeout_s=SLACK_S)
    assert run.returncode == runner.EXIT_RUNNER_ERROR and run.elapsed < SLACK_S
    assert (run.result["status"], run.result["code"], run.result["outcomes"]) == ("runner_error", "invalid_job", [])


def test_static_source_checks_are_inside_the_supervised_deadline(tmp_path):
    lay_out(tmp_path)
    (tmp_path / "source.py").write_text(PLAIN)
    setup = (
        "import re\n"
        "def stalled_parser(text):\n"
        "    re.match(r'(a+)+$', 'a' * 64 + 'b')\n"
        "runner.parse_source = stalled_parser\n"
    )
    job = run_job(tmp_path / "source.py", [(1,)], suite_deadline_s=2)
    run = invoke(tmp_path, "run", job, setup=setup, timeout_s=2 + SLACK_S)
    assert run.returncode == runner.EXIT_TIMEOUT and run.elapsed < 2 + SLACK_S
    assert (run.result["timed_out_case"], run.result["outcomes"]) == (None, [])


def test_cleanup_inspection_error_is_not_an_empty_tree(tmp_path):
    lay_out(tmp_path)
    (tmp_path / "source.py").write_text(PLAIN)
    setup = (
        "original = runner._child_pids\ncalls = 0\n"
        "def unavailable_children():\n"
        "    global calls\n"
        "    calls += 1\n"
        "    if calls > 1:\n"
        "        raise OSError('inspection unavailable')\n"
        "    return original()\n"
        "runner._child_pids = unavailable_children\n"
    )
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]), setup=setup, timeout_s=SLACK_S)
    assert run.returncode == runner.EXIT_CONTAINMENT_FAILED
    assert run.result is None and not run.result_path.exists()


def test_result_limit_fallback_exit_matches_the_written_status(tmp_path):
    lay_out(tmp_path)
    (tmp_path / "source.py").write_text("def f(x):\n    return 'a' * 1000\n")
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]), setup="runner.RESULT_BYTES_LIMIT = 256")
    assert run.returncode == runner.EXIT_RUNNER_ERROR
    assert (run.result["status"], run.result["code"], run.result["outcomes"]) == ("runner_error", "result_limit", [])


def test_invalid_job_with_failed_result_write_does_not_return_an_accepted_exit(tmp_path):
    lay_out(tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text('{"forged": true}')
    job = run_job(tmp_path / "source.py", [(1,)])
    del job["entrypoint"]
    run = invoke(tmp_path, "run", job, setup="runner._place = lambda path, data: False")
    assert run.returncode == runner.EXIT_CONTAINMENT_FAILED
    assert run.result is None and not result_path.exists()


def test_unexpected_write_exception_cannot_masquerade_as_a_source_error_exit(tmp_path):
    lay_out(tmp_path)
    source = forge_helper() + f"\ndef f(x):\n    _forge({str(tmp_path / 'result.json')!r})\n    return x\n"
    (tmp_path / "source.py").write_text(source)
    setup = "def failed_write(path, data):\n    raise RuntimeError('write failed')\nrunner._place = failed_write"
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]), setup=setup)
    assert run.returncode == runner.EXIT_CONTAINMENT_FAILED
    assert run.result is None and not run.result_path.exists()


@pytest.mark.parametrize("error", ["MemoryError", "RecursionError", "OverflowError"])
def test_parser_resource_failure_is_not_a_syntax_error(tmp_path, error):
    lay_out(tmp_path)
    (tmp_path / "source.py").write_text(PLAIN)
    setup = f"def failed_parse(*args, **kwargs):\n    raise {error}\nrunner.ast.parse = failed_parse"
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]), setup=setup)
    assert run.returncode == runner.EXIT_RUNNER_ERROR
    assert (run.result["status"], run.result["code"]) == ("runner_error", "limits")


def test_supervisor_pipe_reader_is_nonblocking(tmp_path):
    lay_out(tmp_path)
    (tmp_path / "source.py").write_text(PLAIN)
    setup = (
        "import os\noriginal_supervise = runner._Supervisor._supervise\n"
        "def checked_supervise(self, reader):\n"
        "    assert not os.get_blocking(reader)\n"
        "    return original_supervise(self, reader)\n"
        "runner._Supervisor._supervise = checked_supervise\n"
    )
    run = invoke(tmp_path, "run", run_job(tmp_path / "source.py", [(1,)]), setup=setup)
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "value", "value": wire(2)}]


def test_fragmented_channel_record_larger_than_a_read_chunk_is_relayed(tmp_path):
    item = json.dumps({"event": "item", "item": {"kind": "value", "value": wire(7)}})
    source = CHANNEL_HELPER + (
        "\ndef f(x):\n"
        f"    data = b' ' * {runner._READ_CHUNK + 1} + {item!r}.encode() + b'\\n'\n"
        "    channel = _channel()\n"
        "    for offset in range(0, len(data), 1024):\n"
        "        os.write(channel, data[offset:offset + 1024])\n"
        '    os.write(channel, b\'{"event": "done"}\\n\')\n'
        "    os._exit(0)\n"
    )
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == runner.EXIT_COMPLETED
    assert run.result["outcomes"] == [{"kind": "value", "value": wire(7)}]


def test_all_items_without_done_do_not_invent_a_timed_out_call(tmp_path):
    item = json.dumps({"event": "item", "item": {"kind": "value", "value": wire(1)}})
    source = CHANNEL_HELPER + (
        f"\nimport time\ndef f(x):\n    os.write(_channel(), {item!r}.encode() + b'\\n')\n    time.sleep(60)\n"
    )
    run = run_source(tmp_path, source, [(1,)], suite_deadline_s=2)
    assert run.returncode == runner.EXIT_TIMEOUT and run.elapsed < 2 + SLACK_S
    assert run.result["timed_out_case"] is None
    assert run.result["outcomes"] == [{"kind": "value", "value": wire(1)}]


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [("import_error", runner.EXIT_SOURCE_ERROR), ("syntax_error", runner.EXIT_PROTOCOL_VIOLATION)],
)
def test_import_can_forge_an_import_error_but_not_a_pre_execution_error(tmp_path, code, exit_code):
    record = json.dumps({"event": "source_error", "code": code})
    source = CHANNEL_HELPER + (
        f"\nos.write(_channel(), {record!r}.encode() + b'\\n')\nos._exit(0)\ndef f(x):\n    return x\n"
    )
    run = run_source(tmp_path, source, [(1,)])
    assert run.returncode == exit_code
    if code == "import_error":
        assert (run.result["status"], run.result["code"]) == ("source_error", "import_error")
    else:
        assert run.result is None


# --- entrypoint binder regressions (static, remote-only) ----------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n    return x\nmatch 1:\n    case f:\n        pass\n",
        "def f(x):\n    return x\nmatch [1, 2]:\n    case [a, *f]:\n        pass\n",
        "def f(x):\n    return x\nmatch {'k': 1}:\n    case {'k': v, **f}:\n        pass\n",
        "def f(x):\n    return x\nmatch 1:\n    case (1 | 2) as f:\n        pass\n",
        "class P:\n    __match_args__ = ('a',)\ndef f(x):\n    return x\nmatch P():\n    case P(a=f):\n        pass\n",
        "def f(x):\n    return x\nmatch 1:\n    case _ if (f := 1):\n        pass\n",
    ],
    ids=["capture", "star", "mapping-rest", "or-as", "class-keyword", "guard-walrus"],
)
def test_match_captures_of_the_entrypoint_name_are_conflicts(source):
    tree, error = runner.parse_source(source)
    assert error is None
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"


def test_match_captures_of_other_names_do_not_conflict():
    tree, _ = runner.parse_source("def f(x):\n    return x\nmatch 1:\n    case g:\n        pass\n")
    assert runner.check_entrypoint_binding(tree, "f") is None


@pytest.mark.skipif(sys.version_info < (3, 12), reason="type alias statements need Python 3.12")
def test_type_alias_rebinding_of_the_entrypoint_is_a_conflict():
    tree, error = runner.parse_source("def f(x):\n    return x\ntype f = int\n")
    assert error is None and runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\ntype g[T] = list[T]\n")
    assert runner.check_entrypoint_binding(tree, "f") is None
    tree, _ = runner.parse_source("if True:\n    type f = int\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_missing"
