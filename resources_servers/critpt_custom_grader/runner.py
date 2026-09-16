# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-side runner: executes supplied source, or the protected comparator, never both in one domain.

Invocation is fixed and data travels only through files:

    python -I -B runner.py run     <job.json> <result.json>
    python -I -B runner.py compare <job.json> <result.json>

The supervisor validates the job, then forks one resource-limited worker for source reading, static checks,
input preparation and task execution, or protected comparison. The supervisor never executes task source or
expressions. Its own monotonic clock and SIGKILL enforce the suite deadline. Reported completions,
exceptions and errors are worker observations, not proof that an invocation returned or raised.

Security. The worker shares the supervisor's UID. This is containment machinery, NOT an authentication
boundary. A nonce, writable file, digest or accepted exit code does not authenticate a result. Every trust
claim additionally REQUIRES an operator-enforced runtime that protects the supervisor, provider and their
code from same-UID interference, prevents privilege escalation and escape, and protects result-path ancestors
and filesystem routing. This runner does not enforce those prerequisites. The controller must preserve
separate reference, candidate and comparator sandboxes, and keep stored expectations authoritative. The
nonce is correlation only.
"""

import ast
import json
import math
import os
import re
import secrets
import select
import signal
import stat
import sys
import time
import types
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


try:
    from .codec import (
        ComplexValue,
        LegacyText,
        RepresentationError,
        SetValue,
        SymbolicText,
        decode_value,
        encode_value,
    )
    from .task_data import (
        MAX_CASES,
        MAX_DEPTH,
        MAX_ELEMENTS,
        MAX_INTEGER_DIGITS,
        MAX_POLICY_BYTES,
        MAX_TASK_BYTES,
        MAX_TEXT_BYTES,
        MAX_WIRE_BYTES,
        bounded_json,
        decimal_text,
    )
except ImportError:  # Executed as a script inside a sandbox: the package directory sits beside this file.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from critpt_grader_rt.codec import (
        ComplexValue,
        LegacyText,
        RepresentationError,
        SetValue,
        SymbolicText,
        decode_value,
        encode_value,
    )
    from critpt_grader_rt.task_data import (
        MAX_CASES,
        MAX_DEPTH,
        MAX_ELEMENTS,
        MAX_INTEGER_DIGITS,
        MAX_POLICY_BYTES,
        MAX_TASK_BYTES,
        MAX_TEXT_BYTES,
        MAX_WIRE_BYTES,
        bounded_json,
        decimal_text,
    )


RUNTIME_PACKAGE = "critpt_grader_rt"
PROTOCOL_VERSION = 1
RUN_MODE = "run"
COMPARE_MODE = "compare"
MAX_SOURCE_BYTES = 1_048_576
MAX_NONCE_LENGTH = 64
MAX_TYPE_NAME = 128
MAX_CODE_LENGTH = 64
# One result holds at most MAX_CASES bounded wire values plus a fixed envelope.
RESULT_BYTES_LIMIT = MAX_CASES * (MAX_WIRE_BYTES + 512) + 65_536
# Reserve two wire/outcome envelopes and one full policy per case, plus the task allowance and 1 MiB of
# job fields.
JOB_BYTES_LIMIT = MAX_TASK_BYTES + MAX_CASES * (2 * (MAX_WIRE_BYTES + 512) + MAX_POLICY_BYTES) + 1_048_576
# The worker RLIMIT_FSIZE per-file allowance. Neither it nor the job read cap bounds aggregate disk use or
# authenticates any file.
FILE_LIMIT_BYTES = max(JOB_BYTES_LIMIT, 4 * RESULT_BYTES_LIMIT)
RUN_STATUSES = ("completed", "source_error", "timeout", "runner_error")
COMPARE_STATUSES = ("completed", "timeout", "runner_error")
SOURCE_ERROR_CODES = (
    "invalid_encoding",
    "source_limit",
    "syntax_error",
    "import_error",
    "entrypoint_missing",
    "entrypoint_conflict",
    "entrypoint_not_callable",
)
RUNNER_ERROR_CODES = ("invalid_job", "limits", "privileged", "missing_dependency", "result_limit")
OUTCOME_KINDS = ("value", "exception", "encoding_error")
VERDICT_STATUSES = ("equal", "mismatch", "invalid_candidate", "invalid_expected", "invalid_reference", "uncertain")
ROLES = ("reference", "candidate")
LIMIT_FIELDS = ("suite_deadline_s", "memory_mib", "cpu_time_s", "processes")
# Necessary transport checks only: these exit statuses do not authenticate files.
EXIT_COMPLETED = 0
EXIT_SOURCE_ERROR = 1
EXIT_RUNNER_ERROR = 2
EXIT_TIMEOUT = 3
EXIT_CONTAINMENT_FAILED = 4
EXIT_WORKER_ABORTED = 5
EXIT_PROTOCOL_VIOLATION = 6
EXIT_USAGE = 64
RESULT_EXIT_CODES = frozenset({EXIT_COMPLETED, EXIT_SOURCE_ERROR, EXIT_RUNNER_ERROR, EXIT_TIMEOUT})
NO_RESULT_EXIT_CODES = {
    EXIT_CONTAINMENT_FAILED: "containment_failed",
    EXIT_WORKER_ABORTED: "worker_aborted",
    EXIT_PROTOCOL_VIOLATION: "protocol_violation",
}
# The two no-result exits the supervisor emits after observing the worker itself fail. In the candidate
# domain this is an authenticated candidate fault. EXIT_CONTAINMENT_FAILED is not here: it means the
# supervisor could not contain or reap the tree, which is ambiguous with a provider fault.
WORKER_ABORT_EXIT_CODES = frozenset({EXIT_WORKER_ABORTED, EXIT_PROTOCOL_VIOLATION})
_STATUS_EXIT_CODES = {
    "completed": EXIT_COMPLETED,
    "source_error": EXIT_SOURCE_ERROR,
    "runner_error": EXIT_RUNNER_ERROR,
    "timeout": EXIT_TIMEOUT,
}
# Worker channel: one JSON record per line, each bounded like one result item plus its envelope.
_RECORD_LIMIT = MAX_WIRE_BYTES + 512
_READ_CHUNK = 65_536
_POLL_INTERVAL_S = 0.05
_REAP_BUDGET_S = 10.0  # bounded wait for the killed worker to become reapable
_SWEEP_BUDGET_S = 5.0  # bounded passes over /proc until no descendant remains
# Public sum of the reap and sweep budgets. execution.py imports this to size the exec wait. A config
# margin below it cuts a clean timeout run short.
CLEANUP_BUDGET_S = _REAP_BUDGET_S + _SWEEP_BUDGET_S
_SWEEP_PAUSE_S = 0.02
_CHILDREN_BYTES_LIMIT = 65_536
_PR_SET_CHILD_SUBREAPER, _PR_GET_CHILD_SUBREAPER = 36, 37
_WORKER_SOURCE_ERRORS = ("import_error", "entrypoint_not_callable")
_STATIC_SOURCE_ERRORS = tuple(code for code in SOURCE_ERROR_CODES if code not in _WORKER_SOURCE_ERRORS)
# Async signals the supervisor turns into "no result". Synchronous fault signals keep their default action.
_SUPERVISOR_SIGNALS = (
    "SIGHUP",
    "SIGINT",
    "SIGQUIT",
    "SIGTERM",
    "SIGUSR1",
    "SIGUSR2",
    "SIGALRM",
    "SIGVTALRM",
    "SIGPROF",
    "SIGXCPU",
    "SIGABRT",
    "SIGTRAP",
    "SIGSYS",
    "SIGIO",
    "SIGPWR",
    "SIGSTKFLT",
    "SIGTSTP",
    "SIGTTIN",
    "SIGTTOU",
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_NONCE = re.compile(r"[A-Za-z0-9]{1,64}\Z")


class RunnerError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# --- host-safe static helpers ------------------------------------------------------------------- #


def validate_job(job: Any, mode: str) -> str | None:
    """Shape-check a job file. Wire payloads are validated later by the codec or the comparator."""
    if type(job) is not dict or job.get("version") != PROTOCOL_VERSION or job.get("mode") != mode:
        return "invalid_job"
    nonce = job.get("nonce")
    if type(nonce) is not str or not _NONCE.fullmatch(nonce):
        return "invalid_job"
    limits = job.get("limits")
    if type(limits) is not dict or set(limits) != set(LIMIT_FIELDS):
        return "invalid_job"
    for value in limits.values():
        if type(value) is not int or value <= 0 or value > 2**31:
            return "invalid_job"
    cases = job.get("cases")
    if type(cases) is not list or not cases or len(cases) > MAX_CASES:
        return "invalid_job"
    if mode == RUN_MODE:
        if set(job) != {
            "version",
            "mode",
            "nonce",
            "entrypoint",
            "source_path",
            "cases",
            "limits",
            "input_conversions",
        }:
            return "invalid_job"
        if type(job["entrypoint"]) is not str or not _IDENTIFIER.fullmatch(job["entrypoint"]):
            return "invalid_job"
        if type(job["source_path"]) is not str or not job["source_path"].startswith("/"):
            return "invalid_job"
        conversions = job["input_conversions"]
        if type(conversions) is not list or len(conversions) > 128:
            return "invalid_job"
        if any(kind is not None and kind not in ("symbol", "function") for kind in conversions):
            return "invalid_job"
        for case in cases:
            if type(case) is not dict or set(case) != {"args", "kwargs"}:
                return "invalid_job"
            if type(case["args"]) is not list or len(case["args"]) > 128:
                return "invalid_job"
            if type(case["kwargs"]) is not dict or len(case["kwargs"]) > 128:
                return "invalid_job"
            if any(type(key) is not str or not _IDENTIFIER.fullmatch(key) for key in case["kwargs"]):
                return "invalid_job"
        return None
    if mode != COMPARE_MODE or set(job) != {"version", "mode", "nonce", "observed_role", "cases", "limits"}:
        return "invalid_job"
    if job["observed_role"] not in ROLES:
        return "invalid_job"
    for case in cases:
        if type(case) is not dict or not {"observed", "expected"} <= set(case) <= {"observed", "expected", "policy"}:
            return "invalid_job"
        if type(case["observed"]) is not dict or type(case["expected"]) is not dict:
            return "invalid_job"
        if "policy" in case and type(case["policy"]) is not dict:
            return "invalid_job"
    return None


def parse_source(text: str) -> tuple[ast.Module | None, str | None]:
    """Parse without executing. Resource failures propagate to the worker's runner-error path."""
    if type(text) is not str:
        return None, "invalid_encoding"
    if "\x00" in text:
        return None, "syntax_error"
    try:
        return ast.parse(text, mode="exec"), None
    except (SyntaxError, ValueError):
        return None, "syntax_error"


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _pattern_names(pattern: ast.AST) -> list[str]:
    """Capture names a match pattern binds in the enclosing scope: ``case x``, ``*rest``, ``**rest``, ``as x``."""
    found: list[str] = []
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name is not None:
                found.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            found.append(node.rest)
    return found


def _module_scope_bindings(statements: list[ast.stmt], nested: bool) -> list[tuple[str, str, bool]]:
    """Return (name, kind, nested) for every module-scope binding, in source order.

    kind is "def" for a plain top-level function definition, "wildcard" for ``from x import *``, and "other"
    for every other binding. Bindings inside compound statements are reported as nested.
    """
    type_alias = getattr(ast, "TypeAlias", None)
    found: list[tuple[str, str, bool]] = []
    for statement in statements:
        if isinstance(statement, ast.FunctionDef):
            found.append((statement.name, "def" if not nested else "other", nested))
        elif isinstance(statement, (ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((statement.name, "other", nested))
        elif isinstance(statement, ast.Assign):
            found.extend((name, "other", nested) for target in statement.targets for name in _target_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            found.extend((name, "other", nested) for name in _target_names(statement.target))
        elif type_alias is not None and isinstance(statement, type_alias):
            found.extend((name, "other", nested) for name in _target_names(statement.name))
        elif isinstance(statement, ast.Delete):
            found.extend((name, "other", nested) for target in statement.targets for name in _target_names(target))
        elif isinstance(statement, ast.Import):
            found.extend((alias.asname or alias.name.partition(".")[0], "other", nested) for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.name == "*":
                    found.append(("*", "wildcard", nested))
                else:
                    found.append((alias.asname or alias.name, "other", nested))
        elif isinstance(statement, (ast.Global, ast.Nonlocal)):
            found.extend((name, "other", nested) for name in statement.names)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            found.extend((name, "other", True) for name in _target_names(statement.target))
            found.extend(_module_scope_bindings(statement.body + statement.orelse, True))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                if item.optional_vars is not None:
                    found.extend((name, "other", True) for name in _target_names(item.optional_vars))
            found.extend(_module_scope_bindings(statement.body, True))
        elif isinstance(statement, (ast.If, ast.While)):
            found.extend(_module_scope_bindings(statement.body + statement.orelse, True))
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            for handler in statement.handlers:
                if handler.name is not None:
                    found.append((handler.name, "other", True))
                found.extend(_module_scope_bindings(handler.body, True))
            found.extend(_module_scope_bindings(statement.body + statement.orelse + statement.finalbody, True))
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                found.extend((name, "other", True) for name in _pattern_names(case.pattern))
                found.extend(_module_scope_bindings(case.body, True))
        found.extend((node.target.id, "other", True) for node in _walrus_targets(statement))
    return found


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _enclosing_scope_children(node: ast.AST) -> list[ast.AST]:
    """The children of a function, class, or lambda that the ENCLOSING scope evaluates.

    Decorators, bases, and argument defaults and annotations evaluate where the definition appears, so a
    walrus in one of them binds in the enclosing scope. The body is the node's own scope and is excluded.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = {id(statement) for statement in node.body}
        return [child for child in ast.iter_child_nodes(node) if id(child) not in body]
    if isinstance(node, ast.Lambda):
        return [node.args]  # a lambda body is in the lambda's own scope, but its defaults are here
    return list(ast.iter_child_nodes(node))


def _walrus_targets(node: ast.AST) -> list[ast.NamedExpr]:
    """Walrus targets that bind in the scope that evaluates ``node``.

    Do not descend into a function, class, or lambda scope, INCLUDING ``node``'s own body when ``node`` is
    itself one. Still visit the parts of such a node the enclosing scope evaluates.
    """
    found: list[ast.NamedExpr] = []
    pending = _enclosing_scope_children(node) if isinstance(node, _SCOPE_NODES) else list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        if isinstance(child, _SCOPE_NODES):
            pending.extend(_enclosing_scope_children(child))
            continue
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            found.append(child)
        pending.extend(ast.iter_child_nodes(child))
    return found


def check_entrypoint_binding(tree: ast.Module, entrypoint: str) -> str | None:
    """Fail closed: exactly one plain top-level def, and nothing else that binds or shadows the name."""
    bindings = _module_scope_bindings(list(tree.body), False)
    definitions = [index for index, (name, kind, _) in enumerate(bindings) if name == entrypoint and kind == "def"]
    if not definitions:
        return "entrypoint_missing"
    if len(definitions) > 1:
        return "entrypoint_conflict"
    for index, (name, kind, _) in enumerate(bindings):
        if name == entrypoint and kind != "def":
            return "entrypoint_conflict"
        if kind == "wildcard" and index > definitions[0]:
            return "entrypoint_conflict"
    return None


def _binary64_component(value: Any) -> float:
    """A complex component as binary64, only when the double keeps its exact value, else reject the input.

    Python's complex is always a pair of binary64 doubles, so an exact component a double cannot hold would
    be narrowed silently. Reject that materialization rather than narrow it, on the runner-error path.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunnerError("invalid_job") from exc
    if not math.isfinite(number) or Fraction(number) != Fraction(value):
        raise RunnerError("invalid_job")
    return number


def materialize_input(value: Any) -> Any:
    """Turn decoded wire carriers into the Python values a task function is called with."""
    kind = type(value)
    if kind is ComplexValue:
        real, imag = materialize_input(value.real), materialize_input(value.imag)
        return complex(_binary64_component(real), _binary64_component(imag))
    if kind is SymbolicText:
        return _symbolic_input(value.text)
    if kind is LegacyText:
        raise RunnerError("invalid_job")
    if kind is list:
        return [materialize_input(child) for child in value]
    if kind is tuple:
        return tuple(materialize_input(child) for child in value)
    if kind is SetValue:
        return {materialize_input(child) for child in value.items}
    if kind is dict:
        return {key: materialize_input(child) for key, child in value.items()}
    return value


def _symbolic_input(text: str) -> Any:
    # Materialize every symbolic input through the same restricted parser the comparator uses, so a bare
    # reserved name becomes the same constant on both sides and an ordinary name becomes a free symbol.
    try:
        import sympy  # noqa: F401  the restricted parser needs it
    except ImportError as exc:
        raise RunnerError("missing_dependency") from exc
    try:
        from .symbolic import parse_expression
    except ImportError:
        from critpt_grader_rt.symbolic import parse_expression
    try:
        return parse_expression(text)
    except RepresentationError as exc:
        raise RunnerError("invalid_job") from exc


def adapt_output(value: Any) -> Any:
    """Map supported library values onto codec types explicitly, never narrowing a dtype.

    Containers are traversed with the codec's element and depth budget and a cycle guard. Only libraries the
    executed source already imported are consulted. A finite SymPy Float is emitted as its exact binary64
    double, or rejected when a double cannot hold it, never rounded. A Float atom inside a larger expression
    stays symbolic. A set or frozenset becomes an inert SetValue carrier the comparator matches unordered.
    """
    numpy = sys.modules.get("numpy")
    sympy = sys.modules.get("sympy")
    budget = 0
    active: set[int] = set()

    def adapt(item: Any, depth: int) -> Any:
        nonlocal budget
        budget += 1
        if budget > MAX_ELEMENTS or depth > MAX_DEPTH:
            raise RepresentationError("structure_limit")
        kind = type(item)
        if kind is list or kind is tuple or kind is dict:
            if id(item) in active or len(item) > MAX_ELEMENTS:
                raise RepresentationError("structure_limit")
            active.add(id(item))
            try:
                if kind is dict:
                    return {key: adapt(child, depth + 1) for key, child in item.items()}
                converted = [adapt(child, depth + 1) for child in item]
            finally:
                active.discard(id(item))
            return converted if kind is list else tuple(converted)
        if kind is set or kind is frozenset:
            if len(item) > MAX_ELEMENTS:
                raise RepresentationError("structure_limit")
            # A set cannot contain itself, so no cycle guard is needed.
            return SetValue(tuple(adapt(child, depth + 1) for child in item))
        if numpy is not None:
            if isinstance(item, numpy.ndarray):
                if item.size > MAX_ELEMENTS:
                    raise RepresentationError("structure_limit")
                if item.dtype.kind == "O":
                    return adapt(item.tolist(), depth + 1)
                if item.dtype.kind in "biu" or (item.dtype.kind == "f" and item.dtype.itemsize <= 8):
                    return adapt(item.tolist(), depth + 1)
                if item.dtype.kind == "c" and item.dtype.itemsize <= 16:
                    return adapt(item.tolist(), depth + 1)
                raise RepresentationError("unsupported_type")
            if isinstance(item, numpy.generic):
                if isinstance(item, numpy.bool_):
                    return bool(item)
                if isinstance(item, numpy.integer):
                    return int(item)
                if isinstance(item, numpy.floating) and item.dtype.itemsize <= 8:
                    return float(item)
                if isinstance(item, numpy.complexfloating) and item.dtype.itemsize <= 16:
                    return complex(item)
                if isinstance(item, numpy.str_):
                    # A numpy text scalar is a str subclass. Normalize to a base str so the codec's exact
                    # type check matches and its text-length bound applies.
                    return str(item)
                # Every other numpy scalar stays refused: the codec does not carry it.
                raise RepresentationError("unsupported_type")
        if sympy is not None and isinstance(item, sympy.Basic):
            if isinstance(item, sympy.Integer):
                return int(item)
            if isinstance(item, sympy.Rational):
                return Fraction(int(item.p), int(item.q))
            if isinstance(item, sympy.Float):
                if item.is_finite:
                    # A bare Float emits the exact binary64 double, or is rejected when a double cannot hold
                    # it, so a bare number goes through numeric tolerance, not the symbolic path.
                    number = float(item)
                    if not math.isfinite(number) or sympy.Rational(item) != sympy.Rational(number):
                        raise RepresentationError("unsupported_type")
                    return number
            # A Float atom inside a larger expression stays symbolic. str(item) round-trips exactly through
            # the bounded grammar, so it does not decimalize to a different rational.
            return SymbolicText(str(item))
        return item

    return adapt(value, 0)


def bounded_type_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name[:MAX_TYPE_NAME] if _IDENTIFIER.fullmatch(name[:MAX_TYPE_NAME]) else "Exception"


# --- sandbox-only: result file ------------------------------------------------------------------ #


class _ResultWriter:
    """Single-shot atomic replacement after cleanup. Neither the path nor this class authenticates a file."""

    def __init__(self, path: str, mode: str, nonce: str):
        self._path = path
        self._envelope = {"version": PROTOCOL_VERSION, "mode": mode, "nonce": nonce}
        self.finished = False
        self.status: str | None = None

    def write(self, status: str, code: str | None, items: list, timed_out_case: int | None) -> bool:
        if self.finished:
            return False
        key = "outcomes" if self._envelope["mode"] == RUN_MODE else "results"
        payload = dict(self._envelope, status=status, code=code, timed_out_case=timed_out_case)
        payload[key] = list(items)
        data = _dump(payload)
        if data is None or len(data) > RESULT_BYTES_LIMIT:
            payload = dict(self._envelope, status="runner_error", code="result_limit", timed_out_case=None)
            payload[key] = []
            data = _dump(payload) or b"{}"
        self.finished = _place(self._path, data)
        self.status = payload["status"] if self.finished else None
        return self.finished


def _dump(payload: dict) -> bytes | None:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None


def _place(path: str, data: bytes) -> bool:
    """Create an exclusive temporary file, sync and replace, moving a squatting directory aside first.

    The final component is not followed through a symlink. Ancestors and filesystem routing must be protected
    by the runtime, which O_NOFOLLOW does not do.
    """
    directory = os.path.dirname(path) or "."
    temporary = os.path.join(directory, f".{os.path.basename(path)}.{secrets.token_hex(8)}.tmp")
    try:
        os.makedirs(directory, exist_ok=True)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError:
        return False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, path)
        except IsADirectoryError:
            if not _move_aside(path):
                return False
            os.replace(temporary, path)
        return True
    except OSError:
        return False
    finally:
        try:
            os.unlink(temporary)  # a no-op after a successful move
        except OSError:
            pass


def _move_aside(path: str) -> bool:
    try:
        os.rename(path, f"{path}.{secrets.token_hex(8)}.squat")
    except OSError:
        return False
    return True


def _discard_result(path: str) -> None:
    """Best effort: leave nothing readable at the result path when no runner-authored result exists."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        _move_aside(path)


# --- sandbox-only: process tree ----------------------------------------------------------------- #


def _child_pids() -> list[int]:
    """Read this single-threaded supervisor's direct children, including adopted orphans, with a byte cap.

    Failure is not an empty tree. A direct child's PID cannot be reused until this process reaps it, so do
    not reap between this snapshot and signalling it.
    """
    with open(f"/proc/self/task/{os.getpid()}/children", "rb") as handle:
        data = handle.read(_CHILDREN_BYTES_LIMIT + 1)
    if len(data) > _CHILDREN_BYTES_LIMIT:
        raise RunnerError("limits")
    fields = data.split()
    if any(not field.isdigit() or len(field) > 10 or int(field) <= 0 for field in fields):
        raise RunnerError("limits")
    return [int(field) for field in fields]


def _become_subreaper() -> bool:
    """Orphans of the worker re-parent to this process instead of init, so the sweep can see and reap them."""
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return False
        flag = ctypes.c_int(0)
        if prctl(_PR_GET_CHILD_SUBREAPER, ctypes.addressof(flag), 0, 0, 0) != 0:
            return False
        return flag.value == 1
    except (ImportError, OSError, AttributeError, ValueError, TypeError):
        return False


def _kill(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False  # a descendant this user may not signal: containment is beyond this runtime
    return True


def _reap_orphans(deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _guard_supervisor(interrupted: Callable[[int, Any], None]) -> None:
    """Signals set a flag. Cleanup runs on the normal control path, never inside a signal handler."""
    for name in _SUPERVISOR_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, interrupted)
            except (OSError, ValueError, RuntimeError):
                pass


def _reset_signals() -> None:
    """Worker side: every supervisor handler goes back to its default action, so RLIMIT_CPU can end the worker."""
    for name in _SUPERVISOR_SIGNALS:
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, signal.SIG_DFL)
            except (OSError, ValueError, RuntimeError):
                pass


def _apply_limits(limits: dict) -> str | None:
    """Worker side: address space, CPU time, file size, core and process-count limits for the task process."""
    import resource

    try:
        memory = int(limits["memory_mib"]) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = int(limits["cpu_time_s"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT_BYTES, FILE_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        processes = int(limits["processes"])
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
    except (KeyError, TypeError, ValueError, OSError):
        return "limits"
    return None


# --- sandbox-only: worker ----------------------------------------------------------------------- #


def _read_file(path: str, limit: int) -> bytes:
    """Bound bytes and refuse FIFOs/devices and final-component symlinks. Ancestors are a runtime prerequisite."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RunnerError("invalid_job")
        return handle.read(limit + 1)


def _read_json(path: str, limit: int) -> Any:
    data = _read_file(path, limit)
    if len(data) > limit:
        raise RunnerError("invalid_job")
    return json.loads(data.decode("utf-8"))


def _load_source(path: str) -> tuple[str | None, str | None]:
    try:
        data = _read_file(path, MAX_SOURCE_BYTES)
    except OSError as exc:
        raise RunnerError("invalid_job") from exc
    if len(data) > MAX_SOURCE_BYTES:
        return None, "source_limit"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "invalid_encoding"


def _input_literal(value: Any) -> Any:
    """An explicit bounded AST visitor, used only in the resource-limited remote worker."""
    if type(value) is not str:
        return value
    if len(value) > MAX_TEXT_BYTES:
        raise ValueError("input literal byte limit")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("input literal byte limit")
    # Bare labels stay strings. Literal-looking but unsupported syntax is a defect.
    if not text or not (text[0] in "[({'\"+-.0123456789" or text in ("True", "False", "None")):
        return value
    if any(len(run) > MAX_INTEGER_DIGITS for run in re.findall(r"[0-9]+", text)):
        raise ValueError("input literal integer limit")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("unsupported input literal syntax") from exc
    count = 0

    def visit(node: ast.AST, depth: int) -> Any:
        nonlocal count
        count += 1
        if count > MAX_ELEMENTS or depth > MAX_DEPTH:
            raise ValueError("input literal structure limit")
        if type(node) is ast.Constant:
            if type(node.value) in (int, float):
                token = ast.get_source_segment(text, node)
                decimal_text(token)
                if type(node.value) is float:
                    if not math.isfinite(node.value) or (node.value == 0 and Decimal(token) != 0):
                        raise ValueError("input literal exceeds binary64 range; use a decimal wire value")
                return node.value
            if node.value is None or type(node.value) in (str, bool):
                return node.value
        elif type(node) in (ast.List, ast.Tuple):
            values = [visit(child, depth + 1) for child in node.elts]
            return tuple(values) if type(node) is ast.Tuple else values
        elif type(node) is ast.Dict:
            out = {}
            for key, child in zip(node.keys, node.values, strict=True):
                name = visit(key, depth + 1)
                if type(name) is not str or name in out:
                    raise ValueError("input literal map keys must be unique strings")
                out[name] = visit(child, depth + 1)
            return out
        elif type(node) is ast.UnaryOp and type(node.op) in (ast.UAdd, ast.USub):
            number = visit(node.operand, depth + 1)
            if type(number) in (int, float):
                return -number if type(node.op) is ast.USub else number
        raise ValueError("unsupported input literal syntax")

    return visit(tree.body, 0)


def _decode_input(wire: dict) -> Any:
    """Parse only a root legacy argument, then bound its representation before materialization."""
    try:
        value = decode_value(wire)
        if type(value) is LegacyText:
            value = _input_literal(value.text)
            encode_value(value)  # Parsed leaves and the complete wire must satisfy the existing codec bounds.
        return materialize_input(value)
    except ValueError as exc:  # Includes RepresentationError. Resource failures keep the worker's limits path.
        raise RunnerError("invalid_job") from exc


def _decode_cases(cases: list) -> list[tuple[list, dict]]:
    decoded = []
    for case in cases:
        args = [_decode_input(item) for item in case["args"]]
        kwargs = {key: _decode_input(item) for key, item in case["kwargs"].items()}
        decoded.append((args, kwargs))
    return decoded


def _apply_input_conversions(cases: list[tuple[list, dict]], conversions: list) -> list[tuple[list, dict]]:
    """Apply per-position symbolic conversions to positional arguments only.

    "symbol" turns a string positional argument into sympy.Symbol(name) and "function" into
    sympy.Function(name). Null or a missing position leaves the argument alone. Keyword arguments are never
    converted. sympy is imported only when a conversion actually runs."""
    if not any(kind in ("symbol", "function") for kind in conversions):
        return cases
    try:
        import sympy
    except ImportError as exc:
        raise RunnerError("missing_dependency") from exc
    converted = []
    for args, kwargs in cases:
        args = list(args)
        for index, kind in enumerate(conversions):
            if index >= len(args) or kind not in ("symbol", "function"):
                continue
            if type(args[index]) is not str:
                continue
            args[index] = sympy.Symbol(args[index]) if kind == "symbol" else sympy.Function(args[index])
        converted.append((args, kwargs))
    return converted


def _invoke(entry: Any, args: list, kwargs: dict) -> dict:
    try:
        value = entry(*args, **kwargs)
    except BaseException as exc:  # Worker observation only. Task code can also forge the same record.
        return {"kind": "exception", "type": bounded_type_name(exc)}
    try:
        return {"kind": "value", "value": encode_value(adapt_output(value))}
    except RepresentationError as exc:
        return {"kind": "encoding_error", "code": exc.code}
    except (MemoryError, RecursionError, OverflowError):
        return {"kind": "encoding_error", "code": "structure_limit"}
    except Exception:
        return {"kind": "encoding_error", "code": "unsupported_type"}


def _send(channel: int, record: dict) -> None:
    data = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    while data:
        data = data[os.write(channel, data) :]


def _run_worker(channel: int, job: dict) -> None:
    """Source parsing and input preparation also run under the supervisor's deadline and worker limits."""
    code = _apply_limits(job["limits"])
    if code is None:
        try:
            source, source_error = _load_source(job["source_path"])
            if source_error is None:
                tree, source_error = parse_source(source)
            if source_error is None:
                source_error = check_entrypoint_binding(tree, job["entrypoint"])
            if source_error is not None:
                _send(channel, {"event": "source_error", "code": source_error})
                return
            cases = _decode_cases(job["cases"])
            cases = _apply_input_conversions(cases, job["input_conversions"])
        except RunnerError as exc:
            code = exc.code
        except (MemoryError, RecursionError, OverflowError):
            code = "limits"
    if code is not None:
        _send(channel, {"event": "runner_error", "code": code})
        return
    _send(channel, {"event": "start"})
    module = types.ModuleType("submission")
    try:
        exec(compile(tree, "<submission>", "exec"), module.__dict__)
    except BaseException:
        _send(channel, {"event": "source_error", "code": "import_error"})
        return
    entry = module.__dict__.get(job["entrypoint"])
    if not callable(entry):
        _send(channel, {"event": "source_error", "code": "entrypoint_not_callable"})
        return
    _send(channel, {"event": "ready"})
    for args, kwargs in cases:
        _send(channel, {"event": "item", "item": _invoke(entry, args, kwargs)})
    _send(channel, {"event": "done"})


def _compare_worker(channel: int, job: dict) -> None:
    """Worker side of a compare job: operator code over untrusted observations, under the same limits."""
    code = _apply_limits(job["limits"])
    if code is None:
        try:
            try:
                from .comparator import compare_request
            except ImportError:
                from critpt_grader_rt.comparator import compare_request
        except ImportError:
            code = "missing_dependency"
    if code is not None:
        _send(channel, {"event": "runner_error", "code": code})
        return
    _send(channel, {"event": "start"})
    _send(channel, {"event": "ready"})
    for case in job["cases"]:
        # Role, expectation and policy come from the controller's job file, never from an observation.
        request = {
            "version": 1,
            "observed_role": job["observed_role"],
            "observed": case["observed"],
            "expected": case["expected"],
        }
        if "policy" in case:
            request["policy"] = case["policy"]
        _send(channel, {"event": "item", "item": compare_request(request)})
    _send(channel, {"event": "done"})


# --- sandbox-only: supervisor ------------------------------------------------------------------- #


def _reject_constant(text: str) -> Any:
    raise ValueError("nonfinite JSON constant")


class _Supervisor:
    """Owns the worker lifecycle: bounded channel reads, the deadline, the tree sweep and the single result write."""

    def __init__(self, mode: str, job: dict, result_path: str):
        self.mode = mode
        self.result_path = result_path
        self.limits = job["limits"]
        self.expected = len(job["cases"])
        self.writer = _ResultWriter(result_path, mode, job["nonce"])
        self.items: list = []
        self.started = False
        self.ready = False
        self.code: str | None = None
        self.worker: int | None = None
        self.ended = False
        self.cpu_seconds = 0.0
        self.deadline = 0.0
        self.interrupted = False

    def interrupt(self, signum: int, frame: Any) -> None:
        self.interrupted = True

    def finish(self, status: str, code: str | None = None, timed_out_case: int | None = None) -> int:
        """Write the single result and map it to the exit status. An unwritable result is a containment failure."""
        if self.interrupted:
            return self.abandon(EXIT_CONTAINMENT_FAILED)
        items = self.items if status in ("completed", "timeout") else []
        if not self.writer.write(status, code, items, timed_out_case) or self.interrupted:
            return self.abandon(EXIT_CONTAINMENT_FAILED)
        return _STATUS_EXIT_CODES[self.writer.status]

    def abandon(self, exit_code: int) -> int:
        _discard_result(self.result_path)
        return exit_code

    def run(self, worker: Callable[[int], None]) -> int:
        _discard_result(self.result_path)
        _guard_supervisor(self.interrupt)
        try:
            # An inherited SIG_IGN for SIGCHLD would auto-reap the worker and make wait4 report it gone while alive.
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        except (OSError, ValueError, RuntimeError):
            pass
        try:
            reader, channel = os.pipe()
        except OSError:
            return self.finish("runner_error", "limits")
        try:
            os.set_blocking(reader, False)  # a task-side reader must not turn select/read into a blocking race
            pid = os.fork()
        except OSError:
            os.close(reader)
            os.close(channel)
            return self.finish("runner_error", "limits")
        if pid == 0:  # worker: never returns into the supervisor's code path
            status = EXIT_WORKER_ABORTED
            try:
                os.close(reader)
                _reset_signals()
                worker(channel)
                status = 0
            except BaseException:
                pass
            finally:
                os._exit(status)
        os.close(channel)
        self.worker = pid
        self.deadline = time.monotonic() + float(self.limits["suite_deadline_s"])
        try:
            outcome = self._supervise(reader)
        except Exception:
            outcome = "interrupted"
        finally:
            os.close(reader)
            cleaned = self._terminate_tree()
        if not cleaned or self.interrupted or outcome == "interrupted":
            return self.abandon(EXIT_CONTAINMENT_FAILED)
        if outcome == "aborted":
            return self.abandon(EXIT_WORKER_ABORTED)
        if outcome == "violation":
            return self.abandon(EXIT_PROTOCOL_VIOLATION)
        if outcome == "timeout":
            timed_out = len(self.items) if self.ready and len(self.items) < self.expected else None
            return self.finish("timeout", None, timed_out)
        return self.finish(outcome, self.code)

    def _supervise(self, reader: int) -> str:
        """Relay records until a terminal one, the deadline, or the worker's end. Every wait is bounded."""
        buffer = bytearray()
        open_channel = True
        while True:
            if self.interrupted:
                return "interrupted"
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            if not self.ended and self._reap_worker():
                # Buffered records may come from the worker OR its descendants. Neither is authenticated.
                outcome = self._drain(reader, buffer) if open_channel else None
                if outcome is not None:
                    return outcome
                return "timeout" if self._cpu_exhausted() else "aborted"
            wait = min(_POLL_INTERVAL_S, remaining)
            if not open_channel:
                time.sleep(wait)
                continue
            if not select.select([reader], [], [], wait)[0]:
                continue
            try:
                chunk = os.read(reader, _READ_CHUNK)
            except BlockingIOError:
                continue
            if not chunk:
                open_channel = False  # every write end is closed, so only the worker's end can still matter
                continue
            outcome = self._consume(buffer, chunk)
            if outcome is not None:
                return outcome

    def _drain(self, reader: int, buffer: bytearray) -> str | None:
        """Read what is already buffered in the pipe, without blocking and within the bytes the worker could
        have legitimately written, so a descendant still holding the write end cannot keep this going."""
        budget = (self.expected - len(self.items) + 2) * _RECORD_LIMIT
        while budget > 0 and select.select([reader], [], [], 0)[0]:
            if self.interrupted:
                return "interrupted"
            if time.monotonic() >= self.deadline:
                return "timeout"
            try:
                chunk = os.read(reader, min(_READ_CHUNK, budget))
            except BlockingIOError:
                return None
            if not chunk:
                return None
            budget -= len(chunk)
            outcome = self._consume(buffer, chunk)
            if outcome is not None:
                return outcome
        return None

    def _consume(self, buffer: bytearray, chunk: bytes) -> str | None:
        buffer += chunk
        while True:
            if self.interrupted:
                return "interrupted"
            if time.monotonic() >= self.deadline:
                return "timeout"
            end = buffer.find(b"\n")
            if end < 0:
                return "violation" if len(buffer) > _RECORD_LIMIT else None
            line = bytes(buffer[:end])
            del buffer[: end + 1]
            outcome = self._accept(line)
            if outcome is not None:
                return outcome

    def _accept(self, line: bytes) -> str | None:
        """One channel record. Anything unexpected, oversized, or out of order ends the run without a result."""
        if len(line) > _RECORD_LIMIT:
            return "violation"
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, TypeError, RecursionError, MemoryError):
            return "violation"
        if type(record) is not dict or type(record.get("event")) is not str:
            return "violation"
        event, keys = record["event"], set(record)
        if event == "start" and keys == {"event"} and not self.started:
            self.started = True
            return None
        if event == "ready" and keys == {"event"} and self.started and not self.ready:
            self.ready = True
            return None
        if event == "runner_error" and keys == {"event", "code"} and not self.started:
            if type(record["code"]) is str and record["code"] in RUNNER_ERROR_CODES:
                self.code = record["code"]
                return "runner_error"
            return "violation"
        if event == "source_error" and keys == {"event", "code"} and self.mode == RUN_MODE and not self.ready:
            codes = _WORKER_SOURCE_ERRORS if self.started else _STATIC_SOURCE_ERRORS
            if type(record["code"]) is str and record["code"] in codes:
                self.code = record["code"]
                return "source_error"
            return "violation"
        if event == "item" and keys == {"event", "item"} and self.ready and len(self.items) < self.expected:
            if self._well_formed(record["item"]):
                self.items.append(record["item"])
                return None
            return "violation"
        if event == "done" and keys == {"event"} and self.ready and len(self.items) == self.expected:
            return "completed"
        return "violation"

    def _well_formed(self, item: Any) -> bool:
        """Bounded and labelled like one result item. The controller performs the full shape check."""
        if type(item) is not dict:
            return False
        if self.mode == RUN_MODE:
            label, labels = item.get("kind"), OUTCOME_KINDS
        else:
            label, labels = item.get("status"), VERDICT_STATUSES
        if type(label) is not str or label not in labels:
            return False
        try:
            bounded_json(item, max_bytes=_RECORD_LIMIT)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return False
        return True

    def _reap_worker(self) -> bool:
        try:
            pid, _, usage = os.wait4(self.worker, os.WNOHANG)
        except ChildProcessError:
            self.ended = True
            return True
        if pid == 0:
            return False
        self.ended = True
        self.cpu_seconds = usage.ru_utime + usage.ru_stime
        return True

    def _cpu_exhausted(self) -> bool:
        """Measured accounting, not the signal number: tasks can self-send SIGXCPU or lower their soft limit."""
        return self.cpu_seconds >= float(self.limits["cpu_time_s"])

    def _terminate_tree(self) -> bool:
        """Kill and reap direct children until adoption exposes no more. Fail closed on error or budget."""
        try:
            if not self.ended and not _kill(self.worker):
                return False
            limit = time.monotonic() + _REAP_BUDGET_S
            while not self.ended:
                if self._reap_worker() or time.monotonic() >= limit:
                    break
                time.sleep(_SWEEP_PAUSE_S)
            if not self.ended:
                return False
            limit = time.monotonic() + _SWEEP_BUDGET_S
            while time.monotonic() < limit:
                _reap_orphans(limit)
                survivors = _child_pids()
                if time.monotonic() >= limit:
                    return False
                if not survivors:
                    return True
                for pid in survivors:
                    if time.monotonic() >= limit or not _kill(pid):
                        return False
                time.sleep(_SWEEP_PAUSE_S)
        except (OSError, RunnerError, ValueError, MemoryError):
            return False
        return False


def _preflight(supervisor: _Supervisor) -> int | None:
    """Check necessary Linux facilities before task execution. This does not qualify runtime integrity."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return supervisor.finish("runner_error", "privileged")
    try:
        if not _become_subreaper():
            return supervisor.finish("runner_error", "limits")
        _child_pids()
    except (OSError, RunnerError, ValueError):
        return supervisor.finish("runner_error", "limits")
    return None


def execute_run(job: dict, result_path: str) -> int:
    supervisor = _Supervisor(RUN_MODE, job, result_path)
    refused = _preflight(supervisor)
    if refused is not None:
        return refused
    return supervisor.run(lambda channel: _run_worker(channel, job))


def execute_compare(job: dict, result_path: str) -> int:
    supervisor = _Supervisor(COMPARE_MODE, job, result_path)
    refused = _preflight(supervisor)
    if refused is not None:
        return refused
    return supervisor.run(lambda channel: _compare_worker(channel, job))


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] not in (RUN_MODE, COMPARE_MODE):
        return EXIT_USAGE
    mode, job_path, result_path = argv
    try:
        job = _read_json(job_path, JOB_BYTES_LIMIT)
        code = validate_job(job, mode)
    except (RunnerError, OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        job, code = None, "invalid_job"
    if code is not None:
        nonce = job.get("nonce") if type(job) is dict and type(job.get("nonce")) is str else ""
        writer = _ResultWriter(result_path, mode, nonce if _NONCE.fullmatch(nonce) else "")
        if not writer.write("runner_error", code, [], None):
            _discard_result(result_path)
            return EXIT_CONTAINMENT_FAILED
        return _STATUS_EXIT_CODES[writer.status]
    return execute_run(job, result_path) if mode == RUN_MODE else execute_compare(job, result_path)


def _entrypoint(argv: list[str]) -> int:
    """Unexpected failures must not become Python's default exit 1, which is also EXIT_SOURCE_ERROR."""
    try:
        return main(argv)
    except BaseException:
        if len(argv) == 3:
            try:
                _discard_result(argv[2])
            except BaseException:
                pass
        return EXIT_CONTAINMENT_FAILED


if __name__ == "__main__":
    os._exit(_entrypoint(sys.argv[1:]))
