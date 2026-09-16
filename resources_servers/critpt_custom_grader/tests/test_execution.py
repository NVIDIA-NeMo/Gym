# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane tests with a fake backend: nothing is executed or compared on the host.

The fake models a control plane the way the pinned SDK behaves: deletion needs the live sandbox object carried by
the handle, an accepted delete is not destruction, and lookups are exact by id or name. Result files use the real
runner envelope, the codec's wire envelope and the comparator's verdict shape, all written by hand: no value is
encoded, decoded or compared here. Tests marked ``sandbox`` need a Daytona API key and an operator snapshot.
Everything else is host-safe.
"""

import asyncio
import json
import logging
import os
import stat
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from pydantic import ValidationError

from nemo_gym import failure_kinds
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle
from resources_servers.critpt_custom_grader import execution, provider_support, runner, task_data
from resources_servers.critpt_custom_grader.execution import (
    CLEANUP_STATUSES,
    DOMAINS,
    LOOKUP_ERROR_KINDS,
    DaytonaBackend,
    ExecutionPolicy,
    Grader,
    GraderBusy,
    LookupFailed,
    MissingRemoteFile,
    OwnershipJournal,
    SandboxCreateRejected,
    SandboxLookup,
    TransferLimitExceeded,
    _correlate_result,
    _describe_sandbox,
    _Domain,
    _missing_file_error,
    _parse_result,
    _reported_exit,
    _Unscorable,
    build_compare_job,
    build_run_job,
    validate_candidate_source,
)
from resources_servers.critpt_custom_grader.runner import MAX_CODE_LENGTH, MAX_NONCE_LENGTH, RESULT_BYTES_LIMIT
from resources_servers.critpt_custom_grader.task_data import (
    MAX_CASES,
    MAX_JSON_NODES,
    MAX_TEXT_BYTES,
    MAX_WIRE_BYTES,
    WIRE_FORMAT,
    TaskData,
)


CANDIDATE = "def f(x):\n    return x + 1\n"
_MISSING = object()


def make_policy(tmp_path, **overrides):
    fields = {
        "snapshot": "snap-abc123",
        "os_user": "grader",
        "owner_id": "unit",
        "journal_dir": str(tmp_path / "journal"),
    }
    fields.update(overrides)
    return ExecutionPolicy(**fields)


def make_task():
    return TaskData.model_validate(
        {
            "problem_id": "p1",
            "reference_source": "def f(x):\n    return x + 1\n",
            "entrypoint": "f",
            "test_cases": [
                {"args": [wire(["int", "1"])], "expected": value_outcome(["int", "2"])},
                {"args": [wire(["int", "2"])], "expected": value_outcome(["int", "3"])},
            ],
        }
    )


def run_result(status, outcomes, code=None, timed_out=None):
    """A run envelope as ``runner._ResultWriter`` writes it. The fake backend fills in the job nonce."""
    return {
        "version": 1,
        "mode": "run",
        "nonce": "",
        "status": status,
        "code": code,
        "timed_out_case": timed_out,
        "outcomes": outcomes,
    }


def exited(code, **fields):
    """What the provider reports for the runner command: an exact process exit, or a sentinel with ``error_type``."""
    return SandboxExecResult(stdout="", stderr="", return_code=code, **fields)


def verdict(status, side="candidate", code="compared", path=""):
    """One verdict as ``comparator._result`` shapes it: only equal and mismatch carry a boolean ``equal``."""
    return {
        "version": 1,
        "status": status,
        "equal": True if status == "equal" else False if status == "mismatch" else None,
        "side": side,
        "code": code,
        "path": path,
    }


def compare_result(statuses, side="candidate"):
    # Status-specific sides from comparator._result call sites. ``side`` selects the observed role otherwise.
    status_sides = {
        "mismatch": "candidate",
        "invalid_candidate": "candidate",
        "invalid_reference": "reference",
        "invalid_expected": "expected",
    }
    return {
        "version": 1,
        "mode": "compare",
        "nonce": "",
        "status": "completed",
        "code": None,
        "timed_out_case": None,
        "results": [verdict(status, side=status_sides.get(status, side)) for status in statuses],
    }


def wire(node):
    """The codec's wire envelope around a hand-written node. Nothing is encoded on the host."""
    return {"format": WIRE_FORMAT, "value": node}


def value_outcome(node):
    return {"kind": "value", "value": wire(node)}


def nested(levels):
    node = ["int", "1"]
    for _ in range(levels):
        node = ["list", [node]]
    return node


VALUE = value_outcome(["int", "2"])
EXCEPTION = {"kind": "exception", "type": "ValueError"}
ENCODING_ERROR = {"kind": "encoding_error", "code": "unsupported_type"}
# The pre-repair fixture shape: a typed scalar envelope the codec never produced.
LEGACY_VALUE = {"kind": "value", "value": {"format": WIRE_FORMAT, "type": "int", "value": "2"}}


class FakeSandbox:
    """Stand-in for the live SDK object a handle carries: what the fake control plane knows about one sandbox."""

    def __init__(self, sandbox_id, name, labels):
        self.id = sandbox_id
        self.name = name
        self.labels = dict(labels)
        self.state = "started"
        self.polls_until_destroyed = 0


class FakeBackend:
    """Scripted per-domain backend over one shared fake control plane. Records calls and the files received.

    Script keys: ``("create"|"exec"|"result", domain)``, ``("close", sandbox_id)``, ``("lookup", key)`` and
    ``("destroy_delay", sandbox_id)``. A list value is consumed one call at a time, an exception value is raised,
    a callable value is awaited (hangs, cancellation, gates) and a bytes result is delivered verbatim. Deletion
    removes the sandbox from the plane immediately unless a destroy delay keeps it ``destroying`` for that many
    lookups.
    """

    instances: list = []
    plane: dict = {}  # sandbox id -> FakeSandbox
    names: dict = {}  # stable name -> sandbox id

    def __init__(self, policy, script, on_create=None):
        self.script = script
        self.on_create = on_create
        self.calls = []
        self.files = {}
        self.jobs = []
        self.downloaded = []
        self.transfers = 0
        self.downloads = 0
        self.closed = False
        self.domain = None
        FakeBackend.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances, cls.plane, cls.names = [], {}, {}

    @classmethod
    def register(cls, sandbox_id, name, labels):
        sandbox = FakeSandbox(sandbox_id, name, labels)
        cls.plane[sandbox_id] = sandbox
        cls.names[name] = sandbox_id
        return sandbox

    @classmethod
    def destroy(cls, sandbox_id):
        sandbox = cls.plane.pop(sandbox_id, None)
        if sandbox is not None:
            cls.names.pop(sandbox.name, None)

    def _scripted(self, key):
        outcome = self.script.get(key, _MISSING)
        if isinstance(outcome, list):
            return outcome.pop(0) if outcome else _MISSING
        return outcome

    async def create(self, spec):
        self.domain = spec.metadata["ng-grader-domain"]
        self.spec = spec
        if self.on_create is not None:
            self.on_create(self)
        outcome = self._scripted(("create", self.domain))
        if isinstance(outcome, BaseException):
            raise outcome
        self.calls.append("create")
        sandbox_id = f"sb-{self.domain}"
        sandbox = FakeBackend.register(sandbox_id, spec.provider_options["extensions"]["daytona.name"], spec.metadata)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name="daytona", raw=sandbox)

    async def exec(self, handle, command, *, cwd, timeout_s):
        self.calls.append(("exec", command))
        outcome = self._scripted(("exec", self.domain))
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome = await outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is _MISSING or outcome is None:
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        return outcome

    async def write_file(self, handle, target_path, data):
        self.transfers += 1
        self.files[target_path] = data
        if target_path == "/tmp/ng-grader/job.json":
            self.jobs.append(json.loads(data))

    async def download_bounded(self, handle, remote_path, *, limit, timeout_s):
        self.transfers += 1
        self.downloads += 1
        payload = self._scripted(("result", self.domain))
        if callable(payload) and not isinstance(payload, BaseException):
            payload = await payload()
        if isinstance(payload, BaseException):
            raise payload
        if type(payload) is not bytes:
            nonce = json.loads(self.files["/tmp/ng-grader/job.json"])["nonce"]
            payload = json.dumps(dict(payload, nonce=nonce)).encode()
        self.downloaded.append(payload)  # Raw bytes, including stale nonces, are never rewritten.
        return payload

    async def close(self, handle):
        if handle.raw is None:
            raise TypeError("the SDK delete needs the live sandbox object, not a bare id")
        outcome = self._scripted(("close", handle.sandbox_id))
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            try:
                await outcome()
            except asyncio.CancelledError:
                self.calls.append(("close_interrupted", handle.sandbox_id))
                raise
        self.calls.append(("close", handle.sandbox_id))
        self.closed = True
        sandbox = FakeBackend.plane.get(handle.sandbox_id)
        if sandbox is None:
            return
        delay = self.script.get(("destroy_delay", handle.sandbox_id), 0)
        if delay:
            sandbox.state, sandbox.polls_until_destroyed = "destroying", delay
        else:
            FakeBackend.destroy(handle.sandbox_id)

    async def lookup(self, key):
        outcome = self._scripted(("lookup", key))
        if outcome is not _MISSING:
            if callable(outcome) and not isinstance(outcome, BaseException):
                outcome = await outcome()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        self.calls.append(("lookup", key))
        sandbox = FakeBackend.plane.get(FakeBackend.names.get(key, key))
        if sandbox is None:
            return None
        if sandbox.state == "destroying":
            sandbox.polls_until_destroyed -= 1
            if sandbox.polls_until_destroyed <= 0:
                FakeBackend.destroy(sandbox.id)
                return None
        return SandboxLookup(
            handle=SandboxHandle(sandbox_id=sandbox.id, provider_name="daytona", raw=sandbox),
            name=sandbox.name,
            labels=dict(sandbox.labels),
            state=sandbox.state,
        )

    async def aclose(self):
        self.calls.append("aclose")


def make_grader(tmp_path, script, on_create=None, **policy_overrides):
    FakeBackend.reset()
    policy = make_policy(tmp_path, **policy_overrides)
    return Grader(policy, backend_factory=lambda p: FakeBackend(p, script, on_create=on_create)), policy


def happy_script():
    return {
        ("result", "reference"): run_result("completed", [VALUE, VALUE]),
        ("result", "comparator"): [
            compare_result(["equal", "equal"], side="reference"),
            compare_result(["equal", "equal"]),
        ],
        ("result", "candidate"): run_result("completed", [VALUE, VALUE]),
    }


def domain_backends():
    return {backend.domain: backend for backend in FakeBackend.instances if backend.domain is not None}


def closes(sandbox_id):
    return sum(backend.calls.count(("close", sandbox_id)) for backend in FakeBackend.instances)


def lookups(key):
    return sum(backend.calls.count(("lookup", key)) for backend in FakeBackend.instances)


def journal_records(tmp_path):
    return [json.loads(path.read_text()) for path in sorted((tmp_path / "journal").glob("*.json"))]


def open_record(
    tmp_path, job_id="x", state="create_uncertain", sandbox_id=None, domain="reference", dispatched_at=None
):
    journal = OwnershipJournal(tmp_path / "journal")
    names = {d: f"ngcg-{job_id}-{d}" for d in DOMAINS}
    labels = {d: {"ng-grader-owner": "unit", "ng-grader-job": job_id, "ng-grader-domain": d} for d in DOMAINS}
    record = journal.open_job(job_id, "unit", names, labels)
    if dispatched_at is not None:
        record["domains"][domain]["dispatched_at"] = dispatched_at
    journal.update(record, domain, state, sandbox_id)
    return journal, record


def identity(entry):
    """A journal entry's reconciliation identity, without the non-deterministic create dispatch timestamp."""
    return {key: value for key, value in entry.items() if key != "dispatched_at"}


async def cancel_self():
    asyncio.current_task().cancel()
    await asyncio.sleep(10)


async def hang():
    await asyncio.sleep(10)


async def wait_until(predicate):
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition not reached")


def stray_tasks():
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]


def assert_job_cleaned(tmp_path, result, *created):
    assert set(domain_backends()) == set(created)
    assert result.cleanup == {domain: "deleted" if domain in created else "not_created" for domain in DOMAINS}
    for domain, backend in domain_backends().items():
        assert backend.calls.count("create") == closes(f"sb-{domain}") == 1
        commands = [call for call in backend.calls if isinstance(call, tuple) and call[0] == "exec"]
        assert len(commands) <= (2 if domain == "comparator" else 1)  # no resubmission
    assert all(backend.calls.count("aclose") == 1 for backend in FakeBackend.instances)
    assert FakeBackend.plane == {} and journal_records(tmp_path) == [] and not stray_tasks()


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(execution, "_ABSENCE_POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(execution, "_ABSENCE_POLL_MAX_S", 0.001)


@pytest.fixture
def sdk():
    """The pinned SDK's error types. The tests that pin the classification skip without the package."""
    return pytest.importorskip("daytona.common.errors")


# --- policy ------------------------------------------------------------------------------------- #


def test_task_fixture_validates_without_legacy_conversion(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("legacy conversion is not part of control-plane tests"))
    monkeypatch.setattr(task_data, "_legacy_wire", forbidden)
    monkeypatch.setattr(runner, "_input_literal", forbidden)
    task = make_task()
    assert len(task.test_cases) == 2
    assert task.test_cases[1].expected.model_dump(mode="json") == value_outcome(["int", "3"])
    forbidden.assert_not_called()


def test_policy_requires_pinned_snapshot_and_rejects_other_providers(tmp_path):
    with pytest.raises(ValidationError):
        make_policy(tmp_path, provider="local")
    with pytest.raises(ValidationError):
        ExecutionPolicy(os_user="grader", owner_id="unit", journal_dir=str(tmp_path))
    with pytest.raises(ValidationError):
        make_policy(tmp_path, network_block_all=False)
    with pytest.raises(ValidationError):
        make_policy(tmp_path, os_user="root")


@pytest.mark.parametrize("field", ["suite_timeout_s", "create_timeout_s", "cleanup_timeout_s", "memory_limit_mib"])
def test_policy_limits_must_be_positive(tmp_path, field):
    with pytest.raises(ValidationError):
        make_policy(tmp_path, **{field: 0})


@pytest.mark.parametrize(
    ("field", "value"),
    [("os_user", "grader"), ("os_user", "_svc-user_2"), ("owner_id", "unit"), ("owner_id", "1team-42")],
)
def test_policy_accepts_well_formed_identifiers(tmp_path, field, value):
    assert getattr(make_policy(tmp_path, **{field: value}), field) == value


@pytest.mark.parametrize("field", ["os_user", "owner_id"])
@pytest.mark.parametrize("value", ["!grader", "Grader", "grader!", "grader\n"])
def test_policy_identifiers_are_anchored_at_both_ends(tmp_path, field, value):
    # Without a start anchor a search would accept the bad prefix. ``$`` alone would accept the trailing newline.
    with pytest.raises(ValidationError):
        make_policy(tmp_path, **{field: value})


def test_policy_os_user_cannot_start_with_a_digit(tmp_path):
    with pytest.raises(ValidationError):
        make_policy(tmp_path, os_user="1grader")


def test_policy_rejects_task_selected_runtime_fields(tmp_path):
    for field in ("image", "mounts", "provider_options", "timeouts", "network_allow_list"):
        with pytest.raises(ValidationError):
            make_policy(tmp_path, **{field: "x"})
    with pytest.raises(ValidationError):
        make_policy(tmp_path, interpreter='/usr/bin/python3"; rm -rf /')


def test_policy_rejects_unsupported_concurrency(tmp_path):
    assert make_policy(tmp_path).max_concurrent_jobs == 1
    with pytest.raises(ValidationError):
        make_policy(tmp_path, max_concurrent_jobs=2)


def test_policy_projects_zero_retries_and_blocked_egress(tmp_path):
    config = make_policy(tmp_path).domain_config()
    assert config["create"]["retries"] == 0
    assert config["operations"]["command_retries"] == 0
    assert config["create"]["network_block_all"] is True and config["create"]["network_allow_list"] is None
    assert config["create"]["ephemeral"] is True and config["create"]["auto_delete_interval"] == 0
    assert config["connection"]["otel_enabled"] is False
    spec = make_policy(tmp_path).sandbox_spec(stable_name="ngcg-x-reference", labels={"a": "b"})
    assert spec.image is None and spec.ttl_s is None
    assert spec.provider_options == {"snapshot_id": "snap-abc123", "extensions": {"daytona.name": "ngcg-x-reference"}}


def test_policy_enables_category_only_provider_diagnostics(tmp_path, monkeypatch):
    from nemo_gym.sandbox.providers.daytona import provider as daytona_provider
    from resources_servers.critpt_custom_grader import provider_support

    sdk = Mock(side_effect=AssertionError("SDK construction is not part of this policy test"))
    monkeypatch.setattr(daytona_provider, "_require_daytona_sdk", sdk)
    policy = make_policy(tmp_path)
    config = policy.domain_config()
    assert config["category_only_diagnostics"] is True
    assert "category_only_diagnostics" not in config["connection"]
    assert "category_only_diagnostics" not in config["create"]
    provider = provider_support.CategoryOnlyDaytonaProvider(**config)
    assert provider._category_only_diagnostics is True
    constructor = Mock(return_value=provider)
    monkeypatch.setattr(provider_support, "CategoryOnlyDaytonaProvider", constructor)
    assert DaytonaBackend(policy)._provider is provider
    constructor.assert_called_once_with(**config)
    sdk.assert_not_called()
    with pytest.raises(ValidationError):
        make_policy(tmp_path, category_only_diagnostics=False)
    spec = policy.sandbox_spec(stable_name="synthetic-owned-name", labels={"synthetic-label": "synthetic-value"})
    assert spec.provider_options == {
        "snapshot_id": "snap-abc123",
        "extensions": {"daytona.name": "synthetic-owned-name"},
    }


class _SessionProcess:
    """A minimal Daytona ``process`` handle for the session-API exec path.

    ``on_execute`` and ``on_get`` are callables invoked with the recorded call so a test can raise or return.
    A returned ``get_session_command`` object carries ``exit_code``. A returned
    ``execute_session_command`` object carries ``cmd_id``.
    """

    def __init__(self, *, on_execute=None, on_get=None):
        self.calls = []
        self._on_execute = on_execute
        self._on_get = on_get

    async def create_session(self, session_id):
        self.calls.append(("create_session", session_id))

    async def execute_session_command(self, session_id, request, timeout=None):
        self.calls.append(("execute_session_command", session_id))
        if self._on_execute is not None:
            return self._on_execute()
        return SimpleNamespace(cmd_id="cmd-8b12")

    async def get_session_command(self, session_id, command_id):
        self.calls.append(("get_session_command", session_id))
        if self._on_get is not None:
            return self._on_get()
        return SimpleNamespace(exit_code=0)

    async def delete_session(self, session_id):
        self.calls.append(("delete_session", session_id))


async def test_session_api_exec_returns_the_reported_process_exit(tmp_path):
    # A command that reports an integer exit returns it as return_code, which the result gate reads through
    # _reported_exit. The session is created for this one command and deleted on the way out.
    process = _SessionProcess(on_get=lambda: SimpleNamespace(exit_code=runner.EXIT_COMPLETED))
    backend = DaytonaBackend(make_policy(tmp_path))
    handle = SandboxHandle("sb-8b12", "daytona", raw=SimpleNamespace(process=process))
    result = await backend.exec(handle, "runner.py run job.json result.json", cwd="/tmp/ng-grader", timeout_s=5)
    assert (result.return_code, result.error_type, result.stdout, result.stderr) == (
        runner.EXIT_COMPLETED,
        None,
        "",
        "",
    )
    assert _reported_exit(result) == runner.EXIT_COMPLETED
    kinds = [name for name, _ in process.calls]
    assert kinds == ["create_session", "execute_session_command", "get_session_command", "delete_session"]
    session_id = process.calls[0][1]
    assert session_id.startswith("critpt-grade-") and all(sid == session_id for _, sid in process.calls)


async def test_session_api_exec_keeps_failure_attribution(tmp_path, caplog):
    # A session error carrying provider text propagates unchanged, so a provider failure never becomes a spoofable
    # process exit that could authorize a result download. The created session is still deleted, and run() maps the
    # exec exception to nonterminal provider uncertainty that leaks none of the SDK error's text.
    marker = "SYNTHETIC-PROVIDER-DIAGNOSTIC-8b12"
    sandbox_id = "synthetic-sandbox-id-8b12"
    error_class = type("SyntheticSessionError8b12", (Exception,), {})
    error = error_class(f"Failed to execute command: {marker} https://synthetic.invalid/8b12 /synthetic-path-8b12")

    def raise_error():
        raise error

    process = _SessionProcess(on_execute=raise_error)
    backend = DaytonaBackend(make_policy(tmp_path))
    handle = SandboxHandle(sandbox_id, "daytona", raw=SimpleNamespace(process=process))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(error_class) as caught:
            await backend.exec(handle, "/synthetic-path-8b12", cwd="/synthetic-path-8b12", timeout_s=5)
    assert caught.value is error  # the provider exception, not a grader-built wrapper that could re-embed its text
    # The command failed to start, so it never advanced to polling, and the created session is deleted.
    kinds = [name for name, _ in process.calls]
    assert kinds == ["create_session", "execute_session_command", "delete_session"]
    session_id = process.calls[0][1]
    assert session_id.startswith("critpt-grade-") and all(sid == session_id for _, sid in process.calls)
    # A session failure is nonterminal provider uncertainty, never a scored task verdict.
    assert execution.UNSCORABLE_CATEGORIES["provider_exec"] == (failure_kinds.PROVIDER_UNAVAILABLE, False)
    # The grader logs nothing on this path, so none of the SDK error's sensitive text can leak through it.
    blob = repr([vars(record) for record in caplog.records])
    for sentinel in (marker, sandbox_id, error_class.__name__, "https://synthetic.invalid/8b12"):
        assert sentinel not in blob


async def test_cancelled_create_keeps_a_leaked_sandbox_id(tmp_path):
    # A create cancelled during its own cleanup can still name a live sandbox on the CancelledError. The create
    # must keep that id and journal ``delete_failed``, the same as the ordinary-exception branch, so cleanup can
    # delete the sandbox by its exact id rather than only by stable name.
    FakeBackend.reset()
    policy = make_policy(tmp_path)
    sandbox_id = "synthetic-cancelled-id-9c31"
    error = asyncio.CancelledError()
    error.sandbox_id = sandbox_id
    journal, record = open_record(tmp_path, state="planned")
    backend = FakeBackend(policy, {("create", "reference"): error})
    domain = _Domain("reference", policy, backend, journal, record)
    with pytest.raises(asyncio.CancelledError):
        await domain.create()
    entry = record["domains"]["reference"]
    assert entry["state"] == "delete_failed" and entry["sandbox_id"] == sandbox_id
    FakeBackend.register(sandbox_id, entry["name"], entry["labels"])
    assert await domain.close() == "deleted"
    assert entry["state"] == "deleted" and FakeBackend.plane == {}


async def test_cancelled_create_without_a_leaked_id_stays_create_uncertain(tmp_path):
    # A plain cancellation with no id attached stays ``create_uncertain``, so the ambiguous-create settle path
    # still governs it.
    FakeBackend.reset()
    policy = make_policy(tmp_path)
    journal, record = open_record(tmp_path, state="planned")
    backend = FakeBackend(policy, {("create", "reference"): asyncio.CancelledError()})
    domain = _Domain("reference", policy, backend, journal, record)
    with pytest.raises(asyncio.CancelledError):
        await domain.create()
    entry = record["domains"]["reference"]
    assert entry["state"] == "create_uncertain" and entry["sandbox_id"] is None


async def test_category_only_cleanup_identity_stays_in_protected_journal(tmp_path):
    from resources_servers.critpt_custom_grader.provider_support import DaytonaCreateCleanupError

    FakeBackend.reset()
    policy = make_policy(tmp_path)
    sandbox_id = "synthetic-unresolved-id-8b12"
    error = DaytonaCreateCleanupError("daytona.create_cleanup_failed", sandbox_id=sandbox_id)
    journal, record = open_record(tmp_path, state="planned")
    backend = FakeBackend(policy, {("create", "reference"): error})
    domain = _Domain("reference", policy, backend, journal, record)
    with pytest.raises(_Unscorable, match="^provider_create$"):
        await domain.create()
    entry = record["domains"]["reference"]
    assert entry["state"] == "delete_failed" and entry["sandbox_id"] == sandbox_id
    assert journal.unresolved()[0]["domains"]["reference"] == entry
    assert stat.S_IMODE(journal.directory.stat().st_mode) == 0o700
    assert entry["name"] == "ngcg-x-reference"
    assert entry["labels"] == {"ng-grader-owner": "unit", "ng-grader-job": "x", "ng-grader-domain": "reference"}
    FakeBackend.register(sandbox_id, entry["name"], entry["labels"])
    assert await domain.close() == "deleted"
    assert backend.calls.count(("lookup", sandbox_id)) == 2
    assert backend.calls.count(("close", sandbox_id)) == 1
    assert entry["sandbox_id"] == sandbox_id and entry["state"] == "deleted"
    assert journal.resolve(record) is True and FakeBackend.plane == {}


def test_suite_timeout_default_preserves_documented_budget(tmp_path):
    assert make_policy(tmp_path).suite_timeout_s == 1800


def test_job_deadline_budget_counts_every_awaited_bound(tmp_path):
    fields = {
        "suite_timeout_s": 100,
        "compare_timeout_s": 10,
        # At the runner cleanup-budget floor (CLEANUP_BUDGET_S = 15 s), so two margins are 30 s.
        "exec_timeout_margin_s": 15.0,
        "cpu_time_limit_s": 50,
        "create_timeout_s": 2.0,
        "transfer_timeout_s": 3.0,
        "cleanup_timeout_s": 4.0,
    }
    policy = make_policy(tmp_path, **fields)
    # Two runs and two comparisons at deadline plus two margins, three creates at create plus four transfers, the 23
    # file transfers at transfer plus grace, and three mid-job deletes each followed by absence confirmation.
    expected = 2 * (100 + 30) + 2 * (10 + 30) + 3 * (2 + 12) + 23 * (3 + 5) + 3 * 2 * 4
    assert execution._JOB_TRANSFERS == 23
    assert policy.minimum_job_timeout_s() == expected == 590
    assert policy.effective_job_timeout_s() == expected
    assert make_policy(tmp_path, job_timeout_s=float(expected), **fields).effective_job_timeout_s() == expected
    with pytest.raises(ValidationError):
        make_policy(tmp_path, job_timeout_s=expected - 1.0, **fields)


# --- job flow ----------------------------------------------------------------------------------- #


def test_outcome_categories_separate_observations_from_scored_contracts():
    assert set(execution.FAILED_CATEGORIES) == {
        "candidate_mismatch",
        "candidate_source_limit",
        "candidate_invalid_output",
        "candidate_source_error",
        "candidate_timeout",
        "candidate_exception",
        "candidate_encoding_error",
        "candidate_worker_abort",
    }
    # The candidate-attributable faults observed after a completed preflight are scored, not unscorable.
    assert set(execution.CANDIDATE_FAULT_CATEGORIES) == {
        "candidate_source_error",
        "candidate_timeout",
        "candidate_exception",
        "candidate_encoding_error",
        "candidate_worker_abort",
    }
    assert set(execution.CANDIDATE_FAULT_CATEGORIES) <= set(execution.FAILED_CATEGORIES)
    assert set(execution.FAILED_CATEGORIES).isdisjoint(execution.UNSCORABLE_CATEGORIES)
    terminal = {category for category, (_, terminal) in execution.UNSCORABLE_CATEGORIES.items() if terminal}
    assert terminal == {"task_invalid", "reference_mismatch"}
    for category in terminal:
        assert execution.UNSCORABLE_CATEGORIES[category] == (failure_kinds.VERIFIER_ERROR, True)
    for category in execution.UNSCORABLE_CATEGORIES.keys() - terminal:
        assert execution.UNSCORABLE_CATEGORIES[category] == (failure_kinds.PROVIDER_UNAVAILABLE, False)


async def test_passing_candidate_scores_one_and_cleans_all_domains(tmp_path):
    grader, _ = make_grader(tmp_path, happy_script())
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("passed", 1.0, "passed")
    assert result.cases_attempted == result.cases_equal == 2
    assert result.cleanup == {"reference": "deleted", "comparator": "deleted", "candidate": "deleted"}
    assert all(backend.closed for backend in FakeBackend.instances)
    # Each domain is deleted exactly once: the final sweep is a no-op after a confirmed mid-job close.
    assert closes("sb-reference") == closes("sb-comparator") == closes("sb-candidate") == 1
    assert all(lookups(f"sb-{domain}") >= 1 for domain in DOMAINS)
    assert FakeBackend.plane == {}
    assert not list((tmp_path / "journal").glob("*.json"))


async def test_transfer_budget_matches_the_transfers_a_job_performs(tmp_path):
    grader, _ = make_grader(tmp_path, happy_script())
    assert (await grader.grade(make_task(), CANDIDATE)).reward == 1.0
    per_domain = {backend.domain: backend.transfers for backend in domain_backends().values()}
    assert per_domain == {"reference": 7, "comparator": 9, "candidate": 7}
    assert sum(per_domain.values()) == execution._JOB_TRANSFERS


@pytest.mark.parametrize("code", runner.SOURCE_ERROR_CODES)
async def test_preflight_failure_never_allocates_candidate(tmp_path, code):
    script = happy_script()
    script[("result", "reference")] = run_result("source_error", [], code=code)
    script[("exec", "reference")] = exited(runner.EXIT_SOURCE_ERROR)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_error")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, "reference")


@pytest.mark.parametrize("observation", [EXCEPTION, ENCODING_ERROR], ids=["exception", "encoding_error"])
@pytest.mark.parametrize("statuses", [["equal", "equal"], ["invalid_reference", "equal"]])
async def test_reference_non_value_observations_never_reach_preflight(tmp_path, observation, statuses):
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, observation])
    preflight = compare_result(statuses, side="reference")
    script[("result", "comparator")] = [preflight]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_error")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert script[("result", "comparator")] == [preflight]  # neither a pass nor an earlier defect may be used
    assert_job_cleaned(tmp_path, result, "reference")


async def test_reference_mismatch_is_unscorable_not_candidate_fault(tmp_path):
    script = happy_script()
    script[("result", "comparator")] = [compare_result(["equal", "invalid_reference"], side="reference")]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "reference_mismatch" and result.reward is None
    assert result.failure_class == failure_kinds.VERIFIER_ERROR and result.terminal is True
    assert_job_cleaned(tmp_path, result, "reference", "comparator")


@pytest.mark.parametrize(
    ("status", "category", "terminal"),
    [
        ("invalid_reference", "reference_mismatch", True),
        ("invalid_expected", "task_invalid", True),
        ("uncertain", "comparator_uncertain", False),
        ("mismatch", "result_invalid", False),  # candidate-role verdicts cannot judge the reference
        ("invalid_candidate", "result_invalid", False),
    ],
)
async def test_reference_preflight_maps_every_comparator_verdict(tmp_path, status, category, terminal):
    script = happy_script()
    script[("result", "comparator")] = [compare_result(["equal", status], side="reference")]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        category,
        terminal,
    )
    assert result.failure_class == (failure_kinds.VERIFIER_ERROR if terminal else failure_kinds.PROVIDER_UNAVAILABLE)
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, "reference", "comparator")
    # The preflight judges observed reference outcomes against the stored expectations, never the other way round.
    compare_job = json.loads(domain_backends()["comparator"].files["/tmp/ng-grader/job.json"])
    task = make_task()
    assert compare_job["observed_role"] == "reference"
    assert [case["observed"] for case in compare_job["cases"]] == [VALUE, VALUE]
    assert [case["expected"] for case in compare_job["cases"]] == [
        case.expected.model_dump(mode="json") for case in task.test_cases
    ]


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("status", runner.VERDICT_STATUSES)
@pytest.mark.parametrize(
    "side", ["reference", "candidate", "expected", "input", "request", "comparator", None, "other"]
)
def test_verdict_batch_role_status_side_contract(role, status, side):
    # Hand-transcribed comparator._result call sites. No comparator import or scientific work.
    valid = {
        "reference": {
            ("equal", "reference"),
            ("invalid_reference", "reference"),
            ("invalid_expected", "expected"),
            ("uncertain", "reference"),
            ("uncertain", "comparator"),
            ("uncertain", "request"),
        },
        "candidate": {
            ("equal", "candidate"),
            ("mismatch", "candidate"),
            ("invalid_candidate", "candidate"),
            ("invalid_expected", "expected"),
            ("uncertain", "candidate"),
            ("uncertain", "comparator"),
            ("uncertain", "request"),
        },
    }
    results = [verdict(status, side=side)]
    if (status, side) in valid[role]:
        assert execution._validate_verdict_batch(results, role, attempted=2) is None
    else:
        with pytest.raises(_Unscorable) as info:
            execution._validate_verdict_batch(results, role, attempted=2)
        assert (info.value.category, info.value.attempted) == ("result_invalid", 2)


@pytest.mark.parametrize(
    ("role", "defect", "invalid"),
    [
        ("reference", "invalid_reference", "mismatch"),
        ("reference", "invalid_reference", "invalid_candidate"),
        ("reference", "invalid_expected", "mismatch"),
        ("reference", "invalid_expected", "invalid_candidate"),
        ("candidate", "mismatch", "invalid_reference"),
        ("candidate", "invalid_candidate", "invalid_reference"),
        ("candidate", "invalid_expected", "invalid_reference"),
    ],
)
@pytest.mark.parametrize("order", list(permutations(range(3))))
async def test_invalid_comparator_role_precedes_disposition_in_every_order(tmp_path, role, defect, invalid, order):
    data = make_task().model_dump(mode="json")
    data["test_cases"].append(data["test_cases"][-1])
    task = TaskData.model_validate(data)
    script = happy_script()
    script[("result", "reference")] = script[("result", "candidate")] = run_result("completed", [VALUE] * 3)
    statuses = ["equal", defect, invalid]
    report = compare_result([statuses[index] for index in order], side=role)
    script[("result", "comparator")] = (
        [report] if role == "reference" else [compare_result(["equal"] * 3, side="reference"), report]
    )
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 3)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, *(DOMAINS if role == "candidate" else ("reference", "comparator")))


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize("side", ["other_role", "expected", "input", "request", "comparator", None])
@pytest.mark.parametrize("wrong_index", [0, 1, None], ids=["first", "last", "all"])
async def test_all_equal_comparator_reports_require_the_observed_role(tmp_path, role, side, wrong_index):
    if side == "other_role":
        side = "candidate" if role == "reference" else "reference"
    report = compare_result(["equal", "equal"], side=role)
    for index in range(2):
        if wrong_index is None or wrong_index == index:
            report["results"][index]["side"] = side
    script = happy_script()
    if role == "reference":
        script[("result", "comparator")] = [report]
    else:
        script[("result", "comparator")][1] = report
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, *(DOMAINS if role == "candidate" else ("reference", "comparator")))


async def test_domain_separation_and_expected_metadata_projection(tmp_path):
    grader, _ = make_grader(tmp_path, happy_script())
    await grader.grade(make_task(), CANDIDATE)
    backends = domain_backends()
    assert len({id(b) for b in backends.values()}) == 3
    candidate_files = backends["candidate"].files
    assert "/tmp/ng-grader/critpt_grader_rt/comparator.py" not in candidate_files
    candidate_job = json.loads(candidate_files["/tmp/ng-grader/job.json"])
    assert set(candidate_job) == {
        "version",
        "mode",
        "nonce",
        "entrypoint",
        "source_path",
        "input_conversions",
        "cases",
        "limits",
    }
    assert candidate_job["input_conversions"] == []  # make_task declares none, so the key is always present
    assert all(set(case) == {"args", "kwargs"} for case in candidate_job["cases"])
    assert candidate_files["/tmp/ng-grader/source.py"] == CANDIDATE.encode()
    assert "source.py" not in "".join(backends["comparator"].files)
    compare_job = json.loads(backends["comparator"].files["/tmp/ng-grader/job.json"])
    task = make_task()
    assert compare_job["observed_role"] == "candidate"
    assert compare_job["cases"][1]["expected"] == task.test_cases[1].expected.model_dump(mode="json")
    assert compare_job["cases"][1]["policy"] == task.comparison_policy(1)
    for backend in backends.values():
        command = backend.calls[1][1]
        assert 'runner.py" ' in command
        assert command.startswith('runuser -u grader -- "/usr/bin/python3" -I -B')


@pytest.mark.asyncio
async def test_configured_server_default_tolerance_reaches_the_comparator(tmp_path):
    # The operator-set silent-leaf default is plumbed app.py -> Grader -> build_compare_job ->
    # comparison_policy, so a non-default yaml value reaches the comparator inside the per-case policy.
    FakeBackend.reset()
    policy = make_policy(tmp_path)
    grader = Grader(
        policy,
        backend_factory=lambda p: FakeBackend(p, happy_script()),
        default_rtol="2e-4",
        default_atol="3e-9",
    )
    await grader.grade(make_task(), CANDIDATE)
    task = make_task()
    comparator_jobs = domain_backends()["comparator"].jobs
    assert [job["observed_role"] for job in comparator_jobs] == ["reference", "candidate"]
    for compare_job in comparator_jobs:
        for index, case in enumerate(compare_job["cases"]):
            assert case["policy"]["default_rtol"] == "2e-4"
            assert case["policy"]["default_atol"] == "3e-9"
            assert case["policy"] == task.comparison_policy(index, "2e-4", "3e-9")


async def test_run_command_drops_privileges_to_configured_os_user(tmp_path):
    # The provider exec runs as root and the runner refuses root, so every domain runs under a runuser drop to
    # the policy os_user. This asserts the configured user reaches the built command, not the default.
    grader, _ = make_grader(tmp_path, happy_script(), os_user="_svc-user_2")
    await grader.grade(make_task(), CANDIDATE)
    backends = domain_backends()
    assert len(backends) == 3
    for backend in backends.values():
        command = backend.calls[1][1]
        assert command.startswith("runuser -u _svc-user_2 -- ")
        assert command.endswith(">/dev/null 2>&1")
        assert '"/usr/bin/python3" -I -B' in command


@pytest.mark.parametrize("value", ["-grader", "gra der", "grader;ls", "grader$x", "grader|x"])
def test_policy_rejects_shell_unsafe_os_user(tmp_path, value):
    # os_user is embedded in the runuser command, so a leading hyphen or any shell metacharacter must be rejected.
    with pytest.raises(ValidationError):
        make_policy(tmp_path, os_user=value)


async def test_candidate_mismatch_scores_zero_with_first_failed_case(tmp_path):
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, value_outcome(["int", "3"])])
    script[("result", "candidate")] = run_result("completed", [VALUE, value_outcome(["int", "4"])])
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["equal", "mismatch"]),
    ]
    grader, _ = make_grader(tmp_path, script)
    source = "def f(x):\n    return x + (1 if x == 1 else 2)\n"
    result = await grader.grade(make_task(), source)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_mismatch")
    assert result.first_failed_case == result.cases_equal == 1 and result.cases_attempted == 2
    assert result.failure_class is None and result.terminal is None
    compare_job = json.loads(domain_backends()["comparator"].files["/tmp/ng-grader/job.json"])
    assert compare_job["cases"][1]["observed"] == value_outcome(["int", "4"])
    assert compare_job["cases"][1]["expected"] == {"kind": "value", "value": wire(["int", "3"])}
    assert_job_cleaned(tmp_path, result, *DOMAINS)


async def test_comparator_validated_invalid_candidate_value_still_scores_zero(tmp_path):
    script = happy_script()
    invalid_value = value_outcome(["int", "not-an-integer"])
    script[("result", "candidate")] = run_result("completed", [VALUE, invalid_value])
    script[("result", "comparator")][1] = compare_result(["equal", "invalid_candidate"])
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_invalid_output")
    assert result.first_failed_case == result.cases_equal == 1 and result.cases_attempted == 2
    assert result.failure_class is None and result.terminal is None
    compare_job = json.loads(domain_backends()["comparator"].files["/tmp/ng-grader/job.json"])
    assert compare_job["cases"][1]["observed"] == invalid_value  # the host forwards it without decoding
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize(
    ("status", "outcome", "category", "first"),
    [
        ("invalid_candidate", "failed", "candidate_invalid_output", 1),
        ("uncertain", "unscorable", "comparator_uncertain", None),
        ("invalid_expected", "unscorable", "comparator_uncertain", None),
        # A reference-role verdict cannot judge the candidate.
        ("invalid_reference", "unscorable", "result_invalid", None),
    ],
)
async def test_candidate_verdicts_map_to_bounded_categories(tmp_path, status, outcome, category, first):
    script = happy_script()
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["equal", status]),
    ]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.first_failed_case) == (outcome, category, first)
    assert result.reward == (0.0 if outcome == "failed" else None)
    assert result.failure_class == (None if outcome == "failed" else failure_kinds.PROVIDER_UNAVAILABLE)
    assert result.terminal is (None if outcome == "failed" else False)
    assert result.cases_attempted == 2
    assert result.cases_equal == (1 if outcome == "failed" else 0)
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize(
    ("observation", "category"),
    [(EXCEPTION, "candidate_exception"), (ENCODING_ERROR, "candidate_encoding_error")],
    ids=["exception", "encoding_error"],
)
@pytest.mark.parametrize(
    "statuses",
    [["equal", "equal"], ["equal", "mismatch"], ["equal", "invalid_candidate"], ["mismatch", "equal"]],
)
async def test_candidate_non_value_observations_score_zero(tmp_path, observation, category, statuses):
    script = happy_script()
    script[("result", "candidate")] = run_result("completed", [VALUE, observation])
    judged = compare_result(statuses)
    script[("result", "comparator")][1] = judged
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A candidate that raises or returns an unencodable object after a completed preflight is a
    # candidate-attributable fault, scored 0 at the first offending case, never an authenticated pass.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, category)
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case == 1
    assert script[("result", "comparator")] == [judged]  # no verdict on the candidate was consumed
    assert domain_backends()["comparator"].downloads == 1
    assert_job_cleaned(tmp_path, result, *DOMAINS)


def _task_with_exception_case():
    # A two-case task whose second case expects an exception. The historical grader scored such a case: the code
    # must raise, and neither the exception type nor its message is compared.
    data = make_task().model_dump(mode="json")
    data["test_cases"][1]["expected"] = {"kind": "exception"}
    return TaskData.model_validate(data)


@pytest.mark.parametrize("domain", ["reference", "candidate"])
async def test_expected_exception_case_admits_a_raised_observation(tmp_path, domain):
    # An exception observed where the stored expectation is an exception is admitted to comparison, not refused.
    # The verdicts are scripted equal, so this is control-flow evidence that the observation reaches comparison,
    # not a qualification of the comparator's own exception rule.
    task = _task_with_exception_case()
    script = happy_script()
    script[("result", domain)] = run_result("completed", [VALUE, EXCEPTION])
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("passed", 1.0, "passed")
    assert result.cases_attempted == 2 and result.cases_equal == 2 and result.first_failed_case is None
    # Every scripted verdict was consumed: preflight over the reference and the judge over the candidate both ran.
    assert script[("result", "comparator")] == []
    assert_job_cleaned(tmp_path, result, *DOMAINS)


async def test_expected_exception_reference_that_does_not_raise_is_a_reference_defect(tmp_path):
    # A reference that returns a value where the case expects an exception is a reference defect. The value is
    # admitted to preflight, and the comparator's invalid_reference verdict maps to the terminal reference_mismatch,
    # the same as any reference that contradicts its stored expectation. No candidate is created.
    task = _task_with_exception_case()
    script = happy_script()  # reference returns [VALUE, VALUE]: it did not raise on the exception case
    script[("result", "comparator")][0] = compare_result(["equal", "invalid_reference"], side="reference")
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "reference_mismatch")
    assert result.failure_class == failure_kinds.VERIFIER_ERROR and result.terminal is True
    assert result.cases_attempted == 0 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, "reference", "comparator")


async def test_expected_exception_candidate_that_does_not_raise_is_a_wrong_answer(tmp_path):
    # A candidate that returns a value where the case expects an exception did not raise. The value is admitted to
    # the judge, and the comparator's mismatch verdict scores it 0 as candidate_mismatch at that case.
    task = _task_with_exception_case()
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [VALUE, EXCEPTION])  # reference raises as required
    script[("result", "candidate")] = run_result("completed", [VALUE, VALUE])  # candidate did not raise
    script[("result", "comparator")][1] = compare_result(["equal", "mismatch"])
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(task, CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_mismatch")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 1 and result.first_failed_case == 1
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize("code", runner.SOURCE_ERROR_CODES)
async def test_candidate_source_error_reports_are_not_host_source_contracts(tmp_path, code):
    script = happy_script()
    script[("result", "candidate")] = run_result("source_error", [], code=code)
    script[("exec", "candidate")] = exited(runner.EXIT_SOURCE_ERROR)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A source-error report after a completed preflight scores 0 with the category preserved. The bounded code
    # and correlated exit still do not authenticate the cause, so a host source contract is not implied.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_error")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert domain_backends()["comparator"].downloads == 1
    assert_job_cleaned(tmp_path, result, *DOMAINS)


async def test_candidate_timeout_correlation_is_not_fault_attribution(tmp_path):
    script = happy_script()
    script[("result", "candidate")] = run_result("timeout", [VALUE], timed_out=1)
    script[("exec", "candidate")] = exited(runner.EXIT_TIMEOUT)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    # A suite timeout after a completed preflight scores 0 with the category preserved. Correlation is not
    # authentication, so the category names what the candidate domain reported, not a proven cause.
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_timeout")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 1 and result.cases_equal == 0 and result.first_failed_case is None
    assert domain_backends()["comparator"].downloads == 1
    assert_job_cleaned(tmp_path, result, *DOMAINS)
    # The same payload behind the provider's exit 0 disagrees with its own status: uncorrelated, never scored.
    script = happy_script()
    script[("result", "candidate")] = run_result("timeout", [VALUE], timed_out=1)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize("exit_code", sorted(runner.RESULT_EXIT_CODES))
async def test_missing_candidate_result_is_infrastructure_uncertainty(tmp_path, exit_code):
    # Behind every result exit, EXIT_TIMEOUT included, an absent file is not a candidate timeout or any verdict.
    script = happy_script()
    script[("result", "candidate")] = MissingRemoteFile()
    script[("exec", "candidate")] = exited(exit_code)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "candidate_no_result" and result.reward is None
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert domain_backends()["candidate"].downloads == 1


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (TransferLimitExceeded(), "transfer_limit"),
        (ConnectionResetError("reset"), "provider_exec"),
        (asyncio.TimeoutError(), "provider_exec"),
        (hang, "provider_exec"),
    ],
    ids=["limit", "transport", "timeout", "hang"],
)
async def test_result_transfer_failures_are_bounded_and_never_scored(tmp_path, monkeypatch, failure, category):
    monkeypatch.setattr(execution, "_TRANSFER_GRACE_S", 0.01)
    script = happy_script()
    script[("result", "candidate")] = failure
    grader, _ = make_grader(tmp_path, script, transfer_timeout_s=0.05)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, category)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS} and not stray_tasks()


async def test_malformed_result_files_are_result_invalid_and_stop_the_job(tmp_path):
    script = happy_script()
    script[("result", "reference")] = run_result("completed", [LEGACY_VALUE, LEGACY_VALUE])
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.terminal) == ("unscorable", "result_invalid", False)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert set(domain_backends()) == {"reference"} and result.cleanup["reference"] == "deleted"

    script = happy_script()
    script[("result", "candidate")] = b"\xff{not json"
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.cases_attempted, result.cases_equal) == ("result_invalid", 0, 0)

    script = happy_script()
    script[("result", "candidate")] = run_result("completed", [VALUE])  # one outcome for two cases
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.cases_attempted) == ("result_invalid", 1)
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


async def test_comparator_uncertain_is_unscorable(tmp_path):
    script = happy_script()
    script[("result", "comparator")] = [
        compare_result(["equal", "equal"], side="reference"),
        compare_result(["uncertain", "equal"]),
    ]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, *DOMAINS)


# A preflight timeout (reference role) or a comparator runner error on either side stays unattributed
# comparator uncertainty. The candidate judge-stage timeout is the one runtime report that becomes a candidate
# fault, covered separately in test_candidate_judge_timeout_scores_invalid_output.
@pytest.mark.parametrize(
    ("role", "status", "code", "exit_code"),
    [
        ("reference", "timeout", None, runner.EXIT_TIMEOUT),
        ("reference", "runner_error", "limits", runner.EXIT_RUNNER_ERROR),
        ("candidate", "runner_error", "limits", runner.EXIT_RUNNER_ERROR),
    ],
)
async def test_comparator_runtime_reports_are_nonterminal(tmp_path, role, status, code, exit_code):
    script = happy_script()
    report = {**compare_result([]), "status": status, "code": code}
    if role == "reference":
        script[("result", "comparator")] = [report]
        script[("exec", "comparator")] = exited(exit_code)
    else:
        script[("result", "comparator")][1] = report
        script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(exit_code)]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert domain_backends()["comparator"].downloads == (1 if role == "reference" else 2)
    if role == "reference":
        assert_job_cleaned(tmp_path, result, "reference", "comparator")
    else:
        assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize("timed_out_case", [1, None], ids=["reported_case", "no_case"])
async def test_candidate_judge_timeout_scores_invalid_output(tmp_path, timed_out_case):
    # A judge-stage timeout after a completed preflight is candidate-attributable: the comparator resolved the
    # reference/expected side cleanly at preflight, so a candidate value it cannot resolve within the CPU budget
    # scores 0 as candidate_invalid_output, at the case the runner reported it timed out on when one is given.
    script = happy_script()
    script[("result", "comparator")][1] = {**compare_result([]), "status": "timeout", "timed_out_case": timed_out_case}
    script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(runner.EXIT_TIMEOUT)]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_invalid_output")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == 0 and result.first_failed_case == timed_out_case
    assert domain_backends()["comparator"].downloads == 2
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize(
    ("role", "first", "uncertain"),
    [
        ("reference", "invalid_reference", "uncertain"),
        ("reference", "invalid_expected", "uncertain"),
        ("candidate", "mismatch", "uncertain"),
        ("candidate", "invalid_candidate", "uncertain"),
        ("candidate", "mismatch", "invalid_expected"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["defect_first", "uncertainty_first"])
@pytest.mark.parametrize("uncertain_side", ["observed_role", "comparator", "request"])
async def test_later_comparator_uncertainty_cannot_be_hidden_by_a_defect(
    tmp_path, role, first, uncertain, reverse, uncertain_side
):
    script = happy_script()
    report = compare_result([first, uncertain], side=role)
    if uncertain == "uncertain":
        report["results"][1]["side"] = role if uncertain_side == "observed_role" else uncertain_side
    if reverse:
        report["results"].reverse()
    if role == "reference":
        script[("result", "comparator")] = [report]
    else:
        script[("result", "comparator")][1] = report
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_uncertain")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == (0 if role == "reference" else 2)
    assert result.cases_equal == 0 and result.first_failed_case is None
    if role == "reference":
        assert_job_cleaned(tmp_path, result, "reference", "comparator")
    else:
        assert_job_cleaned(tmp_path, result, *DOMAINS)


async def test_candidate_source_contract_checked_before_allocation(tmp_path):
    grader, policy = make_grader(tmp_path, happy_script())
    result = await grader.grade(make_task(), "x" * (policy.max_candidate_source_bytes + 1))
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_limit")
    assert result.failure_class is None and result.terminal is None and result.cleanup == {}
    assert FakeBackend.instances == [] and journal_records(tmp_path) == []
    assert validate_candidate_source("def f():\n\x00", policy) == "candidate_source_limit"
    assert validate_candidate_source(CANDIDATE, policy) is None


@pytest.mark.parametrize(
    "source",
    [None, b"source", "", " \n", "x\x00", "\ud800", "é" * 5],
    ids=["not-text", "bytes", "empty", "blank", "nul", "invalid-utf8", "utf8-byte-limit"],
)
async def test_host_source_contract_violations_remain_scored_without_allocation(tmp_path, source):
    grader, policy = make_grader(tmp_path, happy_script(), max_candidate_source_bytes=8)
    assert validate_candidate_source(source, policy) == "candidate_source_limit"
    result = await grader.grade(make_task(), source)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_source_limit")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert result.cleanup == {} and FakeBackend.instances == [] and journal_records(tmp_path) == []
    assert validate_candidate_source("é" * 4, policy) is None  # exactly eight bytes, and source is never parsed here


async def test_cancellation_cleans_created_domains(tmp_path):
    script = happy_script()
    script[("exec", "candidate")] = asyncio.CancelledError()
    grader, _ = make_grader(tmp_path, script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    assert all(backend.closed for backend in domain_backends().values())
    assert closes("sb-comparator") == closes("sb-candidate") == 1
    assert not list((tmp_path / "journal").glob("*.json"))


async def test_journal_persists_stable_name_before_create(tmp_path):
    seen = {}

    def on_create(backend):
        record = json.loads(next((tmp_path / "journal").glob("*.json")).read_text())
        seen[backend.domain] = record["domains"][backend.domain]["state"]

    grader, _ = make_grader(tmp_path, happy_script(), on_create=on_create)
    await grader.grade(make_task(), CANDIDATE)
    assert seen == {"reference": "creating", "comparator": "creating", "candidate": "creating"}


async def test_journal_write_makes_rename_and_unlink_durable(tmp_path, monkeypatch):
    synced_directories = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        synced_directories.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    journal, record = open_record(tmp_path, state="planned")
    assert synced_directories.count(True) >= 2  # open_job and the update each fsync the directory after rename
    synced_directories.clear()
    assert journal.resolve(record) is True
    assert synced_directories == [True]  # the unlink is made durable too


def test_journal_reports_malformed_entries_as_corrupt(tmp_path):
    journal = OwnershipJournal(tmp_path / "journal")
    (tmp_path / "journal" / "bad1.json").write_text(
        json.dumps({"version": 1, "job_id": "bad1", "domains": {"reference": {"state": "created"}}})
    )
    (tmp_path / "journal" / "bad2.json").write_text(json.dumps({"version": 1, "job_id": "other", "domains": {}}))
    assert journal.unresolved() == [{"job_id": "bad1", "corrupt": True}, {"job_id": "bad2", "corrupt": True}]


# --- result shape validation (host-side, values never decoded) ---------------------------------- #


def parse(payload, mode="run"):
    data = payload if type(payload) is bytes else json.dumps(payload).encode()
    return _parse_result(data, mode)


def rejected(payload, mode="run"):
    with pytest.raises(_Unscorable) as info:
        parse(payload, mode)
    return info.value.category == "result_invalid"


def mutate(base, **changes):
    return {**base, **changes}


def without(base, key):
    return {name: value for name, value in base.items() if name != key}


GOOD_RUN = run_result("completed", [VALUE, EXCEPTION, ENCODING_ERROR])
GOOD_COMPARE = compare_result(
    ["equal", "mismatch", "uncertain", "invalid_candidate", "invalid_expected", "invalid_reference"]
)

MALFORMED_RUN = [
    pytest.param(b"{", id="not json"),
    pytest.param(b"\xff\xfe", id="not utf-8"),
    pytest.param(b"[" * 100_000, id="nested past the parser's recursion limit"),
    pytest.param(b"[]", id="not an object"),
    pytest.param(mutate(GOOD_RUN, extra=1), id="extra envelope key"),
    pytest.param(without(GOOD_RUN, "code"), id="missing envelope key"),
    pytest.param(mutate(without(GOOD_RUN, "outcomes"), results=[]), id="results key in run mode"),
    pytest.param(mutate(GOOD_RUN, version=2), id="wrong version"),
    pytest.param(mutate(GOOD_RUN, version="1"), id="text version"),
    pytest.param(mutate(GOOD_RUN, version=True), id="boolean version"),
    pytest.param(mutate(GOOD_RUN, mode="compare"), id="mode mismatch"),
    pytest.param(mutate(GOOD_RUN, nonce="a" * (MAX_NONCE_LENGTH + 1)), id="nonce too long"),
    pytest.param(mutate(GOOD_RUN, nonce=1), id="nonce not text"),
    pytest.param(mutate(GOOD_RUN, status="crashed"), id="unknown status"),
    pytest.param(mutate(GOOD_RUN, code="syntax_error"), id="code on completed"),
    pytest.param(mutate(GOOD_RUN, status="source_error", outcomes=[]), id="source_error without code"),
    pytest.param(
        mutate(GOOD_RUN, status="source_error", code="limits", outcomes=[]), id="source_error with runner code"
    ),
    pytest.param(
        mutate(GOOD_RUN, status="runner_error", code="syntax_error", outcomes=[]),
        id="runner_error with source code",
    ),
    pytest.param(
        mutate(GOOD_RUN, status="runner_error", code="x" * (MAX_CODE_LENGTH + 1), outcomes=[]), id="code too long"
    ),
    pytest.param(mutate(GOOD_RUN, timed_out_case=0), id="timed_out_case on completed"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=MAX_CASES), id="timed_out_case past the cap"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=-1), id="negative timed_out_case"),
    pytest.param(mutate(GOOD_RUN, status="timeout", timed_out_case=False), id="boolean timed_out_case"),
    pytest.param(mutate(GOOD_RUN, outcomes={}), id="outcomes not a list"),
    pytest.param(mutate(GOOD_RUN, outcomes=[VALUE] * (MAX_CASES + 1)), id="too many outcomes"),
    pytest.param(mutate(GOOD_RUN, outcomes=["value"]), id="outcome not an object"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "stdout", "value": "2"}]), id="unknown outcome kind"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": 1}]), id="kind not text"),
    pytest.param(mutate(GOOD_RUN, outcomes=[LEGACY_VALUE]), id="legacy typed wire shape"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**VALUE, "stdout": ""}]), id="value outcome with extra field"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "value"}]), id="value outcome without a wire"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": ["int", "2"]}]), id="bare node without envelope"
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {**wire(["int", "2"]), "type": "int"}}]),
        id="wire with extra key",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {"format": "critpt-value-v0", "value": ["int", "2"]}}]),
        id="wire format mismatch",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "value", "value": {"format": WIRE_FORMAT, "value": {"int": "2"}}}]),
        id="wire node not a list",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[value_outcome(nested(60))]), id="wire past the depth cap"),
    pytest.param(mutate(GOOD_RUN, outcomes=[value_outcome(["int", 2**64])]), id="untagged large integer"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[value_outcome(["str", "x" * (MAX_WIRE_BYTES + 1)])]),
        id="one value past its byte cap",
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[value_outcome(["list", [["int", "1"]] * (MAX_JSON_NODES // 3 + 1)])]),
        id="one value past its node cap",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**EXCEPTION, "traceback": ""}]), id="exception with unknown field"),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "exception", "type": 1}]), id="exception type not text"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "exception", "message": "m" * (MAX_TEXT_BYTES + 1)}]),
        id="exception message too long",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error"}]), id="encoding_error without code"),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error", "code": None}]), id="encoding_error code not text"
    ),
    pytest.param(
        mutate(GOOD_RUN, outcomes=[{"kind": "encoding_error", "code": "x" * (MAX_CODE_LENGTH + 1)}]),
        id="encoding_error code too long",
    ),
    pytest.param(mutate(GOOD_RUN, outcomes=[{**ENCODING_ERROR, "path": ""}]), id="encoding_error with extra field"),
]

MALFORMED_COMPARE = [
    pytest.param(mutate(GOOD_COMPARE, mode="run"), id="mode mismatch"),
    pytest.param(mutate(without(GOOD_COMPARE, "results"), outcomes=[]), id="outcomes key in compare mode"),
    pytest.param(
        mutate(GOOD_COMPARE, status="source_error", code="syntax_error", results=[]),
        id="source_error is not a compare status",
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("uncertain"), equal=False)]), id="legacy boolean on uncertain"
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("mismatch"), equal=None)]), id="mismatch without its boolean"
    ),
    pytest.param(
        mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), equal=1)]), id="integer stands in for the boolean"
    ),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("close_enough")]), id="unknown verdict status"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), status=None)]), id="verdict status not text"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), version=2)]), id="verdict version"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), observed={})]), id="verdict with extra key"),
    pytest.param(mutate(GOOD_COMPARE, results=[without(verdict("equal"), "path")]), id="verdict missing key"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal", path="p" * 1025)]), id="path too long"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), path=None)]), id="path not text"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal", side="s" * 257)]), id="side too long"),
    pytest.param(mutate(GOOD_COMPARE, results=[mutate(verdict("equal"), code=3)]), id="code not text"),
    pytest.param(mutate(GOOD_COMPARE, results=["equal"]), id="verdict not an object"),
    pytest.param(mutate(GOOD_COMPARE, results=[verdict("equal")] * (MAX_CASES + 1)), id="too many verdicts"),
]


@pytest.mark.parametrize("payload", MALFORMED_RUN)
def test_parse_result_rejects_malformed_run_results(payload):
    assert rejected(payload, "run")


@pytest.mark.parametrize("payload", MALFORMED_COMPARE)
def test_parse_result_rejects_malformed_compare_results(payload):
    assert rejected(payload, "compare")


def test_parse_result_accepts_every_documented_envelope():
    assert parse(GOOD_RUN)["outcomes"] == [VALUE, EXCEPTION, ENCODING_ERROR]
    diagnostics = [{"kind": "exception"}, {"kind": "exception", "type": None, "message": "m"}]
    assert parse(run_result("completed", diagnostics))["outcomes"] == diagnostics
    assert parse(run_result("timeout", [VALUE], timed_out=1))["timed_out_case"] == 1
    assert parse(run_result("timeout", []))["status"] == "timeout"
    for code in runner.SOURCE_ERROR_CODES:
        assert parse(run_result("source_error", [], code=code))["code"] == code
    for code in runner.RUNNER_ERROR_CODES:
        assert parse(run_result("runner_error", [], code=code))["code"] == code
    assert parse(run_result("completed", [VALUE] * MAX_CASES))["outcomes"] == [VALUE] * MAX_CASES
    assert parse(run_result("completed", [value_outcome(nested(30))]))["status"] == "completed"
    assert parse(GOOD_COMPARE, "compare")["results"] == GOOD_COMPARE["results"]


def test_parse_result_holds_each_value_to_its_own_caps_not_the_suite():
    # Each value stays under the per-value node and byte caps on its own, but together they exceed both. Applying
    # either cap to the whole suite would refuse a legal result. The counts are derived from the live caps, so
    # this stays a suite-vs-value check if the caps move again.
    per_value_items = 18_000
    heavy = value_outcome(["list", [["int", "1"]] * per_value_items])  # 3 + 3*items JSON nodes, under both caps
    heavy_nodes = 3 + 3 * per_value_items
    count = MAX_JSON_NODES // heavy_nodes + 1  # enough node-heavy values to exceed the node cap in aggregate
    long_text = value_outcome(["str", "x" * (MAX_WIRE_BYTES - 128)])  # just under the byte cap on its own
    outcomes = [heavy] * count + [long_text] * 2
    assert count + 2 <= MAX_CASES  # the whole result still fits the case-count cap
    assert heavy_nodes < MAX_JSON_NODES  # each node-heavy value is under the per-value node cap
    assert count * heavy_nodes > MAX_JSON_NODES  # together they exceed it
    assert 2 * (MAX_WIRE_BYTES - 128) > MAX_WIRE_BYTES  # two long values exceed the byte cap in aggregate
    result = parse(run_result("completed", outcomes))
    assert result["outcomes"] == outcomes  # returned exactly as observed, never decoded


def test_parse_result_enforces_the_total_byte_cap():
    data = json.dumps(run_result("completed", [VALUE])).encode()
    assert parse(data + b" " * (RESULT_BYTES_LIMIT - len(data)))["status"] == "completed"
    assert rejected(data + b" " * (RESULT_BYTES_LIMIT + 1 - len(data)))


# --- provider exit and result correlation (necessary checks that do not authenticate a file) -- #


NON_RESULT_EXITS = [
    pytest.param(exited(runner.EXIT_CONTAINMENT_FAILED), id="containment_failed"),
    pytest.param(exited(runner.EXIT_WORKER_ABORTED), id="worker_aborted"),
    pytest.param(exited(runner.EXIT_PROTOCOL_VIOLATION), id="protocol_violation"),
    pytest.param(exited(runner.EXIT_USAGE), id="usage"),
    pytest.param(exited(137), id="unknown"),
    pytest.param(exited(-9), id="negative"),
]
NO_EXIT_REPORTS = [
    pytest.param(exited(124, error_type="timeout"), id="provider timeout sentinel"),
    pytest.param(exited(125, error_type="sandbox"), id="provider sandbox sentinel"),
    pytest.param(exited(0, error_type="sandbox"), id="sentinel behind a zero code"),
    pytest.param(exited(True), id="boolean"),
    pytest.param(exited(None), id="missing"),
    pytest.param(exited("0"), id="text"),
    pytest.param(exited(0.0), id="float"),
]
# Shape-valid payloads behind a result exit other than the one their status pairs with.
MISPAIRED = [
    pytest.param(runner.EXIT_SOURCE_ERROR, run_result("completed", [VALUE, VALUE]), id="completed behind exit 1"),
    pytest.param(runner.EXIT_RUNNER_ERROR, run_result("completed", [VALUE, VALUE]), id="completed behind exit 2"),
    pytest.param(runner.EXIT_TIMEOUT, run_result("completed", [VALUE, VALUE]), id="completed behind exit 3"),
    pytest.param(
        runner.EXIT_COMPLETED, run_result("source_error", [], code="syntax_error"), id="source_error behind exit 0"
    ),
    pytest.param(
        runner.EXIT_TIMEOUT, run_result("source_error", [], code="syntax_error"), id="source_error behind exit 3"
    ),
    pytest.param(runner.EXIT_COMPLETED, run_result("timeout", [VALUE], timed_out=1), id="timeout behind exit 0"),
    pytest.param(
        runner.EXIT_COMPLETED, run_result("runner_error", [], code="limits"), id="runner_error behind exit 0"
    ),
]


def test_status_exit_pairing_mirrors_the_runner_protocol():
    assert execution._STATUS_EXIT_CODES == runner._STATUS_EXIT_CODES
    assert set(execution._STATUS_EXIT_CODES) == set(runner.RUN_STATUSES) >= set(runner.COMPARE_STATUSES)
    assert set(execution._STATUS_EXIT_CODES.values()) == runner.RESULT_EXIT_CODES
    # One exit per status.
    assert len(set(execution._STATUS_EXIT_CODES.values())) == len(execution._STATUS_EXIT_CODES)
    assert runner.RESULT_EXIT_CODES.isdisjoint(runner.NO_RESULT_EXIT_CODES)
    assert runner.EXIT_USAGE not in runner.RESULT_EXIT_CODES


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (exited(0), 0),
        (exited(runner.EXIT_TIMEOUT), runner.EXIT_TIMEOUT),
        (exited(137), 137),  # reported, but outside the result set: the domain refuses it
        (exited(-9), -9),
        (exited(True), None),  # not an exit, although True == EXIT_SOURCE_ERROR
        (exited(False), None),
        (exited(None), None),
        (exited("0"), None),
        (exited(0.0), None),
        (exited(124, error_type="timeout"), None),
        (exited(125, error_type="sandbox"), None),
        (exited(0, error_type="sandbox"), None),  # a set error_type is never an exit, whatever the code says
        (None, None),
        (SimpleNamespace(error_type=None), None),  # no return_code at all
    ],
)
def test_reported_exit_is_an_exact_integer_process_exit_or_nothing(report, expected):
    assert _reported_exit(report) == expected


def test_correlate_result_requires_the_paired_exit_and_the_job_nonce():
    nonce = "a" * 32
    for status, exit_code in execution._STATUS_EXIT_CODES.items():
        code = "syntax_error" if status == "source_error" else "limits" if status == "runner_error" else None
        result = parse(mutate(run_result(status, [], code=code), nonce=nonce))
        assert _correlate_result(result, exit_code, nonce) is None
        for other in sorted(runner.RESULT_EXIT_CODES - {exit_code}):
            with pytest.raises(_Unscorable) as info:
                _correlate_result(result, other, nonce)
            assert info.value.category == "result_invalid"
        with pytest.raises(_Unscorable):
            _correlate_result(result, exit_code, "b" * 32)
        with pytest.raises(_Unscorable):
            _correlate_result(result, bool(exit_code), nonce)  # True == 1 must never stand in for exit 1
    # A runner that could not read its job writes an empty nonce: shape-valid, but uncorrelated with this job.
    orphan = parse(run_result("runner_error", [], code="invalid_job"))
    assert orphan["nonce"] == ""
    with pytest.raises(_Unscorable) as info:
        _correlate_result(orphan, runner.EXIT_RUNNER_ERROR, nonce)
    assert info.value.category == "result_invalid"
    compared = parse(mutate(compare_result(["equal"]), nonce=nonce), "compare")
    assert _correlate_result(compared, runner.EXIT_COMPLETED, nonce) is None
    with pytest.raises(_Unscorable):
        _correlate_result(compared, runner.EXIT_TIMEOUT, nonce)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("report", NON_RESULT_EXITS)
async def test_non_result_exits_never_read_the_file_and_attribute_nothing(tmp_path, domain, report):
    script = happy_script()  # every scripted file would pass: none may be read behind these exits
    script[("exec", domain)] = report
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    if domain == "candidate" and report.return_code in runner.WORKER_ABORT_EXIT_CODES:
        # A runner-authenticated candidate worker abort is a scored 0 fault, not a no-result. The file is
        # still never read. Pinned in test_candidate_worker_abort_scores_zero_and_is_not_recoverable.
        assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_worker_abort")
        assert result.failure_class is None and result.terminal is None
    else:
        assert (result.outcome, result.reward, result.category) == ("unscorable", None, f"{domain}_no_result")
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    backends = domain_backends()
    assert backends[domain].downloads == 0 and result.cleanup[domain] == "deleted"
    if domain != "candidate":
        assert "candidate" not in backends and result.cleanup["candidate"] == "not_created"
    assert not list((tmp_path / "journal").glob("*.json"))


@pytest.mark.parametrize(
    "code", [runner.EXIT_WORKER_ABORTED, runner.EXIT_PROTOCOL_VIOLATION], ids=["worker_aborted", "protocol_violation"]
)
async def test_candidate_worker_abort_scores_zero_and_is_not_recoverable(tmp_path, code):
    # After a completed preflight the candidate suite exit is a runner-authenticated abort: the supervisor ran
    # to completion and reported the worker died or broke the protocol. That is a candidate fault scored 0 with
    # the category preserved, not a provider-ambiguous no-result. No result file is read, no case is attempted,
    # and the row is a real grade (scored, no failure class), so reverify never re-selects it.
    script = happy_script()
    script[("exec", "candidate")] = exited(code)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("failed", 0.0, "candidate_worker_abort")
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 0 and result.cases_equal == 0 and result.first_failed_case is None
    assert "candidate_worker_abort" in execution.CANDIDATE_FAULT_CATEGORIES
    assert domain_backends()["candidate"].downloads == 0  # the file behind a no-result exit is never read
    assert_job_cleaned(tmp_path, result, *DOMAINS)


async def test_reference_and_comparator_worker_abort_stay_unscorable(tmp_path):
    # Only the candidate domain attributes a worker abort to the candidate. A reference or comparator abort has
    # no candidate to blame, so it stays a provider-ambiguous no-result, unscorable and non-terminal.
    for domain in ("reference", "comparator"):
        script = happy_script()
        script[("exec", domain)] = exited(runner.EXIT_WORKER_ABORTED)
        grader, _ = make_grader(tmp_path, script)
        result = await grader.grade(make_task(), CANDIDATE)
        assert (result.outcome, result.reward, result.category) == ("unscorable", None, f"{domain}_no_result")
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False


@pytest.mark.parametrize("report", NO_EXIT_REPORTS)
async def test_reports_without_a_process_exit_are_provider_failures(tmp_path, report):
    script = happy_script()
    script[("exec", "candidate")] = report
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "provider_exec")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert domain_backends()["candidate"].downloads == 0
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


@pytest.mark.parametrize(("exit_code", "payload"), MISPAIRED)
async def test_mispaired_exit_and_status_is_result_invalid_for_the_candidate(tmp_path, exit_code, payload):
    script = happy_script()
    script[("exec", "candidate")] = exited(exit_code)
    script[("result", "candidate")] = payload
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE
    assert domain_backends()["candidate"].downloads == 1  # read behind a result exit, refused after the shape check
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


@pytest.mark.parametrize(("exit_code", "payload"), MISPAIRED)
async def test_mispaired_exit_and_status_is_result_invalid_for_the_reference(tmp_path, exit_code, payload):
    # Never a terminal reference defect: the file and the exit disagree, so neither says anything about the task.
    script = happy_script()
    script[("exec", "reference")] = exited(exit_code)
    script[("result", "reference")] = payload
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    assert "candidate" not in domain_backends() and result.cleanup["candidate"] == "not_created"


@pytest.mark.parametrize(
    ("exit_code", "payload", "outcome", "category", "reward"),
    [
        (runner.EXIT_COMPLETED, run_result("completed", [VALUE, VALUE]), "passed", "passed", 1.0),
        (
            runner.EXIT_SOURCE_ERROR,
            run_result("source_error", [], code="syntax_error"),
            "failed",
            "candidate_source_error",
            0.0,
        ),
        (runner.EXIT_TIMEOUT, run_result("timeout", [VALUE], timed_out=1), "failed", "candidate_timeout", 0.0),
        (runner.EXIT_RUNNER_ERROR, run_result("runner_error", [], code="limits"), "unscorable", "provider_exec", None),
    ],
    ids=["completed", "source_error", "timeout", "runner_error"],
)
async def test_paired_exit_and_status_reach_the_candidate_mapping(
    tmp_path, exit_code, payload, outcome, category, reward
):
    # Passing the correlation gate does not authenticate a source, timeout or runtime-error report. After a
    # completed preflight a candidate source error or suite timeout scores 0 with the category preserved. A
    # runtime error is unattributed provider uncertainty.
    script = happy_script()
    script[("exec", "candidate")] = exited(exit_code)
    script[("result", "candidate")] = payload
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category, result.reward) == (outcome, category, reward)
    assert result.cases_attempted == len(payload["outcomes"]) and result.first_failed_case is None
    if outcome == "passed":
        assert result.failure_class is None and result.terminal is None and result.cases_equal == 2
        assert domain_backends()["comparator"].downloads == 2  # preflight and candidate judge both ran
    elif outcome == "failed":
        # A whole-run candidate fault scores 0 without a candidate value comparison ever running.
        assert result.failure_class is None and result.terminal is None and result.cases_equal == 0
        assert domain_backends()["comparator"].downloads == 1
    else:
        assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
        assert result.cases_equal == 0 and domain_backends()["comparator"].downloads == 1
    assert domain_backends()["candidate"].downloads == 1
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize(
    ("exit_code", "payload", "category"),
    [
        (runner.EXIT_SOURCE_ERROR, run_result("source_error", [], code="import_error"), "reference_error"),
        (runner.EXIT_TIMEOUT, run_result("timeout", [VALUE], timed_out=1), "reference_timeout"),
        (runner.EXIT_RUNNER_ERROR, run_result("runner_error", [], code="limits"), "reference_error"),
    ],
    ids=["source_error", "timeout", "runner_error"],
)
async def test_paired_exit_and_status_reach_the_reference_mapping(tmp_path, exit_code, payload, category):
    # Even an accepted exit and the correct job nonce cannot attribute the report to a reference defect.
    script = happy_script()
    script[("exec", "reference")] = exited(exit_code)
    script[("result", "reference")] = payload
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, category)
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == result.cases_equal == 0 and result.first_failed_case is None
    assert domain_backends()["reference"].downloads == 1 and "candidate" not in domain_backends()
    assert_job_cleaned(tmp_path, result, "reference")


async def test_comparator_no_result_exit_after_a_completed_candidate_scores_nothing(tmp_path):
    script = happy_script()
    script[("exec", "comparator")] = [exited(runner.EXIT_COMPLETED), exited(runner.EXIT_WORKER_ABORTED)]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "comparator_no_result")
    assert result.cases_attempted == 2 and result.cases_equal == 0
    assert domain_backends()["comparator"].downloads == 1  # the preflight verdicts only
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}


async def test_invocation_nonces_are_distinct_and_match_uploaded_jobs(tmp_path, monkeypatch):
    nonces = [character * 32 for character in "abcd"]
    token = Mock(side_effect=nonces)
    monkeypatch.setattr(execution.secrets, "token_hex", token)
    original_run = _Domain.run
    seen = []

    async def recording_run(domain, mode, deadline_s, nonce):
        job = domain.backend.jobs[-1]
        assert job["nonce"] == nonce and job["mode"] == mode
        seen.append((domain.name, job.get("observed_role"), nonce))
        result = await original_run(domain, mode, deadline_s, nonce)
        assert result["nonce"] == nonce
        return result

    monkeypatch.setattr(_Domain, "run", recording_run)
    grader, _ = make_grader(tmp_path, happy_script())
    result = await grader.grade(make_task(), CANDIDATE, job_id="12345678")
    assert result.reward == 1.0
    assert token.call_args_list == [call(16)] * 4
    assert seen == [
        ("reference", None, nonces[0]),
        ("comparator", "reference", nonces[1]),
        ("candidate", None, nonces[2]),
        ("comparator", "candidate", nonces[3]),
    ]
    assert len({job["nonce"] for backend in domain_backends().values() for job in backend.jobs}) == 4
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize(("origin", "target"), [("reference", "candidate"), ("comparator", "comparator")])
async def test_stale_download_bytes_cannot_replay_across_invocations(tmp_path, monkeypatch, origin, target):
    script = happy_script()
    replayed = []

    async def stale_download():
        data = domain_backends()[origin].downloaded[0]
        assert type(data) is bytes
        replayed.append(data)
        return data  # Not a dict: the fake must not substitute the current job's nonce.

    if target == "comparator":
        script[("result", target)][1] = stale_download
    else:
        script[("result", target)] = stale_download
    correlate = Mock(wraps=execution._correlate_result)
    monkeypatch.setattr(execution, "_correlate_result", correlate)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    backends = domain_backends()
    assert backends[target].downloaded[-1] is replayed[0] is backends[origin].downloaded[0]
    payload, exit_code, nonce = correlate.call_args.args
    assert payload["nonce"] == json.loads(replayed[0])["nonce"] != nonce
    assert nonce == backends[target].jobs[-1]["nonce"] and exit_code == runner.EXIT_COMPLETED
    assert result.cases_equal == 0 and result.first_failed_case is None
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize("target", ["reference", "preflight", "candidate", "comparison"])
async def test_previous_candidate_download_bytes_cannot_replay_in_a_new_job(tmp_path, monkeypatch, target):
    grader, _ = make_grader(tmp_path, happy_script())
    assert (await grader.grade(make_task(), CANDIDATE, job_id="12345678")).reward == 1.0
    domain = "comparator" if target in {"preflight", "comparison"} else target
    origin = "comparator" if domain == "comparator" else "candidate"
    stale = domain_backends()[origin].downloaded[-1]
    script = happy_script()
    if domain == "comparator":
        script[("result", domain)][0 if target == "preflight" else 1] = stale
    else:
        script[("result", domain)] = stale
    correlate = Mock(wraps=execution._correlate_result)
    monkeypatch.setattr(execution, "_correlate_result", correlate)
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE, job_id="87654321")
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "result_invalid")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert domain_backends()[domain].downloaded[-1] is stale
    payload, exit_code, nonce = correlate.call_args.args
    assert payload["nonce"] == json.loads(stale)["nonce"] != nonce
    assert nonce == domain_backends()[domain].jobs[-1]["nonce"] and exit_code == runner.EXIT_COMPLETED
    created = ("reference",) if target == "reference" else ("reference", "comparator")
    assert_job_cleaned(tmp_path, result, *(DOMAINS if target in {"candidate", "comparison"} else created))


async def test_result_must_echo_the_job_nonce(tmp_path):
    # A shape-valid passing file that echoes another nonce is uncorrelated with this invocation and never scored.
    script = happy_script()
    script[("result", "candidate")] = json.dumps(
        dict(run_result("completed", [VALUE, VALUE]), nonce="f" * 32)
    ).encode()
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category, result.terminal) == (
        "unscorable",
        None,
        "result_invalid",
        False,
    )
    # The runner writes an empty nonce when it cannot read its job: shape-valid, uncorrelated, and never a terminal
    # reference defect.
    script = happy_script()
    script[("exec", "reference")] = exited(runner.EXIT_RUNNER_ERROR)
    script[("result", "reference")] = json.dumps(run_result("runner_error", [], code="invalid_job")).encode()
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.category, result.terminal) == ("result_invalid", False) and "candidate" not in domain_backends()
    # Verdict files are correlated the same way.
    script = happy_script()
    script[("result", "comparator")] = [
        json.dumps(dict(compare_result(["equal", "equal"], side="reference"), nonce="f" * 32)).encode()
    ]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.category == "result_invalid" and "candidate" not in domain_backends()


async def test_cancellation_during_the_result_download_propagates_and_cleans_up(tmp_path):
    script = happy_script()
    script[("result", "candidate")] = cancel_self  # the exit was accepted, and the read itself is cancelled
    grader, _ = make_grader(tmp_path, script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    asyncio.current_task().uncancel()
    assert domain_backends()["candidate"].downloads == 1
    assert closes("sb-comparator") == closes("sb-candidate") == 1 and FakeBackend.plane == {}
    assert not list((tmp_path / "journal").glob("*.json")) and not stray_tasks()


# --- bounded download (fake SDK stream, no provider client) ------------------------------------- #


class FakeStream:
    """The SDK's lazy download generator: chunks and errors surface only while iterating. Closure is observable."""

    def __init__(self, steps, on_close=None):
        self.steps = list(steps)
        self.on_close = on_close
        self.close_started = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.steps:
            raise StopAsyncIteration
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return await step()
        return step

    async def aclose(self):
        self.close_started = True
        if self.on_close is not None:
            await self.on_close()
        self.closed = True


class FakeFileSystem:
    """``sandbox.fs`` as the download path uses it: an awaited call that returns the lazy stream."""

    def __init__(self, stream=None, call_outcome=None):
        self.stream = stream
        self.call_outcome = call_outcome
        self.calls = []

    async def download_file_stream(self, remote_path, *, timeout, cancel_event):
        self.calls.append({"path": remote_path, "timeout": timeout, "cancel_event": cancel_event})
        if callable(self.call_outcome) and not isinstance(self.call_outcome, BaseException):
            await self.call_outcome()
        if isinstance(self.call_outcome, BaseException):
            raise self.call_outcome
        return self.stream


def download(stream=None, *, call_outcome=None, limit=1024, timeout_s=1.0):
    fs = FakeFileSystem(stream, call_outcome)
    handle = SandboxHandle(sandbox_id="sb-1", provider_name="daytona", raw=SimpleNamespace(fs=fs))
    backend = DaytonaBackend.__new__(DaytonaBackend)  # the download path touches nothing the constructor builds
    return fs, backend.download_bounded(handle, "/tmp/ng-grader/result.json", limit=limit, timeout_s=timeout_s)


def dedicated_file_not_found(sdk):
    """The dedicated file-absent error, or skip. The pinned 0.183.0 has no such class."""
    cls = getattr(sdk, "DaytonaFileNotFoundError", None)
    if cls is None:
        pytest.skip("SDK has no dedicated DaytonaFileNotFoundError (pinned 0.183.0 predates it)")
    return cls("gone", status_code=404, code="FILE_NOT_FOUND", source="DAYTONA_DAEMON")


def not_found_404(sdk):
    """The generic 404 not-found. Every supported SDK exports it, so this path never skips."""
    return sdk.DaytonaNotFoundError("gone", status_code=404)


def sibling_not_found_404(sdk):
    """A not-found subclass about something else (process), or skip. The pinned 0.183.0 has no such subclass."""
    cls = getattr(sdk, "DaytonaProcessNotFoundError", None)
    if cls is None:
        pytest.skip("SDK has no DaytonaProcessNotFoundError subclass (pinned 0.183.0 predates it)")
    return cls("gone", status_code=404)


def forbidden_403(sdk):
    """A 403 error. The class is ``DaytonaForbiddenError`` on new SDKs and ``DaytonaAuthorizationError`` on 0.183.0."""
    cls = getattr(sdk, "DaytonaForbiddenError", None) or sdk.DaytonaAuthorizationError
    return cls("denied", status_code=403)


async def test_download_streams_within_the_limit_and_closes_the_generator():
    stream = FakeStream([b"ab", b"cd"])
    fs, call = download(stream, limit=4, timeout_s=0.5)
    assert await call == b"abcd"
    assert stream.closed and not stream.steps
    assert fs.calls[0]["path"] == "/tmp/ng-grader/result.json" and fs.calls[0]["timeout"] == 0.5
    assert fs.calls[0]["cancel_event"].is_set()  # the SDK token is released once the transfer ends
    assert not stray_tasks()


@pytest.mark.parametrize("absence", [dedicated_file_not_found, not_found_404], ids=["file_not_found", "not_found_404"])
async def test_download_maps_only_the_sdk_absence_signals_to_missing_file(sdk, absence):
    # ``not_found_404`` runs on every SDK, including the pinned 0.183.0. ``dedicated_file_not_found`` skips
    # on an SDK that predates the dedicated class.
    error = absence(sdk)
    stream = FakeStream([b"partial", error])  # raised lazily, after data started flowing
    _, call = download(stream)
    with pytest.raises(MissingRemoteFile):
        await call
    assert stream.closed


@pytest.mark.parametrize(
    "make_error",
    [
        lambda sdk: sdk.DaytonaError("No file data received for: /tmp/ng-grader/result.json"),
        lambda sdk: sdk.DaytonaError("file not found"),  # exception text is never consulted
        lambda sdk: sdk.DaytonaNotFoundError("gone"),  # no 404 status: not the daemon's per-file verdict
        sibling_not_found_404,  # a not-found about something else, skips on the pinned 0.183.0
        forbidden_403,
        lambda sdk: ConnectionResetError("reset"),  # raw transport error below the SDK
        lambda sdk: ValueError("Truncated multipart response"),
    ],
    ids=["no-data", "text-only", "no-status", "process-not-found", "forbidden", "transport", "value-error"],
)
async def test_download_propagates_other_lazy_errors_unchanged(sdk, make_error):
    error = make_error(sdk)
    stream = FakeStream([b"partial", error])
    _, call = download(stream)
    with pytest.raises(type(error)) as info:
        await call
    assert info.value is error and stream.closed


async def test_download_stops_at_the_byte_limit_and_closes_the_generator():
    stream = FakeStream([b"x" * 10, b"y" * 10, b"never"])
    _, call = download(stream, limit=15)
    with pytest.raises(TransferLimitExceeded):
        await call
    assert stream.closed and stream.steps == [b"never"]  # nothing past the offending chunk is pulled


async def test_download_deadline_covers_the_call_and_the_iteration(monkeypatch):
    monkeypatch.setattr(execution, "_STREAM_CLOSE_FLOOR_S", 0.01)
    stream = FakeStream([b"a", hang])
    _, call = download(stream, timeout_s=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await call
    assert stream.closed and not stray_tasks()
    fs, call = download(call_outcome=hang, timeout_s=0.05)  # the request itself never returns a stream
    with pytest.raises(asyncio.TimeoutError):
        await call
    assert fs.calls[0]["cancel_event"].is_set() and not stray_tasks()


async def test_download_propagates_cancellation_after_closing_the_generator():
    stream = FakeStream([b"a", cancel_self])
    _, call = download(stream)
    with pytest.raises(asyncio.CancelledError):
        await call
    asyncio.current_task().uncancel()
    assert stream.closed and not stray_tasks()


async def test_download_bounds_a_hanging_close_without_replacing_the_outcome(sdk, monkeypatch):
    monkeypatch.setattr(execution, "_STREAM_CLOSE_FLOOR_S", 0.01)
    stream = FakeStream([not_found_404(sdk)], on_close=hang)  # the 0.183.0 absent-file signal, present on every SDK
    _, call = download(stream, timeout_s=0.05)
    with pytest.raises(MissingRemoteFile):
        await call
    assert stream.close_started and not stream.closed and not stray_tasks()


async def test_download_ignores_a_failing_close():
    async def broken_close():
        raise RuntimeError("close failed")

    stream = FakeStream([b"ok"], on_close=broken_close)
    _, call = download(stream)
    assert await call == b"ok"
    assert stream.close_started and not stream.closed


async def test_download_classifies_call_time_errors_like_iteration_errors(sdk):
    fs, call = download(call_outcome=not_found_404(sdk))  # the 0.183.0 absent-file signal, present on every SDK
    with pytest.raises(MissingRemoteFile):
        await call
    assert fs.calls[0]["cancel_event"].is_set()
    _, call = download(call_outcome=ConnectionResetError("reset"))
    with pytest.raises(ConnectionResetError):
        await call


def test_missing_file_classification_uses_sdk_types_only(sdk, monkeypatch):
    # This path never skips. It runs on the pinned 0.183.0, whose only absent-file signal is the generic
    # not-found class with a 404 status. Every classification is by SDK type or the SDK's own status field.
    assert _missing_file_error(sdk.DaytonaNotFoundError("x", status_code=404))
    assert not _missing_file_error(sdk.DaytonaNotFoundError("x"))  # no 404 status: not the per-file verdict
    assert not _missing_file_error(sdk.DaytonaError("x", status_code=404))  # base class, not the not-found type
    assert not _missing_file_error(FileNotFoundError("x"))  # not an SDK type at all

    def no_sdk():
        raise ImportError("daytona")

    # Without the SDK types the classification fails closed, even for a real absent-file signal.
    monkeypatch.setattr(execution, "_daytona_file_types", no_sdk)
    assert not _missing_file_error(sdk.DaytonaNotFoundError("x", status_code=404))


def test_missing_file_classification_recognizes_the_dedicated_class(sdk):
    # The dedicated subclass is the exact signal on the typed-error-model SDKs (0.211.2 has it). It skips on
    # the pinned 0.183.0, which has no such class.
    cls = getattr(sdk, "DaytonaFileNotFoundError", None)
    if cls is None:
        pytest.skip("SDK has no dedicated DaytonaFileNotFoundError (pinned 0.183.0 predates it)")
    assert _missing_file_error(cls("gone", status_code=404, code="FILE_NOT_FOUND", source="DAYTONA_DAEMON"))
    assert _missing_file_error(cls("gone"))  # the dedicated class is recognized by type, without a status
    # A sibling not-found subclass about something else stays unclassified: it is a distinct exact type.
    assert not _missing_file_error(sdk.DaytonaGitRepoNotFoundError("x", status_code=404))


def test_daytona_type_maps_import_only_names_the_real_sdk_exports():
    # No monkeypatching and no fake module: these run against the really installed ``daytona``. The test fails
    # under any SDK generation missing a name the code imports unconditionally. Skips only when no SDK is installed.
    pytest.importorskip("daytona")

    def is_exc_types(value) -> bool:
        classes = value if isinstance(value, tuple) else (value,)
        return bool(classes) and all(isinstance(c, type) and issubclass(c, BaseException) for c in classes)

    lookup = execution._daytona_lookup_types()
    for key in ("not_found", "permission", "timeout", "connection", "error"):
        assert is_exc_types(lookup[key]), key
    assert isinstance(lookup["destroyed"], str) and lookup["destroyed"]

    files = execution._daytona_file_types()
    assert is_exc_types(files["not_found"])
    assert files["file_not_found"] is None or is_exc_types(files["file_not_found"])


# --- ownership lifecycle ------------------------------------------------------------------------ #


async def test_delete_rejects_handles_without_the_live_sdk_object(tmp_path):
    policy = make_policy(tmp_path)
    bare = SandboxHandle(sandbox_id="sb-x", provider_name="daytona", raw=None)
    with pytest.raises(TypeError):
        await FakeBackend(policy, {}).close(bare)
    with pytest.raises(TypeError):  # the real adapter refuses before it reaches the provider or any network
        await DaytonaBackend(policy).close(bare)


def test_describe_sandbox_projects_identity_and_rejects_odd_shapes():
    good = FakeSandbox("sb-1", "ngcg-1-reference", {"a": "b"})
    handle = SandboxHandle(sandbox_id="sb-1", provider_name="daytona", raw=good)
    found = _describe_sandbox(handle, "destroyed")
    assert (found.handle, found.name, found.labels, found.state) == (handle, "ngcg-1-reference", {"a": "b"}, "started")
    good.state = "destroyed"
    assert _describe_sandbox(handle, "destroyed") is None
    good.state = "started"
    good.labels = {"a": 1}
    with pytest.raises(LookupFailed) as info:
        _describe_sandbox(handle, "destroyed")
    assert info.value.kind == "malformed"
    good.labels = {"a": "b"}
    with pytest.raises(LookupFailed):
        _describe_sandbox(SandboxHandle(sandbox_id="sb-2", provider_name="daytona", raw=good), "destroyed")
    with pytest.raises(ValueError):
        LookupFailed("bogus")
    assert all(f"lookup_{kind}" in CLEANUP_STATUSES for kind in LOOKUP_ERROR_KINDS)


async def test_duplicate_close_is_idempotent_after_success(tmp_path):
    FakeBackend.reset()
    policy = make_policy(tmp_path)
    journal, record = open_record(tmp_path, state="planned")
    domain = _Domain("reference", policy, FakeBackend(policy, {}), journal, record)
    await domain.create()
    assert identity(record["domains"]["reference"]) == {
        "name": "ngcg-x-reference",
        "labels": {"ng-grader-owner": "unit", "ng-grader-job": "x", "ng-grader-domain": "reference"},
        "state": "created",
        "sandbox_id": "sb-reference",
    }
    assert isinstance(record["domains"]["reference"]["dispatched_at"], float)  # the create's dispatch time
    assert await domain.close() == "deleted"
    assert domain.handle is None and record["domains"]["reference"]["state"] == "deleted"
    assert await domain.close() == "deleted"
    assert closes("sb-reference") == 1


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_failed_close_is_retried_with_the_same_live_handle(tmp_path, domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("close", f"sb-{domain}")] = [RuntimeError("provider says no")]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    expected = ("passed", 1.0, "passed") if status == "equal" else ("failed", 0.0, "candidate_mismatch")
    assert (result.outcome, result.reward, result.category) == expected
    assert result.failure_class is None and result.terminal is None
    assert result.cases_attempted == 2 and result.cases_equal == (2 if status == "equal" else 1)
    assert result.first_failed_case == (None if status == "equal" else 1)
    assert result.cleanup[domain] == "deleted" and lookups(f"sb-{domain}") == 1
    assert grader._leftover == {}
    assert_job_cleaned(tmp_path, result, *DOMAINS)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_failed_cleanup_is_reported_and_retained(tmp_path, domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("close", f"sb-{domain}")] = RuntimeError("provider says no")
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cases_attempted == 2 and result.cases_equal == (2 if status == "equal" else 1)
    assert result.first_failed_case == (None if status == "equal" else 1)
    assert result.cleanup == {name: "delete_failed" if name == domain else "deleted" for name in DOMAINS}
    records = journal_records(tmp_path)
    assert len(records) == 1 and grader._leftover == {result.job_id: records[0]}
    assert identity(records[0]["domains"][domain]) == {
        "name": f"ngcg-{result.job_id[:12]}-{domain}",
        "labels": {"ng-grader-owner": "unit", "ng-grader-job": result.job_id, "ng-grader-domain": domain},
        "state": "delete_failed",
        "sandbox_id": f"sb-{domain}",
    }
    text = next((tmp_path / "journal").glob("*.json")).read_text()
    assert "def f" not in text and "provider says no" not in text
    assert f"sb-{domain}" not in repr(result) and "provider says no" not in repr(result)
    assert f"sb-{domain}" in FakeBackend.plane
    followup = await grader.grade(make_task(), CANDIDATE)
    assert (followup.category, followup.reward, followup.terminal) == ("ownership_unresolved", None, False)
    # The gate now reconciles on the job path, rebuilding one cleanup client for the still-failing domain.
    # The close still fails, so the leftover is retained and the job is refused. The rebuilt client is the 4th.
    assert len(FakeBackend.instances) == 4 and not stray_tasks()
    del script[("close", f"sb-{domain}")]
    assert await grader.reconcile() == [{"job_id": result.job_id, "domain": domain, "status": "deleted"}]
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_unconfirmed_cleanup_retry_preserves_resolved_verdict(tmp_path, domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("lookup", f"sb-{domain}")] = [LookupFailed("permission")]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    expected = ("passed", 1.0, "passed") if status == "equal" else ("failed", 0.0, "candidate_mismatch")
    assert (result.outcome, result.reward, result.category) == expected
    assert result.failure_class is None and result.terminal is None
    assert result.cleanup == {name: "deleted" for name in DOMAINS}
    assert closes(f"sb-{domain}") == 2 and len(FakeBackend.instances) == 3
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}
    assert not stray_tasks()


@pytest.mark.parametrize("status", ["equal", "mismatch"])
@pytest.mark.parametrize("failure", ["false", "exception", "unlink_sync"])
async def test_unresolved_journal_resolution_fails_closed_and_can_reconcile(tmp_path, monkeypatch, status, failure):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    grader, _ = make_grader(tmp_path, script)
    resolve = grader.journal.resolve
    records = []

    def unresolved(record):
        records.append(record)
        if failure == "false":
            return False
        if failure == "exception":
            raise OSError("synthetic journal failure")
        with monkeypatch.context() as patch:
            patch.setattr(grader.journal, "_sync_directory", Mock(side_effect=OSError("synthetic sync failure")))
            return resolve(record)  # The unlink succeeds, but its durability is unresolved.

    monkeypatch.setattr(grader.journal, "resolve", unresolved)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup == {domain: "deleted" for domain in DOMAINS}
    assert result.cases_attempted == 2 and result.cases_equal == (2 if status == "equal" else 1)
    assert result.first_failed_case == (None if status == "equal" else 1)
    assert len(records) == 1 and grader._leftover[result.job_id] is records[0]
    assert all(entry["state"] == "deleted" for entry in records[0]["domains"].values())
    assert not grader.journal.unresolved()  # Only the retained record can block another job now.
    assert journal_records(tmp_path) == ([] if failure == "unlink_sync" else records)
    assert FakeBackend.plane == {} and all(backend.calls.count("aclose") == 1 for backend in FakeBackend.instances)
    assert (await grader.grade(make_task(), CANDIDATE)).category == "ownership_unresolved"
    assert len(FakeBackend.instances) == 3 and not stray_tasks()
    monkeypatch.setattr(grader.journal, "resolve", resolve)
    assert await grader.reconcile() == []  # Journal-only retry: no new backend, lookup or delete.
    assert len(FakeBackend.instances) == 3 and grader._leftover == {} and journal_records(tmp_path) == []


async def test_journal_resolution_failure_does_not_replace_cancellation(tmp_path, monkeypatch):
    script = happy_script()
    script[("exec", "candidate")] = asyncio.CancelledError()
    grader, _ = make_grader(tmp_path, script)
    resolve = grader.journal.resolve
    monkeypatch.setattr(grader.journal, "resolve", Mock(side_effect=OSError("synthetic journal failure")))
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    assert len(grader._leftover) == 1 and FakeBackend.plane == {} and not stray_tasks()
    record = next(iter(grader._leftover.values()))
    assert all(entry["state"] == "deleted" for entry in record["domains"].values())
    monkeypatch.setattr(grader.journal, "resolve", resolve)
    assert await grader.shutdown() == []
    assert grader._leftover == {} and journal_records(tmp_path) == []


@pytest.mark.parametrize(
    "failure", ["reference_mismatch", "task_invalid", "candidate_exception", "provider_exec", "job_deadline"]
)
async def test_unresolved_cleanup_overrides_non_scored_dispositions(tmp_path, monkeypatch, failure):
    script = happy_script()
    domain = "candidate"
    if failure in {"reference_mismatch", "task_invalid"}:
        domain = "comparator"
        status = "invalid_reference" if failure == "reference_mismatch" else "invalid_expected"
        script[("result", "comparator")] = [compare_result(["equal", status], side="reference")]
    elif failure == "candidate_exception":
        script[("result", "candidate")] = run_result("completed", [VALUE, EXCEPTION])
    elif failure == "provider_exec":
        script[("exec", "candidate")] = ConnectionError("synthetic exec failure")
    else:
        script[("exec", "candidate")] = hang
        monkeypatch.setattr(ExecutionPolicy, "effective_job_timeout_s", lambda self: 1.0)
    script[("close", f"sb-{domain}")] = RuntimeError("synthetic delete failure")
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup[domain] == "delete_failed" and result.cases_equal == 0
    # An unresolved cleanup overrides the disposition but keeps the counts the job observed: a candidate
    # exception was recorded at its offending case before the delete failed.
    assert result.first_failed_case == (1 if failure == "candidate_exception" else None)
    assert result.cases_attempted == (2 if failure == "candidate_exception" else 0)
    assert grader._leftover[result.job_id]["domains"][domain]["sandbox_id"] == f"sb-{domain}"
    assert not stray_tasks()
    del script[("close", f"sb-{domain}")]
    assert await grader.reconcile() == [{"job_id": result.job_id, "domain": domain, "status": "deleted"}]
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}


async def test_proven_pre_dispatch_create_rejection_resolves_clean(tmp_path):
    # Only a create the backend proves was refused before dispatch is ``create_failed_clean``: no sandbox exists,
    # so no lookup or delete is attempted and no ownership intent is retained.
    script = happy_script()
    script[("create", "reference")] = SandboxCreateRejected("refused before dispatch")
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "provider_create")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup == {
        "reference": "create_failed_clean",
        "comparator": "not_created",
        "candidate": "not_created",
    }
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}
    assert lookups("ngcg-" + result.job_id[:12] + "-reference") == 0  # a clean rejection is never looked up
    assert all(backend.calls.count("aclose") == 1 for backend in FakeBackend.instances) and not stray_tasks()


async def test_create_that_registers_then_raises_value_error_keeps_ownership(tmp_path, fast_polls):
    # A ValueError after the backend already registered the sandbox (an SDK decoding a successful create's
    # response, say) is ambiguous, not proof of a clean pre-dispatch rejection. The domain must retain its
    # ownership intent, so cleanup looks the sandbox up by stable name and deletes it rather than dropping it.
    script = happy_script()
    script[("create", "reference")] = ValueError("response decode failed after the sandbox was made")
    name = None

    def on_create(backend):
        nonlocal name
        if backend.domain == "reference":
            name = backend.spec.provider_options["extensions"]["daytona.name"]
            FakeBackend.register("sb-reference", name, backend.spec.metadata)

    grader, _ = make_grader(tmp_path, script, on_create=on_create)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "provider_create")
    # The ambiguous create was reconciled by name, not resolved clean: the sandbox was found and deleted.
    assert result.cleanup["reference"] in {"deleted", "absent"}
    assert lookups(name) >= 1 and closes("sb-reference") == 1
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}
    assert all(backend.calls.count("aclose") == 1 for backend in FakeBackend.instances) and not stray_tasks()


async def test_shutdown_retries_exact_cleanup_of_leftovers_with_a_fresh_client(tmp_path):
    script = happy_script()
    script[("close", "sb-candidate")] = [RuntimeError("no"), RuntimeError("still no")]
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.cleanup["candidate"] == "delete_failed"
    report = await grader.shutdown()
    assert report == [{"job_id": result.job_id, "domain": "candidate", "status": "deleted"}]
    assert closes("sb-candidate") == 1 and FakeBackend.plane == {}
    assert not list((tmp_path / "journal").glob("*.json"))
    assert (await grader.grade(make_task(), CANDIDATE)).category == "busy"  # draining
    assert await grader.shutdown() == []


async def test_shutdown_retries_a_disk_recovered_record_after_a_failed_delete(tmp_path, fast_polls):
    # A crash-persisted record whose first delete fails, with no in-process leftover and no job before shutdown,
    # must still get another delete attempt at shutdown, the way reconcile scans disk, not wait for a restart.
    journal, _ = open_record(tmp_path, state="delete_failed", sandbox_id="sb-x")
    FakeBackend.reset()
    FakeBackend.register(
        "sb-x",
        "ngcg-x-reference",
        {
            "ng-grader-owner": "unit",
            "ng-grader-job": "x",
            "ng-grader-domain": "reference",
            "code-toolbox-language": "python",
        },
    )
    script = {("close", "sb-x"): [RuntimeError("transient delete failure")]}
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert grader._leftover == {} and grader.journal.unresolved()  # disk-only record, nothing in this process
    assert await grader.shutdown() == [{"job_id": "x", "domain": "reference", "status": "delete_failed"}]
    assert grader._leftover == {} and grader.journal.unresolved()  # the failed delete kept the disk record
    assert await grader.shutdown() == [{"job_id": "x", "domain": "reference", "status": "deleted"}]
    assert closes("sb-x") == 1 and FakeBackend.plane == {} and not grader.journal.unresolved()


async def test_ambiguous_create_is_recorded_and_never_retried(tmp_path):
    script = happy_script()
    script[("create", "comparator")] = ConnectionError("socket reset mid-request")
    name = None

    def on_create(backend):
        nonlocal name
        if backend.domain == "comparator":
            name = backend.spec.provider_options["extensions"]["daytona.name"]
            script[("lookup", name)] = [LookupFailed("transport")]

    grader, _ = make_grader(tmp_path, script, on_create=on_create)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup["comparator"] == "lookup_transport"
    record = journal_records(tmp_path)[0]
    assert identity(record["domains"]["comparator"]) == {
        "name": name,
        "labels": record["domains"]["comparator"]["labels"],
        "state": "create_uncertain",
        "sandbox_id": None,
    }
    assert sum(1 for b in FakeBackend.instances if b.domain == "comparator") == 1
    assert "candidate" not in domain_backends()
    FakeBackend.register("sb-found", name, record["domains"]["comparator"]["labels"])
    report = await grader.reconcile()
    assert report == [{"job_id": result.job_id, "domain": "comparator", "status": "deleted"}]
    assert closes("sb-found") == 1 and FakeBackend.plane == {}
    assert not list((tmp_path / "journal").glob("*.json"))


async def test_in_process_settling_create_survives_a_wall_clock_jump(tmp_path, fast_polls, monkeypatch):
    # A create this process dispatched anchors its settle deadline on the monotonic clock, so a forward
    # wall-clock jump while the create is still settling must not trip the deadline and orphan it. The comparator
    # create raises (``create_uncertain``, no id learned) and its sandbox never appears, so its cleanup runs the
    # settle path. The wall clock jumps far past the 360 s deadline (120 + max(240, 60)) while real monotonic
    # barely moves. The comparator must stay unresolved, proving the jump did not resolve a still-settling create.
    clock = {"now": 1000.0}
    monkeypatch.setattr(execution, "_now", lambda: clock["now"])
    script = happy_script()
    script[("create", "comparator")] = ConnectionError("socket reset mid-request")
    captured = {}

    def on_create(backend):
        if backend.domain == "comparator":
            captured["name"] = backend.spec.provider_options["extensions"]["daytona.name"]
            clock["now"] += 10_000.0  # a forward wall-clock jump far past the settle deadline

    grader, _ = make_grader(tmp_path, script, on_create=on_create)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.cleanup["comparator"] == "create_uncertain"  # still settling: the wall jump did not resolve it
    assert journal_records(tmp_path)[0]["domains"]["comparator"]["state"] == "create_uncertain"

    # Same process, no restart: reconcile and shutdown rebuild the cleanup client for this job. The monotonic
    # anchor is kept per job, so the rebuilt client still measures the deadline on the monotonic clock. The wall
    # clock is still jumped, so a wall-clock deadline would resolve the create deleted and drop the record.
    reconcile_report = await grader.reconcile()
    assert reconcile_report == [{"job_id": result.job_id, "domain": "comparator", "status": "create_uncertain"}]
    assert journal_records(tmp_path)[0]["domains"]["comparator"]["state"] == "create_uncertain"  # record retained
    assert await grader.shutdown() == [{"job_id": result.job_id, "domain": "comparator", "status": "create_uncertain"}]
    assert journal_records(tmp_path)[0]["domains"]["comparator"]["state"] == "create_uncertain"  # still retained

    # The create settles remotely afterwards. Because the record was retained, the next reconcile finds the
    # sandbox by its exact stable name and deletes it, so a late-settling create is never orphaned.
    labels = journal_records(tmp_path)[0]["domains"]["comparator"]["labels"]
    FakeBackend.register("sb-late-comparator", captured["name"], labels)
    assert await grader.reconcile() == [{"job_id": result.job_id, "domain": "comparator", "status": "deleted"}]
    assert closes("sb-late-comparator") == 1 and FakeBackend.plane == {}
    assert not list((tmp_path / "journal").glob("*.json"))


async def test_grade_self_reconciles_an_absent_crash_record_then_runs(tmp_path, fast_polls):
    # A crash left an unresolved record whose sandbox is already gone. The next grade must reconcile it once and
    # then run, not refuse every request. reference and comparator stay planned (resolved), so only the candidate
    # domain is unresolved.
    open_record(tmp_path, state="delete_failed", sandbox_id="sb-gone-crash", domain="candidate")
    grader, _ = make_grader(tmp_path, happy_script())
    assert grader.journal.unresolved()  # precondition: the crash record would wedge the gate without a reconcile
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward) == ("passed", 1.0)  # the gate reconciled the record, then ran the job
    assert grader.journal.unresolved() == []  # the crash record cleared and this job resolved


async def test_grade_self_reconciles_an_in_memory_leftover_then_runs(tmp_path, fast_polls):
    # A prior job in this process left an in-memory leftover whose sandbox is already gone on the provider, and
    # its journal file was already unlinked. The gate reconciles the leftover on the next job, clears it, and runs.
    _, record = open_record(tmp_path, state="delete_failed", sandbox_id="sb-gone-leftover", domain="candidate")
    grader, _ = make_grader(tmp_path, happy_script())
    for path in (tmp_path / "journal").glob("*.json"):
        path.unlink()  # only the in-memory leftover remains, and the disk journal is empty
    grader._leftover[record["job_id"]] = record
    assert grader._leftover and not grader.journal.unresolved()  # precondition: only the leftover blocks the gate
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward) == ("passed", 1.0)  # the gate reconciled the leftover, then ran the job
    assert grader._leftover == {} and not grader.journal.unresolved()


async def test_grade_still_refuses_when_a_crash_record_cannot_reconcile(tmp_path, fast_polls, monkeypatch):
    # If the crash record cannot be reconciled, the gate must still refuse. A create still inside its settle
    # deadline whose sandbox is absent stays create_uncertain, so the reconcile does not resolve it.
    monkeypatch.setattr(execution, "_now", lambda: 1005.0)  # 5 s after dispatch, far inside the settle deadline
    open_record(tmp_path, state="create_uncertain", dispatched_at=1000.0, domain="candidate")
    grader, _ = make_grader(tmp_path, happy_script())
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.category) == ("unscorable", "ownership_unresolved")
    assert grader.journal.unresolved()  # the still-settling record was retained, not resolved or run over


async def test_reconcile_ambiguous_create_absent_before_deadline_retains_ownership_exactly(
    tmp_path, fast_polls, monkeypatch
):
    # Before the settle deadline an ambiguous create with no id learned may still be settling remotely, so a name
    # that is absent across the bounded re-lookup window keeps the domain unresolved rather than resolving it
    # deleted. The name match stays exact throughout: a prefix-sharing neighbour is never matched or deleted.
    monkeypatch.setattr(execution, "_now", lambda: 1005.0)  # 5 s after dispatch, far inside the deadline
    journal, _ = open_record(tmp_path, dispatched_at=1000.0)  # reference is ``create_uncertain`` with no id
    FakeBackend.reset()
    neighbour = {"ng-grader-owner": "unit", "ng-grader-job": "x", "ng-grader-domain": "reference"}
    FakeBackend.register("sb-neighbour", "ngcg-x-reference-2", neighbour)  # shares the journaled prefix
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "create_uncertain"}]
    assert journal.unresolved()[0]["domains"]["reference"]["state"] == "create_uncertain"
    assert "sb-neighbour" in FakeBackend.plane and closes("sb-neighbour") == 0
    # One initial lookup plus the bounded settle window, all by the exact stable name, none matching the neighbour.
    assert lookups("ngcg-x-reference") == 1 + execution._CREATE_SETTLE_POLLS
    assert lookups("ngcg-x-reference-2") == 0


async def test_reconcile_ambiguous_create_absent_after_deadline_resolves_deleted(tmp_path, monkeypatch):
    # After the settle deadline the create can no longer be settling, so an absent exact-name lookup is
    # authoritative and resolves the domain deleted, exactly as a learned-id absence does. The grader clears on
    # its own without an operator editing the journal.
    monkeypatch.setattr(execution, "_now", lambda: 1000.0 + 120.0 + 240.0 + 1.0)  # one second past the deadline
    journal, _ = open_record(tmp_path, dispatched_at=1000.0)  # create_timeout 120 s, margin max(2*120, 60) = 240 s
    FakeBackend.reset()
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "absent"}]
    assert not journal.unresolved() and grader._leftover == {}
    assert lookups("ngcg-x-reference") == 1  # no settle window once the deadline has passed


async def test_reconcile_legacy_ambiguous_create_without_timestamp_resolves_deleted(tmp_path):
    # A legacy journal entry predates the dispatch timestamp. It falls back to the record's creation time, so an
    # old record with an absent name resolves deleted rather than staying unresolved forever.
    journal, record = open_record(tmp_path)  # no dispatched_at recorded on the entry
    record["created_at"] = 0.0  # long before any real settle deadline
    journal.update(record, "reference", "create_uncertain")
    assert "dispatched_at" not in journal.unresolved()[0]["domains"]["reference"]
    FakeBackend.reset()
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "absent"}]
    assert not journal.unresolved() and grader._leftover == {}


async def test_reconcile_recovers_a_create_that_settles_after_the_first_lookup(tmp_path, fast_polls, monkeypatch):
    # A create that timed out locally can finish remotely afterwards. Inside the settle deadline the sandbox is
    # invisible on the first cleanup lookup and appears within the re-lookup window. Recovery must then find it by
    # exact name and delete it rather than having already resolved the domain deleted on the first absence.
    monkeypatch.setattr(execution, "_now", lambda: 1005.0)  # well inside the deadline
    journal, record = open_record(tmp_path, dispatched_at=1000.0)  # reference is ``create_uncertain`` with no id
    FakeBackend.reset()
    name = record["domains"]["reference"]["name"]
    FakeBackend.register("sb-late", name, record["domains"]["reference"]["labels"])
    script = {("lookup", name): [None]}  # absent on the first lookup, then visible in the plane
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "deleted"}]
    assert closes("sb-late") == 1 and FakeBackend.plane == {} and not journal.unresolved()


async def test_reconcile_crash_persisted_creating_before_deadline_retains_ownership(tmp_path, fast_polls, monkeypatch):
    # A ``creating`` entry is persisted just before the provider create call and outlives a process crash with no
    # id ever learned. It is as ambiguous as a ``create_uncertain`` entry: a sandbox may still be settling
    # remotely, so an absent name across the bounded re-lookup window keeps the domain unresolved rather than
    # resolving it deleted. The journal state stays ``creating`` for a later reconciliation to revisit.
    monkeypatch.setattr(execution, "_now", lambda: 1005.0)  # 5 s after dispatch, far inside the deadline
    journal, _ = open_record(tmp_path, state="creating", dispatched_at=1000.0)
    FakeBackend.reset()
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "create_uncertain"}]
    assert journal.unresolved()[0]["domains"]["reference"]["state"] == "creating"
    assert lookups("ngcg-x-reference") == 1 + execution._CREATE_SETTLE_POLLS  # initial lookup plus settle window


async def test_reconcile_crash_persisted_creating_after_deadline_resolves_deleted(tmp_path, monkeypatch):
    # Past the settle deadline a crash-persisted ``creating`` create can no longer be settling, so an absent
    # exact-name lookup is authoritative and clears the domain without an operator editing the journal.
    monkeypatch.setattr(execution, "_now", lambda: 1000.0 + 120.0 + 240.0 + 1.0)  # one second past the deadline
    journal, _ = open_record(tmp_path, state="creating", dispatched_at=1000.0)
    FakeBackend.reset()
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "absent"}]
    assert not journal.unresolved() and grader._leftover == {}
    assert lookups("ngcg-x-reference") == 1  # no settle window once the deadline has passed


async def test_reconcile_crash_persisted_creating_that_settles_after_the_first_lookup(
    tmp_path, fast_polls, monkeypatch
):
    # The create this process crashed during finished remotely afterwards. Inside the settle deadline the sandbox
    # is invisible on the first cleanup lookup and appears within the re-lookup window. Recovery must then find it
    # by exact name and delete it rather than having resolved the domain deleted on the first absence.
    monkeypatch.setattr(execution, "_now", lambda: 1005.0)  # well inside the deadline
    journal, record = open_record(tmp_path, state="creating", dispatched_at=1000.0)
    FakeBackend.reset()
    name = record["domains"]["reference"]["name"]
    FakeBackend.register("sb-late", name, record["domains"]["reference"]["labels"])
    script = {("lookup", name): [None]}  # absent on the first lookup, then visible in the plane
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "deleted"}]
    assert closes("sb-late") == 1 and FakeBackend.plane == {} and not journal.unresolved()


async def test_reconcile_deletes_matching_identity_by_id_and_confirms_absence(tmp_path, fast_polls):
    journal, _ = open_record(tmp_path, state="delete_failed", sandbox_id="sb-x")
    FakeBackend.reset()
    FakeBackend.register(
        "sb-x",
        "ngcg-x-reference",
        {
            "ng-grader-owner": "unit",
            "ng-grader-job": "x",
            "ng-grader-domain": "reference",
            "code-toolbox-language": "python",
        },
    )
    script = {("destroy_delay", "sb-x"): 2}
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "deleted"}]
    assert closes("sb-x") == 1 and lookups("sb-x") == 3 and FakeBackend.plane == {}
    assert not journal.unresolved()
    assert all(
        call[0] != "lookup" or call[1] == "sb-x"
        for b in FakeBackend.instances
        for call in b.calls
        if isinstance(call, tuple)
    )


async def test_reconcile_refuses_foreign_resource(tmp_path):
    journal, _ = open_record(tmp_path)
    FakeBackend.reset()
    FakeBackend.register("other", "ngcg-x-reference", {"ng-grader-owner": "someone"})
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "foreign_resource"}]
    assert journal.unresolved() and closes("other") == 0 and "other" in FakeBackend.plane


async def test_reconcile_requires_the_complete_owned_label_set(tmp_path):
    journal, _ = open_record(tmp_path)
    FakeBackend.reset()
    FakeBackend.register(
        "sb-p",
        "ngcg-x-reference",
        {"ng-grader-owner": "unit", "ng-grader-job": "x", "ng-grader-domain": "candidate"},
    )
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "foreign_resource"}]
    assert journal.unresolved() and closes("sb-p") == 0


async def test_reconcile_refuses_id_or_name_mismatch(tmp_path):
    journal, _ = open_record(tmp_path, state="delete_failed", sandbox_id="sb-x")
    FakeBackend.reset()
    labels = {"ng-grader-owner": "unit", "ng-grader-job": "x", "ng-grader-domain": "reference"}
    impostor = SandboxLookup(
        handle=SandboxHandle(
            sandbox_id="sb-y", provider_name="daytona", raw=FakeSandbox("sb-y", "ngcg-x-reference", labels)
        ),
        name="ngcg-x-reference",
        labels=labels,
        state="started",
    )
    script = {("lookup", "sb-x"): impostor}
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "foreign_resource"}]
    assert journal.unresolved() and closes("sb-y") == 0


@pytest.mark.parametrize("kind", LOOKUP_ERROR_KINDS)
async def test_reconcile_keeps_ownership_on_categorized_lookup_errors(tmp_path, kind):
    journal, _ = open_record(tmp_path)
    FakeBackend.reset()
    script = {("lookup", "ngcg-x-reference"): LookupFailed(kind)}
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, script), journal=journal)
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": f"lookup_{kind}"}]
    assert journal.unresolved()[0]["domains"]["reference"]["state"] == "create_uncertain"
    assert all(("close", "ngcg-x-reference") not in b.calls for b in FakeBackend.instances)


async def test_reconcile_maps_unexpected_lookup_exceptions_and_timeouts(tmp_path):
    journal, _ = open_record(tmp_path)
    FakeBackend.reset()
    script = {("lookup", "ngcg-x-reference"): [RuntimeError("boom"), hang]}
    grader = Grader(
        make_policy(tmp_path, cleanup_timeout_s=0.05),
        backend_factory=lambda p: FakeBackend(p, script),
        journal=journal,
    )
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "lookup_provider"}]
    assert await grader.reconcile() == [{"job_id": "x", "domain": "reference", "status": "lookup_timeout"}]
    assert journal.unresolved() and not stray_tasks()


async def test_reconcile_reports_corrupt_records_without_touching_the_provider(tmp_path):
    journal = OwnershipJournal(tmp_path / "journal")
    (tmp_path / "journal" / "bad.json").write_text("{not json")
    FakeBackend.reset()
    grader = Grader(make_policy(tmp_path), backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    assert await grader.reconcile() == [{"job_id": "bad", "status": "corrupt_record"}]
    assert FakeBackend.instances == []


async def test_delayed_destruction_is_confirmed_before_deleted(tmp_path, fast_polls):
    script = happy_script()
    script[("destroy_delay", "sb-candidate")] = 3
    grader, _ = make_grader(tmp_path, script)
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.cleanup["candidate"] == "deleted" and lookups("sb-candidate") == 3
    assert FakeBackend.plane == {} and not list((tmp_path / "journal").glob("*.json"))


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_accepted_but_unconfirmed_deletion_keeps_ownership(tmp_path, fast_polls, domain, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("destroy_delay", f"sb-{domain}")] = 10**6
    grader, _ = make_grader(tmp_path, script, cleanup_timeout_s=0.05)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup == {name: "delete_unconfirmed" if name == domain else "deleted" for name in DOMAINS}
    assert result.cases_attempted == 2 and result.cases_equal == (2 if status == "equal" else 1)
    assert result.first_failed_case == (None if status == "equal" else 1)
    record = journal_records(tmp_path)[0]
    assert identity(record["domains"][domain]) == {
        "name": f"ngcg-{result.job_id[:12]}-{domain}",
        "labels": {"ng-grader-owner": "unit", "ng-grader-job": result.job_id, "ng-grader-domain": domain},
        "state": "delete_unconfirmed",
        "sandbox_id": f"sb-{domain}",
    }
    assert grader._leftover == {result.job_id: record}
    assert (await grader.grade(make_task(), CANDIDATE)).category == "ownership_unresolved"
    # The gate reconciles on the job path, rebuilding one cleanup client for the delete-unconfirmed domain.
    # The destroy is still unconfirmed, so the leftover is retained and the job is refused. That is the 4th.
    assert len(FakeBackend.instances) == 4 and not stray_tasks()
    del script[("destroy_delay", f"sb-{domain}")]
    assert await grader.reconcile() == [{"job_id": result.job_id, "domain": domain, "status": "deleted"}]
    assert grader._leftover == {} and journal_records(tmp_path) == [] and FakeBackend.plane == {}


@pytest.mark.parametrize("status", ["equal", "mismatch"])
async def test_cleanup_timeout_unwinds_the_delete_in_this_task(tmp_path, status):
    script = happy_script()
    script[("result", "comparator")][1] = compare_result(["equal", status])
    script[("close", "sb-candidate")] = [hang, hang]
    grader, _ = make_grader(tmp_path, script, cleanup_timeout_s=0.05)
    result = await grader.grade(make_task(), CANDIDATE)
    assert (result.outcome, result.reward, result.category) == ("unscorable", None, "ownership_unresolved")
    assert result.failure_class == failure_kinds.PROVIDER_UNAVAILABLE and result.terminal is False
    assert result.cleanup["candidate"] == "delete_failed"
    candidate = domain_backends()["candidate"]
    assert candidate.calls.count(("close_interrupted", "sb-candidate")) == 2  # mid-job close and the cleanup retry
    assert not stray_tasks()  # nothing shielded into the background
    assert journal_records(tmp_path)[0]["domains"]["candidate"]["sandbox_id"] == "sb-candidate"
    report = await grader.reconcile()
    assert report == [{"job_id": result.job_id, "domain": "candidate", "status": "deleted"}]
    assert not list((tmp_path / "journal").glob("*.json"))


async def test_cancellation_during_cleanup_finishes_other_domains_then_propagates(tmp_path):
    script = happy_script()
    script[("result", "candidate")] = run_result("timeout", [VALUE], timed_out=1)  # comparator stays open for cleanup
    script[("exec", "candidate")] = exited(runner.EXIT_TIMEOUT)
    script[("close", "sb-comparator")] = [cancel_self]
    grader, _ = make_grader(tmp_path, script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    asyncio.current_task().uncancel()
    record = journal_records(tmp_path)[0]
    assert record["domains"]["comparator"] == {
        **record["domains"]["comparator"],
        "state": "delete_failed",
        "sandbox_id": "sb-comparator",
    }
    assert record["domains"]["candidate"]["state"] == "deleted" and closes("sb-candidate") == 1
    assert domain_backends()["comparator"].calls.count(("close_interrupted", "sb-comparator")) == 1
    assert not stray_tasks()
    report = await grader.reconcile()
    assert report == [{"job_id": record["job_id"], "domain": "comparator", "status": "deleted"}]
    assert FakeBackend.plane == {} and not list((tmp_path / "journal").glob("*.json"))


async def test_close_domains_continues_after_an_ordinary_close_error(tmp_path):
    # A journal-write OSError while resolving one live domain must not abort cleanup of the remaining domains.
    # The second sandbox is still deleted and its client still closed. The accumulated error is raised only after
    # every domain had its turn, so one domain's failure can never strand another domain's live sandbox.
    class FaultyJournal(OwnershipJournal):
        def update(self, record, domain, state, sandbox_id=None):
            if domain == "reference" and state == "deleted":
                raise OSError("synthetic journal write failure while persisting the transition")
            return super().update(record, domain, state, sandbox_id)

    def labels(domain):
        return {"ng-grader-owner": "unit", "ng-grader-job": "j", "ng-grader-domain": domain}

    FakeBackend.reset()
    policy = make_policy(tmp_path)
    journal = FaultyJournal(tmp_path / "journal")
    names = {d: f"ngcg-j-{d}" for d in DOMAINS}
    record = journal.open_job("j", "unit", names, {d: labels(d) for d in DOMAINS})
    live = ("reference", "comparator")
    for domain in live:
        journal.update(record, domain, "delete_failed", f"sb-{domain}")
        FakeBackend.register(f"sb-{domain}", names[domain], labels(domain))
    grader = Grader(policy, backend_factory=lambda p: FakeBackend(p, {}), journal=journal)
    domains = [_Domain(domain, policy, FakeBackend(policy, {}), journal, record) for domain in live]
    statuses: dict = {}
    with pytest.raises(OSError):
        await grader._close_domains(domains, statuses)
    # Both live sandboxes were deleted even though the first domain's resolution raised.
    assert closes("sb-reference") == 1 and closes("sb-comparator") == 1 and FakeBackend.plane == {}
    # The second (and first) domain still had its client closed exactly once.
    assert all(domain.backend.calls.count("aclose") == 1 for domain in domains)
    assert set(statuses) == set(live) and not stray_tasks()


async def test_cancellation_during_mid_job_close_retries_once_then_retains(tmp_path):
    script = happy_script()
    script[("close", "sb-candidate")] = [cancel_self, RuntimeError("still failing")]
    grader, _ = make_grader(tmp_path, script)
    with pytest.raises(asyncio.CancelledError):
        await grader.grade(make_task(), CANDIDATE)
    asyncio.current_task().uncancel()
    record = journal_records(tmp_path)[0]
    assert record["domains"]["candidate"]["state"] == "delete_failed"
    assert record["domains"]["candidate"]["sandbox_id"] == "sb-candidate"
    assert record["domains"]["comparator"]["state"] == "deleted"  # the cancelled job still deleted what it could
    assert closes("sb-candidate") == 0 and "sb-candidate" in FakeBackend.plane
    assert await grader.shutdown() == [{"job_id": record["job_id"], "domain": "candidate", "status": "deleted"}]
    assert FakeBackend.plane == {}


async def test_active_job_blocks_second_job_reconcile_and_shutdown(tmp_path):
    gate = asyncio.Event()

    async def hold():
        await gate.wait()

    script = happy_script()
    script[("exec", "candidate")] = hold
    grader, _ = make_grader(tmp_path, script, cleanup_timeout_s=0.05)
    job = asyncio.ensure_future(grader.grade(make_task(), CANDIDATE))
    await wait_until(
        lambda: any(
            isinstance(call, tuple) and call[0] == "exec"
            for backend in FakeBackend.instances
            if backend.domain == "candidate"
            for call in backend.calls
        )
    )
    live = dict(FakeBackend.plane)
    assert set(live) == {"sb-comparator", "sb-candidate"}
    assert (await grader.grade(make_task(), CANDIDATE)).category == "busy"
    with pytest.raises(GraderBusy):
        await grader.reconcile()
    assert await grader.shutdown() == [{"status": "busy"}]
    assert dict(FakeBackend.plane) == live and closes("sb-comparator") == closes("sb-candidate") == 0
    assert len(FakeBackend.instances) == 3  # no fresh clients were opened against the running job
    gate.set()
    result = await job
    assert result.reward == 1.0 and result.cleanup == {domain: "deleted" for domain in DOMAINS}
    assert FakeBackend.plane == {} and await grader.shutdown() == []


# --- runner static helpers (synthetic sources, never executed) ---------------------------------- #


def test_entrypoint_binding_fails_closed():
    tree, _ = runner.parse_source("def f(x):\n    return x\n")
    assert runner.check_entrypoint_binding(tree, "f") is None
    assert runner.check_entrypoint_binding(tree, "g") == "entrypoint_missing"
    tree, _ = runner.parse_source("def f(x):\n    return x\n\ndef f(y):\n    return y\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\nf = 3\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\nfrom os import *\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("async def f(x):\n    return x\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_missing"
    assert runner.parse_source("def f(:\n") == (None, "syntax_error")


def test_walrus_local_to_the_entrypoint_body_is_not_a_conflict():
    # A walrus inside the entrypoint body binds in the function's own scope, so it does not rebind the
    # module-level entrypoint.
    tree, error = runner.parse_source("def f(x):\n    return (f := x + 1)\n")
    assert error is None and runner.check_entrypoint_binding(tree, "f") is None
    # A walrus that binds the entrypoint name inside a different top-level function's body is also local.
    tree, _ = runner.parse_source("def f(x):\n    return x\ndef g(x):\n    return (f := x)\n")
    assert runner.check_entrypoint_binding(tree, "f") is None
    # A walrus binding a name other than the entrypoint, inside the entrypoint body, is likewise allowed.
    tree, _ = runner.parse_source("def f(x):\n    return (y := x + 1)\n")
    assert runner.check_entrypoint_binding(tree, "f") is None


def test_module_scope_walrus_rebinding_the_entrypoint_stays_a_conflict():
    # A walrus evaluated at module scope still rebinds the entrypoint. A bare module-level expression, and a
    # default argument or decorator expression on a top-level def, are all evaluated in the enclosing scope.
    tree, error = runner.parse_source("def f(x):\n    return x\n(f := 1)\n")
    assert error is None and runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source("def f(x):\n    return x\ndef g(y=(f := 1)):\n    return y\n")
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"
    tree, _ = runner.parse_source(
        "def deco(fn):\n    return fn\ndef f(x):\n    return x\n@(f := deco)\ndef g():\n    return 1\n"
    )
    assert runner.check_entrypoint_binding(tree, "f") == "entrypoint_conflict"


def test_run_job_validation_rejects_unknown_shapes(tmp_path):
    job = build_compare_job(make_task(), "candidate", [VALUE, VALUE], "abc", make_policy(tmp_path))
    assert runner.validate_job(job, "compare") is None
    assert runner.validate_job(dict(job, observed_role="judge"), "compare") == "invalid_job"
    assert runner.validate_job(dict(job, mode="run"), "run") == "invalid_job"


# --- parity: extra keys, bounds, complex carriers, input conversions --------------------------- #


def _delivered_task(**overrides):
    """A minimal valid task dict in the canonical shape. Overrides replace top-level fields."""
    base = {
        "problem_id": "p1",
        "reference_source": "def f(x):\n    return x\n",
        "entrypoint": "f",
        "test_cases": [{"args": [wire(["int", "1"])], "expected": value_outcome(["int", "1"])}],
    }
    base.update(overrides)
    return base


def test_build_run_job_always_carries_input_conversions(tmp_path):
    policy = make_policy(tmp_path)
    # A task that declares none still puts the key in the job as an empty list.
    assert build_run_job(make_task(), "f", "abcd", policy)["input_conversions"] == []
    # A task that declares conversions transports them verbatim, aligned with the positional inputs.
    task = TaskData.model_validate(_delivered_task(input_conversions=[None, "symbol", "function"]))
    assert build_run_job(task, "f", "abcd", policy)["input_conversions"] == [None, "symbol", "function"]


@pytest.mark.parametrize(
    "conversions", [[], [None], ["symbol"], ["function"], [None, "symbol", "function"], [None] * 128]
)
def test_task_data_accepts_valid_input_conversions(conversions):
    assert TaskData.model_validate(_delivered_task(input_conversions=conversions)).input_conversions == conversions


@pytest.mark.parametrize(
    "conversions", [["symbolic"], ["Symbol"], [1], [True], "symbol", {"0": "symbol"}, [None] * 129]
)
def test_task_data_rejects_bad_input_conversions(conversions):
    with pytest.raises(ValidationError):
        TaskData.model_validate(_delivered_task(input_conversions=conversions))


def test_task_data_input_conversions_default_is_none():
    assert TaskData.model_validate(_delivered_task()).input_conversions is None


@pytest.mark.parametrize("comparison_type", ["numeric", "", "x" * 64])
def test_keyword_shape_accepts_and_drops_comparison_type(comparison_type):
    task = TaskData.model_validate(
        _delivered_task(
            test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": comparison_type}]
        )
    )
    # The key is dropped, and the case normalizes to the keyword shape unchanged.
    case = task.test_cases[0]
    assert case.kwargs["v"].value == ["legacy", "1"] and case.expected.value.value == ["legacy", "2"]


def test_keyword_shape_comparison_type_may_be_null():
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": None}])
    )
    assert task.test_cases[0].kwargs["v"].value == ["legacy", "1"]


@pytest.mark.parametrize("comparison_type", [123, ["numeric"], "x" * 65])
def test_keyword_shape_rejects_invalid_comparison_type(comparison_type):
    with pytest.raises(ValidationError):
        TaskData.model_validate(
            _delivered_task(
                test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "comparison_type": comparison_type}]
            )
        )


def test_comparison_type_is_only_accepted_on_the_keyword_shape():
    # The positional shape does not take the extra key. No other unknown key is accepted anywhere.
    with pytest.raises(ValidationError):
        TaskData.model_validate(
            _delivered_task(test_cases=[{"input": [1], "output": 1, "comparison_type": "numeric"}])
        )
    with pytest.raises(ValidationError):
        TaskData.model_validate(_delivered_task(test_cases=[{"inputs": {"v": "1"}, "expected_output": "2", "z": 1}]))


def test_complex_carrier_becomes_a_complex_wire_node_on_args_and_expected():
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"input": [{"__complex__": [1, 2.5]}], "output": {"__complex__": [3, 4]}}])
    )
    case = task.test_cases[0]
    assert case.args[0].value == ["complex", ["int", "1"], ["float", (2.5).hex()]]
    assert case.expected.value.value == ["complex", ["int", "3"], ["int", "4"]]


def test_complex_carrier_nests_through_lists_and_maps():
    task = TaskData.model_validate(
        _delivered_task(
            test_cases=[{"input": [[{"__complex__": [0, -1]}]], "output": {"z": {"__complex__": [1.0, 0.0]}}}]
        )
    )
    case = task.test_cases[0]
    assert case.args[0].value == ["list", [["complex", ["int", "0"], ["int", "-1"]]]]
    assert case.expected.value.value == ["map", [["z", ["complex", ["float", (1.0).hex()], ["float", (0.0).hex()]]]]]


@pytest.mark.parametrize(
    "carrier",
    [
        {"__complex__": [1, 2, 3]},  # not exactly two parts
        {"__complex__": [True, 2]},  # a bool part is not a real
        {"__complex__": [float("inf"), 2]},  # a nonfinite part
        {"__complex__": [1, 2], "x": 1},  # an extra key
        {"__complex__": "nope"},  # not a list
    ],
)
def test_non_complex_dicts_stay_plain_maps(carrier):
    task = TaskData.model_validate(_delivered_task(test_cases=[{"input": [carrier], "expected_error": "boom"}]))
    assert task.test_cases[0].args[0].value[0] == "map"


def test_large_value_of_the_delivered_shape_validates_with_margin():
    # A dict holding a list of pair-lists of small ints: ~12.5k nodes at depth 4, 12 such cases in one task.
    grid = {"grid": [[[i % 1000, (i * 7) % 1000] for _ in range(65)] for i in range(64)]}
    task = TaskData.model_validate(
        _delivered_task(test_cases=[{"inputs": {"n": "1"}, "expected_output": grid} for _ in range(12)])
    )
    assert len(task.test_cases) == 12


# --- remote ------------------------------------------------------------------------------------- #


@pytest.mark.sandbox
async def test_end_to_end_against_daytona(tmp_path):
    snapshot = os.environ.get("NG_CRITPT_GRADER_SNAPSHOT")
    os_user = os.environ.get("NG_CRITPT_GRADER_OS_USER")
    if not os.environ.get("DAYTONA_API_KEY") or not snapshot or not os_user:
        pytest.skip("requires DAYTONA_API_KEY, NG_CRITPT_GRADER_SNAPSHOT and NG_CRITPT_GRADER_OS_USER")
    # cpu_time_limit_s must stay at or below the suite deadline plus margin (120 + 60), or the policy validator
    # rejects the shrunk timeouts before any grading happens. The default 1800 exceeds it.
    # os_user must name the snapshot's real non-root account. A name absent from the snapshot makes the in-sandbox
    # runuser drop fail, so the reference run writes no result and the job ends unscorable, not reward 1.0.
    grader = Grader(
        make_policy(
            tmp_path,
            snapshot=snapshot,
            os_user=os_user,
            suite_timeout_s=120,
            compare_timeout_s=60,
            cpu_time_limit_s=120,
        )
    )
    result = await grader.grade(make_task(), CANDIDATE)
    assert result.reward == 1.0 and result.cleanup == {d: "deleted" for d in DOMAINS}


def test_shrunk_suite_timeouts_need_a_compatible_cpu_time_limit(tmp_path):
    # The opt-in Daytona test shrinks the suite and compare timeouts. The same shrink is invalid while
    # cpu_time_limit_s keeps its 1800 default, and valid once cpu_time_limit_s drops to the suite deadline plus
    # margin. This host-safe test guards the construction the credential skip hides.
    with pytest.raises(ValidationError, match="cpu_time_limit_s must not exceed the suite deadline plus margin"):
        make_policy(tmp_path, suite_timeout_s=120, compare_timeout_s=60)
    policy = make_policy(tmp_path, suite_timeout_s=120, compare_timeout_s=60, cpu_time_limit_s=120)
    assert (policy.suite_timeout_s, policy.compare_timeout_s, policy.cpu_time_limit_s) == (120, 60, 120)


# ---- appended to tests/test_execution.py: local provider subclass ----
# (imports at top: SimpleNamespace, pytest, SandboxExecResult, SandboxHandle, make_policy, provider_support)


def _exec_handle(response=None, *, raise_exc=None):
    async def fake_exec(command, *, cwd=None, env=None, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        return response

    process = SimpleNamespace(exec=fake_exec)
    return SandboxHandle("sb-8b12", "daytona", raw=SimpleNamespace(process=process))


def test_local_provider_requires_a_boolean_flag():
    with pytest.raises(TypeError):
        provider_support.CategoryOnlyDaytonaProvider(category_only_diagnostics="yes")


def test_local_provider_stores_the_flag_from_domain_config(tmp_path):
    config = make_policy(tmp_path).domain_config()
    provider = provider_support.CategoryOnlyDaytonaProvider(**config)
    assert provider._category_only_diagnostics is True


@pytest.mark.parametrize("category_only", [True, False])
async def test_missing_exit_code_becomes_a_sandbox_failure(tmp_path, category_only):
    # A response with no process exit code must never read as success, whatever the diagnostics mode.
    config = make_policy(tmp_path).domain_config()
    config["category_only_diagnostics"] = category_only
    provider = provider_support.CategoryOnlyDaytonaProvider(**config)
    response = SimpleNamespace(exit_code=None, result="out", stderr="err", artifacts=None)
    result = await provider._exec(_exec_handle(response), "cmd", timeout_s=5, retries=0)
    assert result.return_code == provider_support.SANDBOX_RUNTIME_RETURN_CODE
    assert result.error_type == "sandbox"
    assert result.stdout == "out" and "err" in (result.stderr or "")


@pytest.mark.parametrize("exit_code", [True, "0", 1.0, None])
async def test_non_integer_exit_code_becomes_a_sandbox_failure(tmp_path, exit_code):
    provider = provider_support.CategoryOnlyDaytonaProvider(**make_policy(tmp_path).domain_config())
    response = SimpleNamespace(exit_code=exit_code, result=None, stderr=None, artifacts=None)
    result = await provider._exec(_exec_handle(response), "cmd", timeout_s=5, retries=0)
    assert result.return_code == provider_support.SANDBOX_RUNTIME_RETURN_CODE and result.error_type == "sandbox"


@pytest.mark.parametrize("code", [0, 1, 7, 137])
async def test_integer_exit_code_passes_through(tmp_path, code):
    provider = provider_support.CategoryOnlyDaytonaProvider(**make_policy(tmp_path).domain_config())
    response = SimpleNamespace(exit_code=code, result="out", stderr="err", artifacts=None)
    result = await provider._exec(_exec_handle(response), "cmd", timeout_s=5, retries=0)
    assert (result.return_code, result.error_type, result.stdout, result.stderr) == (code, None, "out", "err")


async def test_category_only_exec_error_text_is_a_fixed_category(tmp_path, monkeypatch):
    # A recognised command-exec error returns a typed sandbox failure whose text is a fixed category,
    # never provider-authored free text, when category-only diagnostics are on.
    monkeypatch.setattr(provider_support, "_is_daytona_command_exec_error", lambda exc: True)
    provider = provider_support.CategoryOnlyDaytonaProvider(**make_policy(tmp_path).domain_config())
    marker = "SYNTHETIC-PROVIDER-TEXT-8b12"
    error = RuntimeError(f"Failed to execute command: {marker}")
    result = await provider._exec(_exec_handle(raise_exc=error), "cmd", timeout_s=5, retries=0)
    assert result.return_code == provider_support.SANDBOX_RUNTIME_RETURN_CODE and result.error_type == "sandbox"
    assert result.stderr == "daytona.command_exec_failed"
    assert marker not in (result.stderr or "")


async def test_default_diagnostics_exec_error_uses_shared_provider_text(tmp_path, monkeypatch):
    # With the flag off, the exec-error result is the shared provider's, so behaviour matches core there.
    monkeypatch.setattr(provider_support, "_is_daytona_command_exec_error", lambda exc: True)
    config = make_policy(tmp_path).domain_config()
    config["category_only_diagnostics"] = False
    provider = provider_support.CategoryOnlyDaytonaProvider(**config)
    error = RuntimeError("Failed to execute command: synthetic")
    result = await provider._exec(_exec_handle(raise_exc=error), "cmd", timeout_s=5, retries=0)
    assert result.return_code == provider_support.SANDBOX_RUNTIME_RETURN_CODE and result.error_type == "sandbox"
    assert result.stderr and result.stderr != "daytona.command_exec_failed"
