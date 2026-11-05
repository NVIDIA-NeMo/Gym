# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test support: an in-memory SandboxProvider with scripted outcomes. Nothing executes on the host."""

import json
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from resources_servers.critpt_custom_grader.execution import ExecutionPolicy, Grader
from resources_servers.critpt_custom_grader.task_data import WIRE_FORMAT, TaskData


_MISSING = object()
MISSING_FILE = object()  # a scripted result: the sandbox never wrote the file
DOMAIN_LABEL = "ng-grader-domain"
CANDIDATE = "def f(x):\n    return x + 1\n"


def wire(node):
    return {"format": WIRE_FORMAT, "value": node}


def value_outcome(node):
    return {"kind": "value", "value": wire(node)}


VALUE = value_outcome(["int", "2"])
EXCEPTION = {"kind": "exception", "type": "ValueError"}
ENCODING_ERROR = {"kind": "encoding_error", "code": "unsupported_type"}


def run_result(status, outcomes, code=None, timed_out=None):
    """A run envelope as runner._ResultWriter writes it; the provider fills in the uploaded job's nonce."""
    return {
        "version": 1,
        "mode": "run",
        "nonce": "",
        "status": status,
        "code": code,
        "timed_out_case": timed_out,
        "outcomes": outcomes,
    }


def verdict(status, side="candidate", code="compared", path=""):
    return {
        "version": 1,
        "status": status,
        "equal": True if status == "equal" else False if status == "mismatch" else None,
        "side": side,
        "code": code,
        "path": path,
    }


def compare_result(statuses, side="candidate"):
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


def exited(code, **fields):
    return SandboxExecResult(stdout="", stderr="", return_code=code, **fields)


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


def make_spec(**overrides) -> SandboxSpec:
    fields = {
        "image": "example/grader:test",
        "workdir": "/work",
        "ttl_s": 600,
        "resources": {"cpu": 1, "memory_mib": 2048},
    }
    fields.update(overrides)
    return SandboxSpec(**fields)


def happy_script():
    return {
        ("result", "reference"): [run_result("completed", [VALUE, VALUE])],
        ("result", "comparator"): [
            compare_result(["equal", "equal"], side="reference"),
            compare_result(["equal", "equal"]),
        ],
        ("result", "candidate"): [run_result("completed", [VALUE, VALUE])],
    }


def runner_execs(provider, domain):
    return [call for call in provider.exec_calls if call["domain"] == domain and "runner.py" in call["command"]]


class ScriptedProvider:
    """SandboxProvider over one in-memory plane, driven by a script of outcomes. Nothing executes on the host.

    Script keys are ``("create" | "exec" | "result" | "stop", domain)`` plus ``"uid"`` and ``"imports_ok"``. A list
    value is consumed one entry per call, a scalar repeats on every call, an exception is raised, a callable is
    awaited (its return value is used where the call returns one) and a bytes result is returned verbatim; any other
    result is a JSON object that receives the nonce of the box's last uploaded job.

    Probe commands are answered from the plane, never from the script: ``mkdir -p`` exits 0, ``id -u`` prints
    ``script["uid"]`` (default 1000), ``test -f ... && stat -c %s ...`` reports the next scripted result's size
    (exit 1 for MISSING_FILE, an exception, a callable, a missing key or an exhausted list), and an
    ``-c "import ..."`` probe exits 0 unless ``script["imports_ok"]`` is False.

    ``("download", domain)`` is consulted by ``download_file`` before the result entry: an exception instance or class
    raises (the size probe still answers from the result entry, so the file looks present and only the transfer
    fails); anything else, or no entry, leaves the download to the ``("result", domain)`` script.
    """

    name = "critpt_fake"
    instances: list["ScriptedProvider"] = []

    def __init__(self, script: dict | None = None, **_: Any) -> None:
        self.script: dict = script if script is not None else {}
        self.sandboxes: dict[str, dict] = {}
        self.exec_calls: list[dict] = []
        self.created: list[SandboxSpec] = []
        self.closed: list[str] = []
        self.aclosed = 0
        ScriptedProvider.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def _scripted(self, key):
        outcome = self.script.get(key, _MISSING)
        if isinstance(outcome, list):
            return outcome.pop(0) if outcome else _MISSING
        return outcome

    def _peek(self, key):
        outcome = self.script.get(key, _MISSING)
        if isinstance(outcome, list):
            return outcome[0] if outcome else _MISSING
        return outcome

    @staticmethod
    def _materialize(payload, box) -> bytes:
        if type(payload) is bytes:
            return payload
        jobs = [path for path in box["files"] if path.endswith("/job.json")]
        nonce = json.loads(box["files"][jobs[-1]]).get("nonce", "") if jobs else ""
        return json.dumps(dict(payload, nonce=nonce)).encode()

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        domain = spec.metadata.get(DOMAIN_LABEL, "probe")
        outcome = self._scripted(("create", domain))
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome = await outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        self.created.append(spec)
        sandbox_id = f"fake-{domain}-{len(self.created)}"
        self.sandboxes[sandbox_id] = {"domain": domain, "spec": spec, "files": {}, "live": True}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None) -> SandboxExecResult:
        box = self.sandboxes[handle.sandbox_id]
        self.exec_calls.append(
            {"domain": box["domain"], "command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user}
        )
        if command.startswith("mkdir -p"):
            return exited(0)
        if command.startswith("id -u"):
            return SandboxExecResult(stdout=f"{self.script.get('uid', 1000)}\n", stderr="", return_code=0)
        if command.startswith("test -f"):
            payload = self._peek(("result", box["domain"]))
            if (
                payload is MISSING_FILE
                or payload is _MISSING
                or isinstance(payload, BaseException)
                or callable(payload)
            ):
                return SandboxExecResult(stdout="", stderr="", return_code=1)
            return SandboxExecResult(stdout=f"{len(self._materialize(payload, box))}\n", stderr="", return_code=0)
        if " -c " in command and "import" in command:
            return exited(0 if self.script.get("imports_ok", True) else 1)
        outcome = self._scripted(("exec", box["domain"]))
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome = await outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return exited(0) if outcome is _MISSING or outcome is None else outcome

    async def upload_file(self, handle, source_path: Path, target_path: str) -> None:
        self.sandboxes[handle.sandbox_id]["files"][target_path] = Path(source_path).read_bytes()

    async def download_file(self, handle, source_path: str, target_path: Path) -> None:
        box = self.sandboxes[handle.sandbox_id]
        failure = self._scripted(("download", box["domain"]))
        if isinstance(failure, BaseException) or (isinstance(failure, type) and issubclass(failure, BaseException)):
            raise failure
        payload = self._scripted(("result", box["domain"]))
        if callable(payload) and not isinstance(payload, BaseException):
            payload = await payload()
        if isinstance(payload, BaseException):
            raise payload
        if payload is MISSING_FILE or payload is _MISSING:
            raise FileNotFoundError(source_path)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(self._materialize(payload, box))

    async def status(self, handle) -> SandboxStatus:
        return SandboxStatus.RUNNING if self.sandboxes[handle.sandbox_id]["live"] else SandboxStatus.STOPPED

    async def close(self, handle) -> None:
        box = self.sandboxes[handle.sandbox_id]
        outcome = self._scripted(("stop", box["domain"]))
        if callable(outcome) and not isinstance(outcome, BaseException):
            await outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        box["live"] = False
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        self.aclosed += 1


def make_policy(**overrides) -> ExecutionPolicy:
    fields = {
        "exec_user": "grader",
        "interpreter": "/usr/local/bin/python3",
        "workdir": "/work",
        "create_timeout_s": 5.0,
        "transfer_timeout_s": 5.0,
        "cleanup_timeout_s": 5.0,
        "exec_timeout_margin_s": 20.0,
    }
    fields.update(overrides)
    return ExecutionPolicy(**fields)


def make_grader(script, *, policy=None, spec=None, cache=None, **policy_overrides):
    """(Grader, ScriptedProvider) over one scripted provider; the Grader owns no provider."""
    ScriptedProvider.reset()
    provider = ScriptedProvider(script)
    grader = Grader(
        policy or make_policy(**policy_overrides), provider=provider, spec=spec or make_spec(), reference_cache=cache
    )
    return grader, provider
