# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-native deadlines and unchanged failure delivery, independent of trajectory capture."""

import asyncio
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, time
from typing import Unpack
from uuid import uuid4

from aiohttp import ClientResponse
from aiohttp.client import _RequestOptions
from fastapi import Request
from fastapi.responses import JSONResponse
from multidict import CIMultiDict
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client import ServerClient
from nemo_gym.core.failure_kinds import MODEL_POLICY_RESPONSE_TIMEOUT
from nemo_gym.core.failure_records import FailureReport
from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow
from nemo_gym.rollouts.harness_capabilities import Results
from nemo_gym.sandbox import LocalProvider, SandboxHandle, register_provider

from .probes import Probe


class NativeEvidenceProvider(LocalProvider):
    """Retain actual runtime downloads before ordinary session cleanup removes them."""

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        await super().download_file(handle, source_path, target_path)
        retained = Path(handle.raw["workspace"]).parent / "native_evidence"
        retained.mkdir(exist_ok=True)
        shutil.copy2(target_path, retained / (uuid4().hex + "-" + Path(source_path).name))


register_provider("native_timeout_local", NativeEvidenceProvider, override=True)


@dataclass
class NativeTimeoutProbe(Probe):
    """Leave scripted calls pending until the real harness's shorter native deadline fires."""

    native_timeout_seconds: float = 1
    native_error_pattern: str = ""
    replies: list[dict] = field(default_factory=list)
    caller_replies: list[dict] = field(default_factory=list)

    def save(self) -> None:
        super().save()
        path = self.directory / "witness.json"
        witness = json.loads(path.read_text())
        artifacts = []
        for source in sorted((self.directory / "native_evidence").glob("*")):
            text = source.read_bytes().decode(errors="replace")
            if self.native_error_pattern and (match := re.search(self.native_error_pattern, text, re.IGNORECASE)):
                artifacts.append({"path": source.name, "error": text[max(0, match.start() - 80) : match.end() + 80]})
        witness["native_timeout"] = {
            "seconds": self.native_timeout_seconds,
            "error_pattern": self.native_error_pattern,
            "stall_seconds": self.responses[0]["delay_seconds"],
            "replies": self.replies,
            "caller_replies": self.caller_replies,
            "artifacts": artifacts,
        }
        path.write_text(json.dumps(witness, indent=2) + "\n")

    async def model(self, request: Request) -> JSONResponse:
        body = await request.json()
        script = self.responses[0]
        if script["kind"] != "stall" or script["delay_seconds"] <= self.native_timeout_seconds:
            raise ValueError("native deadline requires a longer scripted stall")
        attempt = {"request": body, "status_code": None, "response": None, "started_at": time()}
        self.attempts.append(attempt)
        self.save()
        await asyncio.sleep(script["delay_seconds"])
        # Reaching this is a fixture failure: the native deadline did not win.
        attempt.update(status_code=504, response={"error": {"message": "scripted stall expired"}})
        self.save()
        return JSONResponse(attempt["response"], status_code=504)


class ReplyWitness:
    """Observe the actual agent's wire reply without changing its bytes or diagnostics."""

    def __init__(self, app: ASGIApp, probe: NativeTimeoutProbe) -> None:
        self.app, self.probe = app, probe

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").endswith(("/responses", "/run")):
            await self.app(scope, receive, send)
            return
        started, headers, body, status = monotonic(), {}, bytearray(), None

        async def observed(message: Message) -> None:
            nonlocal status, headers
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = CIMultiDict((key.decode(), value.decode()) for key, value in message.get("headers", []))
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    report = FailureHeader.decode(headers) or FailureReport()
                    self.probe.replies.append(
                        {
                            "status_code": status,
                            "body": body.decode(errors="replace"),
                            "report": report.model_dump(mode="json"),
                            "elapsed_seconds": monotonic() - started,
                        }
                    )
                    self.probe.save()
            await send(message)

        await self.app(scope, receive, observed)


def _report(value: object) -> FailureReport | None:
    try:
        return FailureReport.model_validate(value)
    except ValidationError:
        return None


def _causes(report: FailureReport | None, target: str | None) -> bool:
    if report is None or target is None:
        return False
    current, seen = report.failure_id, set()
    while current in report.events and current not in seen:
        if current == target:
            return True
        seen.add(current)
        current = report.events[current].cause_id
    return False


def native_timeout_observed(witness: dict, execution: dict) -> bool:
    """Recognize the deliberate native failure, without swallowing setup or outer-deadline errors."""
    native = witness.get("native_timeout", {})
    replies = native.get("replies", [])
    reply = replies[0] if len(replies) == 1 else {}
    attempts = witness.get("attempts", [])
    pattern = native.get("error_pattern")
    return bool(
        bool(attempts)
        and all(attempt.get("status_code") is None for attempt in attempts)
        and not execution.get("timed_out", True)
        and bool(pattern)
        and any(
            re.search(pattern, artifact.get("error", ""), re.IGNORECASE) is not None
            for artifact in native.get("artifacts", [])
        )
        and 0 < native.get("seconds", 0) < reply.get("elapsed_seconds", 0) < native.get("stall_seconds", 0)
    )


def inspect_native_timeout(witness: dict | None, record: dict | None, execution: dict) -> list[dict]:
    """Require a native error and all original events even when turns/model captures are absent."""
    witness = witness or {}
    native = witness.get("native_timeout", {})
    replies, callers = native.get("replies", []), native.get("caller_replies", [])
    reply = replies[0] if len(replies) == 1 else {}
    origin = _report(reply.get("report"))
    caller = _report(callers[0]) if len(callers) == 1 else None
    try:
        saved = FailureReportRow.report(record) if record is not None else None
    except (ValidationError, ValueError):
        saved = None
    results = Results()

    def check(name: str, passed: bool, reason: str) -> None:
        results.add(
            "native_timeout." + name,
            "behavioral",
            "pass" if passed else "fail",
            tier="P1",
            evidence=("TE-1", "TE-8"),
            location="witness.json",
            reason=reason if not passed else "",
        )

    check(
        "trigger",
        native_timeout_observed(witness, execution),
        "the native model-call deadline must win while scripted model calls remain pending",
    )
    selected = origin.events.get(origin.failure_id) if origin is not None else None
    check(
        "failure_event",
        selected is not None and not origin.omitted and selected.failure_kind != MODEL_POLICY_RESPONSE_TIMEOUT,
        "a native timeout needs an accurately sourced agent failure, not an invented Gym response timeout",
    )
    for name, report in (("caller_events", caller), ("saved_events", saved)):
        check(
            name,
            origin is not None
            and bool(origin.events)
            and not origin.omitted
            and report is not None
            and not report.omitted
            and all(report.events.get(key) == event for key, event in origin.events.items()),
            "every original NgFailureEvent must reach " + name + " unchanged",
        )
    check(
        "terminal_failure",
        origin is not None
        and origin.failure_id is not None
        and _causes(caller, origin.failure_id)
        and _causes(saved, origin.failure_id),
        "the native failure must remain selected at the caller and in the canonical saved outcome",
    )
    return results.dump()


def observe_caller(client: ServerClient, probe: NativeTimeoutProbe) -> ServerClient:
    """Observe reports received by the EnvironmentServer, separately from the sender."""

    class ObservedClient(ServerClient):
        async def post(self, server_name: str, url_path: str, **kwargs: Unpack[_RequestOptions]) -> ClientResponse:
            response = await super().post(server_name, url_path, **kwargs)
            if server_name == "probe_agent" and url_path.endswith(("/responses", "/run")):
                report = FailureHeader.decode(response.headers) or FailureReport()
                probe.caller_replies.append(report.model_dump(mode="json"))
                probe.save()
            return response

    return ObservedClient(head_server_config=client.head_server_config, global_config_dict=client.global_config_dict)
