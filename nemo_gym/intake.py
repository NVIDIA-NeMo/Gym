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
"""Client for submitting a finished eval rundir to a benchmark-run data layer.

The service is external and its base URL is configured, never baked in: ``--base-url`` or
``NEMO_GYM_INTAKE_BASE_URL``.

A rundir is named by locator, ``<cluster>:<path>``, and the service reads it off shared storage
itself — no bytes leave this machine. Submission is asynchronous: the call returns a job id once
the request is durable, and the run appears only once the service has decoded it.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import aiohttp


BASE_URL_ENV_VAR_NAME = "NEMO_GYM_INTAKE_BASE_URL"

DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_POLL_TIMEOUT_S = 1800.0


class IntakeError(Exception):
    """The service refused a request, or the request could not be formed."""


@dataclass(frozen=True)
class Submission:
    """An accepted submission and the path its progress is polled at."""

    id: str
    status_path: str
    # The service's accept body, verbatim.
    response: dict


@dataclass(frozen=True)
class IntakeOutcome:
    """Where polling stopped."""

    state: str
    run_ids: list[str]
    detail: dict

    @property
    def succeeded(self) -> bool:
        # A dedup is a success: the bytes are in the catalog under `run_ids`.
        return self.state in {"committed", "deduped"}


def resolve_base_url(explicit: Optional[str] = None) -> str:
    """The data layer's base URL: ``--base-url``, else ``NEMO_GYM_INTAKE_BASE_URL``."""
    base_url = explicit or os.environ.get(BASE_URL_ENV_VAR_NAME)
    if not base_url:
        raise IntakeError(
            f"No intake service configured. Pass --base-url, or set {BASE_URL_ENV_VAR_NAME} to the "
            f"base URL of your benchmark-run data layer."
        )
    return base_url.rstrip("/")


def parse_lustre_locator(locator: str) -> tuple[str, str]:
    """Split ``<cluster>:<absolute path>``. Shape only; which paths are accepted is the service's call."""
    cluster, separator, path = locator.partition(":")
    if not separator or not cluster or not path:
        raise IntakeError(f"Malformed locator {locator!r}: expected '<cluster>:<absolute path>'.")
    if not path.startswith("/"):
        raise IntakeError(f"Malformed locator {locator!r}: the path must be absolute.")
    return cluster, path


async def _read_error(response: aiohttp.ClientResponse) -> str:
    """Keep the service's refusal body verbatim."""
    body = (await response.text()).strip()
    prefix = f"{response.status} {response.reason} from {response.url.path}"
    return f"{prefix}: {body}" if body else prefix


async def _json_or_raise(response: aiohttp.ClientResponse, *, ok: tuple[int, ...]) -> dict:
    if response.status not in ok:
        raise IntakeError(await _read_error(response))
    return await response.json()


async def submit(
    session: aiohttp.ClientSession,
    base_url: str,
    locator: str,
    *,
    label: Optional[str] = None,
) -> Submission:
    """Submit a rundir on shared storage. Moves no bytes."""
    parse_lustre_locator(locator)
    payload: dict[str, str] = {"lustre": locator}
    if label:
        payload["label"] = label
    async with session.post(f"{base_url}/v1/intake", json=payload) as response:
        body = await _json_or_raise(response, ok=(200, 201, 202))
    job_id = body.get("job_id") or body.get("uuid") or body.get("id")
    if not job_id:
        raise IntakeError(f"The service accepted the locator but named no job id: {body}")
    status_path = body.get("status_url") or f"/v1/intake/jobs/{job_id}"
    return Submission(id=str(job_id), status_path=status_path, response=body)


def _absolute(base_url: str, path_or_url: str) -> str:
    return path_or_url if path_or_url.startswith("http") else f"{base_url}{path_or_url}"


# Everything else the poll route reports (queued/claimed/relaying/uploaded) is a wait.
_TERMINAL_STATES = {"committed", "deduped", "failed", "rejected", "quarantined"}


def _read_status(body: dict) -> IntakeOutcome:
    """Normalise one status body into an outcome.

    The job names both `status`, how far the transport got, and `parse_state`, what the decode
    made of the bytes. The decode decides, so it is read first.
    """
    committed = [str(run_id) for run_id in (body.get("run_ids") or [])]
    deduped = [str(run_id) for run_id in (body.get("deduped") or [])]

    state = body.get("parse_state") or body.get("status") or "unknown"
    if state == "parsed":
        state = "deduped" if deduped and not committed else "committed"
    return IntakeOutcome(state=state, run_ids=list(dict.fromkeys(committed + deduped)), detail=body)


async def poll(
    session: aiohttp.ClientSession,
    base_url: str,
    submission: Submission,
    *,
    timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    on_wait: Optional[Callable[[float, IntakeOutcome], None]] = None,
) -> Optional[IntakeOutcome]:
    """Wait for the service to finish with this submission. `None` means the wait ran out."""
    url = _absolute(base_url, submission.status_path)
    deadline = time.monotonic() + timeout_s
    waited = 0.0
    while True:
        async with session.get(url) as response:
            # 404 is terminal: no record of this id, and polling will not create one.
            if response.status == 404:
                return IntakeOutcome(
                    state="failed",
                    run_ids=[],
                    detail={"reason": f"the service has no record of job {submission.id}"},
                )
            body = await _json_or_raise(response, ok=(200,))
        outcome = _read_status(body)
        if outcome.state in _TERMINAL_STATES:
            return outcome
        if time.monotonic() + interval_s >= deadline:
            return None
        if on_wait is not None:
            on_wait(waited, outcome)
        await asyncio.sleep(interval_s)
        waited += interval_s


async def fetch_status(session: aiohttp.ClientSession, base_url: str, job_id: str) -> IntakeOutcome:
    """Read one job's status."""
    async with session.get(f"{base_url}/v1/intake/jobs/{job_id}") as response:
        if response.status == 404:
            raise IntakeError(f"No intake job on this service is named {job_id!r}.")
        return _read_status(await _json_or_raise(response, ok=(200,)))


def new_session(timeout_s: float = 600.0) -> aiohttp.ClientSession:
    """A session of this client's own, not Gym's global one: this is a short-lived CLI, not a
    server serving concurrent rollouts."""
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s))


async def run_intake(
    *,
    base_url: str,
    locator: str,
    label: Optional[str] = None,
    wait: bool = True,
    timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
    interval_s: float = DEFAULT_POLL_INTERVAL_S,
    on_wait: Optional[Callable[[float, IntakeOutcome], None]] = None,
) -> tuple[Submission, Optional[IntakeOutcome]]:
    """Submit one rundir and, unless `wait` is false, wait for the service's decision."""
    async with new_session() as session:
        submission = await submit(session, base_url, locator, label=label)
        if not wait:
            return submission, None
        outcome = await poll(
            session, base_url, submission, timeout_s=timeout_s, interval_s=interval_s, on_wait=on_wait
        )
        return submission, outcome
