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
"""A run-wide cap on provider HTTP attempts, kept in a ledger file that every server process and restart shares.

The ledger holds two fixed-size records, each with the cap, a reserved-call count and a CRC. A reservation takes an
exclusive ``flock`` on the ledger, reads the highest valid count ``k``, writes ``k + 1`` into the record that does not
hold ``k`` and ``fsync``s it before the request is sent. The ledger is never truncated or replaced, so a crash at any
write boundary leaves the record holding ``k`` intact: after a restart the count is ``k`` (the crashed reservation
sent nothing) or ``k + 1``, never lower. A missing ledger is created at 0 with a no-replace link and synced before
the first reservation; a ``.created`` marker beside it then refuses to create it again. An attempt is refused, and
nothing is sent, when the cap is reached or when the ledger is empty, corrupt, records another cap, or cannot be
opened, locked or written. A reserved attempt that then fails still counts.
"""

import fcntl
import os
import struct
import uuid
import zlib

from pydantic import BaseModel, Field

from nemo_gym.client import PermanentEndpointError


_MAGIC = b"NGCB"
_RECORD = struct.Struct("<4sQQI")  # magic, max_calls, reserved, crc32 of the first three fields
_SLOT_BYTES = 512  # one record per 512-byte slot, so a record write never spans a sector boundary


class UpstreamCallBudget(BaseModel):
    ledger_path: str = Field(description="File shared by every process of the run; created on first use.")
    max_calls: int = Field(ge=1, description="Provider HTTP attempts allowed over the run's lifetime.")


class UpstreamCallBudgetExhaustedError(PermanentEndpointError):
    """The run-wide call budget refused an attempt; nothing was sent to the provider."""

    def __str__(self) -> str:
        # No request was made, so there is no request_info for ClientResponseError's own message.
        return f"{self.status}, message={self.message!r}"


def _refuse(reason: str) -> UpstreamCallBudgetExhaustedError:
    return UpstreamCallBudgetExhaustedError(
        request_info=None,
        history=(),
        status=429,
        message=f"Upstream call budget refused the request before sending it: {reason}",
        headers=None,
    )


def _record(max_calls: int, reserved: int) -> bytes:
    head = _MAGIC + struct.pack("<QQ", max_calls, reserved)
    return _RECORD.pack(_MAGIC, max_calls, reserved, zlib.crc32(head))


def _read_records(fd: int) -> list[tuple[int, int]]:
    records = []
    for slot in range(2):
        raw = os.pread(fd, _RECORD.size, slot * _SLOT_BYTES)
        if len(raw) != _RECORD.size:
            continue
        magic, max_calls, reserved, crc = _RECORD.unpack(raw)
        if magic == _MAGIC and crc == zlib.crc32(raw[: _RECORD.size - 4]):
            records.append((max_calls, reserved))
    return records


def _fsync_dir(path: str) -> None:
    dir_fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _create_ledger(budget: UpstreamCallBudget) -> None:
    """Create the ledger at 0 reserved calls, durably, unless another process created it first."""
    marker = budget.ledger_path + ".created"
    if os.path.exists(marker):
        raise _refuse(f"ledger {budget.ledger_path} is missing but {marker} shows it was created")
    temp = f"{budget.ledger_path}.init.{uuid.uuid4().hex}"  # unique per caller, threads included
    fd = os.open(temp, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.pwrite(fd, _record(budget.max_calls, 0), 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, budget.ledger_path)  # fails if the ledger exists: creation never replaces a ledger
    except FileExistsError:
        pass
    finally:
        os.unlink(temp)
    _fsync_dir(budget.ledger_path)


def _mark_created(budget: UpstreamCallBudget) -> None:
    marker = budget.ledger_path + ".created"
    if not os.path.exists(marker):
        os.close(os.open(marker, os.O_WRONLY | os.O_CREAT, 0o644))
        _fsync_dir(marker)


def reserve_call(budget: UpstreamCallBudget) -> int:
    """Reserve one provider attempt and return its 1-based number, or raise without reserving."""
    try:
        try:
            fd = os.open(budget.ledger_path, os.O_RDWR)
        except FileNotFoundError:
            _create_ledger(budget)
            fd = os.open(budget.ledger_path, os.O_RDWR)
    except UpstreamCallBudgetExhaustedError:
        raise
    except OSError as error:
        raise _refuse(f"cannot open or create ledger {budget.ledger_path}: {error}") from error
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        records = _read_records(fd)
        if not records:
            raise _refuse(f"ledger {budget.ledger_path} has no valid record (empty or corrupt)")
        if any(max_calls != budget.max_calls for max_calls, _ in records):
            raise _refuse(f"ledger {budget.ledger_path} records a cap other than max_calls={budget.max_calls}")
        reserved = max(count for _, count in records)
        _mark_created(budget)
        if reserved >= budget.max_calls:
            raise _refuse(f"{reserved} of {budget.max_calls} calls already reserved")
        # Write into the slot that does not hold the current count, so a torn write cannot lose it.
        slot = (reserved + 1) % 2
        data = _record(budget.max_calls, reserved + 1)
        if os.pwrite(fd, data, slot * _SLOT_BYTES) != len(data):
            raise _refuse(f"short write to ledger {budget.ledger_path}")
        os.fsync(fd)
        return reserved + 1
    except UpstreamCallBudgetExhaustedError:
        raise
    except OSError as error:
        raise _refuse(f"ledger {budget.ledger_path} unusable: {error}") from error
    finally:
        os.close(fd)  # also releases the lock


def reserved_calls(ledger_path: str) -> int:
    """Read the reserved-call count of a ledger, for reports; raises ValueError if it has no valid record."""
    fd = os.open(ledger_path, os.O_RDONLY)
    try:
        records = _read_records(fd)
    finally:
        os.close(fd)
    if not records:
        raise ValueError(f"ledger {ledger_path} has no valid record")
    return max(count for _, count in records)
