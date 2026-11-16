# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable replay-review receipts, including an uncertainty marker before any judge dispatch."""

import asyncio
import fcntl
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from resources_servers.partner_if.replay_protocol import PrefixReviewRequest, json_sha256


class ReplayReceiptStore:
    """Serialize identical review slots across tasks and workers without blocking the event loop.

    A surviving in-progress marker is deliberately never retried against the judge: the earlier dispatch may
    already have completed. Files are local to the explicitly configured replay artifact root.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        root = artifacts_dir.expanduser().resolve()
        self.directory = root / ".partner-if-replay-receipts"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink() or self.directory.resolve().parent != root:
            raise ValueError("Replay receipt store must be a real directory beneath the artifact root")

    @staticmethod
    def session_receipt_id(session_id: str) -> str:
        """Use one durable lock for typed seed, all reviews, and resource close."""
        return json_sha256({"kind": "resources_seed", "session_id": session_id})[:32]

    @staticmethod
    def scope(session_id: str, request: PrefixReviewRequest) -> dict:
        """Name the Resources cookie, attempt, task, agent, and prefix before judging."""
        return {
            "session_id": session_id,
            "episode_id": request.episode_id.model_dump(mode="json"),
            "task_id": request.task_id.model_dump(mode="json"),
            "agent_session_id": request.agent_session_id,
            "prefix_index": request.prefix_index,
        }

    @staticmethod
    def receipt_id(scope: dict) -> str:
        """A stable opaque slot identity also survives a lost HTTP reply."""
        return json_sha256(scope)[:32]

    def path(self, receipt_id: str, suffix: str = ".json") -> Path:
        if len(receipt_id) != 32 or any(value not in "0123456789abcdef" for value in receipt_id):
            raise ValueError("Invalid replay receipt identity")
        path = self.directory / (receipt_id + suffix)
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError("Replay receipt path escapes the configured store")
        return path

    @asynccontextmanager
    async def locked(self, receipt_id: str) -> AsyncIterator[None]:
        """Use a process-shared advisory lock while permitting task cancellation during contention."""
        descriptor = os.open(self.path(receipt_id, ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)
            yield
        finally:
            os.close(descriptor)

    def read(self, receipt_id: str) -> dict | None:
        """Read only an atomically published marker or complete receipt."""
        path = self.path(receipt_id)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor) as stream:
            record = json.load(stream)
        if not isinstance(record, dict) or record.get("receipt_id") != receipt_id:
            raise ValueError("Replay receipt identity is corrupt")
        return record

    def write(self, receipt_id: str, record: dict) -> None:
        """Flush file contents and directory metadata before acknowledging a grading decision."""
        target = self.path(receipt_id)
        temporary = self.path(receipt_id, f".{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(record, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
