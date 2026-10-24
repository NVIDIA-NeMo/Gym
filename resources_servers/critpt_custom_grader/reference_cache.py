# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-task cache of the reference-and-preflight outcome, so a task's reference runs once per process.

The key digests everything that can change a reference verdict: the canonical task data, the policy fields that
reach the runner (the interpreter and the exec user included), the runtime files with the operator's sandbox files and
environment (the caller folds them into ``runtime_digest``), the server's default tolerance and the runtime the
reference ran in, named by the image and the provider's class. The local provider ignores the image, so the class
namespaces every entry: an outcome persisted under one provider is never read under another. Only two outcomes are
stored: a passed preflight and a terminal refusal (``task_invalid``, ``reference_mismatch``). Provider faults are never
cached. Entries hold no task content.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .task_data import TaskData


_ENTRY_VERSION = 1
_LOGGER = logging.getLogger(__name__)
_persist_failure_logged = False  # the write-failure warning is logged once per process, not once per grade


def _warn_persist_failure(error: OSError) -> None:
    """Log the first failed write of this process: the exception class name only, never a path, key or message."""
    global _persist_failure_logged
    if _persist_failure_logged:
        return
    _persist_failure_logged = True
    _LOGGER.warning(
        "reference cache entry not persisted (%s); this process keeps it in memory only", type(error).__name__
    )


def provider_class_name(provider: Any) -> str:
    """The provider's module-qualified class name: stable across processes, and free of endpoints and credentials."""
    cls = type(provider)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class CachedReference:
    refusal: str | None  # None: the preflight passed; else the terminal category to raise again


class ReferenceCache:
    def __init__(self, directory: str | None = None, *, max_entries: int = 4096) -> None:
        self._memory: OrderedDict[str, CachedReference] = OrderedDict()
        self._max_entries = max(1, int(max_entries))
        self._directory = Path(directory) if directory else None
        if self._directory is not None:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    @staticmethod
    def key(
        task: TaskData,
        policy: Any,
        runtime_digest: str,
        image: str | None,
        default_rtol: str | None,
        default_atol: str | None,
        *,
        provider_class: str,
    ) -> str:
        material = {
            "task": task.model_dump(mode="json"),
            "policy": {
                name: getattr(policy, name)
                for name in (
                    "interpreter",
                    "exec_user",
                    "memory_limit_mib",
                    "cpu_time_limit_s",
                    "max_processes",
                    "suite_timeout_s",
                    "compare_per_case_s",
                    "compare_timeout_s",
                    "exec_timeout_margin_s",
                )
            },
            "runtime": runtime_digest,
            "image": image,
            "provider": provider_class,  # provider_class_name(): the image alone does not name a local runtime
            "defaults": [default_rtol, default_atol],
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def get(self, key: str) -> CachedReference | None:
        entry = self._memory.get(key)
        if entry is not None:
            self._memory.move_to_end(key)
            return entry
        if self._directory is None:
            return None
        path = self._directory / f"{key}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):  # unreadable, not JSON or nested too deep: a miss
            return None
        if (
            type(data) is not dict
            or type(data.get("version")) is not int
            or data["version"] != _ENTRY_VERSION
            or "refusal" not in data
            or not (data["refusal"] is None or type(data["refusal"]) is str)
        ):
            return None
        entry = CachedReference(refusal=data["refusal"])
        self._remember(key, entry)
        return entry

    def put(self, key: str, value: CachedReference) -> None:
        self._remember(key, value)
        if self._directory is None:
            return
        payload = json.dumps({"version": _ENTRY_VERSION, "refusal": value.refusal}, separators=(",", ":"))
        try:
            fd, temporary = tempfile.mkstemp(dir=self._directory, prefix=".tmp-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.chmod(temporary, 0o600)
                os.replace(temporary, self._directory / f"{key}.json")
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError as error:  # best effort: the memory layer already holds the entry
            _warn_persist_failure(error)

    def _remember(self, key: str, value: CachedReference) -> None:
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self._max_entries:
            self._memory.popitem(last=False)


__all__ = ["CachedReference", "ReferenceCache", "provider_class_name"]
