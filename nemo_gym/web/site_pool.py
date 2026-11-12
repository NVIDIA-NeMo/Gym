# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Site-stack lease boundary and in-process reader/writer scheduling."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from nemo_gym.web.models import WebTask


@dataclass(frozen=True, slots=True)
class SiteLease:
    lease_id: str
    isolated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class SitePool(Protocol):
    async def acquire(self, session_id: str, task: WebTask) -> SiteLease: ...

    async def release(self, lease: SiteLease, *, healthy: bool) -> None: ...

    async def health(self) -> dict[str, Any]: ...


class UnmanagedSitePool:
    """Use externally configured sites without reset or isolation guarantees."""

    def __init__(self) -> None:
        self._active: set[str] = set()

    async def acquire(self, session_id: str, task: WebTask) -> SiteLease:
        self._active.add(session_id)
        return SiteLease(
            lease_id=f"unmanaged:{session_id}",
            isolated=False,
            metadata={"benchmark": task.benchmark.value, "sites": task.sites},
        )

    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        del healthy
        self._active.discard(lease.lease_id.removeprefix("unmanaged:"))

    async def health(self) -> dict[str, Any]:
        return {
            "mode": "unmanaged",
            "isolated": False,
            "active_leases": len(self._active),
        }


class SiteCapacityError(RuntimeError):
    """A site already has its maximum number of concurrent sessions."""


class SiteConcurrencyLimitPool:
    """Cap concurrent sessions per site; tasks on other sites are unaffected.

    Public sites rate-limit by client IP. When many rollouts share one egress
    IP, a burst of sessions on one site (GitHub, Hugging Face) triggers those
    limits and turns tasks into site-access failures. A full site rejects the
    seed instead of queueing it, so the caller retries later and the slot it
    would have held stays free for other sites.
    """

    def __init__(self, max_sessions_per_site: int) -> None:
        if max_sessions_per_site < 1:
            raise ValueError("max_sessions_per_site must be at least 1")
        self._limit = max_sessions_per_site
        self._lock = asyncio.Lock()
        self._active: dict[str, tuple[str, ...]] = {}
        self._counts: Counter[str] = Counter()
        self._rejected = 0

    async def acquire(self, session_id: str, task: WebTask) -> SiteLease:
        sites = LocalSiteLockPool._sites(task)
        lease_id = f"site-limits:{session_id}"
        async with self._lock:
            if lease_id in self._active:
                raise RuntimeError(f"duplicate site lease {lease_id!r}")
            full = [site for site in sites if self._counts[site] >= self._limit]
            if full:
                self._rejected += 1
                raise SiteCapacityError(f"site {', '.join(full)} has {self._limit} active sessions")
            self._counts.update(sites)
            self._active[lease_id] = sites
        return SiteLease(
            lease_id=lease_id,
            isolated=False,
            metadata={"benchmark": task.benchmark.value, "sites": list(sites), "limit": self._limit},
        )

    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        del healthy
        async with self._lock:
            for site in self._active.pop(lease.lease_id, ()):
                self._counts[site] -= 1
                if self._counts[site] <= 0:
                    del self._counts[site]

    async def health(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "mode": "site_limits",
                "scope": "resource_server_process",
                "isolated": False,
                "max_sessions_per_site": self._limit,
                "active_leases": len(self._active),
                "sessions_by_site": dict(sorted(self._counts.items())),
                "rejected_seeds": self._rejected,
            }


@dataclass(frozen=True, slots=True)
class _LocalLeaseState:
    sites: tuple[str, ...]
    access: str


class LocalSiteLockPool:
    """Coordinate shared readers and exclusive writers for configured sites.

    Locks prevent concurrent tasks from mutating the same site stack; they do
    not reset or isolate site data. Cross-site tasks acquire all locks in one
    condition critical section, avoiding partial acquisition and deadlocks.
    """

    _READER_MUTATIONS = frozenset({"read_only", "session_only"})

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers: Counter[str] = Counter()
        self._writers: set[str] = set()
        self._active: dict[str, _LocalLeaseState] = {}

    async def acquire(self, session_id: str, task: WebTask) -> SiteLease:
        sites = self._sites(task)
        mutation_class = str(getattr(task, "mutation_class", "state_changing"))
        access = "reader" if mutation_class in self._READER_MUTATIONS else "writer"
        lease_id = f"local-locks:{session_id}"

        async with self._condition:
            if lease_id in self._active:
                raise RuntimeError(f"duplicate site lease {lease_id!r}")
            await self._condition.wait_for(lambda: self._can_acquire(sites, access))
            if access == "reader":
                self._readers.update(sites)
            else:
                self._writers.update(sites)
            self._active[lease_id] = _LocalLeaseState(sites=sites, access=access)

        return SiteLease(
            lease_id=lease_id,
            isolated=False,
            metadata={
                "benchmark": task.benchmark.value,
                "sites": list(sites),
                "access": access,
                "mutation_class": mutation_class,
            },
        )

    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        del healthy
        async with self._condition:
            state = self._active.pop(lease.lease_id, None)
            if state is None:
                return
            if state.access == "reader":
                for site in state.sites:
                    self._readers[site] -= 1
                    if self._readers[site] <= 0:
                        del self._readers[site]
            else:
                self._writers.difference_update(state.sites)
            self._condition.notify_all()

    async def health(self) -> dict[str, Any]:
        async with self._condition:
            return {
                "mode": "local_locks",
                "isolated": False,
                "active_leases": len(self._active),
                "reader_leases_by_site": dict(sorted(self._readers.items())),
                "writer_sites": sorted(self._writers),
            }

    def _can_acquire(self, sites: tuple[str, ...], access: str) -> bool:
        if access == "reader":
            return all(site not in self._writers for site in sites)
        return all(site not in self._writers and self._readers[site] == 0 for site in sites)

    @staticmethod
    def _sites(task: WebTask) -> tuple[str, ...]:
        configured = getattr(task, "site_locks", None) or task.sites
        sites = tuple(sorted({str(site).strip().lower() for site in configured if str(site).strip()}))
        if sites:
            return sites
        return (f"benchmark:{task.benchmark.value}",)
