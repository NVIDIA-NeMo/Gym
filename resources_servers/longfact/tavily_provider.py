# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Tavily Search adapter with an optional replayable cache."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import random
import stat
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from resources_servers.tavily_search.app import TavilySearchAIOHTTPClient


CacheMode = Literal["live", "read_write", "replay"]
CACHE_FORMAT_VERSION = 2
CACHE_READABLE_FORMAT_VERSIONS = frozenset({1, CACHE_FORMAT_VERSION})
CACHE_ADAPTER_ID = "nemo-gym-longfact-tavily"
CACHE_METADATA_KEY = "_nemo_gym_cache"
CACHE_RESPONSE_KEY = "response"
CACHE_STATS_LOG_INTERVAL = 100


class TavilyProviderError(RuntimeError):
    pass


class TavilyCacheCorruptionError(TavilyProviderError):
    """A cache entry exists but cannot safely be replayed."""


@dataclass(frozen=True)
class TavilyEvidence:
    title: str
    url: str
    snippet: str
    search_score: float | None


class TavilyProvider:
    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str,
        max_results: int,
        timeout_seconds: float,
        max_concurrency: int,
        search_max_attempts: int = 4,
        cache_mode: CacheMode = "live",
        cache_dir: str | None = None,
        client: Any | None = None,
    ) -> None:
        if search_max_attempts < 1:
            raise ValueError("Tavily search attempt count must be positive")
        if client is None and cache_mode != "replay":
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=api_key)
            client._client = TavilySearchAIOHTTPClient.from_httpx_AsyncClient(client._client, debug=False)
        self._client = client
        self._search_depth = search_depth
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._search_max_attempts = search_max_attempts
        self._inflight_guard = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._cache_mode = cache_mode
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_lookups = 0
        self._cache_completed_lookups = 0
        self._cache_persistent_hits = 0
        self._cache_cross_process_hits = 0
        self._cache_inflight_hits = 0
        self._cache_initial_misses = 0
        self._cache_live_fetches = 0
        self._cache_writes = 0
        self._cache_corruptions = 0
        try:
            self.dependency_version = version("tavily-python")
        except PackageNotFoundError:
            self.dependency_version = "missing"
        if cache_mode in {"read_write", "replay"} and self._cache_dir is None:
            raise ValueError("cache_dir is required when cache_mode is read_write or replay")
        if cache_mode == "read_write":
            self._cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if cache_mode in {"read_write", "replay"}:
            self._check_cache_dir_is_trusted(self._cache_dir)
            # All cache I/O goes through the checked resolved path, so no alias can hide a parent.
            self._cache_dir = self._cache_dir.resolve(strict=True)

    @staticmethod
    def _check_cache_dir_is_trusted(cache_dir: Path) -> None:
        # Entries carry only recomputable checksums, so a directory another user can write could forge evidence.
        info = os.lstat(cache_dir)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise TavilyProviderError(f"Tavily cache dir {cache_dir} must be a real directory, not a symlink")
        if info.st_uid != os.getuid():
            raise TavilyProviderError(f"Tavily cache dir {cache_dir} is owned by uid {info.st_uid}, not this user")
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise TavilyProviderError(f"Tavily cache dir {cache_dir} is writable by other users")
        # A parent that others can write (without the sticky bit, as on /tmp) lets them rename the cache away.
        for parent in cache_dir.resolve(strict=True).parents:
            parent_info = os.lstat(parent)
            shared = parent_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not parent_info.st_mode & stat.S_ISVTX
            if parent_info.st_uid not in (0, os.getuid()) or shared:
                raise TavilyProviderError(
                    f"Tavily cache dir {cache_dir} has a parent other users can replace it in: {parent}"
                )

    @property
    def cache_stats(self) -> dict[str, int | float | str]:
        """Return a point-in-time cache and request-coalescing snapshot."""

        effective_hits = self._cache_persistent_hits + self._cache_cross_process_hits + self._cache_inflight_hits
        lookups = self._cache_lookups
        return {
            "mode": self._cache_mode,
            "lookups": lookups,
            "completed_lookups": self._cache_completed_lookups,
            "persistent_hits": self._cache_persistent_hits,
            "cross_process_hits": self._cache_cross_process_hits,
            "inflight_hits": self._cache_inflight_hits,
            "initial_misses": self._cache_initial_misses,
            "live_fetches": self._cache_live_fetches,
            "writes": self._cache_writes,
            "corruptions": self._cache_corruptions,
            "effective_hits": effective_hits,
            "persistent_hit_rate": self._cache_persistent_hits / lookups if lookups else 0.0,
            "effective_hit_rate": effective_hits / lookups if lookups else 0.0,
        }

    def _complete_cache_lookup(self) -> None:
        self._cache_completed_lookups += 1
        if self._cache_completed_lookups % CACHE_STATS_LOG_INTERVAL != 0:
            return
        stats = self.cache_stats
        print(
            "[tavily-cache] stats "
            f"lookups={stats['lookups']} completed={stats['completed_lookups']} "
            f"persistent_hits={stats['persistent_hits']} "
            f"cross_process_hits={stats['cross_process_hits']} "
            f"inflight_hits={stats['inflight_hits']} "
            f"initial_misses={stats['initial_misses']} "
            f"live_fetches={stats['live_fetches']} writes={stats['writes']} "
            f"corruptions={stats['corruptions']} "
            f"persistent_hit_rate={stats['persistent_hit_rate']:.4f} "
            f"effective_hit_rate={stats['effective_hit_rate']:.4f}",
            flush=True,
        )

    async def search(self, query: str) -> list[TavilyEvidence]:
        payload = {
            "query": query,
            "search_depth": self._search_depth,
            "max_results": self._max_results,
            "include_answer": False,
            "include_raw_content": False,
            "topic": "general",
        }
        data = await self._call("search", payload)
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise TavilyProviderError(
                f"Tavily search returned results as {type(raw_results).__name__}, expected array"
            )
        results: list[TavilyEvidence] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise TavilyProviderError(f"Tavily search result {index} was {type(item).__name__}, expected object")
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                TavilyEvidence(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or ""),
                    search_score=item.get("score"),
                )
            )
        return results

    async def _call(self, operation: Literal["search"], payload: dict[str, Any]) -> dict[str, Any]:
        cache_path = self._cache_path(operation, payload)
        if cache_path is None:
            return await self._request(operation, payload)

        self._cache_lookups += 1

        try:
            cached = self._read_cache(
                cache_path,
                operation,
                payload,
                require_checksum=self._cache_mode == "read_write",
            )
        except TavilyCacheCorruptionError:
            if self._cache_mode == "replay":
                raise
            # The per-key inter-process lock below owns quarantine and repair.
            cached = None
        if cached is not None:
            self._cache_persistent_hits += 1
            self._complete_cache_lookup()
            return cached
        self._cache_initial_misses += 1
        if self._cache_mode == "replay":
            self._complete_cache_lookup()
            raise TavilyProviderError(f"Tavily replay cache miss for {operation}: {cache_path}")

        cache_key = str(cache_path)
        async with self._inflight_guard:
            task = self._inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(self._request_and_cache(operation, payload, cache_path))
                self._inflight[cache_key] = task
                task.add_done_callback(lambda completed, key=cache_key: self._clear_inflight(key, completed))
            else:
                self._cache_inflight_hits += 1
        try:
            return await asyncio.shield(task)
        finally:
            # Each caller owns one lookup, including callers coalesced behind
            # the same in-process live request.
            self._complete_cache_lookup()

    async def _request(self, operation: Literal["search"], payload: dict[str, Any]) -> dict[str, Any]:
        max_attempts = self._search_max_attempts
        last_error: Exception | None = None
        # Match the released planner's jitter immediately before it enters the
        # search-provider semaphore. This runs once per live provider operation,
        # not once per provider-internal retry.
        await asyncio.sleep(random.uniform(0, 0.3))
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._semaphore:
                    method = getattr(self._client, operation)
                    data = await asyncio.wait_for(
                        method(
                            **payload,
                            timeout=self._timeout_seconds,
                        ),
                        timeout=self._timeout_seconds,
                    )
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
                    continue
                break

            if not isinstance(data, dict):
                raise TavilyProviderError(f"Tavily {operation} returned {type(data).__name__}, expected object")
            self._validate_response(operation, data)
            return data

        raise TavilyProviderError(
            f"Tavily {operation} failed after {max_attempts} attempts: {last_error}"
        ) from last_error

    async def _request_and_cache(
        self,
        operation: Literal["search"],
        payload: dict[str, Any],
        cache_path: Path,
    ) -> dict[str, Any]:
        async with self._cache_lock(cache_path):
            # Recheck while holding a cross-process lock. This prevents two
            # evaluations sharing a filesystem from paying for the same cache miss.
            try:
                cached = self._read_cache(cache_path, operation, payload, require_checksum=True)
            except TavilyCacheCorruptionError as exc:
                quarantined = self._quarantine_cache(cache_path)
                if quarantined is not None:
                    self._cache_corruptions += 1
                location = f" to {quarantined}" if quarantined is not None else ""
                print(f"[tavily-cache] quarantined corrupt entry {cache_path}{location}: {exc}", flush=True)
                cached = None
            if cached is not None:
                self._cache_cross_process_hits += 1
                return cached

            self._cache_live_fetches += 1
            data = await self._request(operation, payload)
            document = {
                CACHE_METADATA_KEY: {
                    "adapter": CACHE_ADAPTER_ID,
                    "format_version": CACHE_FORMAT_VERSION,
                    "operation": operation,
                    "request_sha256": self._request_digest(operation, payload),
                    "response_sha256": self._response_digest(data),
                },
                CACHE_RESPONSE_KEY: data,
            }
            try:
                if self._publish_cache(cache_path, document):
                    self._cache_writes += 1
            except OSError as exc:
                raise TavilyProviderError(f"Could not write Tavily cache entry {cache_path}: {exc}") from exc

            cached = self._read_cache(cache_path, operation, payload, require_checksum=True)
            if cached is None:  # pragma: no cover - requires external deletion after publication.
                raise TavilyProviderError(f"Tavily cache entry disappeared after publication: {cache_path}")
            return cached

    def _clear_inflight(self, cache_key: str, completed: asyncio.Task[dict[str, Any]]) -> None:
        if self._inflight.get(cache_key) is completed:
            self._inflight.pop(cache_key, None)
        # Retrieve failures even if all waiting callers were cancelled, avoiding
        # an unhandled-task warning. Awaiters still receive the same exception.
        if not completed.cancelled():
            completed.exception()

    def _cache_path(self, operation: str, payload: dict[str, Any]) -> Path | None:
        if self._cache_mode == "live" or self._cache_dir is None:
            return None
        digest = self._request_digest(operation, payload)
        return self._cache_dir / f"{operation}-{digest}.json"

    @staticmethod
    def _request_digest(operation: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps({"operation": operation, "payload": payload}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _response_digest(response: dict[str, Any]) -> str:
        canonical = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_response(operation: str, response: dict[str, Any]) -> None:
        if operation != "search":  # pragma: no cover - operation type is currently search-only.
            raise TavilyProviderError(f"Unsupported Tavily operation: {operation}")
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise TavilyProviderError(
                f"Tavily search returned results as {type(raw_results).__name__}, expected array"
            )
        for index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise TavilyProviderError(f"Tavily search result {index} was {type(item).__name__}, expected object")

    @classmethod
    def _read_cache(
        cls,
        path: Path,
        operation: str,
        payload: dict[str, Any],
        *,
        require_checksum: bool = False,
    ) -> dict[str, Any] | None:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TavilyProviderError(f"Could not read Tavily cache entry {path}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TavilyCacheCorruptionError(f"Could not decode Tavily cache entry {path}: {exc}") from exc

        if not isinstance(document, dict):
            raise TavilyCacheCorruptionError(f"Malformed Tavily cache entry {path}: expected an object")

        metadata = document.get(CACHE_METADATA_KEY)
        if metadata is None:
            # Backward compatibility with the original raw-response cache.
            if require_checksum:
                raise TavilyCacheCorruptionError(
                    f"Unchecksummed legacy Tavily cache entry {path} is not trusted in read_write mode"
                )
            try:
                cls._validate_response(operation, document)
            except TavilyProviderError as exc:
                raise TavilyCacheCorruptionError(f"Malformed Tavily cache entry {path}: {exc}") from exc
            return document
        if not isinstance(metadata, dict) or CACHE_RESPONSE_KEY not in document:
            raise TavilyCacheCorruptionError(f"Malformed Tavily cache envelope {path}")

        expected = {
            "adapter": CACHE_ADAPTER_ID,
            "operation": operation,
            "request_sha256": cls._request_digest(operation, payload),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise TavilyCacheCorruptionError(
                    f"Tavily cache identity mismatch for {path}: expected {key}={value!r}, "
                    f"received {metadata.get(key)!r}"
                )
        format_version = metadata.get("format_version")
        if format_version not in CACHE_READABLE_FORMAT_VERSIONS:
            raise TavilyCacheCorruptionError(f"Unsupported Tavily cache format for {path}: {format_version!r}")
        if require_checksum and format_version != CACHE_FORMAT_VERSION:
            raise TavilyCacheCorruptionError(
                f"Unchecksummed Tavily cache format {format_version!r} for {path} is not trusted in read_write mode"
            )
        response = document[CACHE_RESPONSE_KEY]
        if not isinstance(response, dict):
            raise TavilyCacheCorruptionError(f"Malformed Tavily cache entry {path}: response must be an object")
        try:
            cls._validate_response(operation, response)
        except TavilyProviderError as exc:
            raise TavilyCacheCorruptionError(f"Malformed Tavily cache entry {path}: {exc}") from exc
        if format_version == CACHE_FORMAT_VERSION:
            expected_response_digest = metadata.get("response_sha256")
            actual_response_digest = cls._response_digest(response)
            if expected_response_digest != actual_response_digest:
                raise TavilyCacheCorruptionError(
                    f"Tavily cache response checksum mismatch for {path}: "
                    f"expected {expected_response_digest!r}, received {actual_response_digest!r}"
                )
        return response

    @staticmethod
    def _lock_path(path: Path) -> Path:
        # Stripe locks into 4096 stable files. This avoids one permanent lock
        # inode per query while preserving same-key exclusion across processes.
        stripe = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:3]
        return path.parent / ".locks" / f"{stripe}.lock"

    @asynccontextmanager
    async def _cache_lock(self, path: Path) -> AsyncIterator[None]:
        lock_path = self._lock_path(path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
        try:
            # Poll a non-blocking lock on the event loop: blocked waiters in executor threads could
            # take every thread and leave none for the holders' release.
            delay = 0.005
            while not self._try_lock(stream.fileno()):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.2)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    @staticmethod
    def _try_lock(fileno: int) -> bool:
        try:
            fcntl.flock(fileno, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    @staticmethod
    def _quarantine_cache(path: Path) -> Path | None:
        quarantine_dir = path.parent / ".corrupt"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / f"{path.name}.{time.time_ns()}.{os.getpid()}.corrupt"
        try:
            os.replace(path, target)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TavilyProviderError(f"Could not quarantine Tavily cache entry {path}: {exc}") from exc
        return target

    @staticmethod
    def _publish_cache(path: Path, document: dict[str, Any]) -> bool:
        """Atomically publish a complete entry without replacing an existing winner."""

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(
                    document,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            try:
                os.link(temporary, path)
                return True
            except FileExistsError:
                return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
