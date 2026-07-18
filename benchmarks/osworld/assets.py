# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently prefetch files referenced by OSWorld task JSONL rows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import unquote, urlparse


DEFAULT_SETUP_CACHE = Path(__file__).resolve().parent / ".cache" / "setup"
_HTTP_DOWNLOAD_CACHE = ".downloads"


@dataclass(frozen=True)
class AssetSpec:
    """One remote file and the OSWorld task-cache names that consume it."""

    url: str
    cache_names: tuple[str, ...]


@dataclass(frozen=True)
class AssetSummary:
    task_count: int
    asset_count: int
    materialized_count: int
    cache_dir: Path


def _setup_cache_name(url: str, destination_path: str) -> str:
    return f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{Path(destination_path).name}"


def _download_actions(task: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    action_groups = [task.get("config", [])]
    evaluator = task.get("evaluator")
    if isinstance(evaluator, Mapping):
        action_groups.append(evaluator.get("postconfig", []))
    for actions in action_groups:
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, Mapping) and action.get("type") == "download":
                yield action


def _walk_cloud_files(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("type") == "cloud_file":
            yield value
        for child in value.values():
            yield from _walk_cloud_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_cloud_files(child)


def asset_specs_from_task(task: Mapping[str, Any]) -> list[AssetSpec]:
    """Extract setup, evaluator-postconfig, and evaluator cloud assets."""

    by_url: dict[str, set[str]] = {}
    for action in _download_actions(task):
        parameters = action.get("parameters", {})
        files = parameters.get("files", []) if isinstance(parameters, Mapping) else []
        for file_config in files if isinstance(files, list) else []:
            if not isinstance(file_config, Mapping):
                continue
            url = file_config.get("url")
            destination = file_config.get("path")
            if isinstance(url, str) and url and isinstance(destination, str) and destination:
                by_url.setdefault(url, set()).add(_setup_cache_name(url, destination))

    evaluator = task.get("evaluator")
    if isinstance(evaluator, Mapping):
        for cloud_file in _walk_cloud_files(evaluator):
            paths = cloud_file.get("path")
            destinations = cloud_file.get("dest")
            if not cloud_file.get("multi", False):
                paths = [paths]
                destinations = [destinations]
            if not isinstance(paths, list) or not isinstance(destinations, list):
                continue
            for url, destination in zip(paths, destinations):
                if isinstance(url, str) and url and isinstance(destination, str) and destination:
                    by_url.setdefault(url, set()).add(destination)

    return [AssetSpec(url=url, cache_names=tuple(sorted(names))) for url, names in sorted(by_url.items())]


def _parse_huggingface_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 7 or parts[0] != "datasets" or parts[3] != "resolve":
        return None
    repo_id = "/".join(parts[1:3])
    revision = parts[4]
    filename = "/".join(parts[5:])
    return repo_id, revision, filename


def _manifest_url(url: str) -> str:
    """Retain asset identity without persisting signed query credentials."""

    parsed = urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def _http_download(url: str, download_dir: Path, proxy_url: str | None) -> Path:
    parsed = urlparse(url)
    basename = Path(parsed.path).name or "asset"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    destination = download_dir / f"{digest}_{basename}"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    download_dir.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        if proxy_url
        else urllib.request.ProxyHandler()
    )
    last_error: Exception | None = None
    for attempt in range(5):
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nemo-gym-osworld-prepare/1"})
            with opener.open(request, timeout=300) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(temporary, destination)
            return destination
        except urllib.error.HTTPError as exc:
            last_error = exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            delay = min(2**attempt, 30)
        finally:
            temporary.unlink(missing_ok=True)
        if attempt < 4:
            time.sleep(delay)
    raise RuntimeError(f"Failed to download OSWorld asset after 5 attempts: {url}") from last_error


def _download_asset(url: str, setup_cache_dir: Path, token: str | None, proxy_url: str | None) -> Path:
    hf_asset = _parse_huggingface_url(url)
    if hf_asset is not None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - actionable packaging error
            raise ImportError("OSWorld asset preparation requires `huggingface_hub`") from exc

        repo_id, revision, filename = hf_asset
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename,
                    token=token,
                )
            )
        except Exception:
            if not proxy_url:
                raise
            # Explicit emergency fallback only. Normal downloads retain the HF
            # client's authentication, cache, ETag, and resume behavior.
            return _http_download(url, setup_cache_dir / _HTTP_DOWNLOAD_CACHE, proxy_url)
    return _http_download(url, setup_cache_dir / _HTTP_DOWNLOAD_CACHE, proxy_url)


def _safe_target(task_dir: Path, cache_name: str) -> Path:
    if not cache_name or Path(cache_name).is_absolute():
        raise ValueError(f"Unsafe OSWorld cache destination: {cache_name!r}")
    # Keep this lexical: resolving an existing target follows its symlink into
    # the HF cache and would incorrectly classify a valid second run as an
    # escape from the per-task directory.
    task_root = Path(os.path.abspath(task_dir))
    target = Path(os.path.abspath(task_dir / cache_name))
    if os.path.commonpath([task_root, target]) != str(task_root):
        raise ValueError(f"OSWorld cache destination escapes task directory: {cache_name!r}")
    return target


def _materialize(source: Path, destination: Path) -> bool:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return False
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        os.symlink(source.resolve(), temporary)
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    return True


def _read_tasks(input_jsonl: Path) -> Iterable[Mapping[str, Any]]:
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            metadata = row.get("verifier_metadata", {}) if isinstance(row, Mapping) else {}
            task = metadata.get("osworld_task") if isinstance(metadata, Mapping) else None
            if not isinstance(task, Mapping):
                raise ValueError(f"OSWorld row {line_number} does not contain verifier_metadata.osworld_task")
            yield task


def ensure_osworld_assets(
    input_jsonl: Path | str,
    setup_cache_dir: Path | str = DEFAULT_SETUP_CACHE,
    *,
    token: str | None = None,
    proxy_url: str | None = None,
) -> AssetSummary:
    """Download selected task assets and materialize OSWorld-compatible caches."""

    input_jsonl = Path(input_jsonl).expanduser().resolve()
    setup_cache_dir = Path(setup_cache_dir).expanduser().resolve()
    setup_cache_dir.mkdir(parents=True, exist_ok=True)
    task_count = asset_count = materialized_count = 0
    manifest: list[dict[str, Any]] = []

    for task in _read_tasks(input_jsonl):
        task_count += 1
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            raise ValueError("OSWorld task has no id")
        task_dir = setup_cache_dir / task_id
        for spec in asset_specs_from_task(task):
            asset_count += 1
            source = _download_asset(spec.url, setup_cache_dir, token, proxy_url)
            for cache_name in spec.cache_names:
                materialized_count += int(_materialize(source, _safe_target(task_dir, cache_name)))
            manifest.append(
                {
                    "task_id": task_id,
                    "url": _manifest_url(spec.url),
                    "cache_names": list(spec.cache_names),
                    "bytes": source.stat().st_size,
                }
            )

    manifest_path = setup_cache_dir / "manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return AssetSummary(task_count, asset_count, materialized_count, setup_cache_dir)
