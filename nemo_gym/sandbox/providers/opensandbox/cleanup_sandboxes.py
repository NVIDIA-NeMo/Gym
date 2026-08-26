#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit or delete OpenSandbox sandboxes owned by one exact run and user."""

import argparse
import asyncio
import re
import sys
import urllib.parse
from collections.abc import Mapping
from typing import Any

import aiohttp
import yaml


RUN_METADATA_KEY = "nemo-gym.nvidia.com/run"
USER_METADATA_KEY = "nemo-gym.nvidia.com/user"
REQUEST_TIMEOUT_SECONDS = 30
REAP_CONCURRENCY = 32
REAP_SWEEPS = 3


async def cleanup_sandboxes(
    *,
    domain: str,
    protocol: str,
    access_key: str,
    run_id: str,
    user: str,
    reap: bool,
) -> int:
    """List exact run-owned sandboxes and optionally delete them."""
    base_url = domain.strip().rstrip("/")
    if "://" not in base_url:
        base_url = f"{protocol}://{base_url}"
    parsed_url = urllib.parse.urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"invalid OpenSandbox domain: {domain!r}")

    scope = {}
    for key, value in ((RUN_METADATA_KEY, run_id), (USER_METADATA_KEY, user)):
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
        scope[key] = normalized[:63].strip("._-") or "metadata"

    connector = aiohttp.TCPConnector(limit=REAP_CONCURRENCY, limit_per_host=REAP_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        connector=connector,
        headers={"OPEN-SANDBOX-API-KEY": access_key},
        timeout=timeout,
    ) as session:

        async def list_matches() -> list[dict[str, Any]]:
            matches: list[dict[str, Any]] = []
            page = 1
            while True:
                async with session.get(
                    f"{base_url}/v1/sandboxes",
                    allow_redirects=False,
                    params={"page": page, "pageSize": 100},
                ) as response:
                    if not 200 <= response.status < 300:
                        raise ValueError(f"OpenSandbox list request failed -> HTTP {response.status}")
                    payload = await response.json(content_type=None)

                if not isinstance(payload, dict):
                    raise ValueError("OpenSandbox list response must be an object")
                items = payload.get("items")
                pagination = payload.get("pagination")
                if not isinstance(items, list) or not isinstance(pagination, dict):
                    raise ValueError("OpenSandbox list response is missing items or pagination")
                has_next_page = pagination.get("hasNextPage")
                if not isinstance(has_next_page, bool):
                    raise ValueError("OpenSandbox list response is missing pagination.hasNextPage")

                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("OpenSandbox list response contains an invalid sandbox")
                    metadata = item.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        raise ValueError("OpenSandbox sandbox metadata must be an object")
                    if all(metadata.get(key) == value for key, value in scope.items()):
                        if not isinstance(item.get("id"), str) or not item["id"]:
                            raise ValueError("OpenSandbox list response contains a sandbox without an id")
                        matches.append(item)

                if not has_next_page:
                    return matches
                page += 1

        matches = await list_matches()
        action = "Deleting" if reap else "Would delete"
        print(
            f"{action} {len(matches)} OpenSandbox sandbox(es) "
            f"for run {scope[RUN_METADATA_KEY]!r} and user {scope[USER_METADATA_KEY]!r}"
        )
        if not reap:
            return 0

        semaphore = asyncio.Semaphore(REAP_CONCURRENCY)

        async def delete(item: dict[str, Any]) -> int:
            sandbox_id = item["id"]
            url = f"{base_url}/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}"
            async with semaphore:
                try:
                    async with session.delete(url, allow_redirects=False) as response:
                        await response.read()
                        if response.status == 404:
                            print(f"Sandbox {sandbox_id} was already gone")
                            return 0
                        if not 200 <= response.status < 300:
                            print(f"Failed to delete {sandbox_id} -> HTTP {response.status}", file=sys.stderr)
                            return 1
                        print(f"Deleted {sandbox_id} -> HTTP {response.status}")
                        return 0
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                    print(f"Failed to delete {sandbox_id} -> {error}", file=sys.stderr)
                    return 1

        # Deletes race the cancelled workload's own teardown, and a list taken
        # while the set mutates can skip entries across page boundaries; sweep
        # until a fresh list comes back empty, or a sweep stops progressing.
        for _ in range(REAP_SWEEPS):
            if not matches:
                return 0
            failures = sum(await asyncio.gather(*(delete(item) for item in matches)))
            if failures == len(matches):
                break
            matches = await list_matches()
        if not matches:
            return 0
        print(f"{len(matches)} OpenSandbox sandbox(es) were not reaped", file=sys.stderr)
        return 1


def _run(
    parser: argparse.ArgumentParser,
    domain: str,
    access_key: str,
    protocol: str,
    args: argparse.Namespace,
) -> int:
    """Run the cleanup and turn its failures into a message and a status."""
    try:
        return asyncio.run(
            cleanup_sandboxes(
                domain=domain,
                protocol=protocol,
                access_key=access_key,
                run_id=args.run_id,
                user=args.user,
                reap=args.reap,
            )
        )
    except (aiohttp.ClientError, OSError, TypeError, ValueError) as error:
        print(f"OpenSandbox cleanup failed: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-config",
        help="YAML file containing sandbox.opensandbox.connection, as an alternative to --domain/--api-key.",
    )
    parser.add_argument("--domain", help="OpenSandbox domain, host or full URL.")
    parser.add_argument("--api-key", help="OpenSandbox access key.")
    parser.add_argument(
        "--protocol",
        default="http",
        choices=("http", "https"),
        help="Scheme for --domain when it carries none (default: http).",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--reap", action="store_true", help="Delete exact matches; otherwise only audit them.")
    args = parser.parse_args(argv)

    for name, value in (("run-id", args.run_id), ("user", args.user)):
        if not value.strip():
            parser.error(f"--{name} must not be empty")

    if args.connection_config is None:
        for name, value in (("domain", args.domain), ("api-key", args.api_key)):
            if value is None or not value.strip():
                parser.error(f"--{name} is required when --connection-config is omitted")
        return _run(parser, args.domain.strip(), args.api_key.strip(), args.protocol, args)
    for name, value in (("domain", args.domain), ("api-key", args.api_key)):
        if value is not None:
            parser.error(f"--{name} cannot be combined with --connection-config")

    try:
        with open(args.connection_config, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, Mapping):
            raise ValueError("connection config must contain a YAML object")
        sandbox = config.get("sandbox")
        if not isinstance(sandbox, Mapping):
            raise ValueError("connection config 'sandbox' is required")
        opensandbox = sandbox.get("opensandbox")
        if not isinstance(opensandbox, Mapping):
            raise ValueError("connection config 'sandbox.opensandbox' is required")
        connection = opensandbox.get("connection")
        if not isinstance(connection, Mapping):
            raise ValueError("connection config 'sandbox.opensandbox.connection' is required")

        domain = connection.get("domain")
        access_key = connection.get("api_key")
        protocol = connection.get("protocol") or "http"
        for path, value in (("domain", domain), ("api_key", access_key)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"connection config 'sandbox.opensandbox.connection.{path}' is required")
        if not isinstance(protocol, str) or protocol.strip() not in {"http", "https"}:
            raise ValueError("connection config 'sandbox.opensandbox.connection.protocol' must be http or https")

        return _run(parser, domain.strip(), access_key.strip(), protocol.strip(), args)
    except yaml.YAMLError:
        print("OpenSandbox cleanup failed: invalid YAML connection config", file=sys.stderr)
        return 1
    except (aiohttp.ClientError, OSError, TypeError, ValueError) as error:
        print(f"OpenSandbox cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
