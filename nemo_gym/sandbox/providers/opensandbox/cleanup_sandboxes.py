#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit or delete OpenSandbox sandboxes owned by one exact run and user."""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


RUN_METADATA_KEY = "nemo-gym.nvidia.com/run"
USER_METADATA_KEY = "nemo-gym.nvidia.com/user"
REQUEST_TIMEOUT_SECONDS = 30


def cleanup_sandboxes(
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

    headers = {"OPEN-SANDBOX-API-KEY": access_key}
    matches: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"page": page, "pageSize": 100})
        request = urllib.request.Request(f"{base_url}/v1/sandboxes?{query}", headers=headers)
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)

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
            break
        page += 1

    action = "Deleting" if reap else "Would delete"
    print(
        f"{action} {len(matches)} OpenSandbox sandbox(es) "
        f"for run {scope[RUN_METADATA_KEY]!r} and user {scope[USER_METADATA_KEY]!r}"
    )
    if not reap:
        return 0

    failures = 0
    for item in matches:
        sandbox_id = item["id"]
        url = f"{base_url}/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}"
        request = urllib.request.Request(url, headers=headers, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                print(f"Deleted {sandbox_id} -> HTTP {response.status}")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                print(f"Sandbox {sandbox_id} was already gone")
                continue
            failures += 1
            print(f"Failed to delete {sandbox_id} -> HTTP {error.code}", file=sys.stderr)
        except OSError as error:
            failures += 1
            print(f"Failed to delete {sandbox_id} -> {error}", file=sys.stderr)

    return int(failures > 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=os.environ.get("OPENSANDBOX_DOMAIN"))
    parser.add_argument(
        "--protocol",
        choices=("http", "https"),
        default=(os.environ.get("OPENSANDBOX_PROTOCOL") or "http").strip(),
    )
    parser.add_argument("--run-id", default=os.environ.get("NEMO_GYM_RUN_ID") or os.environ.get("SLURM_JOB_ID"))
    parser.add_argument(
        "--user",
        default=os.environ.get("NEMO_GYM_USER") or os.environ.get("SLURM_JOB_USER") or os.environ.get("USER"),
    )
    parser.add_argument("--reap", action="store_true", help="Delete exact matches; otherwise only audit them.")
    args = parser.parse_args(argv)

    access_key = (os.environ.get("OPENSANDBOX_API_KEY") or "").strip()
    for name, value in (("domain", args.domain), ("run-id", args.run_id), ("user", args.user)):
        if not value or not value.strip():
            parser.error(f"--{name} or its environment fallback is required")
    if not access_key:
        parser.error("OPENSANDBOX_API_KEY is required")

    try:
        return cleanup_sandboxes(
            domain=args.domain,
            protocol=args.protocol,
            access_key=access_key,
            run_id=args.run_id,
            user=args.user,
            reap=args.reap,
        )
    except (OSError, ValueError) as error:
        print(f"OpenSandbox cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
