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

"""Verify that an OpenSandbox OOM is surfaced through the Gym exec error."""

import argparse
import asyncio
import getpass
import json
import os
import shlex
from urllib.parse import urlsplit

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import SandboxBackendUnreachableError


OOM_PROGRAM = r"""
from pathlib import Path

for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
    try:
        raw_limit = path.read_text().strip()
    except OSError:
        continue
    if raw_limit != "max":
        limit = int(raw_limit)
        break
else:
    raise RuntimeError("sandbox has no finite cgroup memory limit")

if not 0 < limit <= 512 * 1024 * 1024:
    raise RuntimeError(f"refusing to allocate against unexpected memory limit: {limit}")

print(f"cgroup memory limit: {limit}", flush=True)
blocks = []
for _ in range((limit * 2) // (8 * 1024 * 1024)):
    block = bytearray(8 * 1024 * 1024)
    for offset in range(0, len(block), 4096):
        block[offset] = 1
    blocks.append(block)
"""


async def run(domain: str, api_key: str) -> int:
    endpoint = urlsplit(domain)
    provider = {
        "opensandbox": {
            "connection": {
                "domain": endpoint.netloc,
                "protocol": endpoint.scheme,
                "api_key": api_key,
                "use_server_proxy": True,
                "request_timeout_s": 30,
            },
            "create": {"timeout_s": 180, "retries": 0},
            "operations": {
                "retries": 0,
                "command_retries": 0,
                "background_exec": True,
                "background_poll_interval_s": 1,
            },
        }
    }
    spec = SandboxSpec(
        image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
        entrypoint=[
            "sh",
            "-c",
            f"while [ ! -e /tmp/oom-trigger ]; do sleep 0.1; done; exec python3 -c {shlex.quote(OOM_PROGRAM)}",
        ],
        ttl_s=300,
        ready_timeout_s=120,
        resources=SandboxResources(cpu=0.5, memory_mib=256),
        provider_options={
            "resource_requests": {"cpu": 0.1, "memory_mib": 128},
            "platform": {"os": "linux", "arch": "amd64"},
        },
        metadata={"purpose": "oom-status-e2e"},
    )

    sandbox = AsyncSandbox(provider, spec)
    gym_error = ""
    oom_error = False
    exec_was_502 = False
    cleanup_ok = False
    sdk_status: dict[str, str | None] = {}
    try:
        await sandbox.start()
        print(f"sandbox_id: {sandbox._require_handle().sandbox_id}")
        try:
            # Trigger the supervised workload allocator only after Gym's command starts.
            result = await sandbox.exec("touch /tmp/oom-trigger; sleep 90", timeout_s=90)
            gym_error = f"return_code={result.return_code}, stderr={result.stderr!r}"
        except Exception as error:
            gym_error = f"{type(error).__name__}: {error}"
            oom_error = isinstance(error, SandboxBackendUnreachableError)
            cause: BaseException | None = error
            seen: set[int] = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                exec_was_502 |= getattr(cause, "status_code", None) == 502
                cause = cause.__cause__
        print(f"gym_exec: {gym_error}")

        raw = sandbox._require_handle().raw
        for _ in range(30):
            status = (await raw.get_info()).status
            sdk_status = {"state": status.state, "reason": status.reason, "message": status.message}
            if "oom" in json.dumps(sdk_status).lower():
                break
            await asyncio.sleep(1)
        print(f"sdk_status: {json.dumps(sdk_status)}")
        print(f"gym_status: {(await sandbox.status()).value}")
    finally:
        try:
            await asyncio.wait_for(sandbox.stop(), timeout=75)
            cleanup_ok = True
            print("cleanup: deleted")
        except Exception as error:
            print(f"cleanup: failed: {type(error).__name__}: {error}")

    oom_status = "oom" in json.dumps(sdk_status).lower()
    surfaced_oom = "OOM-killed" in gym_error and "SandboxResources.memory_mib" in gym_error
    return 0 if oom_status and oom_error and exec_was_502 and surfaced_oom and cleanup_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=os.environ.get("OPENSANDBOX_URL"))
    args = parser.parse_args()
    if not args.domain or not urlsplit(args.domain).netloc:
        parser.error("pass --domain URL or set OPENSANDBOX_URL")
    api_key = os.environ.get("OPENSANDBOX_API_KEY") or getpass.getpass("OpenSandbox API key: ")
    return asyncio.run(run(args.domain, api_key))


if __name__ == "__main__":
    raise SystemExit(main())
