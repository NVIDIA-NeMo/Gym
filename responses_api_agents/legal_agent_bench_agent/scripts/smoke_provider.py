# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the shared sandbox contract used by Legal Agent Bench."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import tempfile
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config


def _provider(args: argparse.Namespace) -> dict:
    if args.config is None:
        return {args.provider: {}}
    config = OmegaConf.load(args.config)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Sandbox config must resolve to a mapping: {args.config}")
    return resolve_provider_config(args.sandbox_name, resolved)


async def _smoke(args: argparse.Namespace) -> dict[str, object]:
    provider = _provider(args)
    provider_name = next(iter(provider))
    token = uuid4().hex
    remote_root = f"{args.workdir.rstrip('/')}/nemo-gym-lab-provider-smoke-{token[:8]}"
    input_path = f"{remote_root}/input.txt"
    output_path = f"{remote_root}/output.txt"
    sandbox = AsyncSandbox(
        provider,
        SandboxSpec(
            image=args.image,
            ttl_s=args.ttl,
            ready_timeout_s=args.ready_timeout,
            workdir=args.workdir,
            metadata={"benchmark": "legal-agent-bench", "purpose": "provider-smoke"},
        ),
    )
    with tempfile.TemporaryDirectory(prefix="legal-agent-bench-provider-smoke-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "input.txt"
        downloaded = temporary_root / "output.txt"
        source.write_text(token)
        try:
            await sandbox.start()
            await sandbox.upload(source, input_path)
            command = (
                f'test "$(cat {shlex.quote(input_path)})" = {shlex.quote(token)} && '
                f"printf %s {shlex.quote(token)} > {shlex.quote(output_path)}"
            )
            result = await sandbox.exec(command, timeout_s=args.timeout)
            if result.return_code != 0 or result.error_type is not None:
                raise RuntimeError(
                    f"sandbox exec failed: return_code={result.return_code}, "
                    f"error_type={result.error_type!r}, stderr={result.stderr!r}"
                )
            await sandbox.download(output_path, downloaded)
            if downloaded.read_text() != token:
                raise RuntimeError("sandbox upload/exec/download round trip changed the payload")
        finally:
            await sandbox.stop()
    return {
        "provider": provider_name,
        "image": args.image,
        "start": "passed",
        "exec": "passed",
        "upload_download": "passed",
        "cleanup": "passed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--provider", help="Provider name with its default configuration")
    selection.add_argument("--config", type=Path, help="Gym provider YAML containing a named sandbox block")
    parser.add_argument("--sandbox-name", default="sandbox", help="Named sandbox block in --config")
    parser.add_argument("--image", required=True, help="Provider-compatible LAB image reference")
    parser.add_argument("--workdir", default="/tmp", help="Writable directory in the sandbox image")
    parser.add_argument("--timeout", type=float, default=300, help="Command timeout in seconds")
    parser.add_argument("--ready-timeout", type=float, default=900, help="Sandbox readiness timeout in seconds")
    parser.add_argument("--ttl", type=float, default=1200, help="Sandbox lifetime in seconds")
    return parser


def main() -> None:
    result = asyncio.run(_smoke(_parser().parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
