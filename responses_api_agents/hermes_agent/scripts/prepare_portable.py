# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage Hermes beside the separate Gym interpreter mounted by AnyTerminal and AnySWE."""

import argparse
import asyncio
import json
import os
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path

from nemo_gym.agents import SandboxPlatform
from responses_api_agents.hermes_agent.runtime import HERMES_SOURCE
from responses_api_agents.hermes_agent.sandbox import _HERMES_RUNTIME, HERMES_MIN_GLIBC_MINOR


async def prepare(prefix: Path, *, target: str, source: str) -> None:
    """Build the target platform's Hermes runtime under the local CLI caller's prefix for mounting in task images."""
    arch, libc = target.split("-unknown-linux-")
    platform = SandboxPlatform(arch, libc, glibc_minor=HERMES_MIN_GLIBC_MINOR)
    runtime = replace(_HERMES_RUNTIME, requirements=(source,))
    interpreter, packages = await runtime.archives(platform)
    destination = prefix / "hermes-runtime"
    with tempfile.TemporaryDirectory(dir=prefix) as temporary:
        staged = Path(temporary)
        for path in (interpreter, packages):
            with tarfile.open(path) as archive:
                archive.extractall(staged, filter="data")
        (staged / "python/hermes-source").write_text(json.dumps({"default": HERMES_SOURCE, "source": source}))
        # The enclosing portable installer serializes rebuilds before publishing the sentinel.
        if destination.exists():
            os.rename(destination, staged / "previous")
        try:
            os.rename(staged / "python", destination)
        except OSError:
            if (staged / "previous").exists():
                os.rename(staged / "previous", destination)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", default=HERMES_SOURCE)
    args = parser.parse_args()
    asyncio.run(prepare(args.prefix, target=args.target, source=args.source))
