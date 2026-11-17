# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the pinned harness's pytest-runner hook, with no local execution fallback.

This module is also the subprocess entrypoint and uses only the standard library;
it runs in the upstream harness's own Python environment, not Gym's environment.
"""

import contextvars
import json
import runpy
import socket
import sys
from pathlib import Path


MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def run_isolated_pytest(
    endpoint: str,
    directory: Path,
    targets: list[str],
    exitfirst: bool,
    noconftest: bool,
    timeout_s: int,
) -> tuple[int | None, str]:
    """Ask the Gym parent to run pytest; an unavailable bridge aborts evaluation."""
    request = {
        "directory": str(directory),
        "targets": targets,
        "exitfirst": exitfirst,
        "noconftest": noconftest,
        "timeout_s": timeout_s,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_s + 120)
        client.connect(endpoint)
        client.sendall(json.dumps(request).encode() + b"\n")
        with client.makefile("rb") as stream:
            line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line.endswith(b"\n") or len(line) > MAX_MESSAGE_BYTES:
        raise RuntimeError("Isolated pytest returned an invalid response")
    response = json.loads(line)
    if response.get("error"):
        raise RuntimeError(response["error"])
    return response["return_code"], response["output"]


def main() -> None:
    """Run the upstream CLI only after installing its required isolation hook."""
    from collab_intel.harness import sandbox

    endpoint, *arguments = sys.argv[1:]
    # Fail before invoking the CLI if a different harness lacks the integration API.
    hook = sandbox.PYTEST_RUNNER
    sandbox.PYTEST_RUNNER = contextvars.ContextVar(
        "gym_isolated_pytest_runner", default=lambda *args: run_isolated_pytest(endpoint, *args)
    )
    try:
        sys.argv = ["collab_intel.harness.cli", *arguments]
        runpy.run_module("collab_intel.harness.cli", run_name="__main__")
    finally:
        sandbox.PYTEST_RUNNER = hook


if __name__ == "__main__":
    main()
