# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry saved W&B authentication into a child command without mounting the user's home."""

import argparse
import contextlib
import io
import os
import sys


def runtime_environment() -> dict[str, str]:
    """Prefer an inherited API key, otherwise load saved login through the W&B SDK.

    This only resolves credentials locally. The training logger performs online
    initialization; there is no login, verification request, or credential write here.
    """
    environment = os.environ.copy()
    if not environment.get("WANDB_API_KEY"):
        # SDK diagnostics may describe malformed credential files. Keep their
        # contents out of launcher logs, including exceptions at this boundary.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from wandb.sdk.lib.wbauth import read_netrc_auth

            key = read_netrc_auth(host=environment.get("WANDB_BASE_URL", "https://api.wandb.ai"))
        if not key:
            raise RuntimeError("W&B authentication is unavailable")
        environment["WANDB_API_KEY"] = key
    environment.pop("WANDB_DISABLED", None)
    environment["WANDB_MODE"] = "online"
    return environment


def main() -> None:
    """Resolve authentication and replace this process with the requested command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not os.path.isabs(command[0]):
        parser.error("Provide a command with an absolute executable path after --")
    try:
        environment = runtime_environment()
    except Exception:
        # Deliberate credential boundary: never serialize the exception or key.
        print(
            "W&B authentication could not be loaded. Use the installed W&B SDK and run wandb login privately "
            "on this host, or supply WANDB_API_KEY through the approved runtime environment.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    try:
        os.execvpe(command[0], command, environment)
    except OSError:
        print("Could not execute the authenticated child command.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
