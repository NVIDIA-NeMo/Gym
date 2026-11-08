# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Carry saved W&B authentication into a child command without mounting the user's home."""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path


def runtime_environment() -> dict[str, str]:
    """Prefer an inherited API key, otherwise load saved login through the W&B SDK.

    This only resolves credentials locally. The training logger performs online
    initialization; there is no login, verification request, or credential write here.
    """
    environment = os.environ.copy()
    if not environment.get("WANDB_BASE_URL"):
        # The saved login may target a private server. Resolve its settings
        # locally and carry that same destination into the container.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from wandb import Settings

            config_dir = Path(environment.get("WANDB_CONFIG_DIR", "~/.config/wandb")).expanduser()
            settings = Settings(settings_system=str(config_dir / "settings"))
            settings.update_from_system_settings()
            environment["WANDB_BASE_URL"] = settings.base_url
    if not environment.get("WANDB_API_KEY"):
        # SDK diagnostics may describe malformed credential files. Keep their
        # contents out of launcher logs, including exceptions at this boundary.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from wandb.sdk.lib.wbauth import read_netrc_auth

            key = read_netrc_auth(host=environment["WANDB_BASE_URL"])
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
            "W&B authentication could not be loaded for the configured server. Check the installed W&B SDK "
            "and saved server settings, or supply WANDB_BASE_URL and WANDB_API_KEY through the approved runtime "
            "environment. No job was dispatched by this wrapper.",
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
