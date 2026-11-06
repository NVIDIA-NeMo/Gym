# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Expose an image's installed Go compiler without replacing its command environment."""

from nemo_gym.agents import exec_idempotent
from nemo_gym.sandbox import AsyncSandbox


# AgentENV may start commands with a standard PATH instead of the image's Docker ENV.
# A link in an existing PATH directory works for subsequent agent and verifier commands,
# including non-login shells, and leaves an already selected Go version alone.
_GO_SETUP = r"""
if command -v go >/dev/null 2>&1; then exit 0; fi
for go_binary in /usr/local/go/bin/go /usr/lib/go/bin/go /usr/lib/go-*/bin/go /opt/go/bin/go; do
    [ -x "$go_binary" ] || continue
    case ":$PATH:" in
        *:/usr/local/bin:*)
            ln -s -- "$go_binary" /usr/local/bin/go || exit 1
            command -v go >/dev/null 2>&1
            exit $?
            ;;
    esac
done
echo 'Go compiler is unavailable on PATH; check the task image and sandbox environment' >&2
exit 127
"""


async def ensure_go_command(sandbox: AsyncSandbox) -> None:
    """Make a preinstalled Go compiler discoverable, or reject an unusable sandbox."""
    result = await exec_idempotent(sandbox, _GO_SETUP, what="exposing the task image's Go compiler", timeout_s=30)
    if result.return_code != 0 or result.error_type is not None:
        raise RuntimeError(
            f"Go runtime setup failed (exit={result.return_code}, error={result.error_type}): "
            f"{result.stderr or result.stdout or ''}"
        )
