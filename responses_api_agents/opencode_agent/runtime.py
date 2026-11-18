# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stock OpenCode plus the same observability patch, locally and in sandboxes."""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nemo_gym.sandbox import SandboxExecResult


OPENCODE_VERSION = "1.17.11"
# OpenCode 2 (npm @opencode/cli, GitHub releases of anomalyco/opencode), selected with opencode_version.
OPENCODE_V2_VERSION = "2.2.7"
# SHA-256 of each Linux release asset the sandbox installer downloads, from the release's SHA256SUMS. A 2.x version
# without an entry here cannot be downloaded into a sandbox; pin a binary with sandbox_executable_path instead.
OPENCODE_RELEASE_SHA256 = {
    "2.2.7": {
        "x64-baseline": "89a6cee227b54bc011179c528a5e63176032462b1a5e21723b616801cddc2ca7",  # pragma: allowlist secret
        "x64-baseline-musl": "c1801a36bf61e03eafac8d7f4521a200e4d3cd76cf43b36621cb78b460b851ee",  # pragma: allowlist secret
        "arm64": "c1682057f0815cb2abdd6e00305f1146ebaf1c167f0335cdb74ddc80008120e7",  # pragma: allowlist secret
        "arm64-musl": "a03405f94ae02db7acb0870cf32030bab86460dc7faf524780e9cae305526b96",  # pragma: allowlist secret
    },
}


def opencode_major(version: str) -> int:
    """The major version of an exact ``opencode_version`` such as ``1.17.11`` or ``2.2.7``."""
    return int(version.split(".", 1)[0])


RUNTIME_INSTALLER_SOURCE = Path(__file__).with_name("install_opencode_runtime.sh")
RUNTIME_INSTALLER_FILE = RUNTIME_INSTALLER_SOURCE.name
TASK_ENVIRONMENT_PLUGIN_SOURCE = Path(__file__).with_name("task_environment.mjs")
TASK_ENVIRONMENT_PLUGIN_FILE = TASK_ENVIRONMENT_PLUGIN_SOURCE.name
ASSISTANT_MESSAGE_PLUGIN_SOURCE = Path(__file__).with_name("assistant_message_header.js")
ASSISTANT_MESSAGE_PLUGIN_FILE = ASSISTANT_MESSAGE_PLUGIN_SOURCE.name
REQUIRED_MCP_PLUGIN_SOURCE = Path(__file__).with_name("required_mcp.mjs")
REQUIRED_MCP_PLUGIN_FILE = REQUIRED_MCP_PLUGIN_SOURCE.name
# Retain the import used by the separate opencode_sandboxed_agent adapter.
OBSERVABILITY_PATCH = ASSISTANT_MESSAGE_PLUGIN_SOURCE
# Merged into the Gym model's entry in sandbox sessions that declare image input.
# OpenCode sends an image to the model only when the model declares image input in modalities.input.
# Otherwise it replaces each image, including images its read tool returns, with an error text
# telling the model it cannot read images. A text-only deployment rejects requests that carry an image.
IMAGE_INPUT_MODEL_OPTIONS = {"modalities": {"input": ["text", "image"], "output": ["text"]}}
# OpenCode 1.17.11 (packages/opencode/src/image/image.ts) shrinks and re-encodes any image over 2000 px on a side
# or 5 MiB of base64, both for prompt images and for images its read tool returns. Raised to the largest value its
# PositiveInt config schema accepts, the limits never apply, so the model receives the image as the environment sent
# it, as on a direct image_url input. `auto_resize: false` is not the setting for this: it fails the turn instead.
# opencode_sandboxed_agent sets the same limits (OPENCODE_IMAGE_LIMITS_UNRESTRICTED).
UNRESTRICTED_IMAGE_ATTACHMENT = {
    "image": {"max_width": 2**31 - 1, "max_height": 2**31 - 1, "max_base64_bytes": 2**31 - 1}
}


def apply_observability_patch(config: dict, *, plugin_path: Path = ASSISTANT_MESSAGE_PLUGIN_SOURCE) -> None:
    """Register the shared correlation plugin at its execution-side location."""
    config.setdefault("plugin", []).append(plugin_path.as_uri())


def command_failure_reason(message: str, result: "SandboxExecResult") -> str:
    """Keep bounded raw diagnostics; an error status need not represent a process exit."""
    return (
        f"{message[:512]}; return_code={result.return_code}; error_type={str(result.error_type)[:128]}; "
        f"stdout_tail={repr(result.stdout or '')[-400:]}; stderr_tail={repr(result.stderr or '')[-400:]}"
    )
