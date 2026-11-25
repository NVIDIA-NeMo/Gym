# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stock OpenCode plus the same observability patch, locally and in sandboxes."""

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nemo_gym.sandbox import SandboxExecResult


OPENCODE_VERSION = "1.19.6"
# OpenCode 2 (npm @opencode/cli, GitHub releases of anomalyco/opencode), selected with opencode_version.
OPENCODE_V2_VERSION = "2.3.2"
# SHA-256 of each Linux release asset the sandbox installer downloads: the `digest` GitHub publishes for each asset of
# the release (`gh api repos/anomalyco/opencode/releases/tags/v<version>`), which OpenCode 2's SHA256SUMS repeats.
# A download of a version listed here is checked against it and needs sha256sum in the sandbox. A 1.x version without an
# entry downloads unchecked; a 2.x version without one cannot be downloaded into a sandbox, so pin a binary with
# sandbox_executable_path instead.
OPENCODE_RELEASE_SHA256 = {
    "1.17.11": {
        "x64-baseline": "36064327d5a3fba662264f5726f517eafa26d1aefb0d59bc95e66bddccf3497b",  # pragma: allowlist secret
        "x64-baseline-musl": "cdb8f1a435f7ef67798276397fc890b762f7946d2d77e5cd9be38ecbe32ae245",  # pragma: allowlist secret
        "arm64": "6c0ae8212401c78f9dcddead35d385953c7a44eb1616309365b58503dbd1b4cd",  # pragma: allowlist secret
        "arm64-musl": "8531805bdbcdd0b912652a989992a8857651fd09d715ee70c0eb9dd2f134953e",  # pragma: allowlist secret
    },
    "1.19.6": {
        "x64-baseline": "8443113b3f3dfebf8b0fa91c4377e5519a62b9231732d9344a5df001e4a852b3",  # pragma: allowlist secret
        "x64-baseline-musl": "bc6b0d702163d859161ee8e43ea3f16d1bd43b305ac7ce17742eb01ba72489f3",  # pragma: allowlist secret
        "arm64": "5b68c1f4c5e75b4134e664cc8f36d634d64208d4fcfe5bcf3b48bab699c2dbac",  # pragma: allowlist secret
        "arm64-musl": "45407388777dfc1894591c9727c169f202ad92c75757038950630a0fbb12ca97",  # pragma: allowlist secret
    },
    "2.2.7": {
        "x64-baseline": "89a6cee227b54bc011179c528a5e63176032462b1a5e21723b616801cddc2ca7",  # pragma: allowlist secret
        "x64-baseline-musl": "c1801a36bf61e03eafac8d7f4521a200e4d3cd76cf43b36621cb78b460b851ee",  # pragma: allowlist secret
        "arm64": "c1682057f0815cb2abdd6e00305f1146ebaf1c167f0335cdb74ddc80008120e7",  # pragma: allowlist secret
        "arm64-musl": "a03405f94ae02db7acb0870cf32030bab86460dc7faf524780e9cae305526b96",  # pragma: allowlist secret
    },
    "2.3.2": {
        "x64-baseline": "563b5d663637226b6e3bb4bb6803281bc39b0d2cd88db06d674b6420aab009b0",  # pragma: allowlist secret
        "x64-baseline-musl": "4e952a67874cfdb4f53eeae5840f9813acc9f109c10e61bc26020271f56d7d84",  # pragma: allowlist secret
        "arm64": "9d10185acdb72336e086e699ca363a6065f410fd3c1caed0dc8046e95ac06ccb",  # pragma: allowlist secret
        "arm64-musl": "2148ba748f5410177868ed67d59ef70626cb6bd4fb2abc5177c807adfdc24fc3",  # pragma: allowlist secret
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
# OpenCode 1 (packages/opencode/src/image/image.ts at 1.17.11 and 1.19.6) shrinks and re-encodes any image over 2000 px
# on a side or 5 MiB of base64, both for prompt images and for images its read tool returns. Raised to the largest value
# its PositiveInt config schema accepts, the limits never apply, so the model receives the image as the environment
# sent it, as on a direct image_url input. 1.19.6 still replaces the oldest images once a request carries more than
# 25 MiB of them (packages/opencode/src/session/message-v2.ts, `boundImages`), which no setting lifts. `auto_resize: false` is not the setting for this: it fails the turn instead.
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
