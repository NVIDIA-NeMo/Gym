# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agentic IF OpenCode plugin bundle: two plugins the Resources Server stages into the task sandbox.

``if-profile.mjs`` applies the row's tool interfaces, read-first guards and planted tool failures inside OpenCode and
records hook receipts. ``if-boundary.mjs`` wraps the policy provider's HTTP call: it presents the row's literal system
prompt, tool selection and descriptions, restores the model's own tool calls in history, and records every model call.
The Resources Server stages them at seed (``stage_bundle``) and reads what they recorded at verify
(``collect_bundle``).

Sandbox layout (``BUNDLE_ROOT``)::

    if-profile.mjs  if-boundary.mjs   plugins; the agent's ``plugins`` lists them in this order
    profile.json  plants.json         the profile plugin's configuration (``compile_profile``, ``compile_plants``)
    boundary.json                     the boundary's configuration (``compile_boundary``)
    receipts/                         hook-applied.jsonl, hook-events-fired.jsonl, model-calls.jsonl
    exports/                          the agent's ``export_dir``
    prefix-export.json                the agent's ``session_import_path`` (continuation rows; the Resources Server)
"""

from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.opencode_bundle.bundle import (
    BundleOutputs,
    BundleReceipt,
    ExecResult,
    SandboxIO,
    bundle_files,
    collect_bundle,
    presentation_gaps,
    stage_bundle,
)
from resources_servers.agentic_if.opencode_bundle.features import UnsupportedFeatureError
from resources_servers.agentic_if.opencode_bundle.layout import (
    BUNDLE_ROOT,
    EXPORT_DIR,
    PLUGIN_PATHS,
    RECEIPTS_DIR,
    SESSION_IMPORT_PATH,
)
from resources_servers.agentic_if.opencode_bundle.profile import compile_plants, compile_profile, profile_sha256


__all__ = [
    "BUNDLE_ROOT",
    "EXPORT_DIR",
    "PLUGIN_PATHS",
    "RECEIPTS_DIR",
    "SESSION_IMPORT_PATH",
    "BundleOutputs",
    "BundleReceipt",
    "ExecResult",
    "SandboxIO",
    "UnsupportedFeatureError",
    "bundle_files",
    "collect_bundle",
    "compile_boundary",
    "compile_plants",
    "compile_profile",
    "presentation_gaps",
    "profile_sha256",
    "stage_bundle",
]
