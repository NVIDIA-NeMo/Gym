# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed in-sandbox paths of the plugin bundle; the OpenCode agent instance config names the same values."""

from pathlib import Path


BUNDLE_ROOT = "/tmp/nemo-gym-agentic-if"
"""The bundle directory the Resources Server creates at seed."""

PROFILE_PLUGIN = "if-profile.mjs"
BOUNDARY_PLUGIN = "if-boundary.mjs"
PROFILE_CONFIG = "profile.json"
PLANTS_CONFIG = "plants.json"
BOUNDARY_CONFIG = "boundary.json"

PLUGIN_PATHS: tuple[str, str] = (f"{BUNDLE_ROOT}/{PROFILE_PLUGIN}", f"{BUNDLE_ROOT}/{BOUNDARY_PLUGIN}")
"""The agent's ``plugins`` list, in this order: the profile plugin, then the boundary."""

EXPORT_DIR = f"{BUNDLE_ROOT}/exports"
"""The agent's ``export_dir``: one raw ``opencode export`` per session plus ``sessions.json``, written at close."""

SESSION_IMPORT_PATH = f"{BUNDLE_ROOT}/prefix-export.json"
"""The agent's ``session_import_path``; written by the Resources Server for continuation rows only."""

WORKSPACE_BASE_PATH = f"{BUNDLE_ROOT}/workspace.json"
"""``{"base_tree": ...}``: the tree the profile plugin's workspace diffs are taken against (null: no diffs);
written by the Resources Server at seed."""

RECEIPTS_DIR = f"{BUNDLE_ROOT}/receipts"
"""Written by the plugins as the identity OpenCode runs as."""

HOOK_RECEIPTS = "hook-applied.jsonl"
"""Profile plugin records: load receipt, tool definitions, tool and presentation receipts."""

EVENT_RECEIPTS = "hook-events-fired.jsonl"
"""Planted-event records: one ``loaded`` (or ``error``) line, then one line per firing."""

CALL_RECORDS = "model-calls.jsonl"
"""Boundary records: load and configuration receipts, then the request, reply and issue lines of every model call."""

EXPORT_INVENTORY = "sessions.json"
"""The agent's session inventory inside ``EXPORT_DIR``: ``[{"id": ..., "parent_id": ...}]``."""

SOURCE_DIR = Path(__file__).parent
"""Where the plugin sources live in this package."""
