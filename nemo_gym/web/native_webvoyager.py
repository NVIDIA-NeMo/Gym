# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the original native WebVoyager contract path."""

from nemo_gym.web.native_visual import (
    NATIVE_VISUAL_SYSTEM_PROMPT,
    NATIVE_VISUAL_TOOLS,
    adapt_native_webvoyager_record,
    native_visual_tools,
)


NATIVE_WEBVOYAGER_SYSTEM_PROMPT = NATIVE_VISUAL_SYSTEM_PROMPT
NATIVE_WEBVOYAGER_TOOLS = NATIVE_VISUAL_TOOLS
native_webvoyager_tools = native_visual_tools


__all__ = [
    "NATIVE_WEBVOYAGER_SYSTEM_PROMPT",
    "NATIVE_WEBVOYAGER_TOOLS",
    "adapt_native_webvoyager_record",
    "native_webvoyager_tools",
]
