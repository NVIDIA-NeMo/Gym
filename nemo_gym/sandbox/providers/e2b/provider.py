# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox provider backed by the E2B Python SDK.

Works against e2b.dev itself and against any e2b-compatible gateway (point
``connection.api_url``/``connection.sandbox_url`` at it).

Two E2B concepts differ from the provider-neutral :class:`SandboxSpec` and are
handled explicitly rather than silently:

**Templates, not images.** E2B starts sandboxes from a pre-built *template*
name or ID, not from an arbitrary registry reference. Because tagged template
names and OCI image references can both contain ``:``, the direct
``SandboxSpec.image`` shortcut is deliberately limited to unambiguous names in
``[A-Za-z0-9_-]``. Use ``provider_options.template`` or
``create.template_map`` for tagged names, IDs, and image references.

**Resources are fixed at template build time.** ``cpu_count``/``memory_mb`` are
arguments to the template *build*, so a per-sandbox
``SandboxSpec.resources`` cannot be honoured at create time. Requests are
reported once per provider instance (or raise, with ``create.strict_resources``)
instead of being dropped quietly.

**Snapshots keep processes.** A snapshot (``create_snapshot``) saves the whole
VM: memory, running processes and disk. A sandbox created from it resumes those
processes where they were, so the provider declares ``capabilities.snapshot_keeps`` with ``"processes"``.
The snapshot is usable as soon as the call returns.

**Commands run in login shells.** The SDK starts every command as
``bash -l -c`` and every terminal as ``bash -i -l``. A process does start with
its image's environment, but a Debian image's ``/etc/profile`` then resets
``PATH``. Each new sandbox therefore gets a ``profile.d`` hook
(``create.restore_login_env``) that gives a login shell back the environment it
started with.
"""

import asyncio
import ipaddress
import json
import logging
import math
import os
import re
import shlex
import uuid
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable, ClassVar, TypeVar

from nemo_gym.sandbox.attribution import AttributionConfig
from nemo_gym.sandbox.providers.base import (
    NETWORK_RULES,
    NetworkRule,
    SandboxCapabilities,
    SandboxCreateError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxSnapshotError,
    SandboxSnapshotInfo,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.e2b._sdk import E2B_SDK_CONSTRAINT, require_e2b_sdk
from nemo_gym.sandbox.providers.utils import require_shell_text
