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

"""Rollout identity: minting, validation, and the per-channel carriers.

One rollout id is minted where the task is scheduled and carried on each
channel in the strongest form that channel supports:

  - model calls from a CLI harness (only a base URL is settable) travel as a
    URL prefix ``/ng-rollout/{rid}/v1`` (matched by NG_ROLLOUT_PREFIX_RE)
  - model calls from in-process Gym agents travel in the
    ``x-nemo-gym-rollout-id`` header
  - environment tool calls over MCP travel as a signed session token bound to
    the id when the session is seeded

The id is meant to be treated as opaque everywhere except this module. Nothing
downstream should parse it: the task/rollout/attempt numbers travel as their
own separate fields on the run request and rollout record. The regex only
constrains the character set (so the id is safe in filenames and URLs); the
determinism comes from make_rollout_id, not from any promise about the string.
"""

from __future__ import annotations

import re


# Charset only: filename- and URL-safe, bounded length. NOT a structure contract.
ROLLOUT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

# Correlation carriers.
ROLLOUT_ID_HEADER = "x-nemo-gym-rollout-id"
NG_ROLLOUT_PREFIX_RE = re.compile(r"^/ng-rollout/(?P<rid>[A-Za-z0-9][A-Za-z0-9._:-]{0,127})(?P<rest>/.*)$")


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower() or "task"


def make_rollout_id(dataset: str, task_index: int, rollout_index: int, attempt: int = 0) -> str:
    """Deterministic id from the (dataset, task, rollout, attempt) coordinates.

    The attempt number is part of the id so that a re-run of the same task
    (for example after a failed generation) gets a fresh, distinct capture
    instead of appending to the previous one.
    """
    rid = f"{slugify(dataset)}.{task_index:05d}.r{rollout_index:02d}.a{attempt}"
    if not ROLLOUT_ID_RE.match(rid):
        raise ValueError(f"invalid rollout id: {rid!r}")
    return rid


def is_valid_rollout_id(rid: str) -> bool:
    return bool(ROLLOUT_ID_RE.match(rid))
