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

"""Name -> harness adapter lookup.

Keeps the orchestrator harness-agnostic: it resolves the adapter from the
configured harness name and never needs to know the concrete adapter classes.
Onboarding a new harness is "add an adapter module and register it here".
"""

from __future__ import annotations

from responses_api_agents.external_harness.adapters.base import HarnessAdapter


REGISTRY: dict[str, HarnessAdapter] = {}


def register_adapter(adapter: HarnessAdapter) -> None:
    REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> HarnessAdapter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown harness {name!r}; registered: {sorted(REGISTRY)}") from None
