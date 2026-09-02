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
"""Run many (dataset, config) pairs as a single Gym job.

A sweep manifest lists entries, each pairing one input JSONL with the configs that
define the agent it dispatches to. Because rollout collection routes every row by its
own `agent_ref`, the entries can be concatenated into one input file and served by one
Gym deployment composed from the union of their configs.
"""

from nemo_gym.sweep.manifest import (
    SweepEntry,
    SweepManifest,
    SweepValidationError,
    load_manifest,
    validate_manifest,
)


__all__ = [
    "SweepEntry",
    "SweepManifest",
    "SweepValidationError",
    "load_manifest",
    "validate_manifest",
]
