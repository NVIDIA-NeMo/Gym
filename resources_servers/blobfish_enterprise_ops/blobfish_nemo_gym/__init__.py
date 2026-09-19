# SPDX-FileCopyrightText: Copyright (c) 2026 Blobfish AI. All rights reserved.
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
"""Blobfish suites as NeMo Gym environments.

The stdlib core (``suite``, ``dataset``, ``emit``) has no third-party dependencies and works
without NeMo Gym installed. ``server`` layers the NeMo Gym resources server on top and imports
``nemo_gym`` lazily, so ``import blobfish_nemo_gym`` never fails on a machine without it.
"""

from .suite import (
    OUTCOME_FAMILIES,
    Criterion,
    Episode,
    Suite,
    SuiteIntegrityError,
    TaskPackage,
    ToolSpec,
    Verdict,
    VerifierError,
    criteria_of,
    outcome_fraction,
    package_digest,
    public_tool_name,
    score_episode,
)


__all__ = [
    "OUTCOME_FAMILIES",
    "Criterion",
    "Episode",
    "Suite",
    "SuiteIntegrityError",
    "TaskPackage",
    "ToolSpec",
    "Verdict",
    "VerifierError",
    "criteria_of",
    "outcome_fraction",
    "package_digest",
    "public_tool_name",
    "score_episode",
]
