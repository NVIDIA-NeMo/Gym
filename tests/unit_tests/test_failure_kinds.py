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

from nemo_gym.failure_kinds import (
    ACTIVE_FAILURE_KINDS,
    FAILURE_KIND_REGISTRY,
    RESERVED_FAILURE_KINDS,
)


def test_every_failure_kind_is_in_exactly_one_registry_tier() -> None:
    assert ACTIVE_FAILURE_KINDS.isdisjoint(RESERVED_FAILURE_KINDS)
    assert ACTIVE_FAILURE_KINDS | RESERVED_FAILURE_KINDS == FAILURE_KIND_REGISTRY.keys()

    for kind in ACTIVE_FAILURE_KINDS:
        assert FAILURE_KIND_REGISTRY[kind].status == "active"
    for kind in RESERVED_FAILURE_KINDS:
        assert FAILURE_KIND_REGISTRY[kind].status == "reserved"


def test_reserved_failure_kinds_name_their_source_pr() -> None:
    for kind in RESERVED_FAILURE_KINDS:
        assert FAILURE_KIND_REGISTRY[kind].source.startswith("#")
