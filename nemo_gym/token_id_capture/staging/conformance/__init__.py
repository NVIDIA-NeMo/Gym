# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installable conformance vectors for independent framework integrations."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from nemo_gym.token_id_capture.staging.digest import compute_extras_digest, compute_staging_digest


def load_golden_vectors() -> dict[str, Any]:
    """Load language-neutral inputs and expected SHA-256 digests."""
    resource = files(__package__).joinpath("golden_vectors.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def assert_golden_vectors() -> None:
    """Raise when this process disagrees with the published wire contract."""
    vectors = load_golden_vectors()
    actual_extras = compute_extras_digest(vectors["extras"])
    if actual_extras != vectors["extras_digest"]:
        raise AssertionError(f"extras digest mismatch: {actual_extras}")
    actual_call = compute_staging_digest(**vectors["staged_call"])
    if actual_call != vectors["staged_call_digest"]:
        raise AssertionError(f"staged call digest mismatch: {actual_call}")


__all__ = ["assert_golden_vectors", "load_golden_vectors"]
