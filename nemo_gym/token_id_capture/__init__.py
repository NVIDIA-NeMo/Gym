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

"""Training-token capture: produce, store, read, and source ``TokenEntry`` records.

This is the per-model-call training data path, kept separate from evaluation
capture. The capture middleware sets a per-request token sink; the model server
records a ``TokenEntry`` from its complete response; a trainer reads a rollout's
entries through a ``TokenSource``.
"""

from typing import Any

from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.records import TOKEN_FIELDS, TokenEntry, extract_token_fields
from nemo_gym.token_id_capture.sink import TokenSink, capture_tokens, reset_token_sink, set_token_sink
from nemo_gym.token_id_capture.store import TokenCaptureStore, validate_rollout_id


# The reader/route/source exports pull in the server stack (fastapi, the
# shared aiohttp client) through nemo_gym.server_utils. They are resolved
# lazily (PEP 562) so the capture core -- records, digest, lineage, rebuild,
# protocols, conformance -- stays importable inside any framework's worker
# process with no serving dependencies (the § 3.0 purity rule, enforced by
# tests/unit_tests/test_token_capture_gate_primitives.py).
_LAZY_EXPORTS = {
    "TokenReader": ("nemo_gym.token_id_capture.reader", "TokenReader"),
    "LocalTokenReader": ("nemo_gym.token_id_capture.reader", "LocalTokenReader"),
    "HttpTokenReader": ("nemo_gym.token_id_capture.reader", "HttpTokenReader"),
    "TokenSource": ("nemo_gym.token_id_capture.source", "TokenSource"),
    "CaptureTokenSource": ("nemo_gym.token_id_capture.source", "CaptureTokenSource"),
    "make_token_store": ("nemo_gym.token_id_capture.routes", "make_token_store"),
    "install_token_capture_routes": ("nemo_gym.token_id_capture.routes", "install_token_capture_routes"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(target[0]), target[1])


__all__ = [
    "TokenIdCaptureConfig",
    "TokenEntry",
    "TOKEN_FIELDS",
    "extract_token_fields",
    "TokenCaptureStore",
    "validate_rollout_id",
    "TokenSink",
    "set_token_sink",
    "reset_token_sink",
    "capture_tokens",
    "TokenReader",
    "LocalTokenReader",
    "HttpTokenReader",
    "TokenSource",
    "CaptureTokenSource",
    "make_token_store",
    "install_token_capture_routes",
]
