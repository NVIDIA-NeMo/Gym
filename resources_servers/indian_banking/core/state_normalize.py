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
#
# This file is derived from Sierra Research's tau2-bench evaluator
# (https://github.com/sierra-research/tau2-bench) and remains under its
# original MIT terms (reproduced below). Modifications Copyright (c) 2026 NVIDIA
# CORPORATION & AFFILIATES and contributors, licensed under the Apache License 2.0
# (SPDX header above).
#
# MIT License
#
# Copyright (c) 2025 Sierra Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Normalize env state for equality checks (tool-response replay + DB hash).

Strips fields that legitimately differ between two correct runs (ids
generated per-call, free text) before comparing, and bag-normalizes maps
keyed only by generated ids.
"""

from __future__ import annotations

import json
import re
from typing import Any


# Per-call generated fields that legitimately differ between two correct runs.
VOLATILE_STATE_KEYS = frozenset(
    {
        "request_id",
        "reference_id",
        "reference",
        "description",  # free-text, gold vs agent wording never matches byte-for-byte
        "note",
        "receipt_ref",  # response-only, not persisted, so no false-pass risk
        "download_url",
        "expiry",
    }
)

_REQUEST_OPTIONAL_KEYS = frozenset({"account_id"})

# Agents freely pick "other" vs "general_query"; treat them as the same category.
_CATEGORY_ALIASES = {
    "other": "general_query",
    "general_query": "general_query",
}

# Dict keys that are only server-assigned refs (e.g. requests["SR-12345678"]).
# When every key in a map matches this, compare as an order-independent bag.
_EPHEMERAL_DICT_KEY = re.compile(
    r"^(?:SR-|TXN|ADR|CHQ|STMT|RCPT|STP|FD|RD)\d+$",
    re.IGNORECASE,
)


def _stable_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _looks_like_request_record(d: dict) -> bool:
    """Heuristic: raise_request / ticket-shaped dicts."""
    return "category" in d and ("request_id" in d or "related_transaction_id" in d or "priority" in d)


def normalize_state(value: Any) -> Any:
    """Return a copy of ``value`` with ephemeral ids/timestamps removed."""
    if isinstance(value, dict):
        if value and all(isinstance(k, str) and _EPHEMERAL_DICT_KEY.match(k) for k in value):
            items = [normalize_state(v) for v in value.values()]
            return sorted(items, key=_stable_sort_key)
        drop = VOLATILE_STATE_KEYS
        if _looks_like_request_record(value):
            drop = VOLATILE_STATE_KEYS | _REQUEST_OPTIONAL_KEYS
        out = {k: normalize_state(v) for k, v in value.items() if k not in drop}
        if _looks_like_request_record(value) and "category" in out:
            cat = out["category"]
            if isinstance(cat, str) and cat in _CATEGORY_ALIASES:
                out["category"] = _CATEGORY_ALIASES[cat]
                # Priority follows category; keep soft SRs comparable.
                if out.get("priority") in ("NORMAL", "URGENT", None) and cat in (
                    "other",
                    "general_query",
                ):
                    out["priority"] = "NORMAL"
        return out
    if isinstance(value, list):
        return [normalize_state(item) for item in value]
    return value
