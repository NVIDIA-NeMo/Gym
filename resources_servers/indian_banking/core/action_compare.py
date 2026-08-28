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
"""Soft argument comparison for ACTION evaluation: ignores ephemeral / free-text
fields so paraphrases, timestamps, and generated ids don't cause a false fail."""

from __future__ import annotations

from typing import Any, Optional


# Keys that must never gate ACTION matching (ephemeral / free-text only).
# Do NOT include product targets like deposit_id, card_id, mandate_id.
VOLATILE_ARG_KEYS = frozenset(
    {
        "updated_at",
        "cancelled_at",
        "blocked_at",
        "booked_at",
        "closed_at",
        "logged_at",
        "requested_at",
        "created_at",
        "timestamp",
        "receipt_ref",
        "request_id",
        "reference_id",
        "reference",
        "download_url",
        "expiry",
        "eta",
        "description",
        "note",
    }
)

OPTIONAL_ARG_KEYS = frozenset({"account_id"})

_MISSING = object()


def args_match(
    gold_args: dict[str, Any],
    pred_args: dict[str, Any],
    compare_args: Optional[list[str]] = None,
) -> bool:
    """Return True if predicted args match gold under soft rules.

    - ``compare_args is None``: compare gold's keys (minus volatile).
    - ``compare_args == []``: always True (name-only match at caller).
    - else: compare listed keys (minus volatile).
    """
    if compare_args is not None and len(compare_args) == 0:
        return True

    if compare_args is None:
        keys = [k for k in gold_args.keys() if k not in VOLATILE_ARG_KEYS]
    else:
        keys = [k for k in compare_args if k not in VOLATILE_ARG_KEYS]

    for key in keys:
        gold_v = gold_args.get(key, _MISSING)
        pred_v = pred_args.get(key, _MISSING)
        if key in OPTIONAL_ARG_KEYS:
            if gold_v in (None, _MISSING) or pred_v in (None, _MISSING):
                continue
        if gold_v is _MISSING:
            continue
        if pred_v is _MISSING or gold_v != pred_v:
            return False
    return True
