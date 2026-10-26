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
"""Failures-sidecar rows for rollouts that produced no result.

An attempt the agent reports as ``kill_shaped`` and a row ``dispatch_budget_s`` never started both
carry ``_ng_no_persist``: there is no result to score or count as an attempt. By default rollout
collection writes nothing for them, so their absence is the only evidence. With
``record_omitted_rollouts`` it writes the row built here to the failures sidecar instead. The row
keeps ``_ng_no_persist``, and every reader of the sidecar skips a row that carries it, so the row
stays out of scoring and the resume attempt count and resume still runs the rollout.
"""

from typing import Any, Dict, Mapping

from nemo_gym.failure_kinds import CANCELLED, KILL_SHAPED


_FAILURE_MESSAGE_KEY = "_ng_failure_message"


def omitted_rollout_record(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The failures-sidecar row for a result marked ``_ng_no_persist``.

    The row is a copy of the result, so the rollout's identity, its agent and whatever diagnostics
    the agent returned stay with it. The failure class the result carries is kept; a result without
    one is ``cancelled`` when the dispatch budget never started it and ``kill_shaped`` otherwise.
    ``_ng_failure_message`` keeps the reason the result gave, else says that no result was produced.
    """
    # Lazy: rollout_collection imports this module.
    from nemo_gym.rollout_collection import NG_DISPATCH_DRAINED_KEY, NG_FAILURE_CLASS_KEY

    record = dict(result)
    failure_class = record.get(NG_FAILURE_CLASS_KEY) or (
        CANCELLED if record.get(NG_DISPATCH_DRAINED_KEY) else KILL_SHAPED
    )
    reason = record.get(_FAILURE_MESSAGE_KEY) or record.get("error_message") or record.get("error")
    record[NG_FAILURE_CLASS_KEY] = failure_class
    record[_FAILURE_MESSAGE_KEY] = str(reason) if reason else f"{failure_class}: the rollout produced no result"
    return record
