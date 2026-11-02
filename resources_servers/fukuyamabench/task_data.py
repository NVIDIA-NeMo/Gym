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

"""Task-data schema for the FukuyamaBench server.

Organic reaction-mechanism pathway prediction. All task fields live inside
``verifier_metadata`` (schema written flat; ``legacy_location`` records that).
Every read in ``verify()`` is a defensive ``meta.get(...)``, so all fields are
Optional, but a row missing its gold pathway or checkpoints scores 0.0 with
status ``bad_gold`` rather than being silently treated as a model failure.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    gt_pathway: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Gold mechanism as an ordered list of {step_id, products}, where products is a list of "
            "SMILES for that elementary step. Checkpoints index into this list by 1-based step ID. "
            "A missing value yields status 'bad_gold' rather than a zero that looks like a model miss."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    checkpoints: Optional[list[list[int]]] = Field(
        default=None,
        description=(
            "Upstream ckpt.txt, one inner list per checkpoint holding the gold step IDs that are "
            "mutually acceptable at that point. Matched in order against the predicted steps; all "
            "checkpoints matched is the reward. Trivial proton transfers are excluded upstream, so "
            "there are usually fewer checkpoints than mechanism steps."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    lenient: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a predicted product set that is a non-empty subset of the gold set counts as a "
            "checkpoint match, crediting a model that names the main product but omits a leaving "
            "group. True upstream by default; upstream's --strict disables it. Which setting produced "
            "the published figures is not stated by the paper. Defaults to True when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Upstream case directory name, e.g. 'B001'. Echoed on the verify response.",
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    case_set: Optional[str] = Field(
        default=None,
        description=(
            "Difficulty tier, 'A', 'B' or 'C'. Echoed for stratified reporting: published scores differ "
            "by roughly an order of magnitude between tiers, so aggregates across sets are not meaningful."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    n_gt_steps: Optional[int] = Field(
        default=None,
        description="Number of gold elementary steps; carried for provenance and length analysis.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Rendered reaction conditions as embedded in the prompt; carried for provenance.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    starting_reactants: Optional[str] = Field(
        default=None,
        description=(
            "Atom-mapped starting reactants as given to the model, matching upstream's SFT/RL input "
            "format. Carried for provenance; the scorer strips atom mapping before comparing."
        ),
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
