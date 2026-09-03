# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""In-gym grading of instruction-following (IF) constraints for the swe_agents app (harness patch 0008).

A data generator embeds the constraints an episode must honour in the row's request metadata
(`metadata.sdg_item`, a JSON string whose `constraints` list carries one `verifier_parameter` per
constraint) together with the tool binding the episode runs under (`metadata.tool_name_overrides`).
`grade_row` grades the recorded episode against every constraint and returns the `if_constraints`
list that `SWEBenchWrapper.run` attaches to the verify response. The outcome reward is not touched;
dataset aggregates are computed by a separate offline script from the attached records.

Provenance of the vendored verifier
-----------------------------------
`verifier.py` in this package is a byte-identical copy of the canonical verifier

  /lustre/fsw/portfolios/llmservice/users/charlwang/cluster/work/logbook/problems/P0000-one-off-task/experiments/E260823-agentic-if-understand-lin-work/runs/2026-08-31-constraint-design/verifier_impl/template_verifiers.py

(its specification is VERIFIER_SPEC.md and its validation record VALIDATION.md in the same
directory). The copy carries no header of its own so that a byte-comparison test can hold it to the
canonical file; do not edit it here, re-copy it from the canonical location instead.

`grader.py` is a port of the grading path of the offline scorer

  /lustre/fsw/portfolios/llmservice/users/charlwang/cluster/work/data/runs/P0000-one-off-task/2026-09-02_r7-sdg-turn-output-samples/score_if.py

(functions `segment`, `to_tv_turns`, `prefix_turn_count`, `score_item`). Both modules use only the
standard library so that they can be tested with a plain interpreter, outside the gym environment.
"""
from .grader import grade_row

__all__ = ["grade_row"]
