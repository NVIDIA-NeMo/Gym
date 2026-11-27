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
"""The evaluation recovery policy: what `gym eval resubmit --recover` decides from a run's saved evidence.

The entry points, each in the module that owns it:

- ``evidence``: ``COLLECT_SCRIPT`` reads a job directory into ``CollectedEvidence``; ``select_managed_evidence`` adds
  the run's own Store selection.
- ``planner``: ``decide_from_collected`` turns the evidence into a ``RecoveryDecision`` (``decision``) under the
  run's ``RecoveryOutcomePolicyConfig`` (``policy``) and budgets (``budgets``).
- ``gate``: ``gate_summary_from_collected`` and ``gate_task_summary`` report the acceptance gate of a decision.
- ``grading``: ``reverification_cause`` says why a saved row is graded again from its saved answer.
- ``continuation``: ``continuation_admission``, which a chain's job script runs in the run's installed runtime.

``rows`` (identity states), ``trials`` (trial observations and fault classes) and ``markers`` (the strings the chain
scripts write) serve those modules. Executing a decision (queueing a follow-up, a judge-only job, a fresh restart) is
the caller's part. This package never imports ``nemo_gym.orchestration``; import-linter enforces it.
"""
