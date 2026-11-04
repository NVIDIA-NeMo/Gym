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
"""The Slurm executor for Pyxis-enabled clusters: ``SlurmExecutor``.

Its parts, each in its own module:

- ``executor``: render a run's job scripts, submit them with sbatch and record the submission;
- ``chain``: recovery of a run's auto-resume chain (``recovery_evidence``, ``resubmit``), mixed into the executor;
- ``judge_job``: judge-only jobs that grade a run's saved answers again (``reverify_judge_failures``), mixed in too;
- ``reader``: the remote Store reader both of them read a run through;
- ``scheduler``: the Slurm commands they share.

Only Slurm implements resubmit today, so these parts are Slurm code with no scheduler-agnostic interface between
them; the executor capabilities give resubmit its seam when a second scheduler needs one.
"""

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.orchestration.executors.slurm.executor import SlurmExecutor


__all__ = ["SlurmExecutor"]

__getattr__ = moved_names_getattr(__name__)
