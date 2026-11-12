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
"""The evidence collector lives in its own module; recovery and the Slurm executor run that same script."""

from nemo_gym.orchestration import collect_evidence, recovery
from nemo_gym.orchestration.executors import slurm


def test_recovery_and_the_slurm_executor_run_the_collector_module_s_script():
    assert recovery.COLLECT_SCRIPT is collect_evidence.COLLECT_SCRIPT
    assert slurm.COLLECT_SCRIPT is collect_evidence.COLLECT_SCRIPT
    compile(collect_evidence.COLLECT_SCRIPT, "<collect_evidence>", "exec")


def test_the_constants_the_collector_embeds_stay_importable_from_recovery():
    for name in (
        "EPISODE_DEADLINE_REASON",
        "JUDGE_JOB_ACK_JOURNAL_NAME",
        "JUDGE_JOB_JOURNAL_NAME",
        "OPERATOR_CONTINUATION_JOURNAL_NAME",
        "UNOBSERVED_CLAIM_REASON",
        "UNREADABLE_CLAIM_REASON",
    ):
        assert getattr(recovery, name) == getattr(collect_evidence, name)
        assert repr(getattr(collect_evidence, name)) in collect_evidence.COLLECT_SCRIPT
