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
"""The chain markers: the strings the auto-resume chain scripts write into a job directory and the recovery
evidence reader parses back.

They are the whole contract between the chain (``nemo_gym.orchestration.executors.resume_script``, which renders
the scripts) and the recovery planner (``nemo_gym.eval.recovery``), so they live below both: the chain imports
them from here, and the planner never imports the chain.
"""

ACCOUNTING_UNKNOWN_PREFIX = "GYM_ACCOUNTING_UNKNOWN:"
# The line prefix the installed continuation reader prints its decision reason with.
CONTINUATION_REASON_PREFIX = "__GYM_CONTINUATION_REASON:"
# The unknown-accounting reason a requeued allocation writes; recovery reconciles it from complete terminal accounting.
REQUEUE_UNKNOWN_REASON = (
    "Requeued allocation elapsed time is unknown; recover only after complete terminal accounting."
)
# The previous state a sentinel gives the installed continuation reader to reconcile a requeued chain, with the
# chain's records in CHAIN_ACCOUNTING_ENV; the reader prints the reconciled walltime after ACCOUNTED_WALLTIME_PREFIX.
REQUEUE_RECONCILE_STATE = "REQUEUED"
CHAIN_ACCOUNTING_ENV = "GYM_CHAIN_ACCOUNTING"
ACCOUNTED_WALLTIME_PREFIX = "__GYM_ACCOUNTED_WALLTIME:"
# The first words of the line a continuation writes to the incomplete marker as soon as it records itself, before
# it has queued its own follow-up. It stays only when the job ended in between, before its guard.
PROVISIONAL_MARKER_PREFIX = "Continuation "
# The words of the line a sentinel writes when its follow-up was cancelled before it started and the parent did
# not finish the run: the chain stopped there, and the line says why.
CANCELLED_FOLLOWUP_MARKER = "was cancelled before it started"
# The scheduler answered the follow-up's sbatch with a refusal (a QOS submit limit, an invalid account): nothing was
# queued, and the accounting is known. The chain stops; `gym eval resubmit --recover` continues it.
FOLLOW_UP_SUBMISSION_REFUSED = "follow_up_submission_refused"
