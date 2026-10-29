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

"""Submitting a run to a cluster and looking after it afterwards.

The modules, by role:

    api.py                     input schema: what to submit (SubmitConfig and its parts)
    jobs.py                    output schema: what was submitted and what is happening (records, statuses)
    operations.py              what you do with a submission: status, info, cancel, logs (no executor imports)
    submit.py                  entry points: submit, resubmit and the wrappers the CLI calls; the only module
                               that knows which executors exist
    executors/base.py          BaseExecutor: the one method a new executor must implement
    executors/capabilities.py  the opt-in interfaces an executor adds, and `require`
    executors/<name>.py        one scheduler
"""
