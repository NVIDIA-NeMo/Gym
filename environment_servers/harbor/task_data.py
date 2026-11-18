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
"""Task-data schema for the harbor environment server.

A Harbor task folder owns the task (instruction, image, tests, timeouts). The row the loader
materializes carries the instruction as the user message in ``responses_create_params`` and, as
its task-owned fields, the content digest of the folder it came from and the digest of the dataset
it belongs to. The server resolves the task as ``<taskset folder>/<task_id>`` and refuses a row
whose folder changed since materialization (409), so a run never grades edited tests against
stale rows. The dataset digest is provenance: rollout collection copies the ``_ng_*`` keys of a
materialized row's ``task_data`` onto the record it writes, so every ``rollouts.jsonl`` row names the
task folder (``_ng_digest``) and the dataset (``_ng_dataset_digest``) it ran.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # The wire key is ``_ng_digest`` (nemo_gym.data.tasks.harbor.DIGEST_KEY); pydantic forbids a leading
    # underscore in a field name, so the field is addressed by alias.
    ng_digest: str = Field(
        alias="_ng_digest",
        description="Content hash of the task folder when the row was materialized.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    # The wire key is ``_ng_dataset_digest`` (nemo_gym.data.tasks.harbor.digest.DATASET_DIGEST_KEY).
    ng_dataset_digest: str | None = Field(
        default=None,
        alias="_ng_dataset_digest",
        description="Digest of the dataset the task belongs to (manifest.toml [dataset].digest) at materialization.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
