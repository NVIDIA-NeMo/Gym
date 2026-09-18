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

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import yaml

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.jobs import MANIFEST_NAME, RESOLVED_CONFIG_NAME, SubmissionRecord


class BaseExecutor(ABC):
    # Whether this executor can auto-resubmit a benchmark that gets killed by
    # the scheduler (time limit, preemption, node failure). False means asking
    # for `resumable` on this executor is a config error, not a silent no-op.
    supports_resumable: ClassVar[bool] = False

    @abstractmethod
    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        """Submit `config` and return the record describing it.

        Returns None on a dry run, which renders the scripts and stops before
        anything is submitted; every other path either returns a record or
        raises.
        """

    def persist(
        self,
        record: SubmissionRecord,
        config: SubmitConfig,
        write_manifest: Callable[[Path, str], None],
    ) -> None:
        """Store the record and the resolved config, in the order that survives a partial failure.

        Shared by every executor because the ordering and the failure handling
        are policy rather than transport. The machine-local index goes first
        precisely because it cannot fail the submit, so if the manifest write
        does fail there is still a parseable record for the by-hand recovery the
        error asks for.

        The resolved config is written here, not by each executor's staging
        step, so that every executor -- today Slurm, tomorrow k8s or a local
        runner -- gets it for free from the one place all of them already call.

        Only the files' transport differs per executor, so it arrives as
        `write_manifest` -- `Connection.write_text` already has this signature --
        rather than this class owning a connection it cannot know how to open.
        Call it while that transport is still open: reopening one here would pay
        a second connection on every submit.

        The jobs are queued by the time this runs, so a failure has to name them
        or they are stranded with no record anywhere.
        """
        record.write_local_index()
        run_dir = Path(record.run_dir)
        resolved_config = run_dir / RESOLVED_CONFIG_NAME
        manifest = run_dir / MANIFEST_NAME
        try:
            write_manifest(resolved_config, yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
            write_manifest(manifest, record.dumps())
        except Exception as error:
            queued = ", ".join(f"{b.benchmark}={b.job_id}" for b in record.benchmarks if b.job_id)
            raise RuntimeError(
                f"Submitted jobs but could not write the manifest/resolved config under {run_dir}: {error}. "
                f"Already queued: {queued or 'nothing'}. Record these by hand before collecting."
            ) from error
