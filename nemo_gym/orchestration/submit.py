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

from nemo_gym.decorators import experimental
from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.slurm import SlurmExecutor

_EXECUTORS = {
    SlurmComputeConfig: SlurmExecutor,
}


@experimental
def submit(config: SubmitConfig, *, dry_run: bool = False) -> None:  # pragma: no cover
    compute = next(iter(config.compute.values()))
    _EXECUTORS[type(compute)]().run(config, dry_run=dry_run)
