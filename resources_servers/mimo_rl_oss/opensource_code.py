# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from mimoagent.environments.datasets import DATASET_REGISTRY, OpenSourceCodeEnvironment


class StrippedOpenSourceCodeEnvironment(OpenSourceCodeEnvironment):
    def _assert_history_truncated(self) -> None:
        if not self._strip_future_commits(self._base_ref):
            raise RuntimeError(f"{self.instance_id}: stripping commits past {self._base_ref[:12]} failed")
        super()._assert_history_truncated()


DATASET_REGISTRY["opensource-code"] = StrippedOpenSourceCodeEnvironment
