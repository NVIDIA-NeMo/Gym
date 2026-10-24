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
import os

from mimoagent.environments.datasets import DATASET_REGISTRY, DatasetEnvironment

from resources_servers.mimo_rl_oss.webdev.eval_mode import grade_eval


os.environ.setdefault("WEBDEV_GRADE_HTTP", "1")


class WebdevEnvironment(DatasetEnvironment):
    REPO_PATH = "/workspace"
    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    _ANTI_HACK_CLEANUP_DEFAULT = False

    @property
    def repo_path(self) -> str:
        return (self.instance.get("cwd") or self.REPO_PATH).rstrip("/")

    def _setup_dataset_specific(self) -> None:
        self.execute(f"mkdir -p {self.repo_path}/dist", cwd="/")

    def _capture_model_diff(self) -> tuple[str, str]:
        return "", ""

    def _do_calculate_reward(self, timeout=None, model_patch: str = "") -> tuple[float, str, dict]:
        result = grade_eval(self.env, f"{self.repo_path}/dist", self.instance["problem_statement"], {})
        grading = result["grading"]
        if result["grader_reward"] is None:
            return 0.0, grading.get("drop_reason", ""), {"error_category": "webdev_drop", "grading": grading}
        return float(result["grader_reward"]), grading.get("reasoning", ""), {"grading": grading}


DATASET_REGISTRY.setdefault("webdev", WebdevEnvironment)
