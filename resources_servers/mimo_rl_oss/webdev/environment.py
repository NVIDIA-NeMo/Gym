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
import os

from mimoagent.environments.datasets import DATASET_REGISTRY, DatasetEnvironment

from resources_servers.mimo_rl_oss.webdev.eval_mode import grade_eval


# MiMo's launcher serves file:// pages over loopback http so runtime-rendered sites (React, ES modules) render.
os.environ.setdefault("WEBDEV_GRADE_HTTP", "1")

DELIVERY_INSTRUCTIONS = """Build the website the user asks for, then DELIVER it to {cwd}/dist/: opening {cwd}/dist/index.html \
in a browser must show the finished site. Always produce a working website; don't ask the user to confirm details.

- Develop however you like under {cwd}. Only {cwd}/dist/ is delivered, so a build step must put its static \
output there.
- The entry point MUST be exactly {cwd}/dist/index.html with <!DOCTYPE html>, and everything it needs (scripts, \
styles, images, fonts) must sit under {cwd}/dist/, referenced by paths relative to the dist/ root.
- Tailwind CSS via CDN (<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>) and/or inline \
<style> is fine.

Build a website for the following request:

{task}"""


class WebdevEnvironment(DatasetEnvironment):
    """MiMo webdev rows, graded with MiMo's pointwise webdev_eval_v1 mode.

    Renders {cwd}/dist in the box with playwright and asks one vision judge for
    (visual + query_fulfillment + premium_assets) / 3. MiMo trains on a group-relative
    pick from a separate grading service instead, which has no per-rollout reward.
    """

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
            # MiMo masks these render or judge infrastructure failures. Gym has no mask, so flag them.
            return 0.0, grading.get("drop_reason", ""), {"error_category": "webdev_drop", "grading": grading}
        return float(result["grader_reward"]), grading.get("reasoning", ""), {"grading": grading}


DATASET_REGISTRY.setdefault("webdev", WebdevEnvironment)
