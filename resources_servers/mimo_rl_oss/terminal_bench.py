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
import base64
import json
import math
import os
import shlex
import tempfile

from mimoagent.environments.datasets import DATASET_REGISTRY, DatasetEnvironment


class TerminalBenchEnvironment(DatasetEnvironment):
    REPO_PATH = "/app"
    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    _ANTI_HACK_CLEANUP_DEFAULT = False
    REWARD_FILE = "/logs/verifier/reward.txt"

    @property
    def repo_path(self) -> str:
        return self.instance.get("cwd") or self.REPO_PATH

    def _setup_dataset_specific(self) -> None:
        pass

    def _capture_model_diff(self) -> tuple[str, str]:
        return "", ""

    def _do_calculate_reward(self, timeout=None, model_patch: str = "") -> tuple[float, str, dict]:
        files = json.loads(self.instance["tests_files"])
        self.execute("rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier", cwd="/")
        for name, data in files.items():
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(base64.b64decode(data))
            try:
                self.env.copy_to(f.name, f"/tests/{name}")
            finally:
                os.unlink(f.name)
        res = self.execute(
            "bash /tests/test.sh",
            cwd=self.repo_path,
            timeout=int(timeout or float(self.instance.get("verifier_timeout_sec") or 900)),
        )
        if res.get("reason") != "ok":
            return 0.0, str(res.get("output", "")), {"transport_error": True}
        raw = self.execute(f"cat {shlex.quote(self.REWARD_FILE)}", cwd="/").get("output", "").strip()
        try:
            reward = float(raw.splitlines()[-1])
            if not (math.isfinite(reward) and 0.0 <= reward <= 1.0):
                raise ValueError(reward)
        except (ValueError, IndexError):
            return (
                0.0,
                str(res.get("output", "")),
                {"error_category": "testbed_corrupted", "reward_error": "no reward file"},
            )
        return reward, str(res.get("output", "")), {"verifier_returncode": res.get("returncode")}


DATASET_REGISTRY.setdefault("terminal_bench", TerminalBenchEnvironment)
