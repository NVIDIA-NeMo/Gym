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
import base64
import json
import os
import shlex
import tempfile

from mimoagent.environments.datasets import DATASET_REGISTRY, DatasetEnvironment


class TerminalBenchEnvironment(DatasetEnvironment):
    """MiMo general/terminal_bench rows: Harbor-style tests shipped base64 in tests_files.

    Tests land in /tests only at reward time so the agent never sees them.
    """

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
            # Bytes, not text: fixtures can be binary.
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(base64.b64decode(data))
            try:
                self.env.copy_to(f.name, f"/tests/{name}")
            finally:
                os.unlink(f.name)
        res = self.execute(
            "sh /tests/test.sh",
            cwd=self.repo_path,
            timeout=int(timeout or float(self.instance.get("verifier_timeout_sec") or 900)),
        )
        if res.get("reason") != "ok":
            return 0.0, str(res.get("output", "")), {"transport_error": True}
        raw = self.execute(f"cat {shlex.quote(self.REWARD_FILE)}", cwd="/").get("output", "").strip()
        try:
            reward = float(raw.splitlines()[-1])
        except (ValueError, IndexError):
            # test.sh always writes the reward file, so its absence is a broken testbed.
            return (
                0.0,
                str(res.get("output", "")),
                {"error_category": "testbed_corrupted", "reward_error": "no reward file"},
            )
        return reward, str(res.get("output", "")), {"verifier_returncode": res.get("returncode")}


DATASET_REGISTRY.setdefault("terminal_bench", TerminalBenchEnvironment)
