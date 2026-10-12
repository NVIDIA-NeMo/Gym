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
"""Starts Nemo Gym inside the sandbox and runs the task e2e with ng_collect_rollouts."""

import json
import subprocess
import time
import urllib.error
import urllib.request


def main() -> None:
    rc = json.load(open("/work/runner_config.json"))
    model_url = open("/work/model_url.txt").read().strip()
    config_paths = "+config_paths=[" + ",".join(rc["config_paths"]) + "]"
    overrides = ["+policy_base_url=" + model_url + "/v1"] + rc["overrides"]

    subprocess.Popen(
        ["ng_run", config_paths] + overrides,
        stdout=open("/work/gym.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(150):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{rc['agent_port']}/health", timeout=2)
            break
        except urllib.error.HTTPError:
            break
        except Exception:
            time.sleep(2)

    try:
        subprocess.run(
            [
                "ng_collect_rollouts",
                config_paths,
                "+input_jsonl_fpath=/work/input.jsonl",
                "+output_jsonl_fpath=/work/rollouts.jsonl",
                "+agent_name=" + rc["agent_name"],
            ]
            + overrides,
            stdout=open("/work/collect.log", "w"),
            stderr=subprocess.STDOUT,
            timeout=rc["timeout"],
            check=True,
        )
    except Exception:
        for f in ("/work/collect.log", "/work/gym.log"):
            try:
                print(f, open(f).read()[-3000:])
            except OSError:
                pass
        open("/work/done", "w").write("1")
        raise
    open("/work/done", "w").write("1")
    print("RUNNER_DONE")


if __name__ == "__main__":
    main()
