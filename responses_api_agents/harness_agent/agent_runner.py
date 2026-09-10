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
import asyncio
import importlib
import json
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR / "gym_mount"))

import nemo_gym  # noqa: E402


assert nemo_gym.__file__.startswith(str(WORK_DIR / "gym_mount")), f"wrong nemo_gym: {nemo_gym.__file__}"

from nemo_gym.agents.config import AgentHarnessConfig  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402


def main() -> None:
    rc = json.loads((WORK_DIR / "runner_config.json").read_text())
    body = json.loads((WORK_DIR / "request.json").read_text())
    model_url = (WORK_DIR / "model_url.txt").read_text().strip()
    cfg_raw = (WORK_DIR / "harness_config.json").read_text().replace("__SANDBOX_MODEL_URL__", model_url)

    module = importlib.import_module(rc["harness_module"])
    harness_class = getattr(module, rc["harness_class"])
    harness = harness_class(AgentHarnessConfig.model_validate_json(cfg_raw))

    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    resp = asyncio.run(harness.run(params, model_base_url=harness.config.model.base_url))
    (WORK_DIR / "response.json").write_text(resp.model_dump_json())
    print("RUNNER_DONE")


if __name__ == "__main__":
    main()
