# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import yaml


def test_module_parses():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    src = app_path.read_text()
    compile(src, str(app_path), "exec")


def test_config_yaml_parses():
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "hermes_agent.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    assert "hermes_agent" in data
    inner = data["hermes_agent"]["responses_api_agents"]["hermes_agent"]
    assert inner["entrypoint"] == "app.py"
    assert inner["max_turns"] == 30
    assert inner["enabled_toolsets"] is None
    assert inner["system_prompt"] is None
    assert inner["terminal_backend"] == "local"
    assert inner["terminal_timeout"] == 60
    assert "disabled_toolsets" not in inner


if __name__ == "__main__":
    test_module_parses()
    test_config_yaml_parses()
    print("OK")
