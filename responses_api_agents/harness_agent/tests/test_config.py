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
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import (
    ALLOW_UNSUPPORTED_PAIRING_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)


def test_terminal_bench_shared_hermes_pairing_resolves(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": ["benchmarks/terminal_bench_2_1/hermes_harness.yaml"]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    block = resolved["terminal_bench_2_1_hermes_harness"]["responses_api_agents"]["harness_agent"]
    resources_name = block["resources_server"]["name"]
    resources = resolved[resources_name]["resources_servers"]["terminal_bench_2_1"]
    assert block["agent"] == "hermes"
    assert block["sandbox_provider"] == resources["sandbox_provider"]
    assert resources["allowed_agents"] == ["harness_agent"]
    assert block["datasets"][0]["prepare_script"] == "benchmarks/terminal_bench_2_1/prepare.py"
    assert block["agent_kwargs"]["api_key"] == "dummy"
    assert block["agent_kwargs"]["terminal_backend"] == "local"
    assert block["token_id_capture"] is False
    assert not resolved.get(ALLOW_UNSUPPORTED_PAIRING_KEY_NAME, False)
