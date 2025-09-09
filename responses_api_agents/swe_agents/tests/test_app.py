# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Add parent directory to path for imports
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))

from app import SWEBenchWrapperConfig


def test_configuration():
    """Test configuration options."""
    config = SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        entrypoint="responses_api_agents/swe_agents",
        agent_framework="swe_agent",
        agent_config="custom/config",
        agent_tools_file="tools.json",
        agent_max_turns=50,
        container_formatter="docker://custom/{instance_id}",
        swebench_tests_timeout=900,
        nemo_skills_config={"custom": "value"},
    )
    assert config.agent_framework == "swe_agent"
    assert config.agent_config == "custom/config"
    assert config.agent_tools_file == "tools.json"
    assert config.agent_max_turns == 50
    assert config.container_formatter == "docker://custom/{instance_id}"
    assert config.swebench_tests_timeout == 900
    assert config.nemo_skills_config["custom"] == "value"


if __name__ == "__main__":
    # Run tests if pytest is available
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        # Run basic tests manually
        print("Running tests manually (pytest not installed)...")

        test_configuration()
        print("✓ test_configuration")

        print("\nBasic tests passed!")
