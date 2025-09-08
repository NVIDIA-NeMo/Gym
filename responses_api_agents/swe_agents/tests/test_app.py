"""Tests for SWE agents."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
import sys
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
        nemo_skills_config={"custom": "value"}
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