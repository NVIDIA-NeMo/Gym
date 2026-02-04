from pathlib import Path
from unittest.mock import MagicMock

import responses_api_agents.swe_agents.app
from nemo_gym.config_types import ModelServerRef, OmegaConf
from nemo_gym.server_utils import ServerClient
from responses_api_agents.swe_agents.app import (
    BaseDatasetHarnessProcessor,
    BaseRunRequest,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
)


class TestSWEBenchWrapper:
    def _setup_wrapper(self, monkeypatch) -> SWEBenchWrapper:
        monkeypatch.setattr(
            responses_api_agents.swe_agents.app, "get_global_config_dict", MagicMock(return_value=OmegaConf.create({}))
        )
        monkeypatch.setattr(BaseDatasetHarnessProcessor, "_run_setup_command", MagicMock(return_value=None))

        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_swe_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_framework="swe_agent",
            agent_config="custom/config",
            agent_tools_file="tools.json",
            agent_max_turns=50,
            container_formatter=["docker://custom/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(
                type="responses_api_models",
                name="test_model",
            ),
        )

        wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))
        return wrapper

    def test_sanity(self, monkeypatch) -> None:
        self._setup_wrapper(monkeypatch)

    async def test_sanity_run(self, monkeypatch) -> None:
        wrapper = self._setup_wrapper(monkeypatch)

        with (Path(__file__).parent / "../data/example.jsonl").open() as f:
            lines = f.readlines()

        await wrapper.run(body=BaseRunRequest.model_validate_json(lines[0]))
