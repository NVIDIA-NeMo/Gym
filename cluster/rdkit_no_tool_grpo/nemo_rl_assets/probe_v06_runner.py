#!/usr/bin/env python3
"""Import-level probe for the RDKit no-tool NeMo-RL v0.6 runner bundle."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import nemo_rl.algorithms.grpo as grpo
import nemo_rl.environments.nemo_gym as nemo_gym
from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from omegaconf import OmegaConf


def main() -> None:
    root = Path.cwd()
    runner_path = root / "cluster/rdkit_no_tool_grpo/nemo_rl_assets/run_grpo_nemo_gym.py"
    config_path = root / "cluster/rdkit_no_tool_grpo/rdkit_no_tool_grpo.yaml"

    spec = importlib.util.spec_from_file_location(
        "rdkit_no_tool_runner_probe", runner_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(str(config_path)), resolve=True)
    assert config["data"]["default"]["dataset_name"] == "NemoGymDataset"
    assert config["data"]["train"]["data_path"].endswith("train.jsonl")
    assert config["data"]["validation"]["data_path"].endswith("test_eval4.jsonl")
    peft = config["policy"]["megatron_cfg"]["peft"]
    assert peft["enabled"] is True
    assert peft["dim"] == 8 and peft["alpha"] == 8
    assert set(peft["target_modules"]) == {
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
    }

    dataset = NemoGymDataset(data_path=config["data"]["train"]["data_path"])
    first_row = json.loads(dataset.dataset[0]["extra_env_info"])
    assert first_row["agent_ref"]["name"] == "rdkit_chemistry_direct_agent"
    assert first_row["responses_create_params"]["tools"] == []

    print("nemo-rl-version", importlib.metadata.version("nemo-rl"))
    print("grpo-file", grpo.__file__)
    print("nemo-gym-file", nemo_gym.__file__)
    print("has-_should_use_nemo_gym", callable(_should_use_nemo_gym))
    print("runner-import-ok", runner_path)
    print("dataset-probe-ok", len(dataset.dataset), dataset.dataset[0]["task_name"])


if __name__ == "__main__":
    main()
