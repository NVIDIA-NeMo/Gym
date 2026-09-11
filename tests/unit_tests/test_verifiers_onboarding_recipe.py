# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "responses_api_agents/verifiers_agent/examples/onboarding"
ENVIRONMENT_ID = "gym-legacy-verifiers-onboarding"
MODULE_NAME = ENVIRONMENT_ID.replace("-", "_")


def test_local_environment_is_shipped():
    assert (EXAMPLE / MODULE_NAME / "__init__.py").is_file(), "Missing local onboarding environment"


def test_documented_model_run_is_valid_for_automatic_serving():
    from nemo_gym.cli.main import build_parser
    from nemo_gym.rollout_collection import E2ERolloutCollectionConfig

    page = (ROOT / "fern/versions/latest/pages/get-started/verifiers-onboarding.mdx").read_text()
    block = re.search(r'```bash title="verifiers-model-run"\n(.*?)```', page, re.S).group(1)
    command = next(line for line in block.replace("\\\n", " ").splitlines() if line.startswith("gym eval run"))
    args, _ = build_parser().parse_known_args(shlex.split(command)[1:])
    values = {"split": args.split, "output_jsonl_fpath": args.output, "agent_name": args.agent}
    if args.input is not None:
        values["input_jsonl_fpath"] = args.input
    config = E2ERolloutCollectionConfig.model_validate(values)
    assert not args.no_serve
    assert config.split == "validation"
    assert args.agent == "verifiers_onboarding"


def test_recipe_collates_one_validation_task_without_servers(tmp_path):
    from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig

    config = OmegaConf.load(EXAMPLE / "config.yaml")
    datasets = config.verifiers_onboarding.responses_api_agents.verifiers_agent.get("datasets", [])
    assert len(datasets) == 1, "Automatic serving needs one declared validation dataset"
    assert datasets[0].type == "validation"
    assert datasets[0].num_repeats == 1
    assert ROOT / datasets[0].jsonl_fpath == EXAMPLE / MODULE_NAME / "input.jsonl"
    # Collation writes sidecars; never modify the bundled canonical input.
    input_path = tmp_path / "input.jsonl"
    input_path.write_bytes((EXAMPLE / MODULE_NAME / "input.jsonl").read_bytes())
    datasets[0].jsonl_fpath = str(input_path)
    config.output_dirpath = str(tmp_path / "collated")
    config.mode = "train_preparation"
    config.should_download = False
    config.task_data_validation = "error"
    processor = TrainDataProcessor()
    processor_config = TrainDataProcessorConfig.model_validate(config)
    instances = processor.load_and_validate_server_instance_configs(processor_config, config)
    processor.load_datasets(processor_config, instances)
    metrics = processor.validate_samples_and_aggregate_metrics(instances, overwrite_metrics_conflicts=False)
    processor.collate_samples(processor_config, instances, metrics)
    rows = [json.loads(line) for line in (tmp_path / "collated/validation.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    expected = json.loads(input_path.read_text())
    assert rows[0]["task_source"] == "verifiers_onboarding"
    for field in ("task_idx", "example_id", "vf_env_id", "answer", "responses_create_params"):
        assert rows[0][field] == expected[field]


@pytest.fixture
def recipe(monkeypatch):
    module_path = EXAMPLE / MODULE_NAME / "__init__.py"
    assert module_path.is_file(), "The local Verifiers onboarding environment has not been added"
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module(MODULE_NAME)


@pytest.mark.parametrize(
    "completion",
    [
        [{"role": "assistant", "content": "42"}],
        [{"role": "assistant", "content": " 42\n"}],
        [{"role": "assistant", "content": "<think>6 times 7 is 42.</think>\n\n42"}],
        [{"role": "assistant", "content": " \n<think>Check the arithmetic.\nIt is 42.</think> 42\n"}],
        [SimpleNamespace(role="assistant", content="42")],
    ],
)
def test_exact_answer_accepts_correct_final_assistant_text(recipe, completion):
    assert recipe.exact_answer(completion=completion, answer="42") == 1.0


@pytest.mark.parametrize(
    "completion",
    [
        [],
        None,
        {"role": "assistant", "content": "42"},
        42,
        "42",
        [{"role": "assistant", "content": "17"}],
        [{"role": "assistant", "content": "The answer is 42"}],
        [{"role": "assistant", "content": "<think>42</think>17"}],
        [{"role": "assistant", "content": "<think>42</think>"}],
        [{"role": "assistant", "content": "<think>42"}],
        [{"role": "assistant", "content": "reasoning</think>42"}],
        [{"role": "assistant", "content": "<think>nested <think>reasoning</think>42"}],
        [{"role": "assistant", "content": "<think>first</think><think>second</think>42"}],
        [{"role": "assistant", "content": "<think>42</think>The answer is 42"}],
        [{"role": "user", "content": "42"}],
        [{"role": "assistant", "content": None}],
        [{"role": "assistant", "content": []}],
        [{"role": "assistant", "content": 42}],
        [{}],
    ],
)
def test_exact_answer_rejects_wrong_or_malformed_completion(recipe, completion):
    assert recipe.exact_answer(completion=completion, answer="42") == 0.0


def test_recipe_config_input_and_loader_share_the_same_task(recipe):
    config = yaml.safe_load((EXAMPLE / "config.yaml").read_text(encoding="utf-8"))
    agent = config["verifiers_onboarding"]["responses_api_agents"]["verifiers_agent"]
    rows = [json.loads(line) for line in (EXAMPLE / MODULE_NAME / "input.jsonl").read_text().splitlines()]
    assert agent["vf_env_id"] == ENVIRONMENT_ID
    assert agent["model_server"]["name"] == "policy_model"
    assert agent["model_name"] == "${policy_model_name}"
    assert agent["max_tokens"] == 256
    assert len(rows) == 1
    assert rows[0]["task_idx"] == rows[0]["example_id"] == 0
    assert rows[0]["vf_env_id"] == ENVIRONMENT_ID
    assert rows[0]["answer"] == "42"
    assert rows[0]["responses_create_params"]["input"] == [
        {"role": "user", "content": "What is 6 * 7? Reply with only the integer."}
    ]


async def test_real_legacy_environment_scores_the_committed_task(recipe):
    vf = pytest.importorskip("verifiers", reason="Run in the documented Verifiers server environment")
    environment = recipe.load_environment()
    dataset = environment.get_dataset()
    assert len(dataset) == 1
    assert dataset[0]["answer"] == "42"
    assert dataset[0]["prompt"] == [{"role": "user", "content": "What is 6 * 7? Reply with only the integer."}]
    assert isinstance(environment, vf.SingleTurnEnv)
    for completion, expected in [
        ([vf.AssistantMessage(content="42")], 1.0),
        ([vf.AssistantMessage(content="<think>6 times 7 is 42.</think>\n\n42")], 1.0),
        ([vf.AssistantMessage(content="17")], 0.0),
        ([], 0.0),
    ]:
        state = vf.State(
            {
                "input": vf.RolloutInput(prompt=[], answer="42", example_id=0),
                "completion": completion,
                "trajectory": [],
            }
        )
        await environment.rubric.score_rollout(state)
        assert state["reward"] == expected


def test_recipe_config_resolves_with_an_explicit_model(recipe):
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(EXAMPLE / "config.yaml"),
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                    ],
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "fixture-model",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agent = config.verifiers_onboarding.responses_api_agents.verifiers_agent
    assert agent.vf_env_id == ENVIRONMENT_ID
    assert agent.model_name == "fixture-model"
    assert config.policy_model.responses_api_models.vllm_model.base_url == "http://127.0.0.1:9999/v1"
