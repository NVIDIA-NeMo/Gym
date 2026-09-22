# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import subprocess
import sys

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from benchmarks.indic.mbpp import prepare as module
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config


@pytest.fixture
def records(monkeypatch):
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    return [
        {
            "task_id": task_id,
            "text": " Add one to x.\n    Preserve the description. ",
            "code": "def add_one(x):\n    return x + 1  # HIDDEN_REFERENCE\n",
            "test_list": ["assert add_one(1) == 2", "assert add_one(2) == 3", "assert add_one(3) == 4"],
            "test_setup_code": "",
            "challenge_test_list": ["assert add_one(4) == 5"],
            **{
                f"text_{name}_translation": f" {code}: x में एक जोड़ें.\n    {{x}} "
                for code, name in module.LANGUAGE_NAMES.items()
            },
        }
        for task_id in (11, 12)
    ]


def test_shared_prompt_and_identical_english_indic_tests(records):
    rows = module.build_rows(records, languages=["en", "hi"])
    prompt = load_prompt_config("benchmarks/mbpp/prompts/default.yaml")
    english, hindi = rows[0], rows[2]
    assert english["verifier_metadata"] == hindi["verifier_metadata"]
    en_input = apply_prompt_to_row(english, prompt)["responses_create_params"]["input"]
    hi_input = apply_prompt_to_row(hindi, prompt)["responses_create_params"]["input"]
    assert hi_input == [
        {
            "role": "user",
            "content": en_input[0]["content"].replace(
                records[0]["text"].replace("    ", "\t"),
                records[0]["text_Hindi_translation"].replace("    ", "\t"),
            ),
        }
    ]
    assert "assert add_one(1) == 2" in hi_input[0]["content"]
    assert "assert add_one(2) == 3" not in hi_input[0]["content"]
    assert "assert add_one(3) == 4" not in hi_input[0]["content"]
    assert "HIDDEN_REFERENCE" not in json.dumps(hindi)
    assert "assert add_one(4) == 5" not in json.dumps(hindi)
    check = ast.parse(hindi["verifier_metadata"]["test"]).body[0]
    assert [ast.dump(node) for node in check.body[1:]] == [
        ast.dump(node) for node in ast.parse("\n".join(records[0]["test_list"])).body
    ]


def test_full_coverage_stable_ids_and_selection(records):
    rows = module.build_rows(records)
    assert len(rows) == 28
    assert len({row["uuid"] for row in rows}) == 28
    assert {row["language"] for row in rows} == set(module.DEFAULT_LANGUAGES)
    selected = module.build_rows(records, languages=["hi"], task_ids=[12])
    assert selected == [
        row for row in rows if row["language"] == "hi" and row["verifier_metadata"]["task_id"] == "Mbpp/12"
    ]


@pytest.mark.parametrize("value", [None, "", " "])
def test_missing_translation_rejected_without_fallback(records, value):
    records[0]["text_Hindi_translation"] = value
    with pytest.raises(ValueError, match="Missing problem text: hi/11"):
        module.build_rows(records, languages=["hi"], task_ids=[12])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"languages": "hi"}, "nonempty sequence"),
        ({"languages": []}, "nonempty sequence"),
        ({"languages": ["xx"]}, "nonempty sequence"),
        ({"languages": ["hi", "hi"]}, "unique"),
        ({"task_ids": []}, "unique integer"),
        ({"task_ids": [True]}, "unique integer"),
        ({"task_ids": [11, 11]}, "unique integer"),
        ({"task_ids": ["11"]}, "unique integer"),
        ({"task_ids": [999]}, "Unknown task_ids"),
    ],
)
def test_bad_selections(records, kwargs, error):
    with pytest.raises(ValueError, match=error):
        module.build_rows(records, **kwargs)


def test_missing_and_duplicate_tasks(records):
    with pytest.raises(ValueError, match="Expected 2"):
        module.build_rows(records[:1])
    with pytest.raises(ValueError, match="Duplicate task_id"):
        module.build_rows([records[0], records[0]])


@pytest.mark.parametrize(
    "change,error",
    [
        ({"task_id": True}, "positive integer"),
        ({"test_list": []}, "three nonempty"),
        ({"test_list": ["", "assert True", "assert True"]}, "three nonempty"),
        ({"code": None}, "reference code"),
        ({"test_setup_code": None}, "setup"),
        ({"code": "def unrelated(): return 0"}, "one tested function"),
    ],
)
def test_malformed_test_contract(records, change, error):
    records[0].update(change)
    with pytest.raises(ValueError, match=error):
        module.build_rows(records)


def run_native(meta, code, tmp_path):
    runner = module.BENCHMARK_DIR.parents[2] / "resources_servers/scicodepile/scp_runner.py"
    result = subprocess.run(
        [sys.executable, str(runner)],
        input=json.dumps({**meta, "code": code, "max_as_limit": 0, "workdir": str(tmp_path)}),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return json.loads(result.stdout)


def test_native_runner_accepts_correct_and_rejects_hidden_test_failure(records, tmp_path):
    meta = module.build_rows(records, languages=["hi"])[0]["verifier_metadata"]
    assert run_native(meta, records[0]["code"], tmp_path)["status"] == "pass"
    # This satisfies the visible example but fails the two remaining original assertions.
    assert run_native(meta, "def add_one(x): return 2", tmp_path)["status"] == "fail"


def test_candidate_named_check_and_recursive_calls_keep_original_binding(records, tmp_path):
    records[0].update(
        {
            "code": "def check(n): return n if n <= 1 else check(n - 1) + 1",
            "test_list": ["assert check(0) == 0", "assert check(2) == 2", "assert check(3) == 3"],
        }
    )
    meta = module.build_rows(records, languages=["hi"])[0]["verifier_metadata"]
    assert run_native(meta, records[0]["code"], tmp_path)["status"] == "pass"
    assert run_native(meta, "def check(n): return 0", tmp_path)["status"] == "fail"


def test_setup_runs_after_candidate_classes(records, tmp_path):
    records[0].update(
        {
            "code": "class Node:\n    def __init__(self, x): self.x = x\ndef value(node): return node.x\n",
            "test_setup_code": "root = Node(3)",
            "test_list": ["assert value(root) == 3", "assert value(Node(4)) == 4", "assert value(Node(5)) == 5"],
        }
    )
    meta = module.build_rows(records, languages=["en"])[0]["verifier_metadata"]
    assert run_native(meta, records[0]["code"], tmp_path)["status"] == "pass"


def test_prepare_pins_source_and_preserves_output_on_failure(records, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "maybe_get_global_config_dict", lambda: None)
    monkeypatch.setattr(module, "get_token", lambda: "test-token")
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "pinned.parquet")
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: Dataset.from_list(records))
    output = tmp_path / "tasks.jsonl"
    assert module.prepare(languages=["hi"], output_fpath=str(output)) == output
    assert calls[0] == {
        "repo_id": module.SOURCE_ID,
        "filename": "test.parquet",
        "repo_type": "dataset",
        "revision": module.SOURCE_REVISION,
        "token": "test-token",
    }
    original = output.read_bytes()
    assert [json.loads(line) for line in output.read_text().splitlines()] == module.build_rows(
        records, languages=["hi"]
    )
    records[0]["test_list"] = []
    with pytest.raises(ValueError):
        module.prepare(languages=["hi"], output_fpath=str(output))
    assert output.read_bytes() == original


def test_native_config_reuses_prompt_agent_and_verifier():
    parser = GlobalConfigDictParser()
    configs = []
    for path in ("benchmarks/mbpp/config.yaml", "benchmarks/indic/mbpp/config.yaml"):
        configs.append(
            parser.parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=OmegaConf.create(
                        {
                            "config_paths": ["responses_api_models/vllm_model/configs/vllm_model.yaml", path],
                            "policy_base_url": "http://unused/v1",
                            "policy_api_key": "dummy",
                            "policy_model_name": "test",
                        }
                    ),
                    skip_load_from_cli=True,
                    skip_load_from_dotenv=True,
                    offline=True,
                )
            )
        )
    en, indic = configs
    original = OmegaConf.to_container(en.mbpp_evalplus_simple_agent.responses_api_agents.simple_agent, resolve=True)
    translated = OmegaConf.to_container(indic.indic_mbpp_simple_agent.responses_api_agents.simple_agent, resolve=True)
    assert original["datasets"][0]["prompt_config"] == translated["datasets"][0]["prompt_config"]
    original["datasets"] = translated["datasets"]
    original["resources_server"]["name"] = "indic_mbpp_resources_server"
    assert original == translated
    assert indic.indic_mbpp_resources_server.resources_servers.scicodepile.subprocess_timeout == 120
    assert {server.name for server in parser.filter_for_server_instance_configs(indic)} == {
        "policy_model",
        "indic_mbpp_resources_server",
        "indic_mbpp_simple_agent",
    }
    benchmark = BenchmarkConfig.from_config_path(module.BENCHMARK_DIR / "config.yaml")
    assert benchmark.agent_name == "indic_mbpp_simple_agent"
    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.resources_server == "scicodepile"
    assert manifest.agent_server == "simple_agent"
    assert manifest.datasets[0].model_dump(exclude_none=True) == {
        key: value for key, value in translated["datasets"][0].items() if key != "license"
    }
