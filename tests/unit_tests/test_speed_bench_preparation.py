# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation and config contracts for every SPEED-Bench split."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from nemo_gym.cli.eval import PrepareBenchmarkConfig, _datasets_to_prepare, _prepare_datasets
from nemo_gym.environment.manifest import dump_manifest, load_manifest
from nemo_gym.environment.validation import _manifest_dataset
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


PREPARE = importlib.import_module("benchmarks.speed-bench.prepare")
COMMON = importlib.import_module("_prepare_common")
ROOT = Path(__file__).resolve().parents[2]
SPLITS = COMMON.ALL_CONFIGS


def source_rows(split):
    return [
        {"turns": [f"{split}: first"], "src_id": "one", "source": "fixture", "sub_category": "test"},
        {
            "turns": [f"{split}: first", f"{split}: second"],
            "src_id": "two",
            "source": "fixture",
            "sub_category": "test",
        },
    ]


def resolve_splits(splits, model_type="dummy_model"):
    paths = [f"benchmarks/speed-bench/config_{split}.yaml" for split in splits]
    if model_type == "local_vllm_model":
        name = "Qwen3-30B-A3B-Instruct-2507-ngram-specdec"
        paths += [
            f"responses_api_models/local_vllm_model/configs/Qwen/{name}.yaml",
            "benchmarks/speed-bench/configs/local_vllm_metrics.yaml",
        ]
        model = {"_inherit_from": name}
    else:
        options = {"entrypoint": "app.py"}
        if model_type == "vllm_model":
            options.update(base_url="http://example.test:8000/v1", api_key="unused", model="test")
        model = {"responses_api_models": {model_type: options}}
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create({"config_paths": paths, "policy_model": model}),
        )
    )


@pytest.fixture(autouse=True)
def working_directory(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize(
    "repo,revision", [(COMMON.DEFAULT_DATASET_REPO, None), ("nvidia/SPEED-Bench-Internal", "abc123")]
)
def test_prepare_sources_and_repeat_overwrite(monkeypatch, tmp_path, split, repo, revision):
    rows = source_rows(split)
    loader = Mock(return_value=Dataset.from_list(rows))
    monkeypatch.setattr("datasets.load_dataset", loader)
    monkeypatch.setattr(COMMON, "DATA_DIR", tmp_path)
    resolver = Mock(side_effect=AssertionError("resolved inputs must not resolve external sources"))
    monkeypatch.setitem(sys.modules, "_resolve_external_data", SimpleNamespace(_resolve_external_data=resolver))
    kwargs = {} if repo == COMMON.DEFAULT_DATASET_REPO else {"dataset_repo": repo, "dataset_revision": revision}
    path = PREPARE.prepare(split, **kwargs)
    assert path == tmp_path / f"speed_bench_{split}_benchmark.jsonl"
    prepared = [json.loads(line) for line in path.read_text().splitlines()]
    for original, row in zip(rows, prepared, strict=True):
        messages = row["responses_create_params"]["input"]
        assert [message["content"] for message in messages] == original["turns"]
        assert all(message["role"] == "user" for message in messages)
        assert row["verifier_metadata"] == {
            "src_id": original["src_id"],
            "source": "fixture",
            "speed_config": split,
            "num_turns": len(original["turns"]),
            "sub_category": "test",
        }
    original = path.read_bytes()
    PREPARE.prepare(split, **kwargs)
    assert path.read_bytes() == original
    loader.assert_called_with(repo, split, split="test", revision=revision)
    resolver.assert_not_called()


def test_placeholders_use_local_resolver_and_must_be_resolved(monkeypatch, tmp_path):
    unresolved = Dataset.from_list([{"turns": [COMMON.TURNS_PLACEHOLDER], "src_id": "one"}])
    monkeypatch.setattr("datasets.load_dataset", Mock(return_value=unresolved))
    monkeypatch.setattr(COMMON, "DATA_DIR", tmp_path)
    resolve = Mock(return_value=Dataset.from_list(source_rows("qualitative")))
    monkeypatch.setitem(sys.modules, "_resolve_external_data", SimpleNamespace(_resolve_external_data=resolve))
    PREPARE.prepare()
    resolve.assert_called_once_with(unresolved, "qualitative")
    resolve.return_value = unresolved
    with pytest.raises(ValueError, match="Unresolved SPEED-Bench placeholder"):
        PREPARE.prepare()


@pytest.mark.parametrize("turns", [[], [""], "not a list"])
def test_invalid_turns_are_rejected(turns):
    with pytest.raises(ValueError, match="nonempty text turns"):
        COMMON._row_to_gym({"turns": turns}, "qualitative")


def test_unknown_split_fails_before_download(monkeypatch):
    loader = Mock()
    monkeypatch.setattr("datasets.load_dataset", loader)
    with pytest.raises(ValueError, match="Unknown speed-bench config"):
        PREPARE.prepare("throughput_100k")
    loader.assert_not_called()


@pytest.mark.parametrize("splits", [(split,) for split in SPLITS] + [SPLITS])
@pytest.mark.parametrize("model_type", ["local_vllm_model", "vllm_model"])
def test_every_split_resolves_and_manifest_matches(tmp_path, splits, model_type):
    config = resolve_splits(splits, model_type)
    OmegaConf.to_container(config, resolve=True)
    datasets = _datasets_to_prepare(config)
    assert len(datasets) == len(splits)
    for benchmark, split in zip(datasets, splits, strict=True):
        dataset = benchmark.dataset
        assert dataset.prepare_dependencies == ["numpy", "pandas", "tiktoken"]
        assert dataset.prepare_script == Path(f"benchmarks/speed-bench/prepare_{split}.py")
        assert str(dataset.jsonl_fpath).endswith(f"speed_bench_{split}_benchmark.jsonl")
        agent = config[f"speed_bench_{split}_simple_agent"].responses_api_agents.speed_bench_agent
        assert agent.model_server.name == "policy_model"
        assert agent.resources_server.name == f"speed_bench_{split}_resources_server"
        env = config[f"speed_bench_{split}_simple_environment_server"].environment_servers.legacy_agent
        assert env.agent_server.name == f"speed_bench_{split}_simple_agent"
        manifest = load_manifest(ROOT / f"benchmarks/speed-bench/config_{split}/manifest.yaml")
        assert _manifest_dataset(dataset) == manifest.datasets[0]
        assert "prepare_script_args:" not in dump_manifest(manifest)
        roundtrip = tmp_path / "manifest.yaml"
        roundtrip.write_text(dump_manifest(manifest))
        assert load_manifest(roundtrip) == manifest
    adapters = config.policy_model.responses_api_models
    assert list(adapters) == [model_type]
    if model_type == "local_vllm_model":
        assert adapters.local_vllm_model.vllm_serve_kwargs.per_request_spec_decode_metrics == "summary"


@pytest.mark.parametrize("splits", [(split,) for split in SPLITS] + [SPLITS, ("throughput_32k", "throughput_1k")])
@pytest.mark.parametrize("arguments", [{}, {"dataset_repo": "other/repo", "dataset_revision": "pinned"}])
def test_selected_wrappers_receive_global_source_arguments(monkeypatch, tmp_path, splits, arguments):
    monkeypatch.setattr("nemo_gym.cli.eval._install_prepare_dependencies", lambda _: None)
    monkeypatch.setattr(COMMON, "DATA_DIR", tmp_path)
    downloads = []

    def load(repo, name, **kwargs):
        downloads.append((repo, name, kwargs))
        return Dataset.from_list(source_rows(name))

    monkeypatch.setattr("datasets.load_dataset", load)
    datasets = _datasets_to_prepare(resolve_splits(splits))
    for benchmark, split in zip(datasets, splits, strict=True):
        benchmark.dataset.jsonl_fpath = tmp_path / f"speed_bench_{split}_benchmark.jsonl"
    _prepare_datasets(datasets, PrepareBenchmarkConfig(prepare_script_args=arguments))
    assert downloads == [
        (
            arguments.get("dataset_repo", COMMON.DEFAULT_DATASET_REPO),
            split,
            {"split": "test", "revision": arguments.get("dataset_revision")},
        )
        for split in splits
    ]
    assert {path.name for path in tmp_path.glob("*.jsonl")} == {
        f"speed_bench_{split}_benchmark.jsonl" for split in splits
    }
    for benchmark, split in zip(datasets, splits, strict=True):
        rows = [json.loads(line) for line in benchmark.dataset.jsonl_fpath.read_text().splitlines()]
        assert len(rows) == 2
        assert all(row["verifier_metadata"]["speed_config"] == split for row in rows)

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: pytest.fail("prepared inputs must be reused"))
    _prepare_datasets(datasets, PrepareBenchmarkConfig(use_cached_prepared_benchmarks=True))
