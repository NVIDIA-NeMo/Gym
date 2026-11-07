# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.healthbench.prepare import HF_REVISION as HB_REVISION
from benchmarks.healthbench.prepare import prepare as prepare_hb
from benchmarks.healthbench.prepare_common import (
    iter_local_records,
    load_jsonl,
    normalize_conversation,
    normalize_rubrics,
    records_from_zip,
    to_gym_row,
    write_gym_jsonl,
)
from benchmarks.healthbench_professional.prepare import HF_REVISION as HBP_REVISION
from benchmarks.healthbench_professional.prepare import prepare as prepare_hbp


def source():
    return {
        "prompt_id": "synthetic",
        "prompt": [
            {"role": "system", "content": "Synthetic system"},
            {"role": "user", "content": "Synthetic first turn"},
            {"role": "assistant", "content": "Synthetic prior answer"},
            {"role": "user", "content": "Synthetic final turn"},
        ],
        "rubrics": [{"criterion": "Synthetic criterion", "points": 1, "tags": ["synthetic"]}],
    }


def test_preserves_native_conversation_and_rubrics():
    data = source()
    row = to_gym_row(data, 1, benchmark_name="healthbench")
    assert row["responses_create_params"]["input"] == data["prompt"]
    assert row["conversation"] == data["prompt"]
    assert row["rubrics"] == data["rubrics"]
    assert row["uuid"] == to_gym_row(data, 2, benchmark_name="healthbench")["uuid"]
    assert row["uuid"] != to_gym_row(data, 1, benchmark_name="healthbench_professional")["uuid"]


@pytest.mark.parametrize("prepare", [prepare_hb, prepare_hbp])
def test_local_preparation_limit_and_atomic_failure(tmp_path, prepare):
    p = tmp_path / "source.jsonl"
    p.write_text(json.dumps(source()) + "\n" + json.dumps(source()) + "\n")
    out = tmp_path / "out.jsonl"
    prepare(p, out, 1)
    assert len(list(load_jsonl(out))) == 1
    before = out.read_text()
    p.write_text(json.dumps(source()) + "\nnot json\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        prepare(p, out)
    assert out.read_text() == before


def test_professional_field_names():
    data = {
        "conversation": source()["prompt"],
        "rubric_items": [{"criterion_text": "Synthetic criterion", "points": 2}],
    }
    row = to_gym_row(data, 1, benchmark_name="healthbench_professional")
    assert row["rubrics"] == [{"criterion": "Synthetic criterion", "points": 2}]


def test_download_revisions_are_pinned(tmp_path):
    p = tmp_path / "source.jsonl"
    p.write_text(json.dumps(source()) + "\n")
    with patch("huggingface_hub.hf_hub_download", return_value=str(p)) as download:
        prepare_hb("synthetic/dataset", tmp_path / "hb.jsonl")
    assert download.call_args.kwargs["revision"] == HB_REVISION
    with patch("datasets.load_dataset", return_value=[source()]) as download:
        prepare_hbp("synthetic/dataset", tmp_path / "hbp.jsonl")
    assert download.call_args.kwargs["revision"] == HBP_REVISION


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": []}],
        [None],
        [{"role": "assistant", "content": "x"}],
    ],
)
def test_invalid_conversations(data):
    with pytest.raises((ValueError, TypeError)):
        normalize_conversation({"conversation": data})


@pytest.mark.parametrize(
    "rubrics",
    [
        [],
        [None],
        [{"criterion": "x", "points": False}],
        [{"criterion": "x", "points": float("inf")}],
        [{"criterion": "x", "points": -1}],
        [{"points": 1}],
    ],
)
def test_invalid_rubrics(rubrics):
    with pytest.raises((ValueError, TypeError)):
        normalize_rubrics({"rubrics": rubrics})


def test_zip_reads_only_named_member(tmp_path):
    p = tmp_path / "assets.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("nested/eval.jsonl", json.dumps(source()) + "\n")
        z.writestr("../../do-not-extract.txt", "unrelated")
    assert list(iter_local_records(p, "eval.jsonl")) == [source()]
    assert not (tmp_path / "do-not-extract.txt").exists()
    with pytest.raises(ValueError, match="exactly one"):
        list(records_from_zip(p, "missing.jsonl"))
    assert iter_local_records(tmp_path / "missing", "eval.jsonl") is None
    with pytest.raises(FileNotFoundError):
        iter_local_records(tmp_path, "missing.jsonl")


def test_invalid_subset_and_limit(tmp_path):
    with pytest.raises(ValueError, match="subset"):
        prepare_hb(tmp_path, subset="unknown")
    with pytest.raises(ValueError, match="positive"):
        write_gym_jsonl([source()], output=tmp_path / "out", benchmark_name="healthbench", limit=0)


@pytest.mark.parametrize("subset", ["oss_eval", "consensus", "hard"])
@pytest.mark.parametrize("variant_first", [False, True])
def test_variant_config_prepares_and_selects_the_same_dataset(tmp_path, subset, variant_first):
    from omegaconf import OmegaConf

    from benchmarks.healthbench.prepare import SOURCE_FILENAMES
    from nemo_gym.cli.eval import _datasets_to_prepare, _multiprocess_benchmark_prepare_fn
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    paths = ["benchmarks/healthbench/config.yaml"]
    if subset != "oss_eval":
        paths.append(f"benchmarks/healthbench/configs/{subset}.yaml")
        if variant_first:
            paths.reverse()
    destinations = {
        name: str(tmp_path / f"{name}.jsonl") for name in ("oss_eval", "consensus", "hard", "oss_meta_eval")
    }
    cfg = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": paths, "healthbench_data_paths": destinations},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    servers = GlobalConfigDictParser().filter_for_server_instance_configs(cfg)
    selected = next(server for server in servers if server.name == "healthbench_benchmark_simple_agent")
    assert selected.datasets[0].jsonl_fpath == Path(destinations[subset])
    datasets = _datasets_to_prepare(cfg)
    assert len(datasets) == 1 and datasets[0].name == "healthbench"
    assert str(datasets[0].dataset.jsonl_fpath) == destinations[subset]
    args = OmegaConf.to_container(cfg.prepare_script_args, resolve=True)
    assert args == {"subset": subset, "output": destinations[subset]}

    local_source = tmp_path / "source.jsonl"
    local_source.write_text(json.dumps(source()) + "\n")
    for other, destination in destinations.items():
        if other != subset:
            Path(destination).write_text("do not overwrite\n")
    with patch("huggingface_hub.hf_hub_download", return_value=str(local_source)) as download:
        _multiprocess_benchmark_prepare_fn((datasets[0], "benchmarks.healthbench.prepare", args))
    assert download.call_args.kwargs["filename"] == SOURCE_FILENAMES[subset]
    row = list(load_jsonl(Path(destinations[subset])))[0]
    assert row["metadata"]["healthbench_subset"] == subset
    assert row["metadata"]["benchmark_name"] == "healthbench"
    for other, destination in destinations.items():
        if other != subset:
            assert Path(destination).read_text() == "do not overwrite\n"


@pytest.mark.parametrize("subset", ["consensus", "hard"])
def test_direct_subset_preparation_has_distinct_default_output(tmp_path, subset):
    import benchmarks.healthbench.prepare as module

    p = tmp_path / "source.jsonl"
    p.write_text(json.dumps(source()) + "\n")
    with patch.object(module, "DATA_DIR", tmp_path):
        output = prepare_hb(p, subset=subset)
    assert output == tmp_path / f"healthbench_{subset}_benchmark.jsonl"
    assert list(load_jsonl(output))[0]["metadata"]["healthbench_subset"] == subset


@pytest.mark.parametrize("name", ["healthbench", "healthbench_professional"])
def test_benchmark_default_repeat_count_is_one(name):
    import yaml

    config = yaml.safe_load((Path("benchmarks") / name / "config.yaml").read_text())
    datasets = config[f"{name}_benchmark_simple_agent"]["responses_api_agents"]["simple_agent"]["datasets"]
    assert datasets[0]["num_repeats"] == 1
    manifest = yaml.safe_load((Path("benchmarks") / name / "manifest.yaml").read_text())
    assert manifest["datasets"][0]["num_repeats"] == 1
