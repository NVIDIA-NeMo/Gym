# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from benchmarks.indic.ifeval import prepare as module
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.instruction_following.app import InstructionFollowingRunRequest


def source_row(key=1, *, tags=None):
    return {
        "key": key,
        "prompt": "नमस्ते लिखिए।",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["नमस्ते"], "num_words": None}],
        "tags": tags if tags is not None else ["correct", "parallel"],
        "resp_lang": "src",
    }


def test_exact_languages_and_aliases():
    assert module.select_languages(None) == ["bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur"]
    assert module.select_languages(["ka", "mar", "mal"]) == ["kn", "mr", "ml"]


@pytest.mark.parametrize("languages", [[], ["en"], ["as"], ["sa"], ["ka", "kn"], "hi"])
def test_reject_out_of_scope_or_duplicate_languages(languages):
    with pytest.raises(ValueError, match="language"):
        module.select_languages(languages)


def test_quality_selection_and_unmodified_source_fields():
    records = [source_row(1), source_row(2, tags=["incorrect"]), source_row(3, tags=["correct"])]
    original = copy.deepcopy(records)
    assert [r["id"] for r in module.build_rows(records, language="hi")] == [1, 3]
    assert [r["id"] for r in module.build_rows(records, language="hi", translation_quality="all")] == [1, 2, 3]
    rows = module.build_rows(records, language="hi", translation_quality="parallel")
    assert [r["id"] for r in rows] == [1]
    row = rows[0]
    parsed = InstructionFollowingRunRequest.model_validate(row)
    assert parsed.verifier_metadata["kwargs"] == records[0]["kwargs"]
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": records[0]["prompt"]}]
    assert row["metadata"]["source_config"] == "indicifeval-trans"
    assert records == original
    assert row["uuid"] != module.build_rows(records, language="bn")[0]["uuid"]


@pytest.mark.parametrize(
    "change", [{"tags": None}, {"tags": ["correct", "incorrect"]}, {"key": None}, {"kwargs": []}, {"prompt": ""}]
)
def test_invalid_source_fails_before_writing(change):
    record = source_row()
    record.update(change)
    with pytest.raises(ValueError):
        module.build_rows([record], language="hi")


def test_duplicate_keys_empty_selection_and_invalid_quality():
    with pytest.raises(ValueError, match="duplicate"):
        module.build_rows([source_row(), source_row()], language="hi")
    with pytest.raises(ValueError, match="no rows"):
        module.build_rows([source_row(tags=["incorrect"])], language="hi")
    with pytest.raises(ValueError, match="translation_quality"):
        module.build_rows([source_row()], language="hi", translation_quality="ground")


def test_prepare_downloads_trans_default_revision_and_round_trips(tmp_path, monkeypatch):
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([source_row(key) for key in range(3)]), parquet)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(parquet)

    monkeypatch.setattr(module, "hf_hub_download", download)
    path = module.prepare(tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 12 * 3
    assert len({row["uuid"] for row in rows}) == len(rows)
    assert all("source_revision" not in row["metadata"] for row in rows)
    assert {row["subset_for_metrics"] for row in rows} == set(module.LANGUAGES)
    assert calls == [
        {
            "repo_id": "ai4bharat/IndicIFEval",
            "repo_type": "dataset",
            "filename": f"indicifeval-trans/{language}-00000-of-00001.parquet",
        }
        for language in module.LANGUAGES
    ]


def test_config_reuses_english_server_and_simple_agent():
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(module.DIRECTORY / "config.yaml")]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert "instruction_following" not in config
    assert "instruction_following_simple_agent" not in config
    server = config["indic_ifeval_instruction_following"]["resources_servers"]["instruction_following"]
    assert server["instruction_backend"] == "indicifeval_trans"
    assert server["entrypoint"] == "app.py"
    agent = config["indic_ifeval_instruction_following_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["entrypoint"] == "app.py"
    assert agent["resources_server"]["name"] == "indic_ifeval_instruction_following"
    assert agent["model_server"]["name"] == "policy_model"
    assert [dataset["name"] for dataset in agent["datasets"]] == ["indic_ifeval_trans", "example"]
    assert agent["datasets"][0]["prepare_script"] == "benchmarks/indic/ifeval/prepare.py"
