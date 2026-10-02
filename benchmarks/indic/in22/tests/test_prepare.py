# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.indic.in22 import prepare as prepare_module
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import fill_prompt, load_prompt_config


BENCHMARK_DIR = Path(__file__).parents[1]


def _record(index: int, subset: str = "gen") -> dict[str, object]:
    spec = prepare_module.SUBSETS[subset]
    row: dict[str, object] = {column: f"metadata-{index}" for column in spec["metadata_columns"]}
    row.update({column: f"{column} text {index}" for column in prepare_module.ALL_TRANSLATION_COLUMNS})
    row["eng_Latn"] = f"  English text {index}  "
    row["hin_Deva"] = f"  हिंदी पाठ {index}  "
    return row


def _patch_expected_rows(monkeypatch: pytest.MonkeyPatch, subset: str, count: int) -> None:
    monkeypatch.setitem(prepare_module.SUBSETS[subset], "rows", count)


def test_build_rows_matches_reference_order_directions_and_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_record(0), _record(1)]
    _patch_expected_rows(monkeypatch, "gen", len(records))

    rows = prepare_module.build_rows(records, subset="gen", languages=["hindi"], directions=["en_xx", "xx_en"])

    assert rows == [
        {
            "text": "English text 0",
            "translation": "हिंदी पाठ 0",
            "source_language": "en",
            "target_language": "hi",
            "source_lang_name": "English",
            "target_lang_name": "Hindi",
        },
        {
            "text": "English text 1",
            "translation": "हिंदी पाठ 1",
            "source_language": "en",
            "target_language": "hi",
            "source_lang_name": "English",
            "target_lang_name": "Hindi",
        },
        {
            "text": "हिंदी पाठ 0",
            "translation": "English text 0",
            "source_language": "hi",
            "target_language": "en",
            "source_lang_name": "Hindi",
            "target_lang_name": "English",
        },
        {
            "text": "हिंदी पाठ 1",
            "translation": "English text 1",
            "source_language": "hi",
            "target_language": "en",
            "source_lang_name": "Hindi",
            "target_lang_name": "English",
        },
    ]


def test_question_ids_select_source_indices_without_changing_pair_order(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_record(0), _record(1), _record(2)]
    _patch_expected_rows(monkeypatch, "gen", len(records))

    rows = prepare_module.build_rows(
        records,
        subset="gen",
        languages=["hindi"],
        directions=["en_xx"],
        question_ids=["2", "0"],
    )

    assert [row["text"] for row in rows] == ["English text 0", "English text 2"]


def test_prepare_writes_selected_subset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = [_record(0, "conv")]
    _patch_expected_rows(monkeypatch, "conv", len(records))
    monkeypatch.setattr(prepare_module, "_load_records", lambda *, subset: records)

    output = prepare_module.prepare(
        tmp_path / "in22.jsonl",
        subset="conv",
        languages=["hindi"],
        directions=["xx_en"],
    )

    assert output == tmp_path / "in22.jsonl"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "text": "हिंदी पाठ 0",
        "translation": "English text 0",
        "source_language": "hi",
        "target_language": "en",
        "source_lang_name": "Hindi",
        "target_lang_name": "English",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"languages": []}, "Select unique languages"),
        ({"languages": ["hindi", "hindi"]}, "Select unique languages"),
        ({"directions": ["sideways"]}, "Select unique directions"),
        ({"question_ids": ["99"]}, "unique existing string row indices"),
    ],
)
def test_invalid_selection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, list[str]],
    message: str,
) -> None:
    records = [_record(0)]
    _patch_expected_rows(monkeypatch, "gen", len(records))

    with pytest.raises(ValueError, match=message):
        prepare_module.build_rows(records, subset="gen", **kwargs)


def test_source_schema_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _record(0)
    row["unexpected"] = "value"
    _patch_expected_rows(monkeypatch, "gen", 1)

    with pytest.raises(ValueError, match="unexpected source schema"):
        prepare_module.build_rows([row], subset="gen", languages=["hindi"])


def test_prompt_matches_reference_exactly() -> None:
    prompt = load_prompt_config(str(BENCHMARK_DIR / "prompts/default.yaml"))
    messages = fill_prompt(
        prompt,
        {
            "text": "Example source",
            "source_lang_name": "English",
            "target_lang_name": "Hindi",
        },
    )

    assert messages == [
        {
            "role": "user",
            "content": (
                "Example source\n\n"
                "Translate the above English text into Hindi.\n"
                "Do not reason, explain, add labels, or use markdown. Return only the translation."
            ),
        }
    ]


@pytest.mark.parametrize(
    ("config_name", "subset"),
    [("config.yaml", "gen"), ("config_conv.yaml", "conv")],
)
def test_config_uses_requested_non_thinking_4k_generation(config_name: str, subset: str) -> None:
    config = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / "config_base.yaml"),
        OmegaConf.load(BENCHMARK_DIR / config_name),
    )
    model = config.policy_model.responses_api_models.vllm_model
    resource = config.in22_wmt_translation_resources_server.resources_servers.wmt_translation

    assert config.prepare_script_args.subset == subset
    assert config.responses_create_params.max_output_tokens == 4096
    assert model.chat_template_kwargs.enable_thinking is False
    assert model.sampling_overrides == {"temperature": 1.0, "top_p": 0.95, "top_k": -1, "seed": 42}
    assert resource.metric_profile == "in22"
    assert resource.compute_comet is False
    assert resource.language_consistency_backend is None


@pytest.mark.parametrize("config_name", ["config.yaml", "config_conv.yaml"])
def test_configs_resolve_through_gym(config_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(BENCHMARK_DIR.parents[2])
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(BENCHMARK_DIR / config_name)],
                    "policy_base_url": "http://127.0.0.1:8022/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test-model",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    model = config.policy_model.responses_api_models.vllm_model
    assert config.responses_create_params.max_output_tokens == 4096
    assert model.chat_template_kwargs.enable_thinking is False
    assert model.uses_reasoning_parser is False
    assert model.sampling_overrides == {"temperature": 1.0, "top_p": 0.95, "top_k": -1, "seed": 42}
