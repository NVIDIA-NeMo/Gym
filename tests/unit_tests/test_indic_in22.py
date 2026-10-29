# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.indic.in22 import prepare as prepare_module
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.data import fill_prompt, load_prompt_config


BENCHMARK_DIR = prepare_module.DIRECTORY


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

    assert [{k: v for k, v in row.items() if k not in {"uuid", "responses_create_params"}} for row in rows] == [
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
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row.pop("uuid") == "indic/in22/conv/xx_en/hindi/0"
    assert row.pop("responses_create_params")["max_output_tokens"] == 4096
    assert row == {
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
def test_config_keeps_defaults_local_to_in22(config_name: str, subset: str) -> None:
    config = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / "config_base.yaml"),
        OmegaConf.load(BENCHMARK_DIR / config_name),
    )
    resource = config.in22_wmt_translation_resources_server.resources_servers.wmt_translation
    assert not {"policy_model", "responses_create_params", "prepare_script_args", "num_repeats"} & set(config)
    assert config[f"in22_{subset}_agent"].responses_api_agents.simple_agent.datasets[0].num_repeats == 1
    assert resource.metric_profile == "in22"
    assert resource.compute_comet is False
    assert resource.language_consistency_backend is None


def _resolve_cli_config(config_names: list[str], flags: list[str] = ()) -> dict:
    from nemo_gym.cli.main import _is_positional_hydra_override, _merge_config_paths, build_parser

    args, extra_overrides = build_parser().parse_known_args(
        [
            "eval",
            "run",
            *[token for name in config_names for token in ("--config", str(BENCHMARK_DIR / name))],
            "--model-type",
            "vllm_model",
            "--model",
            "test-model",
            "--model-url",
            "http://unused/v1",
            "--model-api-key",
            "dummy",
            *flags,
        ]
    )
    # Match the CLI: argparse can consume a Hydra override as the optional TARGET.
    if args.target and _is_positional_hydra_override(args.target):
        extra_overrides = [args.target, *extra_overrides]
        args.target = None
    overrides = _merge_config_paths(
        [token for flag in args._command.flags for token in flag.translate_to_hydra(args)] + extra_overrides
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.from_dotlist([token.lstrip("+") for token in overrides]),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(resolved, resolve=True)


def test_manifest_is_discoverable_under_its_catalog_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.environment import validation
    from nemo_gym.registry import discover_environment_catalog, resolve_catalog_entry

    # Check real catalog identity and configuration without downloading gated benchmark data.
    monkeypatch.setattr(validation, "_validate_dataset", lambda *args, **kwargs: None)
    entry = resolve_catalog_entry("indic/in22", "benchmark", entries=discover_environment_catalog())
    assert entry.path == BENCHMARK_DIR
    assert validation.validate_environment(entry.manifest_path, entry.config_path).name == "indic/in22"


@pytest.mark.parametrize("config_names", [("config.yaml", "config_conv.yaml"), ("config_conv.yaml", "config.yaml")])
def test_gen_and_conv_compose_with_separate_preparation(config_names, monkeypatch, tmp_path) -> None:
    import importlib

    from nemo_gym.cli.eval import _datasets_to_prepare
    from nemo_gym.environment.manifest import load_manifest

    config = _resolve_cli_config(config_names)
    datasets = _datasets_to_prepare(OmegaConf.create(config))
    assert {dataset.name for dataset in datasets} == {"in22_gen", "in22_conv"}
    manifest = load_manifest(BENCHMARK_DIR / "manifest.yaml")
    assert manifest.canonical_split == "benchmark"
    assert manifest.datasets[0].num_repeats == 1
    for dataset in datasets:
        subset = dataset.name.removeprefix("in22_")
        _patch_expected_rows(monkeypatch, subset, 1)
        monkeypatch.setattr(prepare_module, "_load_records", lambda *, subset: [_record(0, subset)])
        module = importlib.import_module(str(dataset.dataset.prepare_script).removesuffix(".py").replace("/", "."))
        output = module.prepare(tmp_path / f"{subset}.jsonl", languages=["hindi"], question_ids=["0"])
        assert {json.loads(line)["uuid"].split("/")[2] for line in output.read_text().splitlines()} == {subset}
        agent = config[dataset.agent_name]["responses_api_agents"]["simple_agent"]
        assert agent["resources_server"]["name"] == "in22_wmt_translation_resources_server"
        env = config[f"in22_{subset}_environment_server"]["environment_servers"]["legacy_agent"]
        assert env["agent_server"]["name"] == dataset.agent_name


@pytest.mark.parametrize("cli_override", [False, True])
def test_cli_defaults_overrides_and_duplicate_conv_tasks(monkeypatch, tmp_path, cli_override) -> None:
    from nemo_gym.data import apply_prompt_to_row
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

    flags = ["--temperature", "0.4", "--top-p", "0.8", "--max-output-tokens", "1024"] if cli_override else []
    config = _resolve_cli_config(["config_conv.yaml"], flags)
    _patch_expected_rows(monkeypatch, "conv", 2)
    source = prepare_module.build_rows([_record(0, "conv")] * 2, subset="conv", languages=["hindi"])
    assert len({row["uuid"] for row in source}) == 4
    prompt = load_prompt_config(str(BENCHMARK_DIR / "prompts/default.yaml"))
    rows = [apply_prompt_to_row(row, prompt) | {"agent_ref": {"name": "in22_conv_agent"}} for row in source]
    # A non-IN22 request must not inherit IN22 sampling or thinking defaults.
    rows.append(
        {
            "agent_ref": {"name": "other_agent"},
            "responses_create_params": {
                "input": [{"role": "user", "content": "unrelated"}],
                "metadata": {"audio_path": "sample.wav"},
            },
        }
    )
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in rows))
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": str(prepared), "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    collected = RolloutCollectionHelper._preprocess_rows_from_config(None, collection)
    assert len(collected) == 5
    assert len({row["_ng_task_index"] for row in collected}) == 5
    for row in collected[:4]:
        request = row["responses_create_params"]
        assert (request["temperature"], request["top_p"], request["max_output_tokens"]) == (
            (0.4, 0.8, 1024) if cli_override else (1.0, 0.95, 4096)
        )
        assert json.loads(request["metadata"]["chat_template_kwargs"]) == {"enable_thinking": False}
        assert json.loads(request["metadata"]["extra_body"]) == {"top_k": -1, "seed": 42}
    other = collected[-1]["responses_create_params"]
    assert other["metadata"] == {"audio_path": "sample.wav"}
    if not cli_override:
        assert not {"temperature", "top_p", "max_output_tokens"} & set(other)


@pytest.mark.parametrize("add_seed", [False, True])
def test_repeat_seeds_are_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, add_seed: bool) -> None:
    from nemo_gym.data import apply_prompt_to_row
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper

    flags = ["--num-repeats", "3"] + (["+num_repeats_add_seed=true"] if add_seed else [])
    config = _resolve_cli_config(["config.yaml"], flags)
    _patch_expected_rows(monkeypatch, "gen", 1)
    source = prepare_module.build_rows([_record(0)], subset="gen", languages=["hindi"], directions=["en_xx"])
    prompt = load_prompt_config(str(BENCHMARK_DIR / "prompts/default.yaml"))
    row = apply_prompt_to_row(source[0], prompt) | {"agent_ref": {"name": "in22_gen_agent"}}
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text(json.dumps(row) + "\n")
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": str(prepared), "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    collected = RolloutCollectionHelper._preprocess_rows_from_config(None, collection)
    metadata = [item["responses_create_params"]["metadata"] for item in collected]
    assert [json.loads(item["extra_body"])["seed"] for item in metadata] == ([0, 1, 2] if add_seed else [42] * 3)
    assert all(json.loads(item["extra_body"])["top_k"] == -1 for item in metadata)
    assert all(json.loads(item["chat_template_kwargs"]) == {"enable_thinking": False} for item in metadata)


@pytest.mark.parametrize("subset", ["gen", "conv"])
def test_download_uses_immutable_revision(monkeypatch, subset) -> None:
    import datasets

    calls = []
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    assert prepare_module._load_records(subset=subset) == []
    spec = prepare_module.SUBSETS[subset]
    assert calls == [((spec["repo_id"], "default"), {"split": "test", "revision": spec["revision"]})]
