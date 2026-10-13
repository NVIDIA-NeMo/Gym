# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import json

import pytest

from benchmarks.wmt24pp import prepare as prepare_module


def _fail_import(monkeypatch, module_name: str, exc: BaseException) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_language_names_are_loaded_from_json() -> None:
    assert prepare_module._WMT24PP_LANG_MAP["es_MX"] == "Spanish (Mexico)"
    assert prepare_module._WMT24PP_LANG_MAP["pt_BR"] == "Portuguese (Brazil)"
    assert prepare_module.DEFAULT_TARGET_LANGUAGES == list(prepare_module._WMT24PP_LANG_MAP)


def test_parse_args_defaults() -> None:
    args = prepare_module._parse_args([])

    assert args.target_languages is None
    assert args.prefetch_comet is True
    assert args.prefetch_spbleu is True


def test_parse_args_accepts_prepare_options() -> None:
    args = prepare_module._parse_args(
        [
            "--target_languages",
            "de_DE",
            "es_MX",
            "--no-prefetch-comet",
            "--no-prefetch-spbleu",
        ]
    )

    assert args.target_languages == ["de_DE", "es_MX"]
    assert args.prefetch_comet is False
    assert args.prefetch_spbleu is False


@pytest.mark.parametrize(
    ("argv", "expected_exit_code"),
    [
        (["--help"], 0),
        (["--definitely-invalid"], 2),
    ],
)
def test_parse_args_exits_before_download(monkeypatch, argv, expected_exit_code) -> None:
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: pytest.fail("load_dataset should not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        prepare_module._parse_args(argv)

    assert exc_info.value.code == expected_exit_code


def test_comet_prefetch_skips_when_comet_is_absent(monkeypatch, capsys) -> None:
    _fail_import(
        monkeypatch,
        "comet",
        ModuleNotFoundError("No module named 'comet'", name="comet"),
    )

    prepare_module._prefetch_comet_model()

    assert "unbabel-comet not installed" in capsys.readouterr().out


def test_comet_prefetch_propagates_missing_transitive_dependency(monkeypatch) -> None:
    _fail_import(
        monkeypatch,
        "comet",
        ModuleNotFoundError("No module named 'torch'", name="torch"),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        prepare_module._prefetch_comet_model()

    assert exc_info.value.name == "torch"


def test_comet_prefetch_propagates_incompatible_api(monkeypatch) -> None:
    _fail_import(
        monkeypatch,
        "comet",
        ImportError("cannot import name 'load_from_checkpoint' from 'comet'"),
    )

    with pytest.raises(ImportError, match="load_from_checkpoint"):
        prepare_module._prefetch_comet_model()


def test_spbleu_prefetch_skips_when_sacrebleu_is_absent(monkeypatch, capsys) -> None:
    _fail_import(
        monkeypatch,
        "sacrebleu.metrics",
        ModuleNotFoundError("No module named 'sacrebleu'", name="sacrebleu"),
    )

    prepare_module._prefetch_spbleu_tokenizer()

    assert "sacrebleu not installed" in capsys.readouterr().out


def test_spbleu_prefetch_propagates_missing_transitive_dependency(monkeypatch) -> None:
    _fail_import(
        monkeypatch,
        "sacrebleu.metrics",
        ModuleNotFoundError("No module named 'regex'", name="regex"),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        prepare_module._prefetch_spbleu_tokenizer()

    assert exc_info.value.name == "regex"


def test_spbleu_prefetch_skips_when_sentencepiece_is_absent(monkeypatch, capsys) -> None:
    monkeypatch.setattr(prepare_module.importlib.util, "find_spec", lambda name: None)

    prepare_module._prefetch_spbleu_tokenizer()

    assert "sentencepiece not installed" in capsys.readouterr().out


def test_spbleu_prefetch_propagates_broken_sentencepiece(monkeypatch) -> None:
    from sacrebleu import metrics

    class BrokenBLEU:
        def __init__(self, **_kwargs):
            raise ImportError("sentencepiece native library failed to load")

    monkeypatch.setattr(prepare_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(metrics, "BLEU", BrokenBLEU)

    with pytest.raises(ImportError, match="native library failed"):
        prepare_module._prefetch_spbleu_tokenizer()


def test_prepare_uses_json_language_name(monkeypatch, tmp_path) -> None:
    def fake_load_dataset(repo_id, config):
        assert repo_id == prepare_module.HF_REPO_ID
        assert config == "en-es_MX"
        return {
            "train": [
                {
                    "source": "Pass the salt.",
                    "target": "Pasa la sal.",
                    "is_bad_source": False,
                }
            ]
        }

    output_path = tmp_path / "wmt24pp_benchmark.jsonl"
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare_module, "prefetch_translation_models", lambda **_kwargs: None)

    result = prepare_module.prepare(
        target_languages=["es_MX"],
        prefetch_comet=False,
        prefetch_spbleu=False,
    )

    assert result == output_path
    assert [json.loads(line) for line in output_path.read_text().splitlines()] == [
        {
            "text": "Pass the salt.",
            "translation": "Pasa la sal.",
            "source_language": "en",
            "target_language": "es_MX",
            "source_lang_name": "English",
            "target_lang_name": "Spanish (Mexico)",
        }
    ]


def test_prepare_deduplicates_target_languages_without_changing_order(monkeypatch, tmp_path) -> None:
    load_calls = []

    def fake_load_dataset(repo_id, config):
        load_calls.append((repo_id, config))
        return {
            "train": [
                {
                    "source": "Pass the salt.",
                    "target": "Pasa la sal.",
                    "is_bad_source": False,
                }
            ]
        }

    output_path = tmp_path / "wmt24pp_benchmark.jsonl"
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(prepare_module, "prefetch_translation_models", lambda **_kwargs: None)

    prepare_module.prepare(
        target_languages=["fr_FR", "es_MX", "fr_FR", "de_DE"],
        prefetch_comet=False,
        prefetch_spbleu=False,
    )

    assert load_calls == [
        (prepare_module.HF_REPO_ID, "en-fr_FR"),
        (prepare_module.HF_REPO_ID, "en-es_MX"),
        (prepare_module.HF_REPO_ID, "en-de_DE"),
    ]
    assert [json.loads(line)["target_language"] for line in output_path.read_text().splitlines()] == [
        "fr_FR",
        "es_MX",
        "de_DE",
    ]


def test_prepare_does_not_publish_output_when_prefetch_fails(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "wmt24pp_benchmark.jsonl"
    output_path.write_text("existing output\n")

    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: {
            "train": [
                {
                    "source": "Pass the salt.",
                    "target": "Pasa la sal.",
                    "is_bad_source": False,
                }
            ]
        },
    )
    monkeypatch.setattr(
        prepare_module,
        "prefetch_translation_models",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("prefetch failed")),
    )

    with pytest.raises(RuntimeError, match="prefetch failed"):
        prepare_module.prepare(target_languages=["es_MX"])

    assert output_path.read_text() == "existing output\n"
    assert list(tmp_path.glob(f".{output_path.name}.*")) == []


def test_prepare_omits_bad_sources_and_preserves_good_row_order(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "wmt24pp_benchmark.jsonl"
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output_path)
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: {
            "train": [
                {
                    "source": "First good source.",
                    "target": "Primera fuente buena.",
                    "is_bad_source": False,
                },
                {
                    "source": "CANARY GUID",
                    "target": "CANARY GUID",
                    "is_bad_source": True,
                },
                {
                    "source": "Second good source.",
                    "target": "Segunda fuente buena.",
                    "is_bad_source": False,
                },
            ]
        },
    )
    monkeypatch.setattr(prepare_module, "prefetch_translation_models", lambda **_kwargs: None)

    prepare_module.prepare(
        target_languages=["es_MX"],
        prefetch_comet=False,
        prefetch_spbleu=False,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [(row["text"], row["translation"]) for row in rows] == [
        ("First good source.", "Primera fuente buena."),
        ("Second good source.", "Segunda fuente buena."),
    ]


def test_prepare_requires_is_bad_source_field(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "wmt24pp_benchmark.jsonl")
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: {
            "train": [
                {
                    "source": "Pass the salt.",
                    "target": "Pasa la sal.",
                }
            ]
        },
    )
    monkeypatch.setattr(prepare_module, "prefetch_translation_models", lambda **_kwargs: None)

    with pytest.raises(KeyError, match="is_bad_source"):
        prepare_module.prepare(
            target_languages=["es_MX"],
            prefetch_comet=False,
            prefetch_spbleu=False,
        )


def test_prepare_rejects_unknown_language_before_download(monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: pytest.fail("load_dataset should not be called"),
    )

    with pytest.raises(ValueError, match="xx_XX"):
        prepare_module.prepare(
            target_languages=["xx_XX"],
            prefetch_comet=False,
            prefetch_spbleu=False,
        )
