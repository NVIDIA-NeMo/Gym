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

from benchmarks.flores200 import prepare as prepare_module


def _fail_import(monkeypatch, module_name: str, exc: BaseException) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_example_rollouts_follow_current_language_and_score_contract() -> None:
    example_path = prepare_module.DATA_DIR / "example_rollouts.jsonl"
    rows = [json.loads(line) for line in example_path.read_text(encoding="utf-8").splitlines()]

    assert rows
    for row in rows:
        assert row["source_language"] == "eng_Latn"
        assert row["target_language"] in prepare_module._FLORES_LANG_MAP
        assert row["language_consistency_score"] is not None
        assert 0.0 <= row["language_consistency_score"] <= 1.0


def test_glotlid_prefetch_skips_when_huggingface_hub_is_absent(monkeypatch, capsys) -> None:
    _fail_import(
        monkeypatch,
        "huggingface_hub",
        ModuleNotFoundError("No module named 'huggingface_hub'", name="huggingface_hub"),
    )

    prepare_module._prefetch_glotlid_model()

    assert "huggingface-hub not installed" in capsys.readouterr().out


def test_glotlid_prefetch_propagates_missing_transitive_dependency(monkeypatch) -> None:
    _fail_import(
        monkeypatch,
        "huggingface_hub",
        ModuleNotFoundError("No module named 'requests'", name="requests"),
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        prepare_module._prefetch_glotlid_model()

    assert exc_info.value.name == "requests"


def test_prepare_uses_full_flores_target_codes_and_english_names(monkeypatch, tmp_path) -> None:
    texts_by_config = {
        "eng_Latn": ["English text"],
        "spa_Latn": ["Texto en español"],
    }

    def fake_load_dataset(_repo_id, config, *, split):
        assert split == "devtest"
        return {"text": texts_by_config[config]}

    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)
    prefetch_calls = []
    monkeypatch.setattr(
        prepare_module,
        "prefetch_translation_models",
        lambda **kwargs: prefetch_calls.append(kwargs),
    )

    output_path = prepare_module.prepare(
        languages=["eng_Latn", "spa_Latn"],
        prefetch_spbleu=False,
        prefetch_comet=False,
        prefetch_glotlid=False,
    )
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert prefetch_calls == [{"prefetch_comet": False, "prefetch_spbleu": False}]
    assert rows == [
        {
            "text": "English text",
            "translation": "Texto en español",
            "source_language": "eng_Latn",
            "target_language": "spa_Latn",
            "source_lang_name": "English",
            "target_lang_name": "Spanish (Latin American)",
        },
        {
            "text": "Texto en español",
            "translation": "English text",
            "source_language": "spa_Latn",
            "target_language": "eng_Latn",
            "source_lang_name": "Spanish (Latin American)",
            "target_lang_name": "English",
        },
    ]


def test_flores_names_distinguish_dialects() -> None:
    assert prepare_module._FLORES_LANG_MAP["apc_Arab_nort3139"] == "Levantine Arabic (North)"
    assert prepare_module._FLORES_LANG_MAP["apc_Arab_sout3123"] == "Levantine Arabic (South)"
    assert prepare_module._flores_lang_code("apc_Arab_sout3123") == "apc_Arab_sout3123"


def test_two_character_language_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="complete code"):
        prepare_module._flores_lang_code("en")


def test_explicit_dev_only_language_is_rejected_before_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        prepare_module,
        "load_dataset",
        lambda *_args, **_kwargs: pytest.fail("load_dataset should not be called"),
    )

    with pytest.raises(ValueError, match="do not provide devtest: brx_Deva"):
        prepare_module.prepare(
            source_languages=["eng_Latn"],
            target_languages=["brx_Deva"],
            prefetch_spbleu=False,
            prefetch_comet=False,
            prefetch_glotlid=False,
        )


def test_dev_split_accepts_language_without_devtest() -> None:
    prepare_module._validate_split_languages(["brx_Deva"], "dev")
