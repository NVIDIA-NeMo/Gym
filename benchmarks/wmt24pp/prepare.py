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
"""Prepare WMT24++ benchmark data.

Two steps:

1. Download the WMT24++ dataset from ``google/wmt24pp`` and write the
   interleaved benchmark JSONL (one config per en-<tgt> pair).
2. Pre-fetch the xCOMET-XXL checkpoint AND its underlying
   xlm-roberta-xxl tokenizer into HF_HOME so the ``wmt_translation``
   resource server can run fully offline. Without this, every fresh
   ``CometActor`` resolves the tokenizer from HF Hub on startup, which
   hits the rate limiter when N actors initialize concurrently.

Per-row fields are referenced by both the prompt template
(``benchmarks/wmt24pp/prompts/default.yaml``) and the
``wmt_translation`` resource server: ``text``, ``translation``,
``source_language``, ``target_language``, ``source_lang_name``,
``target_lang_name``.

Note that we need to somehow specify the dialect/locale we want the
model to generate in. Currently, this is done by setting
``target_lang_name`` to <language> (<country>) (e.g. "Spanish (Mexico)")
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "wmt24pp_benchmark.jsonl"

HF_REPO_ID = "google/wmt24pp"

# Default COMET model used by the wmt_translation resource server.
# Pre-downloading it (plus its xlm-roberta-xxl backbone tokenizer)
# populates HF_HOME so the server can run with HF_HUB_OFFLINE=1 — no
# online tokenizer resolution at verify() time and no per-actor HF
# rate-limit retries.
COMET_MODEL = "Unbabel/XCOMET-XXL"

# Same five targets + same order as Skills' default. Keeping the order
# stable is what makes the interleaved JSONL byte-comparable.
DEFAULT_TARGET_LANGUAGES = ["de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"]

# Hardcoding to avoid both dependency on https://github.com/rspeer/langcodes
#   and network call in prepare.py
# import langcodes #uv pip install langcodes[data]
# from datasets import get_dataset_config_names #uv pip install datasets
# lang_pairs = get_dataset_config_names("google/wmt24pp")
# _WMT24PP_LANG_MAP = {}
# for lang_pair in lang_pairs:
#    _, tgt = lang_pair.split('-')
#    _WMT24PP_LANG_MAP[tgt] = langcodes.Language.get(tgt).display_name()
_WMT24PP_LANG_MAP = {
    "ar_EG": "Arabic (Egypt)",
    "ar_SA": "Arabic (Saudi Arabia)",
    "bg_BG": "Bulgarian (Bulgaria)",
    "bn_IN": "Bangla (India)",
    "ca_ES": "Catalan (Spain)",
    "cs_CZ": "Czech (Czechia)",
    "da_DK": "Danish (Denmark)",
    "de_DE": "German (Germany)",
    "el_GR": "Greek (Greece)",
    "es_MX": "Spanish (Mexico)",
    "et_EE": "Estonian (Estonia)",
    "fa_IR": "Persian (Iran)",
    "fi_FI": "Finnish (Finland)",
    "fil_PH": "Filipino (Philippines)",
    "fr_CA": "French (Canada)",
    "fr_FR": "French (France)",
    "gu_IN": "Gujarati (India)",
    "he_IL": "Hebrew (Israel)",
    "hi_IN": "Hindi (India)",
    "hr_HR": "Croatian (Croatia)",
    "hu_HU": "Hungarian (Hungary)",
    "id_ID": "Indonesian (Indonesia)",
    "is_IS": "Icelandic (Iceland)",
    "it_IT": "Italian (Italy)",
    "ja_JP": "Japanese (Japan)",
    "kn_IN": "Kannada (India)",
    "ko_KR": "Korean (South Korea)",
    "lt_LT": "Lithuanian (Lithuania)",
    "lv_LV": "Latvian (Latvia)",
    "ml_IN": "Malayalam (India)",
    "mr_IN": "Marathi (India)",
    "nl_NL": "Dutch (Netherlands)",
    "no_NO": "Norwegian (Norway)",
    "pa_IN": "Punjabi (India)",
    "pl_PL": "Polish (Poland)",
    "pt_BR": "Portuguese (Brazil)",
    "pt_PT": "Portuguese (Portugal)",
    "ro_RO": "Romanian (Romania)",
    "ru_RU": "Russian (Russia)",
    "sk_SK": "Slovak (Slovakia)",
    "sl_SI": "Slovenian (Slovenia)",
    "sr_RS": "Serbian (Serbia)",
    "sv_SE": "Swedish (Sweden)",
    "sw_KE": "Swahili (Kenya)",
    "sw_TZ": "Swahili (Tanzania)",
    "ta_IN": "Tamil (India)",
    "te_IN": "Telugu (India)",
    "th_TH": "Thai (Thailand)",
    "tr_TR": "Turkish (Türkiye)",
    "uk_UA": "Ukrainian (Ukraine)",
    "ur_PK": "Urdu (Pakistan)",
    "vi_VN": "Vietnamese (Vietnam)",
    "zh_CN": "Chinese (China)",
    "zh_TW": "Chinese (Taiwan)",
    "zu_ZA": "Zulu (South Africa)",
}


def _prefetch_comet_model(model_name: str = COMET_MODEL) -> None:
    """Pre-download xCOMET-XXL + its xlm-roberta-xxl tokenizer to HF_HOME.

    ``download_model`` fetches just the COMET checkpoint files; the
    actual tokenizer + transformer backbone are pulled when
    ``load_from_checkpoint`` instantiates the model. We do both here so
    the cache is fully primed before any actor starts. Skipped silently
    if ``unbabel-comet`` is not installed in the active Python env (e.g.
    when ``prepare()`` is invoked from a venv that doesn't carry the
    server's heavy deps — the resource server's own venv will fetch on
    first use in that case).
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(f"unbabel-comet not installed; skipping {model_name} prefetch")
        return

    print(f"Pre-fetching {model_name} checkpoint...")
    ckpt_path = download_model(model_name)
    # Instantiating the model triggers the xlm-roberta-xxl tokenizer +
    # transformer download into HF_HOME. We don't keep the model object
    # — we just need the cache populated.
    print(f"Loading {model_name} once to populate xlm-roberta-xxl cache...")
    load_from_checkpoint(ckpt_path)
    print(f"{model_name} + tokenizer cached")


def prepare(target_languages: list[str] | None = None, prefetch_comet: bool = True) -> Path:
    """Download and interleave WMT24++ en-<tgt> pairs. Returns the output file path.

    If ``prefetch_comet`` is True (the default), also pre-downloads
    xCOMET-XXL and its tokenizer into HF_HOME so the server can run
    offline. Pass ``prefetch_comet=False`` for benchmark prep on a
    machine without GPU / unbabel-comet (the cache will be populated on
    first server use instead).
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES
    else:
        # check user passed in langs
        unknown = [lang for lang in target_languages if lang not in _WMT24PP_LANG_MAP]
        if unknown:
            raise ValueError(
                f"requested target languages [{','.join(unknown)}] are not in wmt24pp. "
                f"Available languages: [{','.join(list(_WMT24PP_LANG_MAP))}]"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets: dict = {}
    for lang in target_languages:
        print(f"Loading {HF_REPO_ID} config en-{lang}...")
        datasets[lang] = load_dataset(HF_REPO_ID, f"en-{lang}")["train"]

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for tgt_lang in target_languages:
            for src, tgt in zip(
                datasets[tgt_lang]["source"],
                datasets[tgt_lang]["target"],
                strict=True,
            ):
                row = {
                    "text": src,
                    "translation": tgt,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": _WMT24PP_LANG_MAP[tgt_lang],
                }
                fout.write(json.dumps(row) + "\n")
                count += 1

    print(f"Wrote {count} rows to {OUTPUT_FPATH}")

    if prefetch_comet:
        _prefetch_comet_model()

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
