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
"""Prepare FLORES-200 benchmark data.

Two steps:

1. Download the FLORES+ dataset from ``openlanguagedata/flores_plus`` and write
   the interleaved benchmark JSONL covering all directed pairs of the default
   six-language set (excluding self-pairs).
2. Pre-fetch the xCOMET-XXL checkpoint AND its underlying xlm-roberta-xxl
   tokenizer into ``HF_HOME`` so the ``wmt_translation`` resource server can
   run fully offline. Without this, every fresh ``CometActor`` resolves the
   tokenizer from HF Hub on startup, which hits the rate limiter when N
   actors initialize concurrently.

Per-row fields are referenced by both the prompt template
(``benchmarks/flores200/prompts/default.yaml``) and the ``wmt_translation``
resource server: ``text``, ``translation``, ``source_language``,
``target_language``, ``source_lang_name``, ``target_lang_name``.

Mirrors NeMo-Skills' ``nemo_skills/dataset/flores200/prepare.py`` row order
and field set so the output is byte-equivalent under the same default
arguments and split.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).parent / "data"

HF_REPO_ID = "openlanguagedata/flores_plus"

# Default COMET model used by the wmt_translation resource server.
# Pre-downloading it (plus its xlm-roberta-xxl backbone tokenizer)
# populates HF_HOME so the server can run with HF_HUB_OFFLINE=1 — no
# online tokenizer resolution at verify() time and no per-actor HF
# rate-limit retries.
COMET_MODEL = "Unbabel/XCOMET-XXL"

# Same default language set as NeMo-Skills' flores200/prepare.py. Order is
# preserved so the interleaved JSONL is byte-equivalent to Skills' output
# under identical args.
DEFAULT_LANGUAGES = ["en", "de", "es", "fr", "it", "ja"]

DEFAULT_SPLIT = "devtest"

# FLORES+ config name (``iso639-3_iso15924``) for each supported 2-letter
# code. Skills' prepare derives these dynamically via
# ``langcodes.Language``; we hardcode the default six so the prepare step
# adds no runtime dependency to the host Gym venv.
_FLORES_CONFIG = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
}

# Display names for each supported code, byte-equivalent to
# ``langcodes.Language(code).display_name()`` for the defaults above.
_LANG_DISPLAY_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
}


def _flores_lang_code(lang: str) -> str:
    """Map a 2-letter code (e.g. ``en``) to FLORES' ``iso639-3_iso15924`` config name.

    Mirrors NeMo-Skills' use of
    ``f"{Language(lang).to_alpha3()}_{Language(lang).maximize().script}"``
    for the default-six codes; raises a clear error on unsupported inputs so
    users can extend ``_FLORES_CONFIG`` / ``_LANG_DISPLAY_NAMES`` together.
    """
    try:
        return _FLORES_CONFIG[lang]
    except KeyError as exc:
        raise ValueError(
            f"Unknown language code '{lang}'. Supported defaults: {sorted(_FLORES_CONFIG)}. "
            "To add a new code, extend _FLORES_CONFIG and _LANG_DISPLAY_NAMES in "
            "benchmarks/flores200/prepare.py."
        ) from exc


def _display_name(lang: str) -> str:
    try:
        return _LANG_DISPLAY_NAMES[lang]
    except KeyError as exc:
        raise ValueError(
            f"No display name registered for '{lang}'. Add it to _LANG_DISPLAY_NAMES in "
            "benchmarks/flores200/prepare.py (must match the value of "
            "``langcodes.Language(<code>).display_name()`` to keep parity with "
            "NeMo-Skills' flores200 prepare output)."
        ) from exc


def _prefetch_comet_model(model_name: str = COMET_MODEL) -> None:
    """Pre-download xCOMET-XXL + its xlm-roberta-xxl tokenizer to HF_HOME.

    ``download_model`` fetches just the COMET checkpoint files; the actual
    tokenizer + transformer backbone are pulled when ``load_from_checkpoint``
    instantiates the model. We do both here so the cache is fully primed
    before any actor starts. Skipped silently if ``unbabel-comet`` is not
    installed in the active Python env (e.g. when ``prepare()`` is invoked
    from a venv that doesn't carry the server's heavy deps — the resource
    server's own venv will fetch on first use in that case).
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(f"unbabel-comet not installed; skipping {model_name} prefetch")
        return

    print(f"Pre-fetching {model_name} checkpoint...")
    ckpt_path = download_model(model_name)
    print(f"Loading {model_name} once to populate xlm-roberta-xxl cache...")
    load_from_checkpoint(ckpt_path)
    print(f"{model_name} + tokenizer cached")


def prepare(
    languages: list[str] | None = None,
    source_languages: list[str] | None = None,
    target_languages: list[str] | None = None,
    split: str = DEFAULT_SPLIT,
    prefetch_comet: bool = True,
) -> Path:
    """Download and interleave FLORES+ pairs. Returns the output JSONL path.

    ``languages`` is shorthand for setting ``source_languages`` and
    ``target_languages`` to the same list (the Skills default behavior).
    Pass ``source_languages`` / ``target_languages`` explicitly to override
    one direction.

    If ``prefetch_comet`` is True (the default), also pre-downloads
    xCOMET-XXL and its tokenizer into HF_HOME so the server can run offline.
    Pass ``prefetch_comet=False`` for benchmark prep on a machine without GPU
    / unbabel-comet (the cache will be populated on first server use
    instead).
    """
    if source_languages is None:
        source_languages = list(languages) if languages is not None else list(DEFAULT_LANGUAGES)
    if target_languages is None:
        target_languages = list(languages) if languages is not None else list(DEFAULT_LANGUAGES)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / f"flores200_{split}_benchmark.jsonl"

    all_languages = sorted(set(source_languages) | set(target_languages))

    datasets: dict = {}
    for lang in all_languages:
        lang_code = _flores_lang_code(lang)
        print(f"Loading {HF_REPO_ID} config {lang_code} (split={split})...")
        datasets[lang] = load_dataset(HF_REPO_ID, lang_code, split=split)["text"]

    count = 0
    with output_fpath.open("w", encoding="utf-8") as fout:
        # Match Skills' nested-loop ordering: outer src, inner tgt, skip self-pairs.
        for src_lang in source_languages:
            for tgt_lang in target_languages:
                if src_lang == tgt_lang:
                    continue
                for src, tgt in zip(datasets[src_lang], datasets[tgt_lang], strict=True):
                    row = {
                        "text": src,
                        "translation": tgt,
                        "source_language": src_lang,
                        "target_language": tgt_lang,
                        "source_lang_name": _display_name(src_lang),
                        "target_lang_name": _display_name(tgt_lang),
                    }
                    fout.write(json.dumps(row) + "\n")
                    count += 1

    print(f"Wrote {count} rows to {output_fpath}")

    if prefetch_comet:
        _prefetch_comet_model()

    return output_fpath


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=("dev", "devtest"),
        help="FLORES+ split to download.",
    )
    parser.add_argument(
        "--source_languages",
        default=DEFAULT_LANGUAGES,
        nargs="+",
        help="2-letter language codes to translate from.",
    )
    parser.add_argument(
        "--target_languages",
        default=DEFAULT_LANGUAGES,
        nargs="+",
        help="2-letter language codes to translate to.",
    )
    parser.add_argument(
        "--no-prefetch-comet",
        dest="prefetch_comet",
        action="store_false",
        help="Skip xCOMET-XXL prefetch (e.g. when unbabel-comet is unavailable).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        source_languages=args.source_languages,
        target_languages=args.target_languages,
        split=args.split,
        prefetch_comet=args.prefetch_comet,
    )
