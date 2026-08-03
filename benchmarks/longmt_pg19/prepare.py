# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pg19 benchmark for long-document translation.

Downloads emozilla/pg19 (test split, 100 books) and writes pg19_benchmark.jsonl
with one record per (book, target language, truncation length). Each row has a
`target_len` field (int, tiktoken cl100k_base tokens) so rollouts can be
grouped or filtered by context length.

Books are truncated from the end so the model always sees the beginning of the
book without skipping content.

Usage:
    python prepare.py
    python prepare.py --target_languages de_DE fr_FR ja_JP
    python prepare.py --lengths 8 32 65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

HF_REPO_ID = "emozilla/pg19"

# Truncation lengths in tiktoken cl100k_base tokens (powers of 2).
DEFAULT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]

# 6 core languages matching the initial pg19 evaluation runs.
DEFAULT_TARGET_LANGUAGES = ["de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP", "zh_CN"]

# Full 55-language set (same as NeMo-Skills pg19 prepare.py).
ALL_LANGUAGES = [
    "ar_EG",
    "ar_SA",
    "bg_BG",
    "bn_IN",
    "ca_ES",
    "cs_CZ",
    "da_DK",
    "de_DE",
    "el_GR",
    "es_MX",
    "et_EE",
    "fa_IR",
    "fi_FI",
    "fil_PH",
    "fr_CA",
    "fr_FR",
    "gu_IN",
    "he_IL",
    "hi_IN",
    "hr_HR",
    "hu_HU",
    "id_ID",
    "is_IS",
    "it_IT",
    "ja_JP",
    "kn_IN",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "ml_IN",
    "mr_IN",
    "nl_NL",
    "no_NO",
    "pa_IN",
    "pl_PL",
    "pt_BR",
    "pt_PT",
    "ro_RO",
    "ru_RU",
    "sk_SK",
    "sl_SI",
    "sr_RS",
    "sv_SE",
    "sw_KE",
    "sw_TZ",
    "ta_IN",
    "te_IN",
    "th_TH",
    "tr_TR",
    "uk_UA",
    "ur_PK",
    "vi_VN",
    "zh_CN",
    "zh_TW",
    "zu_ZA",
]


def _lang_name(lang_code: str) -> str:
    try:
        from langcodes import Language

        return Language(lang_code.split("_")[0]).display_name()
    except ImportError:
        _FALLBACK = {
            "de_DE": "German",
            "es_MX": "Spanish",
            "fr_FR": "French",
            "it_IT": "Italian",
            "ja_JP": "Japanese",
            "zh_CN": "Chinese",
        }
        return _FALLBACK.get(lang_code, lang_code)


def _sanitize_doc_id(title: str) -> str:
    title = title.replace(" ", "-")
    invalid = set('/\\:*?"<>|\x00')
    return "".join(c for c in title if c not in invalid)


def _truncate_end(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens using tiktoken cl100k_base, dropping from the end.

    The model always sees the beginning of the book. If the text fits within
    max_tokens it is returned unchanged.
    """
    try:
        import tiktoken
    except ImportError:
        print("WARNING: tiktoken not installed — skipping truncation. Install with: pip install tiktoken")
        return text

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def prepare(
    target_languages: list[str] | None = None,
    lengths: list[int] | None = None,
) -> Path:
    """Download emozilla/pg19 test split and write pg19_benchmark.jsonl.

    One row per (book, target_language, truncation_length). The `target_len`
    field (int, tiktoken tokens) lets callers filter rollouts by context length.

    Returns the path to the written file.
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES
    if lengths is None:
        lengths = DEFAULT_LENGTHS

    lengths_tokens = sorted(lengths)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / "pg19_benchmark.jsonl"

    print(f"Loading {HF_REPO_ID} test split...")
    dataset = load_dataset(HF_REPO_ID, split="test", streaming=True)
    books = list(dataset)
    print(f"Loaded {len(books)} books")
    print(f"Lengths (tokens): {lengths_tokens}")

    count = 0
    with output_fpath.open("w", encoding="utf-8") as fout:
        for target_len in lengths_tokens:
            for tgt_lang in target_languages:
                for book in books:
                    text = _truncate_end(book["text"], target_len)
                    row = {
                        "text": text,
                        "source_language": "en",
                        "target_language": tgt_lang,
                        "source_lang_name": "English",
                        "target_lang_name": _lang_name(tgt_lang),
                        "doc_id": _sanitize_doc_id(book["short_book_title"]),
                        "target_len": target_len,
                        "seg_id": 1,
                        "publication_date": int(book["publication_date"]),
                        "url": book["url"],
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1

    n_lengths = len(lengths_tokens)
    print(
        f"Wrote {count} rows "
        f"({len(books)} books × {len(target_languages)} languages × {n_lengths} lengths) "
        f"to {output_fpath}"
    )

    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_languages", nargs="+", default=None, help="Target language codes (default: 6 core languages)"
    )
    parser.add_argument(
        "--all_languages", action="store_true", help="Use all 55 language codes instead of the 6 defaults"
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=f"Truncation lengths in tiktoken tokens (default: {DEFAULT_LENGTHS})",
    )
    args = parser.parse_args()

    langs = ALL_LANGUAGES if args.all_languages else args.target_languages
    prepare(target_languages=langs, lengths=args.lengths)
