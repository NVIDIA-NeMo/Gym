# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare pg19 benchmark for long-document translation.

Downloads emozilla/pg19 (test split, 100 books) and writes pg19_{N}k.jsonl
with one record per (book, target language). Books are tiktoken-truncated to
--max_tokens at prep time. Truncation removes the middle of the book and
preserves the beginning and end — same semantics as NeMo-Skills'
reduce_prompt_from_middle.

Output files are named by token limit (e.g. pg19_100k.jsonl, pg19_50k.jsonl)
so multiple limits can coexist in the data dir. Select the desired file via
--input_file in the run script.

Also pre-fetches the SEGALE judge models (LASER2, ersatz, wmt22-cometkiwi-da)
into their cache directories so the resource server can run with
HF_HUB_OFFLINE=1 from the first verify() call.

Usage:
    python prepare.py
    python prepare.py --target_languages de_DE fr_FR ja_JP
    python prepare.py --max_tokens 50000 --no_prefetch
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

HF_REPO_ID = "emozilla/pg19"

# Default tiktoken cl100k_base tokens to keep per book.
# cl100k_base is used as a model-agnostic approximation — it tends to produce
# similar token counts to GPT-4/Qwen tokenizers and provides a clean round limit.
# The output file is named pg19_{max_tokens//1000}k.jsonl so multiple token limits
# can coexist in the data dir and be selected via --input_file in the run script.
MAX_TIKTOKEN_TOKENS = 20_000

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


def _truncate_middle(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens using tiktoken cl100k_base, removing the middle.

    Keeps the first half and last half of the token budget so the model sees
    both the beginning (title, author, opening) and the end of the book. If the
    text fits within max_tokens it is returned unchanged.
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

    half = max_tokens // 2
    head = enc.decode(tokens[:half])
    tail = enc.decode(tokens[len(tokens) - half :])
    return head + "\n\n[...]\n\n" + tail


def _prefetch_judge_models() -> None:
    """Pre-fetch LASER2, ersatz, and wmt22-cometkiwi-da into their cache dirs."""
    laser_home = os.environ.get("LASER_HOME")
    try:
        from laser_encoders import LaserEncoderPipeline

        print(f"Pre-fetching LASER2 (LASER_HOME={laser_home})...")
        LaserEncoderPipeline(laser="laser2", model_dir=laser_home)
        print("LASER2 cached")
    except ImportError:
        print("laser-encoders not installed; skipping LASER2 prefetch")
    except Exception as exc:
        print(f"LASER2 prefetch failed (will retry at server start): {exc}")

    try:
        import ersatz

        print("Pre-fetching ersatz default-multilingual model...")
        ersatz.split(model="default-multilingual", text=".")
        print("ersatz cached")
    except ImportError:
        print("ersatz not installed; skipping prefetch")
    except Exception as exc:
        print(f"ersatz prefetch failed: {exc}")

    try:
        from comet import download_model, load_from_checkpoint

        print("Pre-fetching Unbabel/wmt22-cometkiwi-da...")
        ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
        load_from_checkpoint(ckpt)
        print("wmt22-cometkiwi-da cached")
    except ImportError:
        print("unbabel-comet not installed; skipping COMETKiwi prefetch")
    except Exception as exc:
        print(f"COMETKiwi prefetch failed: {exc}")


def prepare(
    target_languages: list[str] | None = None,
    max_tokens: int = MAX_TIKTOKEN_TOKENS,
    prefetch: bool = True,
) -> Path:
    """Download emozilla/pg19 test split and write pg19_{max_tokens//1000}k.jsonl.

    Returns the path to the written file.
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_fpath = DATA_DIR / f"pg19_{max_tokens // 1000}k.jsonl"

    print(f"Loading {HF_REPO_ID} test split...")
    dataset = load_dataset(HF_REPO_ID, split="test", streaming=True)
    books = list(dataset)
    print(f"Loaded {len(books)} books")

    count = 0
    with output_fpath.open("w", encoding="utf-8") as fout:
        for tgt_lang in target_languages:
            for book in books:
                text = _truncate_middle(book["text"], max_tokens)
                row = {
                    "text": text,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": _lang_name(tgt_lang),
                    "doc_id": _sanitize_doc_id(book["short_book_title"]),
                    "seg_id": 1,
                    "publication_date": int(book["publication_date"]),
                    "url": book["url"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} rows ({len(books)} books × {len(target_languages)} languages) to {output_fpath}")

    if prefetch:
        _prefetch_judge_models()

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
        "--max_tokens",
        type=int,
        default=MAX_TIKTOKEN_TOKENS,
        help=f"Max tiktoken tokens per book (default: {MAX_TIKTOKEN_TOKENS})",
    )
    parser.add_argument(
        "--no_prefetch", action="store_true", help="Skip judge model prefetch (useful on machines without GPU)"
    )
    args = parser.parse_args()

    langs = ALL_LANGUAGES if args.all_languages else args.target_languages
    prepare(target_languages=langs, max_tokens=args.max_tokens, prefetch=not args.no_prefetch)
