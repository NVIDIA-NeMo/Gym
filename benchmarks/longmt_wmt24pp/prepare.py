# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare wmt24pp benchmark for document-level translation.

Downloads google/wmt24pp and writes wmt24pp_benchmark.jsonl with one record per
(document, target language). Each record contains the full source document as a
single joined string (source_sentences joined by space) plus the individual
source_sentences and reference_sentences lists for downstream analysis.

wmt24pp documents are short (typically < 300 sentences, < 2K tokens) so no
truncation is needed.

Also pre-fetches the SEGALE judge models (LASER2, ersatz, wmt22-cometkiwi-da)
into their cache directories so the resource server can run with
HF_HUB_OFFLINE=1 from the first verify() call.

Usage:
    python prepare.py
    python prepare.py --target_languages de_DE fr_FR ja_JP
    python prepare.py --no_prefetch
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "wmt24pp_benchmark.jsonl"

HF_REPO_ID = "google/wmt24pp"

# All 55 language pairs in google/wmt24pp — same list as NeMo-Skills wmt24pp prepare.py.
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

DEFAULT_TARGET_LANGUAGES = ALL_LANGUAGES


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
    prefetch: bool = True,
) -> Path:
    """Download google/wmt24pp and write wmt24pp_benchmark.jsonl.

    Returns the path to the written file.
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for tgt_lang in target_languages:
            print(f"Loading {HF_REPO_ID} en-{tgt_lang}...")
            dataset = load_dataset(HF_REPO_ID, f"en-{tgt_lang}")["train"]

            # Group rows by document, preserving order.
            docs: dict[str, list] = OrderedDict()
            for row in dataset:
                if row["is_bad_source"]:
                    continue
                doc_id = row["document_id"]
                if doc_id not in docs:
                    docs[doc_id] = []
                docs[doc_id].append(row)

            for rows in docs.values():
                rows.sort(key=lambda r: r["segment_id"])

            for doc_id, rows in docs.items():
                src_sents = [r["source"] for r in rows]
                ref_sents = [r["target"] for r in rows]
                record = {
                    "text": " ".join(src_sents),
                    "source_sentences": src_sents,
                    "reference_sentences": ref_sents,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": _lang_name(tgt_lang),
                    "doc_id": doc_id,
                    "seg_id": 1,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} rows to {OUTPUT_FPATH}")

    if prefetch:
        _prefetch_judge_models()

    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_languages", nargs="+", default=None, help="Target language codes (default: all 55)")
    parser.add_argument(
        "--no_prefetch", action="store_true", help="Skip judge model prefetch (useful on machines without GPU)"
    )
    args = parser.parse_args()
    prepare(target_languages=args.target_languages, prefetch=not args.no_prefetch)
