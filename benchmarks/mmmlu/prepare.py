# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMMLU (multilingual MMLU) benchmark data for NeMo Gym (mcqa).

Ports NeMo-Skills' ``mmmlu`` benchmark: CSVs from OpenAI public blob storage,
with multilingual answer extraction via a combined ``template_metadata.output_regex``.
"""

import argparse
import csv
import json
import uuid
from io import StringIO
from pathlib import Path
from urllib.request import urlretrieve


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmmlu_benchmark.jsonl"

OPENAI_PUBLIC_URL = "https://openaipublic.blob.core.windows.net/simple-evals/{}"

SUPPORTED_LANGUAGES = [
    "AR-XY",
    "BN-BD",
    "DE-DE",
    "ES-LA",
    "FR-FR",
    "HI-IN",
    "ID-ID",
    "IT-IT",
    "JA-JP",
    "KO-KR",
    "PT-BR",
    "ZH-CN",
    "SW-KE",
    "YO-NG",
]

# From NeMo-Skills ``mmmlu_utils.subject2category``.
SUBJECT_TO_CATEGORY: dict[str, str] = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"

MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]

LETTER_REGEX = r"\b\(?\s*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])\s*\)?\.?\b"
GREEDY_REGEX = r"[\s\S]*" + LETTER_REGEX


def _build_mmmlu_output_regex() -> str:
    branches = [MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(rx) for rx in MULTILINGUAL_ANSWER_REGEXES]
    branches.append(GREEDY_REGEX)
    return "(?:" + ")|(?:".join(branches) + ")"


MMMLU_OUTPUT_REGEX = _build_mmmlu_output_regex()


def _format_multichoice_question(row: dict) -> str:
    return QUERY_TEMPLATE_MULTICHOICE.format(
        Question=row["Question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"],
    )


def _download_csv(language: str) -> Path:
    suffix = "mmlu.csv" if language == "EN-US" else f"mmlu_{language}.csv"
    dst = DATA_DIR / suffix
    if not dst.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = OPENAI_PUBLIC_URL.format(suffix)
        print(f"Downloading {suffix} ...")
        urlretrieve(url, dst)
    return dst


def _read_examples(csv_path: Path) -> list[dict]:
    """Parse MMMLU CSV (OpenAI simple-evals), tolerating a leading index column like pandas ``index_col=0``."""
    text = csv_path.read_text(encoding="utf-8")
    reader = csv.reader(StringIO(text))
    header = next(reader)
    required = {"Question", "A", "B", "C", "D", "Answer", "Subject"}
    strip_first = False
    if not required.issubset(header):
        if len(header) > 1 and required.issubset(header[1:]):
            strip_first = True
            keys = header[1:]
        else:
            raise ValueError(f"Unexpected CSV header in {csv_path}: {header!r}")
    else:
        keys = header

    rows: list[dict] = []
    for parts in reader:
        if not parts:
            continue
        if strip_first and len(parts) > len(keys):
            parts = parts[1:]
        if len(parts) != len(keys):
            continue
        rows.append(dict(zip(keys, parts, strict=True)))
    return rows


def format_entry(entry: dict, language: str) -> dict:
    expected = str(entry["Answer"]).strip().upper()
    if expected not in {"A", "B", "C", "D"}:
        raise ValueError(f"Bad answer {expected!r}")
    category = SUBJECT_TO_CATEGORY.get(entry["Subject"], "other")
    letters = ["A", "B", "C", "D"]
    options = [{letters[i]: entry[letters[i]]} for i in range(4)]
    prompt = _format_multichoice_question(entry)
    seed = json.dumps({"lang": language, "q": prompt, "a": expected}, sort_keys=True, ensure_ascii=False)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    return {
        "question": prompt,
        "options": options,
        "expected_answer": expected,
        "template_metadata": {"output_regex": MMMLU_OUTPUT_REGEX},
        "subset_for_metrics": language,
        "category": category,
        "uuid": row_uuid,
    }


def prepare(languages: list[str] | None = None, include_english: bool = False) -> Path:
    if languages is None:
        languages = list(SUPPORTED_LANGUAGES)
    langs = [lang for lang in languages if lang != "EN-US"]
    valid = set(SUPPORTED_LANGUAGES)
    if include_english:
        valid = valid | {"EN-US"}
        langs.append("EN-US")
    invalid = set(langs) - valid
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for language in langs:
            csv_path = _download_csv(language)
            for entry in _read_examples(csv_path):
                row = format_entry(entry, language)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=list(SUPPORTED_LANGUAGES))
    p.add_argument("--include_english", action="store_true")
    args = p.parse_args()
    prepare(languages=args.languages, include_english=args.include_english)
