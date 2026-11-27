# SPDX-FileCopyrightText: Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/open_asr_leaderboard/blob/8d338ca0115a18f90d74e67b29efb480210bbd96/normalizer/normalizer.py
# and .../normalizer/english_abbreviations.py (same commit: `EnglishTextNormalizer`, `EnglishAcronymNormalizer`
# and the name, spelling and compound tables). Changes: the unchanged Whisper number and spelling rules come
# from the `whisper-normalizer` package instead of a copy, so only the leaderboard's additions are kept here;
# the acronym normalizer is a function; names and tables are private. See ATTRIBUTIONS.md.

"""English text normalization of the Hugging Face Open ASR Leaderboard.

``whisper-normalizer`` supplies the Whisper number and spelling rules; this module layers the leaderboard's
replacements on top of them (see the header for the upstream revision).
"""

import re

from whisper_normalizer.english import EnglishTextNormalizer, remove_symbols_and_diacritics


_SPELLING_UPDATES = {
    "archaeological": "archeological",
    "cancellation": "cancelation",
    "cancellations": "cancelations",
    "etcetera": "etc",
    "flyer": "flier",
    "kay": "okay",
    "mm": "hmm",
    "ok": "okay",
    "travelogue": "travelog",
    "travelogues": "travelogs",
    "archaeology": "archeology",
    "pummelled": "pummeled",
    "pummelling": "pummeling",
    "snowploughs": "snowplows",
}

_SPELLING_REMOVALS = {
    "archeological",
    "cancelation",
    "cancelations",
    "flyer / flier",
    "travelog",
    "travelogs",
}

_NAME_NORMALIZER = {
    "alan": "allen",
    "allan": "allen",
    "bridgette": "bridget",
    "charly": "charlie",
    "charley": "charlie",
    "garry": "gary",
    "gregg": "greg",
    "jacky": "jackie",
    "joann": "joanne",
    "joane": "joanne",
    "kellye": "kelly",
    "kelli": "kelly",
    "kelley": "kelly",
    "lilly": "lily",
    "micheal": "michael",
    "michele": "michelle",
    "mollie": "molly",
    "phillip": "philip",
    "sallie": "sally",
    "stacey": "stacy",
    "stacie": "stacy",
    "tracey": "tracy",
    "tracie": "tracy",
    "bret": "brett",
    "carrol": "carol",
    "carole": "carol",
    "carroll": "carol",
    "allison": "alison",
    "alyson": "alison",
    "russel": "russell",
    "douglass": "douglas",
    "dominick": "dominic",
    "robb": "rob",
    "kris": "chris",
    "kristopher": "christopher",
    "cristopher": "christopher",
    "kristina": "christina",
    "kristen": "kristin",
    "karl": "carl",
    "kathy": "cathy",
    "katherine": "catherine",
    "kathryn": "catherine",
    "catharine": "catherine",
    "erik": "eric",
    "erick": "eric",
    "caren": "karen",
    "caryn": "karen",
    "karin": "karen",
    "katelyn": "caitlin",
    "kaitlyn": "caitlin",
    "kaitlin": "caitlin",
    "nikole": "nicole",
    "veronika": "veronica",
    "viktor": "victor",
    "viktoria": "victoria",
    "kevan": "kevin",
    "patrik": "patrick",
    "frederik": "frederick",
    "fredrick": "frederick",
    "lukas": "lucas",
    "ann": "anne",
    "jon": "john",
    "johnathan": "jonathan",
    "jonathon": "jonathan",
    "sara": "sarah",
    "mathew": "matthew",
    "nicolas": "nicholas",
    "rachael": "rachel",
    "rebekah": "rebecca",
    "devorah": "deborah",
    "theresa": "teresa",
    "suzanne": "susanne",
    "antony": "anthony",
    "martyn": "martin",
    "denis": "dennis",
    "laurence": "lawrence",
    "tomas": "thomas",
    "tobey": "toby",
    "macarthur": "mcarthur",
    "macartney": "mccartney",
    "macarthy": "mccarthy",
    "maccarthy": "mccarthy",
    "macdonald": "mcdonald",
    "mackay": "mckay",
    "mackenzie": "mckenzie",
    "macleod": "mcleod",
    "maclean": "mclean",
    "macmillan": "mcmillan",
    "macintosh": "mcintosh",
    "macintyre": "mcintyre",
    "macnamara": "mcnamara",
    "macgowan": "mcgowan",
    "mohamad": "mohammed",
    "mohamed": "mohammed",
    "mohammad": "mohammed",
    "muhammad": "mohammed",
    "muhamad": "mohammed",
    "muhammed": "mohammed",
    "mouhamed": "mohammed",
    "mouhamad": "mohammed",
    "mahomet": "mohammed",
    "fatimah": "fatima",
    "yusuf": "yousef",
    "yusef": "yousef",
    "myriam": "miriam",
    "rajeev": "rajiv",
    "alphonso": "alfonso",
    "bryan": "brian",
    "geoffrey": "jeffrey",
    "jeffery": "jeffrey",
    "geoff": "jeff",
    "neal": "neil",
    "shaun": "sean",
    "shawn": "sean",
    "shayne": "shane",
    "stephen": "steven",
    "toni": "tony",
    "leigh": "lee",
    "lewis": "louis",
    "marc": "mark",
    "meghan": "megan",
    "nathalie": "natalie",
    "robyn": "robin",
    "rodger": "roger",
    "linsey": "lindsay",
    "lindsey": "lindsay",
    "zackary": "zachary",
    "zachery": "zachary",
    "zak": "zach",
    "sheri": "sherry",
    "cheri": "sherry",
    "sherrie": "sherry",
    "terri": "terry",
    "lori": "laurie",
    "jaime": "jamie",
    "jayson": "jason",
    "lesley": "leslie",
    "lynda": "linda",
    "lynne": "lynn",
    "gayle": "gail",
    "rhonda": "ronda",
    "yvonne": "ivonne",
    "stewart": "stuart",
    "walther": "walter",
    "symon": "simon",
    "collin": "colin",
    "dillon": "dylan",
    "aron": "aaron",
    "artur": "arthur",
    "henri": "henry",
    "josef": "joseph",
    "pieter": "peter",
}

_COMPOUND_NORMALIZER = {
    r"\bet\s+cetera\b": "etc",
    r"\bal\s+right\b": "alright",
    r"\ball\s+right\b": "alright",
    r"\bhow\s+ever\b": "however",
    r"\bwi\s+fi\b": "wifi",
    r"\bhi\s+fi\b": "hifi",
    r"\blo\s+fi\b": "lofi",
    r"\bsci\s+fi\b": "scifi",
    r"\be\s+mail\b": "email",
    r"\be\s+book\b": "ebook",
    r"\be\s+commerce\b": "ecommerce",
    r"\bx\s+ray\b": "xray",
    r"\bt\s+shirt\b": "tshirt",
    r"\ba\s+m\b": "am",
    r"\bp\s+m\b": "pm",
    r"\bo\s+k\b": "okay",
}


def _normalize_acronyms(text: str) -> str:
    words = text.split()
    result: list[str] = []
    index = 0
    while index < len(words):
        if len(words[index]) == 1 and words[index].isalnum():
            run = [words[index]]
            next_index = index + 1
            while next_index < len(words) and len(words[next_index]) == 1 and words[next_index].isalnum():
                run.append(words[next_index])
                next_index += 1
            minimum_run = 3 if any(character in ("a", "i") for character in run) else 2
            if len(run) >= minimum_run:
                result.append("".join(run))
            else:
                result.extend(run)
            index = next_index
        else:
            result.append(words[index])
            index += 1
    return " ".join(result)


class OpenASREnglishTextNormalizer:
    """Whisper English normalization with the Open ASR Leaderboard replacements applied."""

    def __init__(self):
        whisper_normalizer = EnglishTextNormalizer()
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um|ah|aha|ahh|ahm|eh|ehehe|em|hm|huh|hum|mhum|uhm|umm|uhuh)\b"
        self.replacers = whisper_normalizer.replacers.copy()
        for pattern in (r"\bkinda\b", r"\bsorta\b", r"\bdunno\b", r"\bcause\b", r"'s\b"):
            self.replacers.pop(pattern)
        self.replacers[r"\b(it|he|she|what|that|who|here|there|how|when|where|why|this)'s\b"] = r"\1 is"
        self.standardize_numbers = whisper_normalizer.standardize_numbers
        self.standardize_spellings = whisper_normalizer.standardize_spellings
        for word in _SPELLING_REMOVALS:
            self.standardize_spellings.mapping.pop(word)
        self.standardize_spellings.mapping.update(_SPELLING_UPDATES)

    def __call__(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = re.sub(self.ignore_patterns, "", text)
        text = re.sub(r"\s+'", "'", text)

        for pattern, replacement in self.replacers.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = re.sub(r"\.([^0-9]|$)", r" \1", text)
        text = remove_symbols_and_diacritics(text, keep=".%$¢€£")

        for pattern, replacement in _COMPOUND_NORMALIZER.items():
            text = re.sub(pattern, replacement, text)

        text = self.standardize_numbers(text)
        text = self.standardize_spellings(text)
        text = " ".join(_NAME_NORMALIZER.get(word, word) for word in text.split())
        text = _normalize_acronyms(text)

        text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
        text = re.sub(r"([^0-9])%", r"\1 ", text)
        return re.sub(r"\s+", " ", text)
