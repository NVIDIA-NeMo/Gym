# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned SciKnowEval rubrics and retain the documented benchmark adaptations."""

import copy
import hashlib
import re
from pathlib import Path
from urllib.request import urlopen

import yaml


REVISION = "53addee640092d667439e9a8901bf55f2d92c9ed"  # pragma: allowlist secret
URL = f"https://raw.githubusercontent.com/HICAI-ZJU/SciKnowEval/{REVISION}/evaluation/utils/prompts/prompt.yaml"
SHA256 = "9ab53fe9062bc879b029e8daf5a9d4ad77543683a315e793b0a68bdcbf623199"  # pragma: allowlist secret
RELATION_TASKS = {
    "drug_drug_relation_extraction": "identifying all drug-drug interactions in a text and extracting them as (drug, interaction, drug) triplets",
    "compound_disease_relation_extraction": "identifying all compound-disease relations in an abstract and extracting them as [compound, disease] pairs",
}


def load_rubrics(cache_dir: Path) -> dict[str, dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "judge_prompts.yaml"
    source = path.read_bytes() if path.is_file() else b""
    if hashlib.sha256(source).hexdigest() != SHA256:
        with urlopen(URL, timeout=120) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != SHA256:
            raise ValueError("SciKnowEval rubric source checksum mismatch")
        path.write_bytes(source)
    rubrics = yaml.safe_load(source)
    # Upstream uses {prompt}, which renders an instruction dict instead of the problem.
    for name in ("formula_derivation", "problem_solving"):
        rubrics[name]["user"] = rubrics[name]["user"].replace("{prompt}", "{question}")
    # Preserve the documented evaluation protocol; these two scores differ from upstream's
    # word-vector relation scorer. All other rubric text is unchanged.
    for name, description in RELATION_TASKS.items():
        rubric = copy.deepcopy(rubrics["extract_doping"])
        rubric["user"] = re.sub(
            r"The question involves .*?\. Your task",
            f"The question involves {description}. Your task",
            rubric["user"],
            count=1,
        )
        rubrics[name] = rubric
    rubrics.pop("property_and_usage_analysis", None)  # Not used by V2.
    return rubrics
