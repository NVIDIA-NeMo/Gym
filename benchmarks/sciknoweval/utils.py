# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared routing tables for SciKnowEval (https://github.com/HICAI-ZJU/SciKnowEval).

SciKnowEval ships one flat JSONL with six answer types. `SUB_BENCHMARKS` maps them
to five prompt families; both multiple-choice types share the MCQ template.
Preparation combines all families into one dataset with per-row verifier metadata.
"""

from typing import Any


DATASET_NAME = "hicai-zju/SciKnowEval"
# V2 (Jul 2025): 28,392 samples / 58 tasks. Identical to raw_data/ in the upstream repo.
DATA_FILE = "data/v2/sciknoweval_test_v2.jsonl"


# answer `type` -> sub-benchmark directory
SUB_BENCHMARKS = {
    "mcq-4-choices": "mcq",
    "mcq-2-choices": "mcq",
    "true_or_false": "true_false",
    "filling": "filling",
    "relation_extraction": "relation_extraction",
    "open-ended-qa": "open_ended",
}

DOMAIN_SPLITS = {"Biology": "biology", "Chemistry": "chemistry", "Material": "material", "Physics": "physics"}

# L1 memory and L2 comprehension are recall-style; L3 reasoning, L4 safety/ethics and L5
# application are the levels worth spending a reasoning model's budget on. Written as a separate
# split so the full benchmark stays intact under test.jsonl.
LEVEL_SPLIT_NAME = "test_l345"
LEVEL_SPLIT_LEVELS = ("L3", "L4", "L5")

# Upstream tells the model to answer with no explanation and no other characters, which makes the
# benchmark unusable for reasoning models: `get_single_score_MCQ` in evaluation/metrics.py only
# handles a one-character response cleanly, and otherwise scans for "D."/"C."/"B."/"A." in that
# order and returns on the first hit anywhere in the text, so a chain of thought that discusses
# and rejects D is scored as D.
#
# These phrases are removed and the answer-format contract is re-stated by the prompt config
# instead (`Answer: ...` on the last line, as in eval/aai/mcq-4choices). Everything else in the
# instruction - the role framing, the required list/CSV format, "summarize in one sentence" - is
# task-defining and is kept verbatim. `original_instruction` keeps the untouched string.
#
# Exact substrings, not regexes, so that this stays auditable: there are only 30 distinct
# instructions in the whole dataset and `test_muzzle_stripping` in prepare.py checks that none
# of them still contains a muzzle after stripping.
MUZZLE_PHRASES = {
    "Please directly give the answer without any explanation.": "",
    "Please directly give the answer, DO NOT output any other characters.": "",
    "Directly give me the list, DO NOT output any other characters.": "",
    "Do not output any other characters.": "",
    # infix forms, where a clause of the sentence has to survive
    ", do not output other characters.": ".",
    " without any explanation": "",
}

# Answer-format sentences that the prompt config now owns. Dropped so the model is not given two
# conflicting output contracts (e.g. one true/false task asks for "true"/"false" while the gold
# labels are "Yes"/"No" - upstream's grader silently maps between them).
FORMAT_SENTENCES = [
    'Your answer should be "A", "B", "C" or "D".',
    'Your answer should be "A" or "B".',
    'Your answer should be "Yes" or "No".',
    'Your answer should be "true" or "false".',
]

# details.task -> judge rubric in judge_prompts.yaml.
# Mirrors `task_trans_dict` in upstream evaluation/metrics.py, plus the two relation-extraction
# rubrics that replace the word2vec scorer (see judge_prompts.yaml for why).
TASK_TO_RUBRIC = {
    # open-ended-qa
    "L2_General": "text_summary",
    "L2_Material": "csv_extraction",
    "harmful_QA": "harmful_QA",
    "procedure_generation": "procedure_generation",
    "reagent_generation": "reagent_generation",
    "crystal_structure_and_composition_analysis": "crystal_design",
    "specified_band_gap_material_generation": "material_generation",
    "physics_formula_derivation": "formula_derivation",
    "physics_problem_solving": "problem_solving",
    # relation_extraction
    "L2_Chemistry": "extract_doping",
}
# the two biology relation-extraction tasks share details.task == "L2_Biology", so they are
# routed by subtask instead
SUBTASK_TO_RUBRIC = {
    "drug_drug_relation_extraction": "drug_drug_relation_extraction",
    "compound_disease_relation_extraction": "compound_disease_relation_extraction",
}


def strip_muzzle(instruction: str) -> str:
    """Remove the 'answer with nothing else' directives and the answer-format sentences."""
    for phrase, replacement in MUZZLE_PHRASES.items():
        instruction = instruction.replace(phrase, replacement)
    for sentence in FORMAT_SENTENCES:
        instruction = instruction.replace(sentence, "")
    return " ".join(instruction.split())


def get_rubric_name(record: dict[str, Any]) -> str:
    subtask = record["details"].get("subtask", "")
    if subtask in SUBTASK_TO_RUBRIC:
        return SUBTASK_TO_RUBRIC[subtask]
    task = record["details"]["task"]
    if task not in TASK_TO_RUBRIC:
        raise KeyError(f"No judge rubric for task={task!r} subtask={subtask!r}")
    return TASK_TO_RUBRIC[task]
