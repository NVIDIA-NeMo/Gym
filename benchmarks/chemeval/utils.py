# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared routing tables for ChemEval (https://github.com/USTC-StarTeam/ChemEval).

The HuggingFace release is one flat table whose only task label is a `filename` column holding
the path the question came from in the authors' working tree - Chinese task names, sometimes
inside a Windows path that also encodes the level and the capability dimension:

    选择任务.json
    BBBP_test.json
    3.分子理解\\3.分子性质预测\\2.6.1分子性质回归预测熔点_自建\\熔点_test.json
    3shot_4.科学知识推演\\2.反应条件推荐\\4.催化剂推荐_自建\\催化剂推荐_test_3shot.json

`task_key` normalizes that to one stable key per task, and `TASKS` maps every key to the level,
the dimension, and the grader family. The grader assignment mirrors `split_jsonl_by_filename` in
upstream's `Textual/code evaluate/2_Extract.py`, which is the only place the benchmark writes
down which metric each task is scored with.
"""

import re


DATASET_NAME = "Ooo1/ChemEval"
# The textual release; prepare.py excludes three-shot and multimodal questions.
DATA_FILE = "data/text-00000-of-00001.parquet"


# The four progressive levels of the paper, in order.
LEVELS = ["L1", "L2", "L3", "L4"]
LEVEL_NAMES = {
    "L1": "Advanced Knowledge Question Answering",
    "L2": "Literature Understanding",
    "L3": "Molecular Understanding",
    "L4": "Scientific Knowledge Deduction",
}

# Grader family -> preparation group within the single ChemEval dataset.
#
# Families are grouped by what running them costs and by what their score means, not by how many
# of them there are:
#   - `judged` is the only group that needs an LLM judge, so keeping it apart means 1,760 of the
#     2,210 questions are never sent to one;
#   - `regression` is scored from a numeric error rather than from a match, and its raw error
#     carries the unit of the task (log-solubility, Kelvin, hours, ...), so it is kept out of the
#     pooled rule-based number;
#   - `mcq` and `true_false` are the only tasks whose prompt has to supply an answer format, since
#     they are the only ones whose question text does not already state one.
SUB_BENCHMARKS = {
    "mcq": "mcq",
    "true_false": "true_false",
    "classification": "rule_based",
    "classification_subset": "rule_based",
    "entity_extraction": "rule_based",
    "entity_recognition": "rule_based",
    "relation_extraction": "rule_based",
    "reagent_selection": "rule_based",
    "sider": "rule_based",
    "molecule_smiles": "rule_based",
    "molecule_formula": "rule_based",
    "molecule_iupac": "rule_based",
    "range_overlap": "rule_based",
    "regression": "regression",
    "judged": "judged",
}

# task_key -> (level, capability dimension, grader family, judge rubric or None)
#
# The dimension names are the paper's 13. For the 23 in-house tasks they are also spelled out in
# the `filename` path (`2.文献理解\\1.信息抽取\\...` is L2 / Information Extraction); for the 25
# adapted open-source tasks and the 5 L1 tasks the path is flat and the dimension comes from the
# paper's task table.
TASKS = {
    # --- L1: Advanced Knowledge Question Answering -------------------------------------------
    "选择任务": ("L1", "Objective Questions", "mcq", None),
    "判断任务": ("L1", "Objective Questions", "true_false", None),
    "填空任务": ("L1", "Subjective Questions", "judged", "fill_in_the_blank"),
    "简答任务": ("L1", "Subjective Questions", "judged", "short_answer"),
    "计算任务": ("L1", "Subjective Questions", "judged", "calculation"),
    # --- L2: Literature Understanding ---------------------------------------------------------
    "化学论文摘要生成": ("L2", "Inductive Generation", "judged", "abstract_generation"),
    "研究内容提纲生成": ("L2", "Inductive Generation", "judged", "outline_generation"),
    # upstream scores this one with `gold in prediction` rather than equality, because the label
    # set contains names that are prefixes of each other ("Chemistry Education" / "Chemistry")
    "化学文献主题分类": ("L2", "Inductive Generation", "classification_subset", None),
    "化学反应类型识别归纳": ("L2", "Inductive Generation", "entity_extraction", None),
    "催化类型抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "产率性能抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "合成反应添加剂抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "合成反应溶剂抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "合成反应温度抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "合成反应时间抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "表征手段抽取": ("L2", "Information Extraction", "entity_extraction", None),
    "化学实体关系分类": ("L2", "Information Extraction", "relation_extraction", None),
    "合成反应产物抽取": ("L2", "Information Extraction", "entity_recognition", None),
    "合成反应底物抽取": ("L2", "Information Extraction", "entity_recognition", None),
    "化学命名实体识别": ("L2", "Molecular Name Recognition", "entity_extraction", None),
    # --- L3: Molecular Understanding ----------------------------------------------------------
    "基于文本描述生成分子名称": ("L3", "Molecular Name Generation", "molecule_smiles", None),
    "IUPAC转SMILES": ("L3", "Molecular Name Translation", "molecule_smiles", None),
    "SMILES与SELFIES互译": ("L3", "Molecular Name Translation", "molecule_smiles", None),
    "SMILES转IUPAC": ("L3", "Molecular Name Translation", "molecule_iupac", None),
    "IUPAC转分子式": ("L3", "Molecular Name Translation", "molecule_formula", None),
    "SMILES转分子式": ("L3", "Molecular Name Translation", "molecule_formula", None),
    "BBBP": ("L3", "Molecular Property Prediction", "classification", None),
    "ClinTox": ("L3", "Molecular Property Prediction", "classification", None),
    "HIV": ("L3", "Molecular Property Prediction", "classification", None),
    "SIDER": ("L3", "Molecular Property Prediction", "sider", None),
    "ESOL": ("L3", "Molecular Property Prediction", "regression", None),
    "HOMO": ("L3", "Molecular Property Prediction", "regression", None),
    "LUMO": ("L3", "Molecular Property Prediction", "regression", None),
    "Lipo": ("L3", "Molecular Property Prediction", "regression", None),
    "分子性质分类预测极性": ("L3", "Molecular Property Prediction", "classification", None),
    "分子性质回归预测极性": ("L3", "Molecular Property Prediction", "regression", None),
    "分子性质回归预测熔点": ("L3", "Molecular Property Prediction", "regression", None),
    "分子性质回归预测沸点": ("L3", "Molecular Property Prediction", "regression", None),
    "基于分子结构描述分子的物理化学性质": ("L3", "Molecular Description", "judged", "physicochemical"),
    # --- L4: Scientific Knowledge Deduction ---------------------------------------------------
    "反应底物推荐": ("L4", "Retrosynthetic Analysis", "reagent_selection", None),
    "单步合成路径推荐": ("L4", "Retrosynthetic Analysis", "judged", "single_step_synthesis"),
    "多步合成路径推荐": ("L4", "Retrosynthetic Analysis", "judged", "multi_step_synthesis"),
    "合成难度评估": ("L4", "Retrosynthetic Analysis", "regression", None),
    "试剂推荐": ("L4", "Reaction Condition Recommendation", "reagent_selection", None),
    "溶剂推荐": ("L4", "Reaction Condition Recommendation", "reagent_selection", None),
    "配体推荐": ("L4", "Reaction Condition Recommendation", "reagent_selection", None),
    "催化剂推荐": ("L4", "Reaction Condition Recommendation", "reagent_selection", None),
    "反应温度推荐": ("L4", "Reaction Condition Recommendation", "regression", None),
    "反应时间推荐": ("L4", "Reaction Condition Recommendation", "regression", None),
    "反应产物预测": ("L4", "Reaction Outcome Prediction", "reagent_selection", None),
    "产物产率预测": ("L4", "Reaction Outcome Prediction", "classification", None),
    "反应活化能刻画": ("L4", "Reaction Outcome Prediction", "range_overlap", None),
    "反应中间体推导": ("L4", "Reaction Mechanism Analysis", "judged", "reaction_intermediate"),
}

# English names, for readers of the output rows and of the metrics tables. Only used for display.
TASK_NAMES_EN = {
    "选择任务": "Multiple Choice",
    "判断任务": "True or False",
    "填空任务": "Fill in the Blank",
    "简答任务": "Short Answer",
    "计算任务": "Calculation",
    "化学论文摘要生成": "Chemical Paper Abstract Generation",
    "研究内容提纲生成": "Research Outline Generation",
    "化学文献主题分类": "Chemical Literature Topic Classification",
    "化学反应类型识别归纳": "Reaction Type Recognition and Induction",
    "催化类型抽取": "Catalysis Type Extraction",
    "产率性能抽取": "Yield Extraction",
    "合成反应添加剂抽取": "Synthetic Reaction Additive Extraction",
    "合成反应溶剂抽取": "Synthetic Reaction Solvent Extraction",
    "合成反应温度抽取": "Reaction Temperature Extraction",
    "合成反应时间抽取": "Reaction Time Extraction",
    "表征手段抽取": "Characterization Method Extraction",
    "化学实体关系分类": "Chemical Entity Relationship Classification",
    "合成反应产物抽取": "Reaction Product Extraction",
    "合成反应底物抽取": "Synthetic Reaction Substrate Extraction",
    "化学命名实体识别": "Chemical Named Entity Recognition",
    "基于文本描述生成分子名称": "Molecular Name Generation from Text Description",
    "IUPAC转SMILES": "IUPAC to SMILES",
    "SMILES与SELFIES互译": "SMILES and SELFIES Conversion",
    "SMILES转IUPAC": "SMILES to IUPAC",
    "IUPAC转分子式": "IUPAC to Molecular Formula",
    "SMILES转分子式": "SMILES to Molecular Formula",
    "BBBP": "BBBP",
    "ClinTox": "ClinTox",
    "HIV": "HIV",
    "SIDER": "SIDER",
    "ESOL": "ESOL",
    "HOMO": "HOMO",
    "LUMO": "LUMO",
    "Lipo": "Lipo",
    "分子性质分类预测极性": "Molecular Property Classification (Polarity)",
    "分子性质回归预测极性": "Molecular Property Regression (Polarity)",
    "分子性质回归预测熔点": "Molecular Property Regression (Melting Point)",
    "分子性质回归预测沸点": "Molecular Property Regression (Boiling Point)",
    "基于分子结构描述分子的物理化学性质": "Physicochemical Property Description from Structure",
    "反应底物推荐": "Substrate Recommendation",
    "单步合成路径推荐": "Single-step Synthetic Pathway Recommendation",
    "多步合成路径推荐": "Multi-step Synthetic Pathway Recommendation",
    "合成难度评估": "Synthetic Difficulty Evaluation",
    "试剂推荐": "Reagent Recommendation",
    "溶剂推荐": "Solvent Recommendation",
    "配体推荐": "Ligand Recommendation",
    "催化剂推荐": "Catalyst Recommendation",
    "反应温度推荐": "Reaction Temperature Recommendation",
    "反应时间推荐": "Reaction Time Recommendation",
    "反应产物预测": "Reaction Product Prediction",
    "产物产率预测": "Product Yield Prediction",
    "反应活化能刻画": "Reaction Activation Energy Characterization",
    "反应中间体推导": "Reaction Intermediate Derivation",
}

# Upstream's answer-suppressing directives, removed so that the benchmark is usable by a reasoning
# model. Exact substrings, not regexes, so this stays auditable and cannot fire on question text -
# "without any" and "explanation" both occur inside real chemistry passages in this dataset.
# `check_muzzle_stripping` in conversion.py fails preparation if a new one appears.
#
# What is *not* removed is the answer format each question states, usually `{"answer": "..."}`:
# that is the contract the graders parse, and it is restated rather than dropped, with the added
# sentence in `ANSWER_FORMAT_INSTRUCTIONS` pinning it to the end of the response so that it can be
# lifted out of a chain of thought.
MUZZLE_PHRASES = {
    "I don't need any explanation, you just need to output your judgment in format.": "",
    "I don't need any explanation, you just need to output your judgment in the format.": "",
    "I don't need any explanations, you just need to output the answer according to the format.": "",
    "I don't need any explanations, just output the answer according to the format.": "",
    "I don't need anything unrelated to this, you just need to output as required.": "",
    "Please strictly follow the format, no other information can be provided.": "",
    "You only need to output the requested content, do not need to output other explanations and introductions.": "",
    "You only need to output the requested content, no additional explanation content.": "",
    "You only need to provide your prediction without any additional explanation or introduction.": "",
    "You simply output as required, no additional explanatory information is required.": "",
    # infix forms, where the rest of the sentence is a real format instruction and has to survive
    "You do not need to output too much explanation, you must output a number": "You must output a number",
    "Please only provide the missing part in your response, without any additional content.": (
        "Please only provide the missing part."
    ),
}

# Markers that mean a muzzle survived stripping. Checked against the *stripped* instruction in
# conversion.py; the phrases above cover the pinned source release.
MUZZLE_MARKERS = (
    "don't need any explanation",
    "no other information can be provided",
    "do not need to output other explanation",
    "no additional explanat",
    "without any additional explanation",
    "too much explanation",
)

# Appended to the question, per sub-benchmark, to say where the answer goes. Only `mcq` and
# `true_false` introduce a format: their questions are bare ("Please answer the following
# questions: ...") and upstream recovers the answer by prompting a second LLM to extract it. For
# every other family the question already states the format, so this only pins it to the last line.
ANSWER_FORMAT_INSTRUCTIONS = {
    "mcq": ("The last line of your response should be in the following format: 'Answer: X' (e.g. 'Answer: C')."),
    "true_false": (
        "The last line of your response should be in the following format: 'Answer: Correct' or 'Answer: Incorrect'."
    ),
    "judged_open": "",
    "formatted": (
        "You may reason through the problem first. The last line of your response must be the "
        "answer in exactly the format requested above, and nothing else."
    ),
}


# Stable task slugs stored alongside the level in `subset_for_metrics`.
# The resource server's compute_metrics groups rewards by the original task key
# and derives task, level, dimension, and family averages from verifier metadata.
TASK_SLUGS = {
    key: re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_") for key, name in TASK_NAMES_EN.items()
}
assert len(set(TASK_SLUGS.values())) == len(TASKS), "task slugs must stay unique"
assert set(TASK_SLUGS) == set(TASKS), "TASK_NAMES_EN and TASKS must cover the same tasks"


def task_key(filename: str) -> str:
    """Map a raw `filename` value to the stable task key used by `TASKS`.

    The 23 in-house tasks carry a Windows path whose *parent directory* is the task name, prefixed
    with its numbering and suffixed with 自建 ("in-house") - `...\\2.6.1分子性质回归预测熔点_自建\\熔点_test.json`
    is the melting-point task, and the leaf name alone (熔点) collides with the boiling-point
    task's sibling. The other 30 are a flat file name.
    """
    name = filename.removeprefix("3shot_")
    parts = name.replace("\\", "/").split("/")
    if len(parts) > 1:
        return re.sub(r"_自建$", "", re.sub(r"^[\d.]+", "", parts[-2]))
    leaf = re.sub(r"\.jsonl?$", "", parts[-1])
    return re.sub(r"_test(_3shot)?$|_3shot$", "", leaf)


def is_three_shot(filename: str) -> bool:
    """Whether a row is from the three-shot variant of its task.

    Almost all of them are marked by a `3shot_` prefix, but 反应活化能刻画 is marked only by the
    `_3shot` suffix on the leaf file name, so both have to be checked.
    """
    return filename.startswith("3shot_") or "_3shot" in filename


# The last number in a string, which is where every numeric answer in this benchmark ends up:
# "{'answer': '3.9'}", "56 ℃", "110K". Upstream's `extract_number` is the same expression without
# the lookbehind and the exponent, so it reads "180-370" as -370 (a minus that follows a digit
# separates a range, it does not negate what comes after it) and "8.8e-2" as -2.
NUMBER = re.compile(r"(?<![\d.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?=\D*$)")


def extract_number(text: object) -> float | None:
    """The last number in `text`, or None if it holds none."""
    if text is None:
        return None
    match = NUMBER.search(str(text))
    return float(match.group()) if match else None


def strip_muzzle(instruction: str) -> str:
    """Remove the 'answer with nothing else' directives, leaving the answer format in place."""
    for phrase, replacement in MUZZLE_PHRASES.items():
        instruction = instruction.replace(phrase, replacement)
    # collapse the double spaces and dangling newlines the removals leave behind, without
    # flattening the questions that are laid out over several lines
    instruction = re.sub(r"[ \t]{2,}", " ", instruction)
    instruction = re.sub(r"\n{3,}", "\n\n", instruction)
    return "\n".join(line.rstrip() for line in instruction.split("\n")).strip()
