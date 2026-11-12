# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemEval answer extraction and grading; no external scorer file is needed."""

import ast
import json
import re
from collections import Counter
from typing import Any


SIDER_LABELS = [
    "Hepatobiliary disorders",
    "Metabolism and nutrition disorders",
    "Eye disorders",
    "Musculoskeletal and connective tissue disorders",
    "Gastrointestinal disorders",
    "Immune system disorders",
    "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Endocrine disorders",
    "Vascular disorders",
    "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders",
    "Congenital, familial and genetic disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders",
    "Cardiac disorders",
    "Nervous system disorders",
]

ENTITY_ALIASES = {
    "化学命名实体识别": {
        "H ( 2 ) O ( 2 ), SiO ( 2 )": "H2O2, SiO2",
        "4 - hydroxy - 1 - ( 3 - pyridyl ) - 1 - butanone (HPB)": "HPB",
        "side chains of compound ( - ) - 1a (SCH 900229)": "SCH 900229",
        "nitric oxide (( . ) NO)": "( . ) NO",
        "4 - hydroxy - 1 - ( 3 - pyridyl ) - 1 - butanone": "HPB",
        "side chains of compound ( - ) - 1a": "SCH 900229",
        "nitric oxide": "( . ) NO",
        "HAND - SANT - SLIDE": "HSS",
        "glucose - dependent insulinotropic polypeptide": "GIP",
        "monoacylglycerol lipase": "MAGL",
        "short - chain fatty acid": "SCFA",
        "total polyphenols": "TP",
        "structure activity relationships": "SAR",
        " life and octanol - water distribution coefficient": "log D",
        "glycyrrhetinic acid": "GA",
        "Lysyl oxidase": "LO",
        "Manganese": "Mn",
    },
    "化学实体关系分类": {
        "tranexamic acid (tAMCA)": "tAMCA",
        "thrombotic microangiopathy (TMA)": "TMA",
        "trimethaphan (TMP)": "TMP",
        "prostaglandin E1 (PGE1)": "PGE1",
        "calcium chloride (CaCl(2))": "CaCl(2)",
        "thoracic aortic aneurysm (TAA)": "TAA",
        "tranexamic acid": "tAMCA",
        "thrombotic microangiopathy": "TMA",
        "trimethaphan": "TMP",
        "prostaglandin E1": "PGE1",
        "calcium chloride": "CaCl(2)",
        "thoracic aortic aneurysm": "TAA",
        "counter cyanide": "CN",
        "acetylcholine": "ACh",
        "organophosphorus": "OP",
        "diisopropylfluorophosphate": "DFP",
        "acetylcholinesterases": "AChEs",
        "butyrylcholinesterases": "BChEs",
        "N-methyl-D-aspartate": "NMDA",
        "venous thromboembolism": "VTE",
        "antiepileptic drug": "AED",
        "Carbamazepine": "CBZ",
        "gamma-Vinyl GABA": "GVG",
        "Parkinson's disease": "PD",
        "metalloproteinase": "ADAM",
        "metalloproteinases": "MMPs",
    },
    "合成反应添加剂抽取": {
        "pivalic acid (PivOH)": "PivOH",
        "2,2,6,6-tetramethylpiperidinooxy (TEMPO)": "TEMPO",
        "pivalic acid": "PivOH",
        "2,2,6,6-tetramethylpiperidinooxy": "TEMPO",
    },
    "合成反应溶剂抽取": {
        "N,N-dimethylformamide (DMF)": "DMF",
        "1,1,1,3,3,3-hexafluoroisopropanol (HFIP)": "HFIP",
        "N,N-dimethylformamide": "DMF",
        "1,1,1,3,3,3-hexafluoroisopropanol": "HFIP",
    },
}

NUMBER = re.compile(r"(?<![\d.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?=\D*$)")


def extract_number(text: object) -> float | None:
    """The last number in `text`, or None if it holds none."""
    if text is None:
        return None
    match = NUMBER.search(str(text))
    return float(match.group()) if match else None


ANSWER_LINE = re.compile(r"(?im)^[\s>*_]*answer[\s*_]*:[\s*_]*(.+?)\s*$")

YES_NO = re.compile(r"(?i)\b(yes|no)\b")

CORRECT_INCORRECT = re.compile(r"(?i)\b(?P<negation>(?:not\s+)*)(?P<verdict>incorrect|correct|false|true|no|yes)\b")

MCQ_LETTER = re.compile(r"\b([A-D])\b")

TUPLE = re.compile(r"\([^()]*\)")

FORMULA_ATOM = re.compile(r"([A-Z][a-z]*)(\d*)")

SELFIES_LIKE = re.compile(r"^(?:\[[^\[\]]+\]|\.)+$")


def _balanced_json_objects(text: str):
    """Yield every balanced `{...}` span in the text, outermost first, left to right.

    Upstream's `extract_balanced_json_string` returns only the first one, which picks up a JSON
    block quoted from the question before the model's own answer.
    """
    depth, start = 0, None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]


PARSE_ERRORS = (ValueError, SyntaxError, TypeError, MemoryError, RecursionError)


def _parse_answer_object(span: str):
    """Read the `answer` value out of one `{...}` span, or None if it is not an answer object."""
    # several questions ask for a single-quoted, Python-literal style dict, so json is not enough
    for loads in (json.loads, ast.literal_eval):
        try:
            parsed = loads(span)
        except PARSE_ERRORS:
            continue
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                # the questions ask for `{"answer": ...}` but write the key with stray spaces in
                # several tasks (`{' answer ': ' effective catalyst '}`)
                if str(key).strip().strip("\"'").lower() == "answer":
                    return value
    return None


def extract_answer(generation: str) -> object:
    """Pull the model's answer out of the response.

    In order: the `answer` field of the last balanced JSON object, the last `Answer:` line, the
    last non-empty line. Returns None when the response is empty.
    """
    if not generation or not generation.strip():
        return None

    for span in reversed(list(_balanced_json_objects(generation))):
        value = _parse_answer_object(span)
        if value is not None:
            return value

    lines = ANSWER_LINE.findall(generation)
    if lines:
        return lines[-1].strip()

    tail = [line.strip() for line in generation.strip().splitlines() if line.strip()]
    return tail[-1] if tail else None


def as_text(value: object) -> str:
    """Flatten an extracted answer to the string the graders compare.

    An answer may come back as a list (`{"answer": ["CCO", "CCN"]}`) or a dict (SIDER), because
    that is what some questions ask for; upstream's extractors do the same flattening.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ".".join(as_text(item) for item in value)
    if isinstance(value, dict):
        return "".join(as_text(item) for item in value.values())
    return str(value)


def parse_literal(text: object) -> object:
    """Best-effort parse of a gold or predicted value that may be a Python literal string."""
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(text)
    except PARSE_ERRORS:
        return text


def set_f1(gold: set, predicted: set) -> float:
    """Upstream's `calculate_f1_score`, shared by every extraction family.

    An empty set on either side scores 0, as it does upstream: no ChemEval question has an empty
    answer, so an empty gold means the reference could not be parsed and an empty prediction means
    the model's could not be.
    """
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_range(text: str) -> tuple[float, float] | None:
    """Upstream's `parse_range`: the first two numbers of a string, as (low, high)."""
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text.replace("-", " "))
    if len(numbers) < 2:
        return None
    low, high = float(numbers[0]), float(numbers[1])
    return (low, high) if low <= high else (high, low)


def parse_formula(formula: str) -> dict[str, int]:
    """Upstream's `parse_molecular_formula`: element symbol -> atom count."""
    counts = Counter()
    for element, count in FORMULA_ATOM.findall(formula):
        counts[element] += int(count) if count else 1
    return dict(counts)


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, so that the IUPAC family does not need the `Levenshtein` package."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (lchar != rchar)))
        previous = current
    return previous[-1]


def canonical_smiles(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the string is not a parsable molecule."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def tanimoto(predicted: str, gold: str) -> float:
    """Morgan (radius 2, 2048 bits) Tanimoto similarity, as in upstream's `smiles_to_fps`."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    gold_molecule = Chem.MolFromSmiles(gold) if gold else None
    predicted_molecule = Chem.MolFromSmiles(predicted) if predicted else None
    if gold_molecule is None or predicted_molecule is None:
        return 0.0
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        for molecule in (predicted_molecule, gold_molecule)
    ]
    return DataStructs.TanimotoSimilarity(*fingerprints)


def normalize_molecule_text(value: object) -> str:
    """Flatten a gold or predicted molecule field to a '.'-separated string of molecules.

    Several tasks write the gold as a Python list literal (`"['CCO','CCN']"`), some use ';' as the
    fragment separator and the catalyst task writes two molecules joined by " and ". Upstream's
    `design_S` and `select` normalize exactly these three.
    """
    text = as_text(parse_literal(value) if isinstance(value, str) else value)
    return text.replace(" and ", ".").replace(" ", "").replace(";", ".")


def resolve_selfies(gold: str, predicted: str) -> tuple[str, str]:
    """If the gold is written as SELFIES, decode both sides to SMILES.

    Which notation a question uses is not recorded anywhere in the release: the SMILES/SELFIES
    task runs in both directions inside one file, and 24 of the reaction-product and substrate
    answers are SELFIES while the rest of that task is SMILES. Upstream keys this off the row
    index (`i < 25`) in one place and off `sf.decoder` returning a non-empty string in the other;
    the second test is used here, guarded by the shape of a SELFIES string so that it cannot fire
    on a SMILES.
    """
    if not SELFIES_LIKE.match(gold):
        return gold, predicted
    decoded_gold = decode_selfies(gold)
    if decoded_gold is None:
        return gold, predicted
    return decoded_gold, decode_selfies(predicted) or ""


def decode_selfies(text: str) -> str | None:
    """SELFIES -> SMILES, or None if it does not decode.

    Only reached by the SMILES/SELFIES task, so `selfies` is imported here rather than at module
    level and the error names the package to install.
    """
    try:
        import selfies
    except ImportError as error:  # pragma: no cover - depends on the eval environment
        raise ImportError(
            "The ChemEval SMILES/SELFIES task needs the `selfies` package. Install it alongside "
            "rdkit, e.g. --installation_command='pip install rdkit selfies'."
        ) from error
    try:
        decoded = selfies.decoder(text)
    except (selfies.DecoderError, ValueError, TypeError, IndexError, KeyError):
        return None
    return decoded or None


def grade_mcq(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Single-letter match. Upstream prompts a second LLM to pull the letter out of the response."""
    text = as_text(answer).strip()
    # Prefer the leading answer label over letters in its explanation (e.g. "a tropane").
    leading = re.match(
        r"(?i)^[\s*_`$]*(?:(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*)?"
        r"(?:(?:option|choice)\s+)?[\s(*_`$]*([A-D])\b",
        text,
    )
    if leading:
        # A single-choice answer cannot name alternatives or several labels.
        alternatives = re.match(r"(?i)^[)\s*_`$]*(?:,|/|or\b|and\b)\s*\(?[A-D]\b", text[leading.end() :])
        predicted = None if alternatives else leading.group(1).upper()
    else:
        # Keep case here: the article "a" is not an option label.
        letters = set(MCQ_LETTER.findall(text))
        predicted = next(iter(letters)) if len(letters) == 1 else None
    return float(predicted == sample["expected_answer"].strip().upper()), {"predicted_letter": predicted}


def grade_true_false(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Correct/Incorrect match, accepting the yes/no and true/false synonyms upstream maps."""
    match = CORRECT_INCORRECT.search(as_text(answer))
    if match is None:
        return 0.0, {"predicted_verdict": None}
    positive = match.group("verdict").lower() in ("correct", "true", "yes")
    if len(match.group("negation").split()) % 2:
        positive = not positive
    predicted = "Correct" if positive else "Incorrect"
    return float(predicted == sample["expected_answer"].strip()), {"predicted_verdict": predicted}


def grade_classification(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Yes/No match. A response containing both words counts as no answer, as upstream does."""
    words = {word.lower() for word in YES_NO.findall(as_text(answer))}
    predicted = None if words != {"yes"} and words != {"no"} else words.pop().capitalize()
    return float(predicted == sample["expected_answer"].strip()), {"predicted_label": predicted}


def grade_classification_subset(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Upstream's `calculate_accuracy2`: the gold label has to appear inside the answer.

    Case-insensitive here, where upstream compares case-sensitively - the 15 topic names are
    listed verbatim in the question, so casing is not part of what is being tested.
    """
    text = as_text(answer)
    return float(sample["expected_answer"].strip().lower() in text.lower()), {}


def _entity_normalize(task: str, gold: str, predicted: str):
    """The per-task rewrites upstream applies to both sides before the F1, in `entity_extract`."""
    for full, abbreviation in ENTITY_ALIASES.get(task, {}).items():
        gold = gold.replace(full, abbreviation)
        predicted = predicted.replace(full, abbreviation)

    if task in ("催化类型抽取", "化学反应类型识别归纳"):
        # upstream also turns ',' into '.' here, which means the answer is compared as one item
        for text in ("reactions", "reaction"):
            gold, predicted = gold.replace(text, ""), predicted.replace(text, "")
        gold, predicted = gold.replace(",", "."), predicted.replace(",", ".")
    elif task == "合成反应温度抽取":
        for unit in ("℃", "°C"):
            gold, predicted = gold.replace(unit, ""), predicted.replace(unit, "")
        predicted = predicted.replace("degrees Celsius", "")
    elif task == "合成反应时间抽取":
        gold = gold.replace("h", "")
        minutes = re.search(r"([\d.]+)\s*min", predicted)
        if minutes and "h" not in predicted.replace("hours", "").replace("hour", ""):
            predicted = f"{float(minutes.group(1)) / 60:g}"
        else:
            for text in ("hours", "hour", "h", ".0"):
                predicted = predicted.replace(text, "")
    elif task == "产率性能抽取":
        predicted = predicted.replace("answer:", "").replace("percent", "%").replace("% to ", "-").replace("%-", "-")
    return gold, predicted


def grade_entity_extraction(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Set F1 over the comma-separated entity list, lowercased and whitespace-stripped."""

    def to_set(text):
        return {item for item in text.lower().replace(" ", "").split(",") if item}

    # Entity lists are comma-separated, unlike molecule lists joined by as_text.
    parsed = parse_literal(answer)
    if isinstance(parsed, (list, tuple)):
        predicted = ",".join(parsed) if all(isinstance(item, str) for item in parsed) else ""
    else:
        predicted = as_text(answer)
    gold, predicted = _entity_normalize(sample["task"], sample["expected_answer"], predicted)
    score = set_f1(to_set(gold), to_set(predicted))
    return score, {"entity_f1": score}


def grade_relation_extraction(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Set F1 over the `(subject, object)` pairs found in the answer."""

    def to_set(text):
        return set(TUPLE.findall(text.lower().replace(" ", "")))

    score = set_f1(to_set(sample["expected_answer"]), to_set(as_text(answer)))
    return score, {"relation_f1": score}


def grade_entity_recognition(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Per-token accuracy over the BIO tag list, against the gold list's length.

    Upstream reports this micro-averaged over all tokens of a task; averaging the per-question
    numbers, as the metrics do here, weights every sentence equally instead. `bio_tokens` is
    written per row so the micro average stays recoverable.
    """
    gold = parse_literal(sample["expected_answer"])
    if not isinstance(gold, list):
        return 0.0, {"bio_tokens": 0, "bio_correct": 0}
    predicted = parse_literal(answer if not isinstance(answer, str) else answer.strip())
    if not isinstance(predicted, list):
        # Prompts also allow unquoted BIO tags, e.g. [O, B-X]. Keep literal-list
        # parsing first, then recover only the last bracketed answer list.
        brackets = re.findall(r"\[([^\[\]]*)\]", answer) if isinstance(answer, str) else []
        predicted = [tag.strip().strip("\"'").strip() for tag in brackets[-1].split(",")] if brackets else []
    correct = sum(1 for g, p in zip(gold, predicted) if isinstance(p, str) and p.strip() == g)
    return correct / len(gold), {"bio_tokens": len(gold), "bio_correct": correct}


def grade_reagent_selection(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Set F1 over canonical SMILES, splitting both sides on '.' as upstream's `select` does."""

    def to_set(text):
        return {smiles for smiles in (canonical_smiles(part) for part in text.split(".")) if smiles}

    gold, predicted = resolve_selfies(
        normalize_molecule_text(sample["expected_answer"]), normalize_molecule_text(answer)
    )
    gold_set = to_set(gold)
    score = set_f1(gold_set, to_set(predicted))
    # 25 of the 270 gold answers do not parse as molecules at all, which caps this task below 1.0.
    # Upstream `break`s out of its loop on the first of them, silently truncating the task.
    return score, {"smiles_f1": score, "gold_parsed": bool(gold_set)}


def grade_sider(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Fraction of the 20 side-effect labels answered with the gold Yes/No."""
    gold = parse_literal(sample["expected_answer"])
    predicted = parse_literal(answer) if isinstance(answer, str) else answer
    if not isinstance(gold, dict):
        return 0.0, {"sider_correct": 0}
    if not isinstance(predicted, dict):
        return 0.0, {"sider_correct": 0}
    lowered = {str(key).strip().lower(): value for key, value in predicted.items()}
    correct = sum(1 for label in SIDER_LABELS if str(lowered.get(label.lower(), "")).strip() == gold.get(label))
    return correct / len(SIDER_LABELS), {"sider_correct": correct}


def grade_molecule_smiles(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Tanimoto similarity to the gold molecule, after decoding whichever side is SELFIES."""
    gold, predicted = resolve_selfies(
        normalize_molecule_text(sample["expected_answer"]), normalize_molecule_text(answer)
    )
    canonical_prediction = canonical_smiles(predicted)
    score = tanimoto(canonical_prediction or "", canonical_smiles(gold) or gold)
    return score, {"tanimoto": score, "valid_smiles": canonical_prediction is not None}


def grade_molecule_formula(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Cosine similarity of the two atom-count vectors, plus exact match and the L1/L2 variants."""
    gold_atoms = parse_formula(sample["expected_answer"])
    text = as_text(answer).strip()
    predicted_atoms = parse_formula(text) if text else {}
    if not predicted_atoms:
        return 0.0, {"formula_cosine": 0.0, "formula_l1": 0.0, "formula_l2": 0.0, "formula_exact": False}

    elements = sorted(set(gold_atoms) | set(predicted_atoms))
    gold_vector = [gold_atoms.get(element, 0) for element in elements]
    predicted_vector = [predicted_atoms.get(element, 0) for element in elements]
    dot = sum(g * p for g, p in zip(gold_vector, predicted_vector))
    norms = (sum(g * g for g in gold_vector) ** 0.5) * (sum(p * p for p in predicted_vector) ** 0.5)
    cosine = dot / norms if norms else 0.0
    l1 = 1 / (1 + sum(abs(g - p) for g, p in zip(gold_vector, predicted_vector)))
    l2 = 1 / (1 + sum((g - p) ** 2 for g, p in zip(gold_vector, predicted_vector)) ** 0.5)
    return cosine, {
        "formula_cosine": cosine,
        "formula_l1": l1,
        "formula_l2": l2,
        "formula_exact": gold_atoms == predicted_atoms,
    }


def grade_molecule_iupac(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Case-insensitive exact match on the IUPAC name, with the edit similarity alongside.

    Upstream also reports a Tanimoto for this task, by resolving each predicted name to a SMILES
    through the PubChem web API. That is left out: it turns grading into thousands of network
    calls whose results depend on when they were made.
    """
    gold = sample["expected_answer"].strip().lower()
    predicted = as_text(answer).strip().lower()
    exact = float(bool(predicted) and predicted == gold)
    longest = max(len(gold), len(predicted)) or 1
    return exact, {
        "iupac_exact": bool(exact),
        "iupac_edit_similarity": 1 - edit_distance(gold, predicted) / longest,
    }


def grade_range_overlap(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Upstream's `calculate_overlap`: intersection over union of the two numeric ranges."""
    gold_range = parse_range(sample["expected_answer"])
    predicted_range = parse_range(as_text(answer))
    if gold_range is None or predicted_range is None:
        return 0.0, {"range_overlap": 0.0}
    intersection = max(0.0, min(gold_range[1], predicted_range[1]) - max(gold_range[0], predicted_range[0]))
    union = max(gold_range[1], predicted_range[1]) - min(gold_range[0], predicted_range[0])
    score = intersection / union if union else 0.0
    return score, {"range_overlap": score}


def grade_regression(sample: dict[str, Any], answer: object) -> tuple[float, dict[str, object]]:
    """Normalized error: `max(0, 1 - |predicted - gold| / span)` over the task's gold spread.

    Upstream reports RMSE, RAE, R^2 and an NRMSE normalized by the same spread, all corpus-level.
    The raw values are written per row (`predicted_value`, `expected_value`, `abs_error`) so those
    stay computable; the 0-1 score is what the level rollup can average.

    Two upstream quirks are not reproduced. It discards any prediction outside [-1000, 1000] as
    invalid, which silently drops legitimate melting points in Kelvin. And for the temperature
    task it subtracts 273.15 from the gold value, on a line that reads `gold_num` before it is
    assigned - so the conversion actually applies the *previous* question's gold and raises on the
    first row. The gold values there are written `110K` while the question asks for degrees
    Celsius and 110 K would be absurd for a Buchwald coupling, so the number is taken at face
    value and compared against the Celsius answer the question asks for.
    """
    predicted = extract_number(as_text(answer))
    gold = extract_number(sample["expected_answer"])
    if predicted is None or gold is None:
        return 0.0, {"predicted_value": predicted, "expected_value": gold, "abs_error": None}
    error = abs(predicted - gold)
    score = max(0.0, 1 - error / sample["gold_span"])
    return score, {"predicted_value": predicted, "expected_value": gold, "abs_error": error}


GRADERS = {
    "mcq": grade_mcq,
    "true_false": grade_true_false,
    "classification": grade_classification,
    "classification_subset": grade_classification_subset,
    "entity_extraction": grade_entity_extraction,
    "entity_recognition": grade_entity_recognition,
    "relation_extraction": grade_relation_extraction,
    "reagent_selection": grade_reagent_selection,
    "sider": grade_sider,
    "molecule_smiles": grade_molecule_smiles,
    "molecule_formula": grade_molecule_formula,
    "molecule_iupac": grade_molecule_iupac,
    "range_overlap": grade_range_overlap,
    "regression": grade_regression,
}


def grade(sample: dict[str, Any]) -> None:
    """Score one deterministic row in place; the resource server handles LLM judging."""
    generation = sample.get("generation") or ""
    answer = extract_answer(generation)
    sample["predicted_answer"] = as_text(answer) if answer is not None else None
    score, extra = GRADERS[sample["family"]](sample, answer)
    sample["score"] = score
    sample.update(extra)
