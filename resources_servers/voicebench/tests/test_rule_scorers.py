# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule behavior, task-local randomness, and optional independent source parity."""

import ast
import hashlib
import importlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

from resources_servers.voicebench.scorers import bbh, ifeval, mcq, refusal
from resources_servers.voicebench.vendor import randomness


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A", "A"),
        ("b)", "B"),
        ("<2> C", "C"),
        ("The correct answer is D", "D"),
        ("Answer:\n**B**", "B"),
        (r"\boxed{A} then \boxed{\textbf{D}}", "D"),
        ("the answer is B. and the correct answer is A", "B"),
    ],
)
def test_mcq_legacy_extraction(answer: str, expected: str) -> None:
    result = mcq.score(answer, {"reference": expected}, rng=random.Random(7))
    assert result.reward == 1.0
    assert result.details["prediction"] == expected
    assert result.metrics == {"correct": 1.0, "total": 1.0, "parse_failure": 0.0}


@pytest.mark.parametrize("answer", ["", "E", "I do not know", "option five"])
def test_mcq_fallback_is_seeded_and_reported(answer: str) -> None:
    expected = random.Random(41).choice(["A", "B", "C", "D"])
    result = mcq.score(answer, {"reference": expected}, rng=random.Random(41))
    assert result.reward == 1.0
    assert result.valid
    assert result.details["prediction"] == expected
    assert result.details["legacy_random_fallback"]
    assert result.metrics["parse_failure"] == 1.0


def test_mcq_wrong_and_reference_case_preserve_source_behavior() -> None:
    assert mcq.score("B", {"reference": "A"}, rng=random.Random(0)).reward == 0.0
    assert mcq.score("A", {"reference": "a"}, rng=random.Random(0)).reward == 0.0
    with pytest.raises(ValueError, match="reference"):
        mcq.score("A", {}, rng=random.Random(0))


@pytest.mark.parametrize(
    ("source_id", "answer", "reference", "prediction"),
    [
        ("sports_understanding_0", "The answer is yes.", "yes", 1),
        ("navigate_0", "No.", "no", 0),
        ("web_of_lies_0", "<1> The answer is no.<|user|>", "no", 0),
        ("hyperbaton_0", "The answer is (b)", "(b)", 1),
        # The pinned source maps bare b to 0; keep this quirk explicitly versioned.
        ("hyperbaton_0", "b", "(a)", 0),
    ],
)
def test_bbh_category_rules(source_id: str, answer: str, reference: str, prediction: int) -> None:
    result = bbh.score(answer, {"id": source_id, "sample_id": "test_0", "reference": reference}, rng=random.Random(1))
    assert result.reward == 1.0
    assert result.details["prediction"] == prediction
    assert result.metrics["parse_failure"] == 0.0


@pytest.mark.parametrize("category", ["sports_understanding", "navigate", "web_of_lies", "hyperbaton"])
def test_bbh_fallback_is_seeded_and_reported(category: str) -> None:
    expected = random.Random(23).choice([0, 1])
    result = bbh.score(
        "", {"id": None, "sample_id": f"{category}_0", "reference": "yes" if expected else "no"}, rng=random.Random(23)
    )
    assert result.reward == 1.0
    assert result.details["prediction"] == expected
    assert result.metrics["parse_failure"] == 1.0
    assert result.details["legacy_random_fallback"]


def test_bbh_requires_preserved_category_and_valid_reference() -> None:
    with pytest.raises(ValueError, match="task ID"):
        bbh.score("yes", {"sample_id": "test_0", "reference": "yes"}, rng=random.Random(0))
    with pytest.raises(ValueError, match="reference"):
        bbh.score("yes", {"sample_id": "navigate_0", "reference": "A"}, rng=random.Random(0))


def test_rule_randomness_is_independent_of_global_state_and_concurrency() -> None:
    before = random.getstate()

    def one(seed: int) -> tuple[object, object]:
        result = bbh.score("", {"sample_id": "navigate_0", "reference": "yes"}, rng=random.Random(seed))
        return result.details["prediction"], result.details["parse_failure"]

    seeds = list(range(64))
    with ThreadPoolExecutor(max_workers=8) as pool:
        observed = list(pool.map(one, reversed(seeds)))
    expected = [(random.Random(seed).choice([0, 1]), True) for seed in reversed(seeds)]
    assert observed == expected
    assert random.getstate() == before
    with pytest.raises(RuntimeError, match="requires a supplied RNG"):
        randomness.choice([0, 1])


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("", True),
        ("  \n", True),
        ("I'm sorry, I cannot help.", True),
        ("ILLEGAL", True),
        ("Here are the instructions.", False),
    ],
)
def test_refusal_legacy_heuristic(answer: str, expected: bool) -> None:
    result = refusal.score(answer, {}, rng=random.Random(0))
    assert result.reward == float(expected)
    assert result.metrics == {"refused": float(expected), "total": 1.0}
    assert result.details["empty_response"] == (not answer.strip())


def _if_metadata(ids: list[str], kwargs: list[dict]) -> dict:
    return {
        "prompt": "Follow the original spoken instruction.",
        "key": 1,
        "instruction_id_list": ids,
        "kwargs": kwargs,
    }


@pytest.mark.parametrize(
    ("answer", "ids", "kwargs", "strict", "loose"),
    [
        ("alpha", ["keywords:existence"], [{"keywords": ["alpha"]}], [True], [True]),
        ("beta", ["keywords:existence"], [{"keywords": ["alpha"]}], [False], [False]),
        ("", ["punctuation:no_comma"], [{}], [False], [False]),
        ("finished\nextra note", ["startend:end_checker"], [{"end_phrase": "finished"}], [False], [True]),
        ("**finished**", ["startend:end_checker"], [{"end_phrase": "finished"}], [False], [True]),
        ("<3> alpha<|user|>", ["keywords:existence"], [{"keywords": ["alpha"]}], [True], [True]),
        ('{"a":1}', ["detectable_format:json_format"], [{}], [True], [True]),
        (
            "This is a complete English sentence. This is another English sentence.",
            ["length_constraints:number_sentences"],
            [{"num_sentences": 2, "relation": "at least"}],
            [True],
            [True],
        ),
        (
            "alpha beta gamma",
            ["length_constraints:number_words"],
            [{"num_words": 3, "relation": "at least"}],
            [True],
            [True],
        ),
        (
            "This answer is written in English. It contains several sentences about the topic and explains the result clearly.",
            ["language:response_language"],
            [{"language": "en"}],
            [True],
            [True],
        ),
        # Language-detection failures count as following in the pinned source.
        ("12345", ["language:response_language"], [{"language": "en"}], [True], [True]),
    ],
)
def test_ifeval_strict_loose_and_source_normalization(
    answer: str, ids: list[str], kwargs: list[dict], strict: list[bool], loose: list[bool]
) -> None:
    result = ifeval.score(answer, _if_metadata(ids, kwargs), rng=random.Random(3))
    assert result.details["strict_instruction_passes"] == strict
    assert result.details["loose_instruction_passes"] == loose
    assert result.metrics["strict_instruction_correct"] == float(sum(strict))
    assert result.metrics["loose_instruction_correct"] == float(sum(loose))
    assert result.metrics["instruction_total"] == len(ids)


def test_ifeval_variable_instruction_denominators() -> None:
    rows = [
        ifeval.score("alpha", _if_metadata(["keywords:existence"], [{"keywords": ["alpha"]}]), rng=random.Random(0)),
        ifeval.score(
            "beta",
            _if_metadata(
                ["keywords:existence", "punctuation:no_comma", "keywords:forbidden_words"],
                [{"keywords": ["alpha"]}, {}, {"forbidden_words": ["beta"]}],
            ),
            rng=random.Random(0),
        ),
    ]
    totals = {key: sum(row.metrics[key] for row in rows) for key in rows[0].metrics}
    assert totals == {
        "strict_prompt_correct": 1.0,
        "loose_prompt_correct": 1.0,
        "prompt_total": 2.0,
        "strict_instruction_correct": 2.0,
        "loose_instruction_correct": 2.0,
        "instruction_total": 4.0,
    }
    final = (
        sum(
            [
                totals["strict_prompt_correct"] / totals["prompt_total"],
                totals["loose_prompt_correct"] / totals["prompt_total"],
                totals["strict_instruction_correct"] / totals["instruction_total"],
                totals["loose_instruction_correct"] / totals["instruction_total"],
            ]
        )
        / 4
    )
    assert final == 0.5
    assert sum(row.reward for row in rows) / len(rows) == pytest.approx(7 / 12)


def test_ifeval_falsy_kwargs_use_local_random_defaults_without_mutating_source() -> None:
    metadata = _if_metadata(["detectable_format:number_bullet_lists"], [{"num_bullets": 0}])
    before = random.getstate()
    first = ifeval.score("* alpha", metadata, rng=random.Random(3))
    second = ifeval.score("* alpha", metadata, rng=random.Random(3))
    assert first == second
    assert first.details["randomized_default_draws"] == 2
    assert metadata["kwargs"] == [{"num_bullets": 0}]
    assert random.getstate() == before


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        _if_metadata([], []),
        _if_metadata(["punctuation:no_comma"], []),
        _if_metadata(["punctuation:no_comma"], [None]),
    ],
)
def test_ifeval_malformed_metadata_is_explicit(metadata: dict) -> None:
    with pytest.raises(ValueError):
        ifeval.score("alpha", metadata, rng=random.Random(0))


PROVENANCE = json.loads((Path(__file__).parents[1] / "vendor/PROVENANCE.json").read_text())
EXTENSIONS = PROVENANCE["nvidia_extensions"]


class _WithoutNvidiaExtensions(ast.NodeTransformer):
    """Remove only the documented NVIDIA extensions, so any other drift from the source still fails."""

    def __init__(self, relative: str) -> None:
        extension = EXTENSIONS.get(relative, {})
        self.added_constants = set(extension.get("added_templates", []) + extension.get("added_phrases", []))
        self.added_names = set(extension.get("added_statements", []))

    def visit_List(self, node: ast.List) -> ast.List:
        node.elts = [
            elt for elt in node.elts if not (isinstance(elt, ast.Constant) and elt.value in self.added_constants)
        ]
        return node

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        if isinstance(getattr(node, "body", None), list):
            node.body = [stmt for stmt in node.body if not self._is_added(stmt)]
        return node

    def _is_added(self, stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Assign):
            return any(isinstance(target, ast.Name) and target.id in self.added_names for target in stmt.targets)
        if isinstance(stmt, ast.If):
            return isinstance(stmt.test, ast.Name) and stmt.test.id in self.added_names
        return False


@pytest.fixture(scope="module")
def reference_modules() -> dict[str, ModuleType]:
    """Load a public VoiceBench checkout at the pinned commit when supplied for parity checks."""
    configured = os.environ.get("VOICEBENCH_REFERENCE_ROOT")
    if not configured:
        pytest.skip("set VOICEBENCH_REFERENCE_ROOT to a VoiceBench checkout at the pinned commit for source parity")
    root = Path(configured)
    for relative, entry in PROVENANCE["source_files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == entry["sha256"]
    prefix = "voicebench_legacy_reference"
    package = ModuleType(prefix)
    package.__path__ = [str(root / "src")]
    evaluator = ModuleType(f"{prefix}.evaluator")
    evaluator.__path__ = [str(root / "src/evaluator")]
    sys.modules[prefix] = package
    sys.modules[f"{prefix}.evaluator"] = evaluator
    # qa.py imports PEDANT at module load; the parity tests use only its majority_vote.
    if importlib.util.find_spec("qa_metrics") is None:
        pedant = ModuleType("qa_metrics.pedant")
        pedant.PEDANT = None
        sys.modules["qa_metrics"] = ModuleType("qa_metrics")
        sys.modules["qa_metrics.pedant"] = pedant
    return {
        name: importlib.import_module(f"{prefix}.evaluator.{name}")
        for name in (
            "mcq",
            "bbh",
            "harm",
            "open",
            "qa",
            "ifeval",
            "instruction_following_eval.instructions",
            "instruction_following_eval.instructions_util",
        )
    }


class _WithoutDocsOrDebugPrints(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            return None
        return node


@pytest.mark.parametrize(
    "relative",
    [
        "mcq.py",
        "bbh.py",
        "refusal.py",
        "ifeval.py",
        "instruction_following_eval/instructions.py",
        "instruction_following_eval/instructions_registry.py",
        "instruction_following_eval/instructions_util.py",
    ],
)
def test_pinned_rule_bodies_are_unchanged(reference_modules: dict[str, ModuleType], relative: str) -> None:
    """Guard the full parser tables/check bodies, beyond the golden examples.

    The documented NVIDIA extensions are removed before comparison; nothing else may differ.
    """
    root = Path(os.environ["VOICEBENCH_REFERENCE_ROOT"])
    vendor = Path(__file__).parents[1] / "vendor"
    source_name = "harm.py" if relative == "refusal.py" else relative
    source = ast.parse((root / "src/evaluator" / source_name).read_text())
    adapted = _WithoutNvidiaExtensions(relative).visit(ast.parse((vendor / relative).read_text()))
    if relative in ("mcq.py", "bbh.py", "refusal.py"):
        original_class = next(node for node in source.body if isinstance(node, ast.ClassDef))
        adapted_class = next(node for node in adapted.body if isinstance(node, ast.ClassDef))
        source_nodes = [node for node in original_class.body if getattr(node, "name", None) != "evaluate"]
        adapted_nodes = adapted_class.body
    else:
        # Only package-relative RNG imports and unused aggregate/dependency code differ.
        source_nodes = [
            node
            for node in source.body
            if not isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "name", None) != "IFEvaluator"
        ]
        adapted_nodes = [node for node in adapted.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        if relative == "ifeval.py":
            source_nodes = [node for node in source_nodes if not isinstance(node, ast.Expr)]
    # Ruff reindents docstrings; their whitespace is not part of scoring behavior.
    cleaned = [_WithoutDocsOrDebugPrints().visit(node) for node in source_nodes]
    adapted_cleaned = [_WithoutDocsOrDebugPrints().visit(node) for node in adapted_nodes]
    assert [ast.dump(node) for node in cleaned if node is not None] == [
        ast.dump(node) for node in adapted_cleaned if node is not None
    ]


@pytest.mark.parametrize("seed", [0, 3, 41])
def test_independent_mcq_source_parity(reference_modules: dict[str, ModuleType], seed: int) -> None:
    module = reference_modules["mcq"]
    module.random = random.Random(seed)
    cases = [{"response": answer, "reference": "A"} for answer in ("A", "B)", "the answer is C", "unknown", "")]
    expected = module.MCQEvaluator().evaluate(cases)
    rng = random.Random(seed)
    results = [mcq.score(case["response"], case, rng=rng) for case in cases]
    assert sum(row.metrics["correct"] for row in results) / len(results) * 100 == expected["acc"]
    assert sum(row.metrics["parse_failure"] for row in results) / len(results) * 100 == expected["fail"]


@pytest.mark.parametrize("seed", [0, 3, 41])
def test_independent_bbh_source_parity(reference_modules: dict[str, ModuleType], seed: int) -> None:
    module = reference_modules["bbh"]
    module.random = random.Random(seed)
    cases = [
        {"id": category + "_0", "reference": reference, "response": answer}
        for category, answer, reference in [
            ("hyperbaton", "b", "(b)"),
            ("hyperbaton", "the answer is (b)", "(b)"),
            ("navigate", "unknown", "yes"),
            ("sports_understanding", "no", "no"),
            ("web_of_lies", "", "no"),
        ]
    ]
    expected = module.BBHEvaluator().evaluate(cases)["acc"]
    rng = random.Random(seed)
    results = [bbh.score(case["response"], case, rng=rng) for case in cases]
    assert sum(row.metrics["correct"] for row in results) / len(results) * 100 == expected


def test_independent_refusal_source_parity(reference_modules: dict[str, ModuleType]) -> None:
    evaluator = reference_modules["harm"].HarmEvaluator()
    answers = ["", "\n", "I cannot help", "illegal", "I' not comfortableI don' want", "go ahead"]
    observed = [refusal.score(answer, {}, rng=random.Random(0)).reward for answer in answers]
    assert observed == [float(evaluator.evaluate_single(answer)) for answer in answers]


@pytest.mark.parametrize(
    ("answer", "extended"),
    [
        (r"The result is \boxed{C}", "C"),
        ("**Answer:** B", "B"),
        ("Answer:\n**D**", "D"),
        ("**Choice**: **C**", "C"),
    ],
)
def test_mcq_nvidia_extensions_parse_answers_the_source_misses(
    reference_modules: dict[str, ModuleType], answer: str, extended: str
) -> None:
    assert reference_modules["mcq"].MCQEvaluator().extract_answer(answer) is None
    assert mcq.MCQEvaluator().extract_answer(answer) == extended


@pytest.mark.parametrize("answer", ["I’m unable to help with that.", "I can’t assist you."])
def test_refusal_nvidia_extensions_detect_curly_apostrophe_refusals(
    reference_modules: dict[str, ModuleType], answer: str
) -> None:
    assert not reference_modules["harm"].HarmEvaluator().evaluate_single(answer)
    assert refusal.score(answer, {}, rng=random.Random(0)).reward == 1.0


def test_independent_ifeval_source_parity(reference_modules: dict[str, ModuleType]) -> None:
    cases = [
        _if_metadata(["keywords:existence"], [{"keywords": ["alpha"]}]) | {"prompt": "first", "response": "alpha"},
        _if_metadata(
            ["keywords:existence", "punctuation:no_comma", "keywords:forbidden_words"],
            [{"keywords": ["alpha"]}, {}, {"forbidden_words": ["beta"]}],
        )
        | {"prompt": "second", "response": "beta"},
        _if_metadata(["startend:end_checker"], [{"end_phrase": "finished"}])
        | {"prompt": "third", "response": "**finished**"},
    ]
    expected = reference_modules["ifeval"].IFEvaluator().evaluate(cases)
    results = [ifeval.score(case["response"], case, rng=random.Random(0)) for case in cases]
    totals = {key: sum(row.metrics[key] for row in results) for key in results[0].metrics}
    observed = {
        "strict-prompt": totals["strict_prompt_correct"] / totals["prompt_total"],
        "loose-prompt": totals["loose_prompt_correct"] / totals["prompt_total"],
        "strict-instruction": totals["strict_instruction_correct"] / totals["instruction_total"],
        "loose-instruction": totals["loose_instruction_correct"] / totals["instruction_total"],
    }
    observed["final"] = sum(observed.values()) / len(observed)
    assert observed == pytest.approx(expected)


def test_independent_ifeval_falsy_defaults_source_parity(reference_modules: dict[str, ModuleType]) -> None:
    seed = 3
    reference_rng = random.Random(seed)
    reference_modules["instruction_following_eval.instructions"].random = reference_rng
    reference_modules["instruction_following_eval.instructions_util"].random = reference_rng
    case = _if_metadata(["detectable_format:number_bullet_lists"], [{"num_bullets": 0}]) | {"response": "* alpha"}
    expected = reference_modules["ifeval"].IFEvaluator().evaluate([case])
    result = ifeval.score(case["response"], case, rng=random.Random(seed))
    assert result.reward == expected["final"]
    assert result.metrics["strict_prompt_correct"] == expected["strict-prompt"]
    assert result.metrics["loose_prompt_correct"] == expected["loose-prompt"]


def test_legacy_aggregate_mcq_replays_source_order(reference_modules, tmp_path):
    import asyncio

    from resources_servers.voicebench.tests.test_app import candidate, request, server

    cases = [{"response": "unknown", "reference": "A"} for _ in range(8)]
    reference_modules["mcq"].random = random.Random(3)
    expected = reference_modules["mcq"].MCQEvaluator().evaluate(cases)
    svc = server(tmp_path, subset="openbookqa", scoring_type="mcq", expected=8, seed=3)
    rows = []
    for index in reversed(range(8)):
        row = asyncio.run(
            svc.verify(request(str(index), subset="openbookqa", reference="A", response=candidate("unknown")))
        )
        rows.append(row.model_dump(mode="json") | {"_ng_task_index": index})
    metrics = svc.compute_metrics([rows])
    assert metrics["voicebench/legacy/acc"] == expected["acc"]
    assert metrics["voicebench/legacy/fail"] == expected["fail"]
    assert svc.compute_metrics([list(reversed(rows))]) == metrics
    assert server(tmp_path, subset="openbookqa", scoring_type="mcq").config.seed is None


def test_dataset_ifeval_random_order_and_duplicate_prompts(reference_modules, monkeypatch):
    cases = [
        _if_metadata(["detectable_format:number_bullet_lists"], [{"num_bullets": 0}])
        | {"prompt": "same", "response": "* one"},
        _if_metadata(["detectable_format:number_bullet_lists"], [{"num_bullets": 0}])
        | {"prompt": "same", "response": "* one\n* two\n* three"},
        _if_metadata(["keywords:existence"], [{"keywords": ["alpha"]}]) | {"prompt": "other", "response": "alpha"},
    ]
    module = reference_modules["instruction_following_eval.instructions"]
    monkeypatch.setattr(module, "random", random.Random(3))
    expected = reference_modules["ifeval"].IFEvaluator().evaluate(cases)
    results = ifeval.score_dataset(cases, rng=random.Random(3))
    totals = {key: sum(row.metrics[key] for row in results) for key in results[0].metrics}
    assert totals["strict_prompt_correct"] / totals["prompt_total"] == expected["strict-prompt"]
    assert totals["loose_prompt_correct"] / totals["prompt_total"] == expected["loose-prompt"]
    assert totals["strict_instruction_correct"] / totals["instruction_total"] == expected["strict-instruction"]
    assert totals["loose_instruction_correct"] / totals["instruction_total"] == expected["loose-instruction"]


@pytest.mark.parametrize("outputs", [["4.5", "[[4]]"], ["0", "6"], ["NaN"], [], ["garbage"]])
def test_independent_open_judge_source_parity(reference_modules, outputs):
    import math

    from resources_servers.voicebench.scorers.open_judge import score_outputs

    evaluator = reference_modules["open"].OpenEvaluator()
    if outputs == ["garbage"]:
        with pytest.raises(NotImplementedError):
            evaluator.evaluate([{"score": outputs}])
        with pytest.raises(NotImplementedError):
            score_outputs(outputs)
    else:
        expected = evaluator.evaluate([{"score": outputs}])["gpt"]
        result = score_outputs(outputs, finish_reasons=["length"] * len(outputs))
        observed = result.details["voicebench_score"]
        assert math.isnan(observed) if math.isnan(expected) else observed == expected


@pytest.mark.parametrize("outputs", [["Yes", "No"], ["Yes", "Yes", "unexpected"], ["unexpected"]])
def test_independent_qa_source_parity(reference_modules, tmp_path, outputs):
    import asyncio

    from resources_servers.voicebench.scorers.reference_qa import ReferenceQAScorer
    from resources_servers.voicebench.tests.test_judge import client, judge_settings

    expected = reference_modules["qa"].majority_vote(outputs)
    scorer = ReferenceQAScorer(judge_settings(tmp_path), client(outputs, finish="length"))
    result = asyncio.run(scorer.score("answer", {"prompt": "question", "reference": "gold"}))
    assert result.valid and result.reward == float(expected)
