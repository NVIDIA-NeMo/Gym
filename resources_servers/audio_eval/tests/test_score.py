# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corpus scoring helpers: evaluator settings, finiteness checks, batching, flattening and response text."""

import asyncio
from pathlib import Path

import pytest

from resources_servers.audio_eval import score
from resources_servers.audio_eval._scoring.evaluation.evaluator.audio import AudioEvaluator
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.audio_metrics import AudioMetrics
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics
from resources_servers.audio_eval.score import (
    assistant_text,
    check_finite,
    evaluator_settings,
    flatten_metrics,
    generation_failure,
    locate,
    resolve,
    score_rows,
    score_samples,
)


SCORING = Path(score.__file__).parent / "_scoring"


def test_evaluator_settings_parse_the_profile_arguments():
    contract = {"skills": {"EVAL_ARGS": "++eval_type=audio ++eval_config.reference_fields='[a,b]' ++eval_config.n=3"}}
    assert evaluator_settings(contract) == ("audio", {"reference_fields": ["a", "b"], "n": 3})
    assert evaluator_settings({"skills": {}}) == (None, {})


@pytest.mark.parametrize(
    "arg",
    [
        "++eval_conifg.x=1",
        "eval_type=audio",
        "++eval_type",
        "++eval_type=",
        "++eval_config.=1",
        "++eval_config.x",
        "--flag",
    ],
)
def test_an_unrecognized_profile_argument_raises_instead_of_being_ignored(arg):
    with pytest.raises(ValueError, match="Unrecognized argument"):
        evaluator_settings({"skills": {"EVAL_ARGS": f"++eval_type=audio {arg}"}})
    with pytest.raises(ValueError, match="Unrecognized argument"):
        evaluator_settings({"skills": {"GENERATION_ARGS": arg}})


def test_check_finite_rejects_nonfinite_values_anywhere():
    check_finite({"a": [1.0, {"b": 2}]})
    with pytest.raises(ValueError, match=r"metrics.a\[1\]"):
        check_finite({"a": [1.0, float("nan")]})


def test_flatten_metrics_builds_scalar_keys():
    assert flatten_metrics({"p": {"pass@1": {"wer": 1.5, "n": 2}}, "p-x": {"pass@1": {"wer": 0.0}}}) == {
        "audio/p/pass@1/wer": 1.5,
        "audio/p/pass@1/n": 2,
        "audio/p-x/pass@1/wer": 0.0,
    }


def test_score_samples_keeps_row_order_across_batches():
    class Scorer:
        def __init__(self, settings):
            self.settings = settings

        async def eval_single(self, row):
            return {"double": row["i"] * 2}

    rows = [{"i": i} for i in range(1100)]
    scored = asyncio.run(score_samples(rows, Scorer, {}, 8))
    assert [row["double"] for row in scored] == [i * 2 for i in range(1100)]


def test_score_rows_validates_input_and_batch_evaluator_row_counts(monkeypatch):
    contract = {"skills": {"EVAL_ARGS": "++eval_type=fake", "METRICS_TYPE": "audio"}}
    with pytest.raises(ValueError, match="nonempty rows"):
        score_rows([], profile="p", contract=contract)

    def drop_rows(config):
        open(config["input_file"], "w").close()

    monkeypatch.setattr(score, "resolve", lambda kind, name: drop_rows)
    with pytest.raises(ValueError, match="changed the row count"):
        score_rows([{"generation": "x"}], profile="p", contract=contract)


def test_generation_failure_and_assistant_text():
    assert generation_failure({"status": "completed"}) is None
    assert generation_failure({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}) is None
    assert "content_filter" in generation_failure(
        {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}
    )
    assert "incomplete response (None)" == generation_failure({"status": "incomplete"})
    assert "failed" in generation_failure({"status": "failed"})
    assert "cancelled" in generation_failure({"status": "cancelled"})
    # The model ended its turn with reasoning only, or with nothing: a measurement (an empty hypothesis).
    for reason in ("incomplete_reasoning", "empty_output"):
        assert generation_failure({"status": "incomplete", "metadata": {"ng_termination_reason": reason}}) is None
    other = {"status": "incomplete", "metadata": {"ng_termination_reason": "other"}}
    assert "incomplete response" in generation_failure(other)
    # A content filter is not made a measurement by a stale termination reason.
    filtered = {
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "metadata": {"ng_termination_reason": "empty_output"},
    }
    assert "content_filter" in generation_failure(filtered)
    output = [
        {"type": "reasoning"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "a"}, {"type": "refusal"}],
        },
        {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "no"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "b"}]},
    ]
    assert assistant_text({"output": output}) == "ab"


# --- scorers are found by name, with no registry file ----------------------------------------------------------------


def test_the_audio_scorers_resolve_by_name():
    assert resolve("evaluators", "audio") is AudioEvaluator
    assert resolve("metrics", "audio") is AudioMetrics
    module, evaluator = locate("evaluators", "audio")
    assert module.__name__.endswith("evaluation.evaluator.audio") and evaluator is AudioEvaluator


@pytest.mark.parametrize(
    "kind, directory, suffix",
    [("evaluators", "evaluation/evaluator", ""), ("metrics", "evaluation/metrics", "_metrics")],
)
def test_every_scoring_module_resolves_by_its_file_name(kind, directory, suffix):
    """Each scorer module that is present resolves from its own name (dashes and underscores are equivalent)."""
    base = BaseEvaluator if kind == "evaluators" else BaseMetrics
    names = sorted(
        path.stem.removesuffix(suffix)
        for path in (SCORING / directory).glob(f"*{suffix}.py")
        if path.stem not in ("__init__", "base") and not path.stem.endswith("_utils")
    )
    assert "audio" in names
    for name in names:
        for spelling in {name, name.replace("_", "-")}:
            module, found = locate(kind, spelling)
            assert module.__name__.endswith(f"{name}{suffix}")
            assert callable(found) and (not isinstance(found, type) or issubclass(found, base)), spelling


@pytest.fixture
def declared_variants(monkeypatch):
    """The audio metrics module declares the variant suffixes its benchmarks use."""
    module = locate("metrics", "audio")[0]
    monkeypatch.setattr(module, "VARIANTS", ("open_ended", "instruction_following_v2"), raising=False)


def test_a_metrics_name_may_carry_a_variant_suffix_the_module_declares(declared_variants):
    assert resolve("metrics", "audio_open_ended") is AudioMetrics
    assert resolve("metrics", "audio_instruction_following_v2") is AudioMetrics
    assert resolve("metrics", "audio") is AudioMetrics


@pytest.mark.parametrize("name", ["audio_wer", "audio_open", "audio_open_ended_extra"])
def test_an_undeclared_metrics_variant_is_a_key_error_not_a_silent_fallback(declared_variants, name):
    """A typo, or a benchmark whose own metrics module has not merged yet, must not resolve to the generic class."""
    with pytest.raises(KeyError, match="does not declare the variant"):
        resolve("metrics", name)


def test_a_module_that_declares_no_variants_rejects_every_suffix():
    with pytest.raises(KeyError, match="declared: none"):
        resolve("metrics", "audio_open_ended")


@pytest.mark.parametrize(
    "kind, name",
    [
        ("evaluators", "no-such-scorer"),
        ("metrics", "no_such_scorer"),
        ("evaluators", "../audio"),
        ("evaluators", "audio.py"),
        ("evaluators", ""),
        ("evaluators", "nemo_skills.dataset.x/y.scorer::Z"),
        ("evaluators", "nemo_skills.dataset.no-such-dataset.scorer::Z"),
    ],
)
def test_an_unknown_or_malformed_name_is_a_key_error(kind, name):
    with pytest.raises(KeyError):
        resolve(kind, name)


def test_kind_must_be_evaluators_or_metrics():
    with pytest.raises(ValueError, match="kind must be"):
        resolve("scorers", "audio")


def test_a_dataset_scoped_key_resolves_to_the_scorer_of_that_dataset(monkeypatch):
    import importlib.machinery
    import sys
    import types

    module = types.ModuleType("resources_servers.audio_eval._scoring.dataset.demo-set.scorer")
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)

    class DemoEvaluator(BaseEvaluator):
        pass

    module.DemoEvaluator = DemoEvaluator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    key = "nemo_skills.dataset.demo-set.scorer::DemoEvaluator"
    assert resolve("evaluators", key) is DemoEvaluator
    assert locate("metrics", key)[0] is module


def test_a_module_must_define_exactly_one_scorer_class(monkeypatch):
    import importlib.machinery
    import sys
    import types

    name = "resources_servers.audio_eval._scoring.evaluation.evaluator.twice"
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)

    class First(BaseEvaluator):
        pass

    class Second(BaseEvaluator):
        pass

    First.__module__ = Second.__module__ = name
    module.First, module.Second, module.Imported = First, Second, AudioEvaluator
    monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(KeyError, match="exactly one BaseEvaluator subclass, found 2"):
        resolve("evaluators", "twice")
    # A function named eval_<name> wins, and imported classes never count.
    module.eval_twice = lambda config: None
    assert resolve("evaluators", "twice") is module.eval_twice


def test_a_dataset_scoped_key_naming_a_missing_class_is_a_key_error(monkeypatch):
    import importlib.machinery
    import sys
    import types

    module = types.ModuleType("resources_servers.audio_eval._scoring.dataset.demo-set.scorer")
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(KeyError, match="has no Missing"):
        resolve("evaluators", "nemo_skills.dataset.demo-set.scorer::Missing")
