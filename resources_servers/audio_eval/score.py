# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corpus scoring shared by individual Gym audio benchmarks.

Scoring is batch-oriented because session merging and corpus BLEU are not
reducible to averages of per-rollout rewards. It is pure CPU work: judge calls
happen in ``verify`` and arrive here as row fields.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import math
import re
import shlex
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics
from resources_servers.audio_eval._scoring.text import cleanup_generation_loops


SCORING = "resources_servers.audio_eval._scoring"
_PLAIN_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_DATASET_KEY = re.compile(r"nemo_skills\.dataset\.([A-Za-z0-9][A-Za-z0-9_-]*)\.scorer::([A-Za-z_]\w*)")


def load_jsonl(path: Path) -> list[dict]:
    """Read physical JSONL records without splitting Unicode inside strings."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package of the module does not exist
        return False


def _defined_in(module: ModuleType, base: type) -> list[type]:
    """The classes ``module`` itself defines (not the ones it imports) that derive from ``base``."""
    return [
        value
        for value in vars(module).values()
        if inspect.isclass(value) and value.__module__ == module.__name__ and issubclass(value, base)
    ]


def locate(kind: str, name: str) -> tuple[ModuleType, Any]:
    """The module and the evaluator or metrics class (or function) a profile names under ``kind``.

    ``kind`` is ``evaluators`` (``++eval_type=<name>``) or ``metrics`` (``METRICS_TYPE``). There is no registry file:
    a name resolves by where its module lives, so adding a scorer means adding a module.

    * ``foo-bar`` or ``foo_bar`` -> ``_scoring/evaluation/evaluator/foo_bar.py``, which defines a function
      ``eval_foo_bar`` or exactly one ``BaseEvaluator`` subclass.
    * ``foo-bar`` or ``foo_bar`` -> ``_scoring/evaluation/metrics/foo_bar_metrics.py``, which defines exactly one
      ``BaseMetrics`` subclass. A metrics name may carry a variant suffix: ``foo_bar_open_ended`` finds
      ``foo_bar_metrics.py`` when no module matches the full name (the longest matching prefix wins), and only when
      that module declares the suffix in a module-level ``VARIANTS`` (``VARIANTS = ("open_ended",)``). A name whose
      suffix the module does not declare, such as the typo ``audio_wer``, is a ``KeyError``.
    * ``nemo_skills.dataset.<dataset>.scorer::<Class>`` -> ``_scoring/dataset/<dataset>/scorer.py``, attribute
      ``<Class>`` (the dataset-scoped scorers of the Skills layout).
    """
    if kind not in ("evaluators", "metrics"):
        raise ValueError(f"kind must be 'evaluators' or 'metrics', got {kind!r}")
    dataset_key = _DATASET_KEY.fullmatch(name)
    if dataset_key is not None:
        dataset, attribute = dataset_key.groups()
        if not _module_exists(f"{SCORING}.dataset.{dataset}.scorer"):
            raise KeyError(f"Unknown {kind[:-1]} {name!r}: no module {SCORING}.dataset.{dataset}.scorer")
        module = importlib.import_module(f"{SCORING}.dataset.{dataset}.scorer")
        if not hasattr(module, attribute):
            raise KeyError(f"Unknown {kind[:-1]} {name!r}: {module.__name__} has no {attribute}")
        return module, getattr(module, attribute)
    if not _PLAIN_KEY.fullmatch(name):
        raise KeyError(f"Invalid {kind[:-1]} name {name!r}")
    stem = name.replace("-", "_")
    if kind == "evaluators":
        package, base = f"{SCORING}.evaluation.evaluator", BaseEvaluator
        candidates = [stem]
    else:
        package, base = f"{SCORING}.evaluation.metrics", BaseMetrics
        parts = stem.split("_")
        candidates = [f"{'_'.join(parts[:end])}_metrics" for end in range(len(parts), 0, -1)]
    for candidate in candidates:
        if _module_exists(f"{package}.{candidate}"):
            module = importlib.import_module(f"{package}.{candidate}")
            break
    else:
        raise KeyError(f"Unknown {kind[:-1]} {name!r}: no module {package}.{candidates[0]}")
    if kind == "metrics" and candidate != candidates[0]:
        variant = stem.removeprefix(candidate.removesuffix("_metrics")).lstrip("_")
        if variant not in getattr(module, "VARIANTS", ()):
            raise KeyError(
                f"Unknown {kind[:-1]} {name!r}: {module.__name__} does not declare the variant {variant!r} "
                f"(declared: {', '.join(getattr(module, 'VARIANTS', ())) or 'none'})"
            )
    if kind == "evaluators" and callable(function := getattr(module, f"eval_{stem}", None)):
        return module, function
    classes = _defined_in(module, base)
    if len(classes) != 1:
        raise KeyError(f"{module.__name__} must define exactly one {base.__name__} subclass, found {len(classes)}")
    return module, classes[0]


def resolve(kind: str, name: str) -> Any:
    """The evaluator or metrics class (or function) a profile names under ``kind``; see :func:`locate`."""
    return locate(kind, name)[1]


def evaluator_settings(contract: dict) -> tuple[str | None, dict]:
    """The evaluator name and ``++eval_config.*`` settings a profile contract declares.

    The contract keeps NeMo Skills' command-line strings, but only two token forms mean anything here:
    ``++eval_type=<name>`` and ``++eval_config.<key>=<value>``. Any other token (a typo such as
    ``++eval_conifg.x=1``, a bare ``eval_type=audio``) raises ``ValueError`` instead of being ignored.
    """
    settings = {}
    evaluator = None
    skills = contract["skills"]
    for arg in shlex.split((skills.get("GENERATION_ARGS") or "") + " " + (skills.get("EVAL_ARGS") or "")):
        if arg.startswith("++eval_type=") and len(arg) > len("++eval_type="):
            evaluator = arg.split("=", 1)[1]
        elif arg.startswith("++eval_config.") and "=" in arg and arg.split("=", 1)[0] != "++eval_config.":
            key, value = arg.removeprefix("++eval_config.").split("=", 1)
            settings[key] = yaml.safe_load(value)
        else:
            raise ValueError(
                f"Unrecognized argument {arg!r} in the profile's EVAL_ARGS/GENERATION_ARGS: "
                "use ++eval_type=<name> and ++eval_config.<key>=<value>"
            )
    return evaluator, settings


def check_finite(value: Any, path: str = "metrics") -> None:
    """Raise ``ValueError`` naming the path of the first nonfinite float in nested dicts and lists."""
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite value at {path}: {value}")


def clear_mecab_tokenizer_caches() -> None:
    """Release sentence-BLEU tokenizer instances retained by method LRU caches.

    SacreBLEU creates a tokenizer per sentence, and the method cache keys include
    self. Keeping 65,536 such keys can retain tens of GiB of MeCab state. Cache
    eviction changes no tokenization parameters or returned scoring values.
    Only touch already-loaded tokenizers; optional dependencies stay optional.
    """
    for suffix, class_name in (("ja", "TokenizerJaMecab"), ("ko", "TokenizerKoMecab")):
        module = sys.modules.get(f"sacrebleu.tokenizers.tokenizer_{suffix}_mecab")
        if module is not None:
            getattr(module, class_name).__call__.cache_clear()


async def score_samples(rows: list[dict], evaluator: Any, settings: dict, concurrency: int) -> list[dict]:
    """Run an evaluator class's ``eval_single`` on every row, ``concurrency`` rows at a time, keeping row order."""
    scorer = evaluator(settings)

    async def one(row: dict) -> dict:
        updates = await scorer.eval_single(dict(row))
        return {**row, **updates}

    result = []
    # Bound tasks and retained tokenizer instances without changing row order.
    since_eviction = 0
    clear_mecab_tokenizer_caches()
    try:
        for start in range(0, len(rows), concurrency):
            batch = rows[start : start + concurrency]
            result.extend(await asyncio.gather(*(one(row) for row in batch)))
            since_eviction += len(batch)
            if since_eviction >= 512:
                clear_mecab_tokenizer_caches()
                since_eviction = 0
        return result
    finally:
        clear_mecab_tokenizer_caches()


def aggregate(rows: list[dict], profile: str, contract: dict) -> dict:
    """Metrics per scoring group: the profile itself and each of its ``subset_for_metrics`` values."""
    cls = resolve("metrics", contract["skills"]["METRICS_TYPE"])
    calculators = {"_all_": cls()}
    calculators["_all_"].reset()
    for row in rows:
        subset = row.get("subset_for_metrics", "_all_")
        if subset not in calculators:
            calculators[subset] = cls()
            calculators[subset].reset()
        calculators["_all_"].update([dict(row)])
        if subset != "_all_":
            calculators[subset].update([dict(row)])
    subset_metrics = {}
    for subset, calculator in calculators.items():
        metrics = calculator.get_metrics()
        metrics.pop("pass@1[avg-of-1]", None)
        subset_metrics[subset] = metrics
    result = {
        profile if subset == "_all_" else f"{profile}-{subset}": metrics for subset, metrics in subset_metrics.items()
    }
    check_finite(result)
    return result


def flatten_metrics(results: dict, prefix: str = "audio") -> dict[str, Any]:
    """``{group: {mode: {metric: value}}}`` as flat ``audio/<group>/<mode>/<metric>`` keys with scalar values."""
    flat = {}

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}/{key}", item)
        else:
            flat[path] = value

    visit(prefix, results)
    return flat


def score_rows(rows: list[dict], *, profile: str, contract: dict, concurrency: int = 8) -> tuple[list[dict], dict]:
    """Score the rows of one profile: per-row evaluator, then corpus metrics. Returns the scored rows and metrics."""
    if not rows or concurrency < 1:
        raise ValueError("Scoring requires nonempty rows and positive concurrency")
    name, settings = evaluator_settings(contract)
    # Scorer-independent opt-in (++eval_config.cleanup_generation_loops=true).
    if settings.pop("cleanup_generation_loops", False):
        rows = [
            {**row, "generation_raw": row["generation"], "generation": cleanup_generation_loops(row["generation"])}
            for row in rows
        ]
    if name is None:
        # Judged profiles carry their per-row judgement from ``verify``; metrics read it directly.
        scored = rows
    else:
        evaluator = resolve("evaluators", name)
        if inspect.isclass(evaluator):
            scored = asyncio.run(score_samples(rows, evaluator, settings, concurrency))
        else:
            with tempfile.TemporaryDirectory(prefix="audio-score-") as directory:
                path = Path(directory) / "predictions.jsonl"
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                evaluator({**settings, "input_file": str(path)})
                scored = load_jsonl(path)
            if len(scored) != len(rows):
                raise ValueError("Batch evaluator changed the row count")
    return scored, aggregate(scored, profile, contract)


# ``ng_termination_reason`` values the agent sets when the model ended its turn without an answer.
_NO_ANSWER_REASONS = ("incomplete_reasoning", "empty_output")


def generation_failure(response: dict) -> str | None:
    """Why a Gym response carries no measurable generation, or None when it does.

    These are measurements, not infrastructure outcomes, and are scored: a completed response, a response truncated
    at the output-token limit, and an ``incomplete`` response the agent marked ``incomplete_reasoning`` or
    ``empty_output`` (the model ended its turn with reasoning only, or with nothing; the hypothesis is empty, as in
    NeMo Skills). Masking those would drop exactly the rows the model failed and improve the headline metric.
    Everything else (``failed``, ``cancelled``, or an ``incomplete`` response with another reason such as
    ``content_filter``) is an infrastructure outcome and is masked.
    """
    status = response.get("status")
    if status == "completed":
        return None
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            return None
        if reason is None and (response.get("metadata") or {}).get("ng_termination_reason") in _NO_ANSWER_REASONS:
            return None
        return f"incomplete response ({reason})"
    return f"unsuccessful response status {status}"


def assistant_text(response: dict) -> str:
    """The concatenated assistant output text of a Gym response."""
    text = []
    for item in response.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text.append(content["text"])
    return "".join(text)
