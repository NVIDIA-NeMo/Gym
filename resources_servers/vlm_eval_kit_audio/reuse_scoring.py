# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Pilot B: reuse VLMEvalKitMcore's NATIVE scorers inside NeMo Gym (no torch).

Wraps the mcore fork's own scoring code (imported, not rewritten) for:
  * OCRBench_v2   -- rule-engine scoring via
    ``vlmeval.dataset.utils.ocrbrnch_v2_eval.process_predictions``
  * CharXiv (reasoning) -- LLM-judge prompt construction + judge-response
    parsing via ``vlmeval.dataset.charxiv.auxeval``

The mcore fork lives at /home/mj/repos/forks/VLMEvalKitMcore (branch
mmikulski/gym-generic-audio) and is installed editable (--no-deps) in the Gym
venv. ``vlmeval.dataset``'s package __init__ imports every dataset module,
several of which import torch/torchvision/cv2/decord/av at module scope even
though the scoring code paths never touch them. ``_install_import_stubs``
below registers a meta-path finder that serves inert stub modules for those
heavy roots so the import chain completes without GPU/vision deps.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import sys
import types

# ---------------------------------------------------------------------------
# Import stubs for heavy optional deps (torch & friends).
# ---------------------------------------------------------------------------

_STUB_PREFIXES = ("torch", "torchvision", "cv2", "decord", "av", "timeout_decorator")


class _Any:
    def __getattr__(self, k):
        return _Any()

    def __call__(self, *a, **kw):
        return _Any()

    def __mro_entries__(self, bases):
        return (object,)


class _StubMeta(type):
    """Attributes of stub modules are real classes (scipy probes
    ``issubclass(x, torch.Tensor)``, which requires an actual class)."""

    def __getattr__(cls, k):
        return _Any()


class _StubModule(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__") and k != "__version__":
            raise AttributeError(k)
        v = _StubMeta(
            k,
            (),
            {
                "__init__": lambda self, *a, **kw: None,
                "__call__": lambda self, *a, **kw: _Any(),
                "__getattr__": lambda self, k2: _Any(),
            },
        )
        setattr(self, k, v)
        return v


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        m.__version__ = "0.0.0"
        return m

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in _STUB_PREFIXES):
            return importlib.machinery.ModuleSpec(name, _StubLoader(), is_package=True)
        return None


def _install_import_stubs() -> None:
    # Only stub modules that are genuinely absent; never shadow real installs.
    global _STUB_PREFIXES
    present = []
    for p in _STUB_PREFIXES:
        try:
            if importlib.util.find_spec(p) is not None:
                present.append(p)
        except (ImportError, ValueError):
            pass
    _STUB_PREFIXES = tuple(p for p in _STUB_PREFIXES if p not in present)
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


_install_import_stubs()

# Native mcore imports (AFTER stubs). These are the actual scoring entry
# points -- no logic is rewritten here.
from vlmeval.dataset.charxiv import auxeval as _charxiv_auxeval  # noqa: E402
from vlmeval.dataset.utils.ocrbrnch_v2_eval import (  # noqa: E402
    process_predictions as _ocrbench_v2_process_predictions,
)

# ---------------------------------------------------------------------------
# OCRBench_v2 -- rule-engine scorer
# ---------------------------------------------------------------------------


def _maybe_literal(value, sentinel):
    """Mirror of the field decoding in OCRBench_v2.evaluate
    (vlmeval/dataset/image_vqa.py, class OCRBench_v2)."""
    if isinstance(value, str) and value != sentinel:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def score_ocrbench_v2(verifier_extra: dict, prediction: str) -> float:
    """Score one OCRBench_v2 sample with mcore's native rule engine.

    ``verifier_extra`` carries the original TSV row fields:
    ``category`` (a.k.a. type), ``question``, ``answer`` (list or its repr),
    ``eval``, ``bbox``, ``content``.

    Glue only: this rebuilds the exact per-sample dict that
    ``OCRBench_v2.evaluate`` builds from the eval xlsx, then calls
    ``process_predictions`` on a single-item list. The score is written
    in-place by mcore code into ``entry['score']``.
    """
    answers = _maybe_literal(verifier_extra.get("answer"), None)
    if not isinstance(answers, list):
        answers = [answers]
    entry = {
        "type": verifier_extra.get("category"),
        "question": verifier_extra.get("question"),
        "predict": prediction if isinstance(prediction, str) else str(prediction),
        "answers": answers,
        "bbox": _maybe_literal(verifier_extra.get("bbox", "without bbox"), "without bbox"),
        "content": _maybe_literal(verifier_extra.get("content", "without content"), "without content"),
    }
    eval_field = verifier_extra.get("eval", "without eval")
    if eval_field and eval_field != "without eval":
        entry["eval"] = eval_field
    scored = _ocrbench_v2_process_predictions([entry])
    return float(scored[0].get("score", 0.0))


# ---------------------------------------------------------------------------
# CharXiv (reasoning) -- LLM-judge prompt + parsing
# ---------------------------------------------------------------------------


class _EchoJudge:
    """Minimal judge-model stand-in whose .generate() returns a pre-recorded
    judge response, so mcore's ``auxeval`` can be reused verbatim for
    parsing (vlmeval/dataset/charxiv.py::auxeval)."""

    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, **kwargs):  # signature mirrors judge models
        return self._text


def charxiv_judge_prompt(verifier_extra: dict, prediction: str) -> str:
    """Build the CharXiv judge prompt exactly as mcore does.

    Verbatim logic from vlmeval/dataset/charxiv.py::auxeval (line 27):
        prompt = line["grading_query"].replace("{PREDICTION}", prediction)
    The ``grading_query`` column ships in the CharXiv TSV and already embeds
    the question and gold answer.
    """
    grading_query = verifier_extra["grading_query"]
    return grading_query.replace(
        "{PREDICTION}", prediction if isinstance(prediction, str) else ""
    )


def charxiv_parse_judge(judge_text: str) -> float:
    """Parse a CharXiv judge response into a score using mcore's own parser.

    Calls ``vlmeval.dataset.charxiv.auxeval`` with an echo judge, so the
    json.loads + schema validation ("score" / "extract_answer" keys, dict
    check, failure fallback of 0.0) is mcore code, not a copy.
    """
    import pandas as pd

    line = pd.Series({"grading_query": "{PREDICTION}", "prediction": ""})
    result = _charxiv_auxeval(_EchoJudge(judge_text), line, retry=1)
    return float(result.get("score", 0.0))
