"""Verifier robustness — fuzz, determinism, and anti-reward-hacking fixtures.

Motivated by the RLVR-testing literature: verifiers should be fuzzed BEFORE a
model trains against them (a verifier bug becomes a learned behavior), scores
must be deterministic, no input may crash or hang a checker, and known gaming
strategies must score 0.
"""

from __future__ import annotations

import random
import time

import pytest

from responses_api_agents.swe_agents_constrained.grading.if_format.constraints import (
    AGENTIC_CONSTRAINT_REGISTRY,
    CONVERSATIONAL_CONSTRAINT_REGISTRY,
)
from responses_api_agents.swe_agents_constrained.grading.verifiers import (
    AGENTIC_VERIFIER_REGISTRY,
    CONVERSATIONAL_VERIFIER_REGISTRY,
)
from responses_api_agents.swe_agents_constrained.grading.verifiers.base import VerifierResult

_ALL_VERIFIERS = [
    (ctype.value, verifier,
     (AGENTIC_CONSTRAINT_REGISTRY.get(ctype).parameters
      if ctype in AGENTIC_CONSTRAINT_REGISTRY else {}))
    for ctype, verifier in AGENTIC_VERIFIER_REGISTRY.items()
] + [
    (ctype.value, verifier,
     (CONVERSATIONAL_CONSTRAINT_REGISTRY.get(ctype).parameters
      if ctype in CONVERSATIONAL_CONSTRAINT_REGISTRY else {}))
    for ctype, verifier in CONVERSATIONAL_VERIFIER_REGISTRY.items()
]

_FUZZ_INPUTS = [
    "",
    " \n\t \n",
    "a",
    "\U0001f389 \u00fcn\u00efc\u00f6d\u00e9 \u202ereversed\u202c text \x00-ish \uffff",
    "{" * 500 + "}" * 499,                      # unbalanced JSON braces
    '{"name": ',                                # truncated JSON
    "```python\n" * 50,                         # unclosed fences
    "Step 999999999999999999:",                 # huge int
    "-" * 10_000,                               # long diff-ish run
    "\\d{3} $ ^ ( [ regex metachars",           # regex-looking garbage
]


def _seeded_garbage(n_cases: int = 20, size: int = 2000) -> list[str]:
    rng = random.Random(1234)
    alphabet = "abc {}[]()\"':,.\n\t<>|#-*`$\\/=0123456789"
    return ["".join(rng.choice(alphabet) for _ in range(size)) for _ in range(n_cases)]


@pytest.mark.parametrize("name,verifier,params", _ALL_VERIFIERS,
                         ids=[v[0] for v in _ALL_VERIFIERS])
def test_verifier_never_crashes_and_returns_result(name, verifier, params):
    ctx = {"constraint_params": params, "prior_steps": [], "step_index": 0}
    for text in _FUZZ_INPUTS + _seeded_garbage():
        result = verifier.check(text, ctx)
        assert isinstance(result, VerifierResult), f"{name} returned {type(result)}"
        assert isinstance(result.passed, bool)


@pytest.mark.parametrize("name,verifier,params", _ALL_VERIFIERS,
                         ids=[v[0] for v in _ALL_VERIFIERS])
def test_verifier_is_deterministic(name, verifier, params):
    ctx = {"constraint_params": params, "prior_steps": [], "step_index": 0}
    for text in _FUZZ_INPUTS[:4] + _seeded_garbage(3):
        first = verifier.check(text, ctx)
        second = verifier.check(text, dict(ctx))
        assert first.passed == second.passed, f"{name} non-deterministic on {text[:40]!r}"


def test_no_catastrophic_regex_backtracking():
    # Every verifier must finish quickly on adversarially repetitive input.
    bomb = ("a" * 60 + "!") * 800 + "Action Input: " + "{" * 200
    ctx = {"prior_steps": [], "step_index": 0}
    for name, verifier, params in _ALL_VERIFIERS:
        start = time.monotonic()
        verifier.check(bomb, {**ctx, "constraint_params": params})
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"{name} took {elapsed:.1f}s on adversarial input"


# NOTE: the task-verifier anti-reward-hacking fixtures (gsm8k question echo,
# humaneval string injection, envfactory bounds) live in the agentic-if repo's
# copy of this test file — they exercise the task-reward axis
# (agentic-if infrastructure/task_verifiers.py), which is not part of the
# constraint grading core here.
