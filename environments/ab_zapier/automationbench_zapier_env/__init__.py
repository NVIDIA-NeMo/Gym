"""AutomationBench with upstream Zapier scoring, exposed as a verifiers env.

The upstream `automationbench` package ships an eval CLI rather than a
`load_environment` entry point, so this thin module adapts it to the
`vf_env_id` interface the verifiers_agent uses.

Scoring is upstream `create_rubric()` verbatim. The toolset defaults to
`api`, matching upstream's CLI default; upstream's Python signatures default
to `zapier`, so it is set explicitly here rather than inherited.
"""

from automationbench.domains import DEFAULT_DOMAINS, get_combined_dataset
from automationbench.rubric import create_rubric
from automationbench.runner import AutomationBenchEnv


def load_environment(domains=None, max_turns: int = 50, toolset: str = "api",
                     search_top_k=None, **kwargs):
    dataset = get_combined_dataset(list(domains) if domains else list(DEFAULT_DOMAINS))
    return AutomationBenchEnv(dataset=dataset, rubric=create_rubric(),
                              max_turns=max_turns, toolset=toolset,
                              search_top_k=search_top_k, **kwargs)
