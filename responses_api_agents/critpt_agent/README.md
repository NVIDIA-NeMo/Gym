# CritPt Agent

Custom two-turn agent for the [CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt)
research-level physics benchmark. The built-in `simple_agent` does not work for CritPt; this agent
drives the two LLM calls required per problem:

- **Turn 1 — solve:** the policy model reasons about the physics problem.
- **Turn 2 — populate template:** the model fills the problem's code template from its Turn 1
  solution (`turn2_prompt_fpath`). The result is submitted to the resources server for batched
  evaluation via the [Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api).

## Composition

Wired together in [`benchmarks/critpt/config.yaml`](../../benchmarks/critpt/config.yaml):

- `responses_api_agents/critpt_agent` — this agent
- `responses_api_models/*` — typically `policy_model`
- `resources_servers/critpt` — the verifier (see [its README](../../resources_servers/critpt/README.md))

## Tests

```bash
ng_test +entrypoint=responses_api_agents/critpt_agent
```
