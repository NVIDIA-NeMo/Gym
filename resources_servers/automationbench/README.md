# AutomationBench resources server

Serves [AutomationBench](https://github.com/zapier/AutomationBench) tasks as typed resources sessions, for any agent
that calls tools over direct HTTP. `benchmarks/automationbench/config.yaml` pairs it with `simple_agent` through the
`single_agent_turn` Environment Server; see [the benchmark README](../../benchmarks/automationbench/README.md).

- **`/seed_session`** builds the task's simulated SaaS world (Gmail, Salesforce, Google Sheets, ...) from the row's
  `info.initial_state`, scoped to the services the task uses, as upstream's verifiers environment does.
- **`POST /api_search`, `/api_fetch`, `/base64_encode`** are upstream's `api` toolset, called on the session's world
  through Gym's `automationbench_env` (`api_fetch` takes `params` and `body` as JSON objects). The model gets each
  tool's text as it is, and an unknown tool, bad arguments or a failing call as error text, as under verifiers.
- **`/verify`** scores the final world with the benchmark's rubric. The reward is `aa_headline` by default: the share
  of objectives met, or 0 when any guardrail broke. The response also carries upstream's `partial_credit`,
  `task_completed_correctly`, the guardrail and objective counts, `task_name` and `domain`. A rubric that raises
  masks the sample as `verifier_error`.
- **`/close_session`** drops the world.

Configuration (`configs/automationbench.yaml`):

- `reward_fn`: `aa_headline` (default) or `partial_credit`. Both are reported either way.
- `export_assertion_results`: add every assertion's verdict to the verify response as `assertion_results`.

A row whose `responses_create_params` carries no tools was prepared before this server existed; `/verify` masks its
sample as `verifier_error` and says to re-run `gym eval prepare --benchmark automationbench`.

Sessions are kept in the worker's memory and are not exported to partial-rollout checkpoints: an episode interrupted
by a crash starts over.

The tool dispatch comes from verifiers, as a transitional choice for exact tool parity with the `verifiers_agent`
recipe: `tool_env()` builds Gym's `AutomationBenchEnv` on a one-row placeholder dataset, and each call goes through
its `update_tool_args` and `tool_map`. That keeps verifiers' argument handling and `automationbench_env`'s `api_fetch`
overrides. The cost is the verifiers dependency: `pyproject.toml` pins verifiers at the same git revision as
`verifiers_agent` and copies its uv constraint, override and exclude lists. Dropping verifiers would mean calling
upstream `automation-bench`'s tools directly and carrying those `api_fetch` overrides here.

## Licensing information

Code: Apache 2.0
Data: MIT (AutomationBench)

Dependencies

- nemo_gym: Apache 2.0
- automation-bench: MIT
- verifiers: Apache 2.0
