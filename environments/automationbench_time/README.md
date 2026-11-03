# AutomationBench over simulated business days

A training environment on the AutomationBench world (`benchmarks/automationbench`) where a task spans several
simulated days. The agent gets a `wait` tool that moves the world clock forward. Each task's dated events (an
approval arrives, a ticket is reopened, a record changes) are applied when the clock passes their time, so the agent
has to wait for the right day, read the world again and act then.

Every successful state-changing `api_fetch` call is logged with the simulated time. Besides the usual world
assertions, a task can grade when the agent acted:

- `acted_on_day` (`record`, `day`, optional `body_contains`): the first write whose URL names `record` happened on
  `day` (YYYY-MM-DD). Acting early fails it.
- `not_acted` (`record`): no write named `record`. It is a guardrail.

The reward is the guardrail-gated headline used for AutomationBench (`aa_headline`) over the world assertions plus
these day assertions: a broken guardrail scores 0, otherwise the score is the share of objectives met.

## Task rows

Gym rows with `responses_create_params.input` and an `info` object that holds, next to AutomationBench's
`initial_state` and `assertions`:

| key | meaning |
|---|---|
| `period_end` | the latest time the clock can reach (ISO 8601) |
| `events` | `[{"at": iso time, "patch": {...}}]`, a patch sets fields on a record (`app`, `collection`, `id`, `set`) or adds one (`app`, `collection`, `add`) |
| `wait_tool_description` | the `wait` tool's description shown to the policy |
| `wait_until_description` | the description of its `until` parameter |

The tool text comes from the rows, so the dataset owns all text the policy reads. `replay(info, calls)` scores a
scripted plan against a row, which checks that a task is solvable before training on it.

## Run

```bash
uv pip install -e benchmarks/automationbench -e environments/automationbench_time
gym env start --config environments/automationbench_time/config.yaml --model-type vllm_model \
    ++automationbench_time.responses_api_agents.verifiers_agent.vf_env_args.tasks_path=$PWD/environments/automationbench_time/data/train.jsonl
```

No dataset is shipped. Put the rows at `data/train.jsonl`.

## Tests

```bash
pytest environments/automationbench_time/tests
```
