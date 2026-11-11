# Synthetic workflow tasks on the AutomationBench app simulator

A training environment for synthetic SaaS workflow tasks. It runs on the AutomationBench app simulator (world state,
API routes and assertion checks from `benchmarks/automationbench`). Tasks come from your own dataset; no benchmark
tasks are shipped or used. One environment serves three kinds of rows, chosen by keys in each row's `info`:

| kind | `info` keys | what changes |
|---|---|---|
| single-day | `initial_state`, `assertions` | one session, scored on the final world |
| multi-day | + `period_end`, `events`, `wait_tool_description`, `wait_until_description` | a `wait` tool and simulated time |
| multi-turn | + `user_card` and/or `requester_services` | a simulated requester answers when the agent ends its turn |

## Multi-day

The `wait` tool moves the clock forward. Each dated event (an approval arrives, a ticket is reopened, a record
changes) is applied when the clock passes its time, so the agent has to wait for the right day, read the world again
and act then. `events` is `[{"at": iso time, "patch": {...}}]`: a patch sets fields on a record (`app`, `collection`,
`id`, `set`) or adds one (`app`, `collection`, `add`).

Every successful state-changing `api_fetch` call is logged with the simulated time, and two more assertion types grade
when the agent acted:

- `acted_on_day` (`record`, `day`, optional `body_contains`): the first write whose URL names `record` happened on
  `day` (YYYY-MM-DD). Acting early fails it.
- `not_acted` (`record`): no write named `record`. It is a guardrail.

## Multi-turn

When the agent ends a turn without tool calls, a requester model answers as the person who sent the request:

- `user_card`: the requester's private notes. It answers questions from them and replies DONE when it has nothing to
  add (at most `max_user_turns` replies).
- `requester_services`: apps only the requester can use. They leave the agent's connected services, and when the agent
  asks, the requester operates them with `api_search` and `api_fetch` (at most `max_requester_tool_calls` calls per
  reply) and reports back. The agent has to hand those steps over with every value they need.

The requester is any OpenAI-compatible chat endpoint (`requester_base_url`, `requester_model`), called through Gym's
HTTP client. `requester_body` fields are merged over the built-in sampling values, so a `max_tokens` or `temperature`
there wins. An endpoint that needs a key gets it as a Bearer token, read from the environment variable named by
`requester_api_key_env` or from the file at `requester_api_key_file`. The requester sees only the visible
conversation, never reasoning or tool traffic. The environment refuses to load multi-turn rows when
`requester_base_url` or `requester_model` is unset. Each try ends after `requester_timeout_s` (default 300). A requester that still fails after
`requester_max_tries`, or keeps calling tools after its tool budget, ends the episode and sets the `requester_failed`
metric. A dated event that cannot be applied (a missing record, an invalid field) does the same, since it is a dataset
failure and not the agent's. The agent config in `config.yaml` lists
it in `mask_sample_metrics`, so `verifiers_agent` returns that rollout with `mask_sample: true`
(`failure_kind: verifiers_agent:requester_failed`) and the reward is not counted as a measurement.

## Reward and metrics

The reward is the guardrail-gated share of objectives over the world and day assertions, with the same split as
AutomationBench's headline: a broken guardrail scores 0, otherwise the score is the share of objectives met. With
`unavailable_call_break: true`, a call to a tool the agent was not given, or to a service not connected for it (a
requester-owned app included), also breaks a guardrail.

Metrics: `user_turns`, `requester_failed`, `empty_turn`, `unknown_tool_calls`, `out_of_scope_calls`,
`requester_service_calls`, `requester_tool_calls`, `violation`.

`replay(info, calls)` scores a scripted plan against a row, which checks that a task is solvable before training on it.

The dataset owns all text the policy reads: the `wait` tool's text comes from each row, and the requester's replies come
from the requester model.

## Limitations

Simulated time reaches the world's clock (`meta.current_time`, which upstream date queries read) and the write log
that `acted_on_day` and `not_acted` grade. It does not reach records the agent creates: many upstream AutomationBench
routes and schema defaults stamp new records with the wall clock (`datetime.now()`), for example QuickBooks `txn_date`
and Google Ads `conversion_time`. Only `acted_on_day` and `not_acted` grade timing. The environment refuses to load a row whose
world assertion names a date or time field (a key such as `txn_date` or `created_at`, or a `field`, `property` or
`column` such as `closeDate`), since its result would depend on the date of the run.

## Run

```bash
uv pip install -e benchmarks/automationbench -e environments/ab_synth
gym env start --config environments/ab_synth/config.yaml --model-type vllm_model \
    ++ab_synth.responses_api_agents.verifiers_agent.vf_env_args.tasks_path=$PWD/environments/ab_synth/data/train.jsonl
```

Set `requester_base_url` and `requester_model` in `vf_env_args` for multi-turn rows. No dataset is shipped. Put the rows
at `data/train.jsonl`, one JSON object per line, for example a single-day row:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Mark contact 101 in HubSpot as a customer."}]},
 "info": {"initial_state": {"hubspot": {"contacts": [{"id": "101", "email": "a@example.com"}]}},
          "assertions": [{"type": "hubspot_contact_has_property", "contact_id": "101", "property": "lifecyclestage",
                          "value": "customer"}]},
 "task_idx": 0, "agent_ref": {"name": "ab_synth"}}
```

With the servers up, collect rollouts:

```bash
gym eval run --no-serve --config environments/ab_synth/config.yaml --model-type vllm_model \
    --agent ab_synth --input environments/ab_synth/data/train.jsonl --output results/ab_synth.jsonl
```

## Tests

The tests live with the agent that runs this environment, in
`responses_api_agents/verifiers_agent/tests/test_ab_synth_env.py`:

```bash
gym env test +entrypoint=responses_api_agents/verifiers_agent
```
