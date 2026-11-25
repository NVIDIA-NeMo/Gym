# Synthetic workflow tasks on the AutomationBench app simulator

A training environment for synthetic SaaS workflow tasks. It runs on the AutomationBench app simulator (world state,
API routes and assertion checks from `benchmarks/automationbench`). Tasks come from your own dataset, and no benchmark
tasks are shipped or used. One environment serves three kinds of rows, chosen by keys in each row's `info`:

| kind | `info` keys | what changes |
|---|---|---|
| single-day | `initial_state`, `assertions` | one session, scored on the final world |
| multi-day | + `period_end`, `events`, `wait_tool_description`, `wait_until_description` | a `wait` tool and simulated time |
| multi-turn | + `user_card` and/or `requester_services` | a simulated requester answers when the agent ends its turn |

## Multi-day

The `wait` tool moves the clock forward, and each dated event (an approval arrives, a ticket is reopened) is applied
when the clock passes its time, so the agent has to wait for the right day, read the world again and act then.
`events` is `[{"at": iso time, "patch": {...}}]`: a patch sets fields on a record (`app`, `collection`, `id`, `set`) or
adds one (`app`, `collection`, `add`). The `wait` reply gives the new time in the seed's zone (naive UTC for a naive
seed).

Every `api_fetch` write that changes the world is logged with the simulated time and the records it added, changed or
deleted. A failed write or a read-only POST (a search) is not an act. Two more assertion types grade when the agent
acted:

- `acted_on_day` (`record`, `day`, optional `body_contains`): the first write that changed `record` happened on `day`
  (YYYY-MM-DD, in the zone of the seed's `meta.current_time`). Acting early fails it.
- `not_acted` (`record`): no write changed `record`. It is a guardrail.

`record` names a record of the seed or one an event adds: by its key (`id`, else `sid`, else the collection's primary
key such as Xero's `contact_id`), a Jira issue by its `key` (`OPS-1`), a sheet row by its `row_id` or `id`. A string
that names no record is read as an A1 reference to rows of one tab (`B7`, `Sheet1!A7:C7`, `'Q4 2025'!B7`,
`Sheet1!A2:B5`), where a row's number is its `row_id`, or its position in a tab of string row ids (the first row is 2).
A write is graded on the records it changed, not on its URL. It also acts on a few parents, such as a calendar event's
calendar, a Slack message's channel, a sheet row's spreadsheet and tab. A write an app keeps only in its action log acts
on the records its params name, such as a Jira comment's issue or a Drive move's file and folders. Apps that keep only
an action log and no records (Trello, Notion, Asana, ...) cannot be graded with these two assertions.

## Refused and masked rows

Load checks every row and leaves out the ones that cannot be graded, logging the count (it fails only when no row is
left): a `record` that names nothing or more than one record, an `acted_on_day` `day` that is not YYYY-MM-DD, a
multi-day row without its two wait texts or with a time that is not ISO 8601, a dated event that cannot apply to the
seed (load runs the events in time order on a copy of it), and a world assertion on a field the simulator stamps from
the wall clock (see Limitations). Each rollout repeats these checks on the row Gym sends it, so a refused row that still
arrives ends before its first turn, masked as `env_row_invalid`, never a server error.

An event on a record the agent (or the requester it asked) deleted is skipped and the rollout is scored. An event that
still fails when it fires, or a requester that fails after `requester_max_tries`, ends the episode with
`requester_failed` set. `config.yaml` lists both metrics in `mask_sample_metrics`, so `verifiers_agent` returns such a
rollout with `mask_sample: true` and its reward is not counted.

## Multi-turn

When the agent ends a turn without tool calls, a requester model answers as the person who sent the request:

- `user_card`: the requester's private notes. It answers questions from them and replies DONE when it has nothing to
  add (at most `max_user_turns` replies).
- `requester_services`: apps only the requester can use. They leave the agent's connected services, and when the agent
  asks, the requester operates them with `api_search` and `api_fetch` (at most `max_requester_tool_calls` calls per
  reply) and reports back. The agent has to hand those steps over with every value they need.

The requester is any OpenAI-compatible chat endpoint (`requester_base_url`, `requester_model`), called through Gym's
HTTP client, and sees only the visible conversation, never reasoning or tool traffic. `requester_body` fields are
merged over the built-in sampling values. A key, if the endpoint needs one, is read from the environment variable named
by `requester_api_key_env` or the file at `requester_api_key_file` and sent as a Bearer token. Each try ends after
`requester_timeout_s` (default 300). Multi-turn rows do not load without `requester_base_url` and `requester_model`.

## Reward and metrics

The reward is the guardrail-gated share of objectives over the world and day assertions, with the same split as
AutomationBench's headline: a broken guardrail scores 0, otherwise the score is the share of objectives met. With
`unavailable_call_break: true`, a call to a tool the agent was not given, or to a service not connected for it (a
requester-owned app included), also breaks a guardrail.

Metrics: `user_turns`, `requester_failed`, `env_row_invalid`, `empty_turn`, `unknown_tool_calls`, `out_of_scope_calls`,
`requester_service_calls`, `requester_tool_calls`, `violation`.

`replay(info, calls)` scores a scripted plan against a row, which checks that a task is solvable before training on it.

The dataset owns all text the policy reads: the `wait` tool's text comes from each row, and the requester's replies come
from the requester model.

## Limitations

Simulated time reaches the world's clock (`meta.current_time`, which upstream date queries read) and the write log
that `acted_on_day` and `not_acted` grade. It does not reach the timestamps the simulator stamps on records with the
wall clock (`datetime.now()`): `created_at`, `updated_at`, `sent_at`, `ts` and the others in `STAMPED_FIELDS`. A world
assertion on one of those (as a key, or as the `field` or `property` it checks) cannot match a fixed value, so its row
is refused. Date fields the agent writes from the task (`due_date`, `start.dateTime`, `close_date`, a sheet column) are
graded like any other field. A field the simulator fills from the clock only when the agent leaves it out (QuickBooks
`txn_date`) is graded too, so a task that checks one must name the date.

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
`responses_api_agents/verifiers_agent/tests/test_ab_synth_env.py` and `test_ab_synth_records.py`:

```bash
gym env test +entrypoint=responses_api_agents/verifiers_agent
```
