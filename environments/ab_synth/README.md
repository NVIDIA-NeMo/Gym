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
`id`, `set`, where `id` is the record's key as below) or adds one (`app`, `collection`, `add`). The `wait` reply gives the new time in the seed's zone (naive
UTC for a naive seed). A multi-day row without the two wait texts, with a time that is not ISO 8601, or with an
`acted_on_day` whose `day` is not YYYY-MM-DD is refused. A refused row is left out at load (each refusal is counted and
logged, and the load fails only when no row is left) and, when it reaches a rollout anyway, masked as `env_row_invalid`.

Every `api_fetch` write that changes the world is logged with the simulated time and the records it added, changed
or deleted (a failed write or a read-only POST such as a search changes nothing and is not an act). Two more
assertion types grade when the agent acted:

- `acted_on_day` (`record`, `day`, optional `body_contains`): the first write that changed `record` happened on
  `day` (YYYY-MM-DD, in the time zone of the seed's `meta.current_time`, UTC for a naive one). Acting early fails it.
- `not_acted` (`record`): no write changed `record`. It is a guardrail.

A write is graded on the world, not on its URL, so a record named only in the body (a QuickBooks delete, a Slack
channel) counts the same as one in the path. When a rollout starts, each `record` is resolved against the seed and the
records its events add, to the record with that key (`id`, else `sid`, else the collection's primary key, such as Xero's
`contact_id`), the Jira issue with that key (`OPS-1`), or the sheet row with that `row_id` or row id. Only a string that names no record is
read as an A1 reference to rows of one tab: `B7`, `Sheet1!A7:C7` or `'Q4 2025'!B7` (any columns), or rows
`Sheet1!A2:B5`. A row's number is its `row_id`, or its position in a tab of string row ids (the first row is 2), the
way the simulator addresses it, and a single row past a named tab's end is the row an append adds. A row whose record
names nothing, or more than one record (two collections, or a row number without a tab in a sheet with several tabs),
cannot be graded: the environment logs the count of such rows at load, and a rollout of one ends before its first turn
masked as `env_row_invalid`. Each rollout repeats the load checks on the row it receives (Gym sends the row's own
`info`, which need not come from `tasks_path`), and a row that fails them, or whose setup raises, ends the same way:
masked, never a server error. Apps that keep only an action log (Trello, Notion,
Asana, ...) hold no records, so their writes cannot be graded with these two assertions.

A write changes the records it adds, edits or deletes, and only these parents: a calendar event's calendar, a Facebook
post's page, a Slack message's channel, a sheet row's spreadsheet and tab. Writes an app keeps only in its action log
name their record through the action's params: a Jira comment its issue, a Google Drive move, rename, copy or delete
its file and the folders it adds to or leaves, a new Drive file its folder, a Pipefy card move or field update its card,
an Airtable record create, update or comment its base. Records that only point at
another (an opportunity's account, a ticket's requester) do not act on it. A row written through a URL that names its
spreadsheet or tab by title (in any case, as the simulator reads titles) is filed under, and credits, the same
spreadsheet and tab as one written by id: titles are mapped to ids by the simulator's own resolvers, so they match the
way it reads them. A tab is keyed within its spreadsheet, and a write to a tab acts on its spreadsheet.

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
failure and not the agent's. An event on a record that existed and is gone (the agent, or the requester it asked,
deleted it) is skipped instead, and the rollout is scored. The agent config in `config.yaml` lists
it in `mask_sample_metrics`, so `verifiers_agent` returns that rollout with `mask_sample: true`
(`failure_kind: verifiers_agent:requester_failed`) and the reward is not counted as a measurement.

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
`responses_api_agents/verifiers_agent/tests/test_ab_synth_env.py`:

```bash
gym env test +entrypoint=responses_api_agents/verifiers_agent
```
