# Description

OfficeEnv simulates a set of connected SaaS apps (Slack, Gmail, Google Calendar, Google Sheets,
HubSpot, Jira and about a hundred more; `GET /officeenv/apps` lists them) for each rollout. A
row's `officeenv` task spec says which apps are connected, the data they hold, the acting user,
how the apps reach the policy (`tools`, `api_search`, `cli`, `mcp`), the faults and data
injections that happen during the episode, and the final-state checks that produce the reward.
Episodes are deterministic: the same spec and the same tool calls give the same world and reward.

The environment is the `officeenv` package (pydantic and the standard library), a separate
repository. This directory is the Gym integration: `app.py` routes Gym's endpoints to OfficeEnv's
framework-neutral episode service (`officeenv.service.EpisodeService`), and `requirements.txt`
pins the release it runs.

## The OfficeEnv version

`requirements.txt` pins a release tag:

```
officeenv @ git+ssh://git@gitlab-master.nvidia.com:12051/dmolinari/officeenv.git@v1.9.0
```

OfficeEnv is not published yet, so building this server's venv needs SSH read access to that
repository (`git+https://gitlab-master.nvidia.com/...` works instead where a git credential helper
or `~/.netrc` holds a token). On a cluster, build the venv on a login node and reuse it
(`skip_venv_if_present=true`), or bake it into the container with `gym env prefetch`.

Release numbers follow OfficeEnv's `docs/RELEASING.md`: a patch release changes no verdict for an
existing row, a minor one may add features or move verdicts (each listed in OfficeEnv's
`CHANGELOG.md`), a major one may refuse rows emitted by an older major version.

To move to a new release:

1. change the tag in `requirements.txt`;
2. emit `data/example.jsonl` again with that release (below), collect `data/example_rollouts.jsonl`
   again (see Run), and regenerate `data/example_metrics.json`:
   `gym dataset collate "+config_paths=[resources_servers/officeenv/configs/officeenv.yaml]" +output_dirpath=data/officeenv +mode=example_validation`;
3. `gym env test +entrypoint=resources_servers/officeenv +should_validate_data=true`.

To run an unreleased OfficeEnv, put its source first on `PYTHONPATH` when starting the servers
(`PYTHONPATH=<officeenv checkout>/src gym env start ...`). Gym keeps the caller's `PYTHONPATH` for
every server, so that tree shadows the pinned release, and `/verify` reports its version.

## Rows

```json
{"responses_create_params": {"input": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "tools": [...]},
 "officeenv": {"task_id": "...", "seed": 1, "actor": {...}, "apps": {...}, "exposure": {"mode": "api_search"}, "checks": [...]},
 "officeenv_version": "1.9.0",
 "task_id": "..."}
```

`officeenv emit <task.json>... -o rows.jsonl` writes them (`officeenv.emit.gym_row` from Python).
The tool list is the one the emitting version serves for the spec, and `officeenv_version` records
that version:

- `/seed_session` refuses a row from another major version (400, `incompatible_row`); rows
  without a stamp (emitted before 1.0.0) run as they are;
- a row whose tool list differs from what the server serves (a renamed tool, other parameters)
  still runs; `/seed_session` and `/verify` report the tools in `tool_drift`.

The example rows are five of OfficeEnv's built-in examples: one per exposure mode, plus a task
with a scripted user.

```bash
E=<officeenv checkout at the pinned tag>/src/officeenv/examples
officeenv emit $E/renewal_escalation.json $E/schedule_kickoff.json $E/vendor_terms_review.json \
  $E/ticket_triage.json $E/sprint_handover.json -o data/example.jsonl
```

The example rollouts were collected on 2026-10-28 with `azure/openai/gpt-5.5` against OfficeEnv
1.9.0, through this server and the simple agent (concurrency two). All five score 1.0. Every
error the metrics count is an authored fault (a transient error, rate limits) or a tool mistake
the model recovered from. They are integration smokes, not a benchmark: `schedule_kickoff` scores
0.75 whenever the model books its slot over a meeting injected during the episode
(`gcalendar.no_conflicts`), as it did on earlier runs.

Training and validation sets are not committed.

## Artifacts and corpora

A row may name its TaskSpec's large strings (a seeded workbook's base64, a long CSV) by SHA-256
instead of carrying them, in OfficeEnv's artifact format (`officeenv.artifacts`):

```json
{"$artifact": "sha256:<64 hex digits>", "encoding": "base64", "bytes": 10240}
```

A corpus is one directory, its rows files beside the store of the artifacts they name:

```
<corpus>/<rows>.jsonl
<corpus>/artifacts/<64 hex digits>
```

so it moves as one directory and is read with only its rows path changed.
`python -m officeenv.artifacts externalize rows.jsonl -o <corpus>/rows.jsonl` writes one, and
`check`, `pack` and `unpack` verify a corpus, write it as one archive and install it.

The server reads the references back at seed, for an Agent's seed and an Environment Server's
alike, from the stores in `artifacts_dirs`: absolute paths (the server runs in its own
directory), searched in order. Each artifact is checked
against its hash and size before OfficeEnv reads the TaskSpec. A row whose artifacts cannot be
read is refused with 400 (`missing_artifact`, `artifact_mismatch` or `invalid_artifact`) and
nothing is seeded; a row without references seeds as before. To serve a corpus:

```bash
gym env start --resources-server officeenv --model-type openai_model \
  "++officeenv_resources_server.resources_servers.officeenv.artifacts_dirs=[/data/corpus/artifacts]"
```

## Run

```bash
gym env start --resources-server officeenv --model-type openai_model
gym eval run --no-serve --agent officeenv_simple_agent \
  --input resources_servers/officeenv/data/example.jsonl \
  --output results/officeenv_rollouts.jsonl
```

The simple agent stops after `max_steps` (50) turns, or once the model answers without calling a
tool. Tasks explored through `api_search` spend a search and a fetch on every endpoint before
calling it, so long ones need more:
`+officeenv_simple_agent.responses_api_agents.simple_agent.max_steps=150`.

## Environment Server episodes and MCP

An Environment Server (for example `single_agent_turn`) seeds with a `ResourcesSeedSessionRequest`
whose `task_data` holds the row's fields other than `responses_create_params`. The OfficeEnv
session takes the request's `resources_session_id`, which the seed's session cookie also names;
the agent calls the same `POST /<tool_name>` routes with that cookie, and `/verify` grades the
episode as for an Agent. Seeding the same ID again for the same episode returns the same session,
and `/close_session` frees a world that verification has not. After a close, a seed of the same
ID is refused for `session_close_retry_window_seconds`. These seeds carry no tool list, so they
report no `tool_drift`.

With `expose_tools_over_mcp: true`, Gym's MCP endpoint serves every tool of OfficeEnv's `tools`
exposure under its tool name, and each episode's MCP token (in the seed response) admits only
that episode's tools. MCP clients read the tool definitions from that endpoint rather than from
the row, so these episodes need rows in the `tools` exposure with the default tool-name
separator; a seed whose tools differ from what the endpoint serves fails with 400
(`unsupported_exposure`).

## Verification

`/verify` grades the final state of the rollout's session and frees it. Besides `reward` (a
guarded weighted mean by default), it returns:

- `officeenv_checks`: per check, pass or fail with observed and expected values;
- `officeenv_metrics`, with flat numeric copies (`checks_passed`, `checks_total`,
  `guards_failed`, `policy_calls`, `errors`, `faults`, `injections`, `discovery_calls`,
  `unknown_tool_calls`, `write_events`, `tool_drift`) for metric aggregation;
- `officeenv_scenario_log`, `officeenv_integration_log`, `officeenv_timer_failures`: which faults
  and injections fired (or were held), cross-app effects, and timers that failed;
- `officeenv_server_version`: the OfficeEnv version that graded the rollout.

Checks on the policy's replies read every message it sent, including any it sent after completing
the task. A rollout without a session to grade (never seeded, verified already, closed, or lost
to a server restart) gets reward 0 with `mask_sample: true` and `failure_kind: session_lost`, and
a tool call without a session answers 404 with the same `failure_kind`.

## Training length penalty

`length_penalty.enabled=true` on the resources server deducts AutomationBench's training length
penalty (`benchmarks/automationbench`, same knobs and defaults) from rewards above a threshold:

```python
if reward <= reward_threshold:  # 0.5
    return reward
return reward - min(max_penalty, turns / turn_scale + tokens / token_scale)  # 0.5, 200, 200000
```

Turns are the model's responses in the episode; tokens are the completion tokens it generated
over all of them, reasoning included, from the response usage the simple agent accumulates (the
penalty fails explicitly when a rewarded episode has none). With the penalty on, `/verify` also
returns

- `reward_components`: `{"task_reward": <OfficeEnv's reward>, "length_penalty": -<deduction>}`,
  which sum to `reward`, so a trainer can apply the penalty on its own schedule (NeMo RL:
  `grpo.phasic_reward_components` with `components=[length_penalty]`);
- `task_reward`, `length_penalty` (the deduction), `model_turns` and `generated_tokens` as
  metrics.

Set it on training jobs only; with it off, rewards are OfficeEnv's. From NeMo RL, at
AutomationBench's half strength:

```
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.enabled=true
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.turn_scale=400
++env.nemo_gym.officeenv_resources_server.resources_servers.officeenv.length_penalty.token_scale=400000
```

# Licensing information
Code: this server, Apache 2.0; the `officeenv` package, NVIDIA, not yet published (its open-source release is under review)
Data: the example rows are synthetic, emitted from OfficeEnv's built-in examples

Dependencies
- nemo_gym: Apache 2.0
- officeenv: NVIDIA, not yet published; its only dependency is pydantic (MIT)
