# example_session_state_mgmt

A counter environment that keeps per-episode state on the resources server. Each episode gets a session ID, the
server maps that ID to one integer, tools change it, and `/verify` scores the stored value. The model never holds
the count.

```text
cookie -> request.session[SESSION_ID_KEY] -> session_id_to_counter[session_id] -> int
```

For the annotated walkthrough and a rollout transcript, see the
[Stateful Environment](https://docs.nvidia.com/nemo/gym/main/environment-tutorials/stateful-environment) tutorial.
This README covers what is in this folder and how to run it.

## What it demonstrates

| Idea | In this example |
|---|---|
| Several tool calls in one episode | `increment_counter`, then `get_counter_value` |
| State per episode | `session_id_to_counter: Dict[str, int]` |
| Episodes stay separate | Different session IDs get different counters |
| Reward reads server state | `reward = 1.0` only if the stored count equals `expected_count` |
| Checkpoints | `checkpoint_mode = "exported"`: the counters can be saved and restored |

The reward comes from the stored counter, not from the model's answer: if the model says "7" but the counter is 5,
the reward is `0.0`.

One episode from `client.py`:

| Step | Request | Counter after |
|---|---|---|
| Seed | `POST /seed_session` `{"initial_count": 0}` | `0` |
| Tool | `POST /increment_counter` `{"count": 4}` | `4` |
| Tool | `POST /increment_counter` `{"count": 3}` | `7` |
| Tool | `POST /get_counter_value` | returns `{"count": 7}` |
| Score | `POST /verify` `{"expected_count": 7}` | `reward = 1.0` |

## Files

| File | Purpose |
|---|---|
| [`app.py`](app.py) | `StatefulCounterResourcesServer`: the counter store, tools, seed, verify, and checkpoint hooks |
| [`nooa_app.py`](nooa_app.py) | `NOOAStatefulCounterResourcesServer`: the same counter behind typed resource sessions |
| [`task_data.py`](task_data.py) | Row fields `initial_count` and `expected_count` |
| [`configs/example_session_state_mgmt.yaml`](configs/example_session_state_mgmt.yaml) | This server with `simple_agent` |
| [`configs/example_session_state_mgmt_nooa.yaml`](configs/example_session_state_mgmt_nooa.yaml) | `nooa_app.py` with the NOOA agent, routed as `nooa-counter` |
| [`data/example.jsonl`](data/example.jsonl) | Five example tasks, written by [`create_examples.py`](create_examples.py) |
| [`data/example_nooa_tasks.jsonl`](data/example_nooa_tasks.jsonl) | The same tasks as NOOA task rows |
| [`client.py`](client.py) | One live episode against running servers |
| [`tests/`](tests/) | Session isolation, checkpoint export, and NOOA session tests; no model needed |

## How it works

### Session state in `app.py`

`SimpleResourcesServer` installs Gym's session middleware, which gives each episode a session ID. Every handler
reads it with `request.session[SESSION_ID_KEY]`:

1. `setup_webserver` adds `/increment_counter` and `/get_counter_value` to the inherited `/seed_session` and `/verify`.
2. `seed_session` stores `initial_count` with `setdefault`, so a second seed of the same session keeps the old value.
   The tutorial recommends direct assignment for new environments.
3. `increment_counter` and `get_counter_value` fall back to `0` for an unseeded session.
4. `verify` returns `reward = 1.0` only if the session exists and its counter equals `expected_count`.

### Checkpoints

The server declares `checkpoint_mode = "exported"` and implements three hooks:

- `export_session_states` returns `{"count": ...}` for each requested session it still holds and leaves out the rest.
- `restore_session_states` validates each saved state and loads it back into `session_id_to_counter`.
- `retire_session_state` drops a retired attempt's counter, so a later restore of the same ID still works.

### Several workers

With `num_workers > 1`, each worker has its own `session_id_to_counter`. Gym routes every request to the worker that
created the session, so an episode keeps seeing its own counter. The module-level `app` at the bottom of `app.py`
is what each uvicorn worker serves.

### NOOA resource sessions

`nooa_app.py` uses the typed `/seed_session` and `/close_session` that the NOOA agent and Environment Server call:

- The seed names its session with `resources_session_id`. Retrying it with the same body is safe; different task data
  gets a conflict.
- Close removes the counter and refuses reuse of that ID for `session_close_retry_window_seconds`.
- The tool routes refuse a session that was never seeded or is already closed.

## When to use in-memory state

The counters are a Python dict in the server process. A stop or crash loses them unless a checkpoint exported them
first.

| Fits | Use something else |
|---|---|
| Small state (a counter, a few flags) | Large world state, such as databases or file systems |
| Tutorials, unit tests, local evaluation | State that several separate servers must share |

Larger environments keep richer per-session state the same way, for example
[Workplace Assistant](../workplace_assistant/README.md), [ToolSandbox](../toolsandbox/README.md),
[Calendar](../calendar/README.md), and [ns_tools](../ns_tools/README.md).

## How to run it

From the Gym repository root, with `policy_base_url`, `policy_api_key`, and `policy_model_name` set in `env.yaml`
(see the [quickstart](https://docs.nvidia.com/nemo/gym/main/get-started)):

```bash
gym env start \
    --resources-server example_session_state_mgmt \
    --model-type openai_model
```

In a second terminal:

```bash
source .venv/bin/activate
gym eval run --no-serve \
    --agent example_session_state_mgmt_simple_agent \
    --input resources_servers/example_session_state_mgmt/data/example.jsonl \
    --output results/session_state_rollouts.jsonl
```

To run the NOOA variant end to end with a real model, set `POLICY_BASE_URL`, `POLICY_API_KEY`, and
`POLICY_MODEL_NAME`, then run:

```bash
bash responses_api_agents/nooa_agent/smoke/run_stateful_e2e.sh
```

## How to test it

```bash
gym env test --resources-server example_session_state_mgmt
```

`tests/test_app.py` turns off automatic cookie handling, then checks that two sessions on one server keep separate
counters (`2` and `7`) and that a checkpoint export leaves out a session the server no longer holds.
`tests/test_nooa_app.py` covers seed retries, close fencing, and tool routes refusing unseeded sessions.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
