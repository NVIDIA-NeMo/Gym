# Description

A minimal tool-calling agent. For each task it calls the model server, runs every returned
`function_call` as `POST /<name>` on the resources server, appends the results as
`function_call_output` items, and repeats until the model answers without calling a tool, the
response is incomplete, or `max_steps` is reached.

Tool calls are only sent to the resources server when the name is a plain identifier, is not one of
the server's own endpoints (`seed_session`, `verify`, `aggregate_metrics`), and, when the request
declares function tools, is one of them. Arguments must be a JSON object. Other calls get an error
result the model can see.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `max_steps` | unlimited | Maximum number of model calls per task |
| `echo_items` | `off` | Print each task's items as they happen: `pretty` or `json` (see below) |
| `echo_file` | stdout | Append the echo to this file instead |
| `echo_max_chars` | 2000 | Truncate each item's text in `pretty` output (0 = no limit) |

## Echoing items

With `echo_items` set, the agent prints each task's items as they happen: the input messages when
the task starts, every item of each model response (assistant text, function calls, reasoning),
and each tool result.

- `pretty` prints a readable block per item: a `[simple_agent:<step>] <kind>` header line, the
  text, and a blank line. A final `[simple_agent] episode <status> after <n> step(s)` line ends
  each task.
- `json` prints each item exactly as it arrives, one JSON object per line, in the Responses API
  format.

Neither format identifies the task, so the output of tasks running at the same time is
interleaved.

The global overrides `+simple_agent_echo_items=pretty|json` and `+simple_agent_echo_file=<path>`
apply to every `simple_agent` in a run, whatever its config name:

```bash
gym env start --resources-server mcqa --model-type openai_model +simple_agent_echo_items=pretty
```

[`examples/simple_agent.py`](../../examples/simple_agent.py) sets these with `--echo` and prints the
echo to its own output.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
