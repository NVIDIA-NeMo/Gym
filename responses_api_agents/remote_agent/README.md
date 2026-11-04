# Remote Agent

An agent server that drives an agent service you host yourself — in your own repo, on your own
infrastructure. Your service implements one endpoint **compliant with the OpenAI
`/v1/responses` contract**, and the two servers compose as Responses-speaking agents: each call
your service receives the conversation so far and returns what it wants to do next. Gym runs
the loop.

## Who does what

Gym: seeds a fresh environment session per rollout and holds its cookies (session state and
`verifier_metadata` never reach your service), executes the tool calls your service asks for
against the resources server, appends the results and calls your service again, converts every
failure into a reward-0 sidecar row (never a crashed run), and verifies the finished trajectory.

Your service: answers each call with a valid Responses API object. Within a call it can do
anything — its own model turns, its own tools, sub-agents; there are exactly two reasons to
return: it needs a Gym-hosted tool executed, or the rollout is finished. It tolerates being
called N times per rollout (each request carries the full conversation; set a cookie if you want
per-rollout state — Gym echoes your cookies back within the rollout) and reports full `usage` or
omits it.

## The response contract

- `function_call` items **without** a matching `function_call_output` (same `call_id`, same
  response) are asks: Gym executes them on the resources server and feeds the results back as
  `function_call_output` items on the next call.
- `function_call` + `function_call_output` **pairs** are your own internal tool records; they
  pass into the trajectory untouched.
- An assistant `message` with no unpaired calls finishes the rollout; Gym merges the whole
  conversation into one trajectory and verifies it.
- Unknown tool names and malformed arguments are not crashes: the error text comes back to you
  as that call's output and the rollout continues. An invalid Responses object is a terminal,
  non-retried failure.

The tool schemas your service may ask for arrive in every request's `tools` field, verbatim from
the dataset row.

## Run

```bash
gym env start --resources-server <your_env> # plus this agent's config
gym eval run --no-serve +agent_name=remote_agent +input_jsonl_fpath=... +output_jsonl_fpath=...
```

Knobs and the full contract: see the "Drive a Remote Agent" docs page.
