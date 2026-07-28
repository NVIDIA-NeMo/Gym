# Remote Agent

A thin agent server that brokers rollouts to an agent service you host yourself — in your own
repo, on your own infrastructure. Your service implements one endpoint, `POST /v1/responses`:
it receives the task's `responses_create_params`, runs its own agent loop (its own model and
tools, however many turns it needs), and returns one finished Responses API trajectory.

Gym keeps everything else on its side: this server seeds the session, holds the session
cookies, verifies the trajectory on the resources server (`verifier_metadata` never leaves
Gym), and reports the verify response — so your rollouts land in the standard artifacts and
work with `gym eval profile`, aggregation, and (when your service routes its model calls
through a Gym model server) token-id capture for training.

## Contract for your service

- `POST {agent_base_url}/v1/responses` with the row's `responses_create_params` as the JSON body.
- Return a single finished Responses API object: the last output item is an assistant message,
  no dangling tool calls, `usage` populated (`{input_tokens, output_tokens, total_tokens}`).
- Failures on Gym's side never crash a collection run: they are recorded as reward-0 rows in
  the failures sidecar and retried on resume.

## Gym-hosted tools (optional)

With `forward_session: true`, each request to your service carries two headers:
`X-NeMo-Gym-Resources-Server-Url` and `X-NeMo-Gym-Session-Cookie`. Echo the cookie on every
tool call you make against that URL and stateful environments work end to end. If your service
runs on a different machine, set `advertised_resources_url` to an externally reachable URL —
the default resolves to the resources server's bind address, which is typically a loopback
address only valid on the Gym host. Without
forwarding, tasks that declare tools are refused up front (instead of silently scoring 0)
unless you set `assume_remote_tools: true` because your service implements the declared tools
itself.

## Run

```bash
gym env start --resources-server <your_env> # plus this agent's config
gym eval run --no-serve +agent_name=remote_agent +input_jsonl_fpath=... +output_jsonl_fpath=...
```
