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

`tools_mode` decides who serves the tools a dataset declares:

- `refuse` (default): tool-declaring tasks are rejected up front with a clear error instead of
  silently scoring zero against untouched session state.
- `forward`: each request to your service carries two headers, `X-NeMo-Gym-Resources-Server-Url`
  and `X-NeMo-Gym-Session-Cookie`. Echo the cookie on every tool call you make against that URL
  and stateful environments work end to end. The URL is re-sent per request because Gym assigns
  servers random ports on every start; the cookie is minted per rollout.
- `remote`: your service implements the declared tools itself; nothing is forwarded.

### Running the service off-host (`tools_mode: forward`)

The advertised URL must be reachable *from your service's machine* — Gym serves the tools and
tells your service where they are; making that address route to Gym is on you:

1. Bind the resources server on all interfaces and pin its port (`host: 0.0.0.0`, `port: <fixed>`).
2. Make the path route (internal DNS / firewall rule / SSH tunnel / load balancer — your infra).
3. Verify once from the remote machine: `curl http://<address>:<port>/` should connect.
4. Set `advertised_resources_url: http://<address>:<port>` on this agent. It changes only the
   header string; the default advertises the bind address, which is typically a loopback address
   other machines cannot reach (you'll see a warning for that combination).
5. Recommended: have your service probe the advertised URL on its first request and fail loudly —
   reachability is only testable from your side (see the self-check in the docs example).

## Run

```bash
gym env start --resources-server <your_env> # plus this agent's config
gym eval run --no-serve +agent_name=remote_agent +input_jsonl_fpath=... +output_jsonl_fpath=...
```
