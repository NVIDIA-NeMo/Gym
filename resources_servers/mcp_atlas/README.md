# MCP-Atlas Resources Server

[MCP-Atlas](https://arxiv.org/abs/2602.00933) (Scale AI) measures multi-step tool use against real MCP servers.
This server runs the public set (500 tasks, [`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas)
at revision `8c563b55`) with the servers of the upstream agent-environment image
(`ghcr.io/scaleapi/mcp-atlas:1.2.7`, [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas)).

## How it is measured

- **Tools.** Each task offers the model its `ENABLED_TOOLS` (7 to 37 tools from 2 to 11 servers), with the schemas
  the image serves (`tool_catalog.json`). Tools of servers the image does not ship are dropped, as upstream does.
- **Sandbox.** Every rollout gets a private agent-environment sandbox that serves only the task's servers, so
  stateful servers (filesystem, memory, git, desktop-commander) start clean. Tool calls go to its `/call-tool`
  on sandbox-local loopback through the host's OpenSandbox exec connection, with no declared MCP port and
  caching off. This transport alone does not establish outbound network isolation. Except for configuration
  failures described below, a tool error reaches the model as the sandbox's error text and an empty result
  as `success`. Bridge/provider transport timeouts are infrastructure failures.
- **Limits.** 256 model turns (`max_steps`) and 100 tool calls; calls past the 100th return an error that asks
  for the final answer. The [pinned upstream evaluation driver](https://github.com/scaleapi/mcp-atlas/blob/5c11ac44e96d31b91b92a0affad5f2648255907a/run_eval.py#L344-L345)
  uses a default 1800 s per-task request deadline and records a timeout as an `ERROR` response. This
  resources server sets no per-task wall-clock deadline. Its bridge timeout defaults to 1860 s, giving a
  response the upstream task budget plus 60 s of transport headroom; this is not a tool execution limit.
  The bridge timeout and sandbox TTL do not enforce the upstream task deadline.
- **Concurrency.** Tool calls within one rollout are serialized under its session lock. Verification and
  close wait for the active call before stopping its sandbox. `tool_call_timeout_s` is a fresh per-call
  bridge/provider budget (1860 s by default); cancellation cleanup can add delay, and the sandbox's
  remaining TTL can expire earlier. Different rollouts can run concurrently.
- **Answer.** The final answer is the last assistant message with non-empty text.
- **Judge.** Each ground-truth claim is judged on its own against the final answer only, with upstream's
  `score_claims.py` prompt, structured output and temperature 0. The default judge is
  `gemini-3.1-pro-preview` (the upstream default) on Google's OpenAI-compatible endpoint. Set
  `mcp_atlas_judge_api_key`, and `mcp_atlas_judge_base_url` / `mcp_atlas_judge_model_name` to use another endpoint.
- **Score.** A claim scores 1 / 0.5 / 0 (fulfilled / partially / not). `coverage` is the mean, rounded to three
  places. `reward` is 1 when coverage is at least 0.75, so `mean/reward` is the upstream pass rate.
  `passed_at_0_50` and `coverage` are also reported. An empty answer, or one that starts with `ERROR:`, scores 0
  without a judge call.

## Who failed

The verify response records `mask_sample` and `failure_kind` to separate environment failures from
model outcomes:

| `failure_kind` | Meaning | Masked |
|---|---|---|
| `mcp_atlas:sandbox_start_failed` | The sandbox did not start or never became healthy. | always |
| `mcp_atlas:sandbox_unreachable` | A bridge/provider request failed or exceeded its transport deadline. | always |
| `mcp_atlas:tools_missing` | A server of the task did not come up, so some task tools were missing. | always |
| `mcp_atlas:session_lost` | Verify found no seeded session. | always |
| `mcp_atlas:service_error` | A service answered with a quota, rate-limit, TLS or server error. | if `mask_service_errors` |
| `mcp_atlas:tool_timeout` | A timeout marker saved by an earlier server version. | if `mask_service_errors` |
| `judge_failed` | A judge call failed, or `judge_max_attempts` calls gave no readable verdict for a claim (set by Gym's `judge_failsafe`; upstream scores such a claim 0). | always |

Recognized authentication failures from credentialed, enabled services on trusted startup or tool-error
paths are configuration failures. The resources server's tool route returns HTTP 422 with
`detail: "MCP-Atlas authentication failed for <tool_name>; check credentials/scopes. The environment is unavailable; give your final answer."`.
The affected session does not make further upstream tool calls or judge its answer, and verification
rejects it without producing a reward or masked result. An ordinary closed or lost session instead
returns a masked `session_lost` result; closing an authentication-rejected session does not erase its
rejection during the configured close retry window. HTTP 422 is a resource-boundary signal. Simple Agent
presents the tool's 422 body to the model as tool output and may continue model turns. It is the later
`/verify` 422 that becomes `/run` HTTP 500. A startup `/seed_session` 422 can instead produce `/run` HTTP
500 before model turns. With the collector default
`route_failures_to_sidecar: false`, this failed `/run` stops collection after preserving unscored failure
evidence. With the option set to `true`, collection records an ungraded, nonterminal agent failure row and
continues. Inspect that evidence and fix the credentials/scopes before using results.

Successful HTTP 200 tool content is opaque data: text resembling an authentication or service error is
not evidence of failure. Tools that need no credentials cannot establish a service-credential failure.
On tool calls to a credentialed, enabled service, a trusted HTTP 401 is an authentication failure. A bare
HTTP 403 or `Forbidden` diagnostic remains a service failure unless it provides explicit credential or
scope evidence.
Recognized GitHub primary or secondary rate-limit diagnostics with HTTP 403 remain `service_error` outcomes,
subject to `mask_service_errors`, rather than credential rejections.
During startup, a 403 without credential/scope evidence does not establish an authentication failure.
If health or tool listing cannot become ready, it remains a masked `sandbox_start_failed` outcome.

An API can report an auth or quota failure inside HTTP 200 content without setting the MCP error flag.
Such content is not automatically rejected or masked. Before benchmark scoring, make a known-good call
to every credentialed service used by the task set and capture its actual success and error formats
without credential values.

An execution-deadline error actually returned by an MCP tool for model-written code remains a model
observation and is scored normally. Returned errors from `calculator`, `cli-mcp-server`,
`desktop-commander`, `filesystem`, `git`, `memory`, `mcp-code-executor` and `mcp-server-code-runner` are
also scored, even when their text says `Client failed to connect` or `session was closed`: model code
and files can produce arbitrary text. An actual bridge or provider transport timeout is instead
`mcp_atlas:sandbox_unreachable`, an infrastructure failure. A legacy saved `tool_timeout` marker does not
establish that a model program timed out; it retains its historical service-failure masking policy.
Calling a local program tool before a transport failure does not establish that the program caused the
failure: provider loss, sandbox TTL expiry and a terminated bridge can report the same error. Attribution
of model-caused environment destruction requires runtime evidence beyond that ordering; it remains an
acceptance check before scoring, rather than turning unknown infrastructure failures into model zeroes.
The attribution gap also applies when a command terminates an enabled server outside that local-tool
list: returned errors with a recognized leading connection/session diagnostic remain `service_error`,
masked when `mask_service_errors` is set. Returned errors from the listed local tools are still scored;
for example, a returned connection error from `memory` or `filesystem` is a model observation.
`service_errors` in a scored verify response lists its recorded failures. `gym eval reverify` recomputes
the score and mask from saved `service_errors` and `tool_calls`, preserving that legacy attribution.

### Check rollouts before publishing

Before committing genuine rollout artifacts, use the service credential values used for collection,
provided through approved environment configuration, and scan every JSONL file being published. This
includes rollout, failure and materialized-input sidecars. For a dedicated output directory, run from
the repository root:

```bash
python -m resources_servers.mcp_atlas.validate_rollouts results/mcp_atlas/*.jsonl
```

The scanner checks raw JSONL and decoded JSON strings for exact nonblank service credential values.
It reports only file/line locations, credential names and counts, never values or surrounding text.
It exits 1 for a match and 2 for missing or malformed inputs. A successful scan reports how many
credential values were checked. Zero checked values is legitimate for a task set using only key-free
services, but does not establish that credential values are absent from artifacts. Confirm the count
matches the actual configured services. This check does not detect arbitrary encoded exfiltration or
replace credential scoping and enforced network isolation. Run it on the actual artifacts; absent or
synthetic example rollouts do not establish publication readiness.

## Setup

The current integration is pending validation. The five Slack schemas match the pinned package and upstream
export (see Licensing). An authenticated runtime listing, working Slack credentials and the seeded workspace
still need validation. The OpenSandbox transport and policy configuration require real deployment and
rollout validation.
Collect and inspect five genuine example rollouts with a model before treating the integration as validated.
An evaluated Python tool must be unable to reach other rollouts or the NeMo Gym control plane. See the
[evaluation tutorial](https://docs.nvidia.com/nemo/gym/main/tutorials/evaluation-tutorials/mcp-atlas).

1. Include the approved OpenSandbox provider config and set `sandbox_image` to the OCI reference
   `ghcr.io/scaleapi/mcp-atlas:1.2.7` (or an approved deployment rewrite of it). The benchmark config includes
   `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml` and selects its named `sandbox` block.
   Provider credentials stay in the host client. Validate image architecture, user permissions, `/root`
   tool/config access and writable private `/tmp` on the actual deployment before using this path.
2. Supply an operator-reviewed `sandbox_options.network_policy` with explicit `defaultAction: deny`.
   The server rejects policy fields the installed SDK silently drops or changes, including unsupported
   port restrictions. Configure the full policy through `mcp_atlas_sandbox_options`; no external-service
   allowlist is invented here. Required task services and bootstrap hosts must work under the same policy.
   A policy request is not enforcement evidence: verify IP-level filtering (DNS-only filtering is
   insufficient), peer/host/Gym/provider/metadata protection, IPv6 handling and protected policy-management
   access from evaluated code. Prove this with concurrent A/B private-marker tests and host-proxy positive
   controls before benchmark scoring. No network or isolation acceptance is claimed by unit tests.
3. Export the service credentials under the upstream names (`service_env_names`, for example `EXA_API_KEY`,
   `GITHUB_TOKEN`, `NOTION_TOKEN`). Each sandbox receives only known service credentials required by its
   `ENABLED_SERVERS`; unrelated known credential names are removed even when supplied through `sandbox_env`.
   Startup preflight reads available datasets declared by this resources server or agents using it,
   including materialized task inputs and canonical `task_data` rows. Relative JSONL paths follow the
   controller's search root order captured before worker launch, including configured `--search-dir`
   roots. The resolved files must be readable on the resources server host. A resolution failure gives
   guidance to supply an absolute `jsonl_fpath` or an appropriate `--search-dir`.
   Missing credentials fail startup with `CredentialConfigurationError`. Inputs awaiting a declared
   preparation script are checked at each seed: missing credentials, unknown server names or an empty
   server list reject the seed with HTTP 422 before provider allocation. Recognized authentication failures
   in health or tool listing also reject configuration after sandbox cleanup, rather than becoming masked
   startup failures.
   Nonblank values establish presence, not valid scopes or account state.
   411 of the 500 ground-truth trajectories call at least one key-gated server; five servers (Airtable,
   Google Calendar, Notion, MongoDB, Slack) also need the upstream `data_exports/` seeded into the accounts.
   Keep management, model and judge credentials host-side; do not put them in `sandbox_env`.
4. Allow the task's service endpoints and any bootstrap hosts needed when a server misses its image cache.
   `sandbox_files` pins package resolution to the image date (2026-07-17). OpenSandbox defaults preserve
   the image's private uv/npm caches; Apptainer development keeps writable scratch caches. An explicit
   `sandbox_files` mapping replaces these defaults, including an empty mapping.

`sandbox_ttl_s` is passed to the provider and controls the in-process idle reaper. Verify the deployment's
TTL behavior. Shutdown reports failed stops and propagates the error, retaining ownership for a later
cleanup retry while the process is alive. If cleanup cannot finish before process exit, an operator must
inspect and stop the remaining instances.

### Development transports

The explicit `configs/mcp_atlas_apptainer_development.yaml` selects Apptainer and
`sandbox_development_mode: true`. Build a local SIF with readable/writable image paths:

```bash
apptainer build --sandbox mcp-atlas docker://ghcr.io/scaleapi/mcp-atlas:1.2.7
chmod -R a+rX mcp-atlas && chmod -R a+rwX mcp-atlas/root mcp-atlas/data mcp-atlas/agent-environment
apptainer build mcp-atlas-1.2.7-nonroot.sif mcp-atlas
```

Set `sandbox_image` to that file. The component requires `--cleanenv`/`-e` or `--containall`/`-C` on both
instance-start and exec paths, including custom provider configurations. Externally inherited
`APPTAINERENV_` and `SINGULARITYENV_` variables are rejected: unset them before starting this server.
Raw `--env`/`--env-file` options in `create.extra_start_args` or `exec.extra_exec_args` are also rejected.
Set workload values through `sandbox_env` so task credential scoping applies; the provider's internal
environment file generated from `SandboxSpec.env` is retained. These guards constrain environment
forwarding; they do not establish isolation.
The provider shares host networking, so separate filesystems/ports are not a scoring boundary. Apptainer
does not enforce provider `ttl_s`; an exited host process can leave instances requiring operator cleanup.

For development, `sandbox_base_url` points at one running agent-environment
(`docker run -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.7`). Every rollout then shares it, so
stateful servers leak between tasks. This also requires `sandbox_development_mode: true`. Neither
development transport may be used for benchmark scores.

## Differences from upstream

- The upstream harness stops at the 100th tool call; here the model gets an error and may still answer.
- The upstream evaluation driver's default per-task request deadline is 1800 s (linked above); this
  resources server sets no per-task wall-clock deadline.
- Recognized service-authentication failures reject configuration before judging, instead of becoming
  scored model observations or masked service failures.
- The local bridge/provider deadline is an infrastructure failure. An error returned by a tool for a
  model's program remains scored; real package deadline behavior still needs qualification before
  protocol comparisons.
- Upstream counts a claim whose judge call fails 8 times as not fulfilled; here the rollout is masked.

## Licensing

The claim-normalization rules in `benchmarks/mcp_atlas/prepare.py` and the prompt, verdict schema and scoring
rules in `verifier.py` are adapted from
[`services/scoring/score_claims.py`](https://github.com/scaleapi/mcp-atlas/blob/5c11ac44e96d31b91b92a0affad5f2648255907a/services/scoring/score_claims.py)
at upstream revision `5c11ac44e96d31b91b92a0affad5f2648255907a`, under the
[MIT license](https://github.com/scaleapi/mcp-atlas/blob/5c11ac44e96d31b91b92a0affad5f2648255907a/LICENSE),
Copyright (c) 2026 Scale. Those files retain the MIT notice; NVIDIA modifications and the local NeMo Gym
integration are Apache-2.0. The component and source-to-file mapping are recorded in
[`ATTRIBUTIONS.md`](../../ATTRIBUTIONS.md#mcp-atlas).

The server-to-credential-name registry in `credentials.py` is derived from upstream's
[`mcp_server_template.json`](https://github.com/scaleapi/mcp-atlas/blob/5c11ac44e96d31b91b92a0affad5f2648255907a/services/agent-environment/src/agent_environment/mcp_server_template.json)
at the same revision, under MIT, Copyright (c) 2026 Scale. The file retains that notice; credential
validation, per-task scoping and dataset preflight are independently implemented NVIDIA modifications.

The public dataset and the example tasks are from Scale AI's
[`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas/tree/8c563b55d7c967755f474299848049834d624617)
at revision `8c563b55d7c967755f474299848049834d624617`, under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Preparation changes the row format, extracts and
normalizes the claims, and selects the image-supported tool schemas. These are changes to the data's
representation; they do not change its attribution or license. `tool_catalog.json` records the image from
which its MCP schemas were exported. Its 5 `slack_*` schemas are copied from the upstream README's
[list-tools export](https://gist.githubusercontent.com/geobio/d0272d41ea395376233f1617a3988860/raw/4444343be47c0874fd19631b20caf2738b0168a4/list-tools.json)
(Gist revision `3c562bd9468516a1d80dfadfc8a4dc1e24ed967a`, SHA-256
`ca173815297dece47eb5384b53b9f6b77ad3e74987751e9000cf7ef0aa32fea5`), keeping only `name`, `description` and
`inputSchema`. They are the static tool definitions of `slack-mcp-server@1.1.23`, the version the image pins
([korotovsky/slack-mcp-server](https://github.com/korotovsky/slack-mcp-server/blob/21eaf72a3be33e7f3fd44c61a07645bedc376ff4/pkg/server/server.go)
at `21eaf72a3be33e7f3fd44c61a07645bedc376ff4`, MIT, Copyright (c) 2025 Dmitrii Korotovskii), which registers
the same five tools with matching parameters, types and required fields. The catalog header retains the
source revision and full MIT notice. The image and its installed services are external runtime dependencies.
