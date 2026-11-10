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
  caching off. This transport alone does not establish outbound network isolation. A tool error reaches
  the model as the sandbox's error text, an empty result as `success`,
  and a call that takes more than 60 s as `Tool call timed out after 60s`, as in the upstream harness.
- **Limits.** 256 model turns (`max_steps`) and 100 tool calls; calls past the 100th return an error that asks
  for the final answer.
- **Concurrency.** Tool calls within one rollout are serialized under its session lock. Verification and
  close wait for an active call, up to `tool_call_timeout_s` (60 s by default), before stopping its sandbox.
  Different rollouts can run concurrently.
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

The verify response sets `mask_sample` and a `failure_kind` when the environment failed, not the model:

| `failure_kind` | Meaning | Masked |
|---|---|---|
| `mcp_atlas:sandbox_start_failed` | The sandbox did not start or never became healthy. | always |
| `mcp_atlas:sandbox_unreachable` | A tool call could not reach the sandbox. | always |
| `mcp_atlas:tools_missing` | A server of the task did not come up, so some task tools were missing. | always |
| `mcp_atlas:session_lost` | Verify found no seeded session. | always |
| `mcp_atlas:service_error` | A service answered with an auth, quota, rate-limit, TLS or server error. | if `mask_service_errors` |
| `mcp_atlas:tool_timeout` | A tool call took longer than `tool_call_timeout_s`. | if `mask_service_errors` |
| `judge_failed` | A judge call failed, or `judge_max_attempts` calls gave no readable verdict for a claim (set by Gym's `judge_failsafe`; upstream scores such a claim 0). | always |

`service_errors` in the verify response lists every failure the rollout met. `gym eval reverify` judges the
saved final answers again; the saved `service_errors` and `tool_calls` take the place of the live rollout state,
so the masks stay the same.

## Setup

The current integration is pending validation. The five Slack schemas match the pinned package and upstream
export (see Licensing). An authenticated runtime listing, working Slack credentials and the seeded workspace
still need validation. The OpenSandbox transport and policy configuration require real deployment and
rollout validation.
An evaluated Python tool must be unable to reach other rollouts or the Gym control plane. See the
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
   `GITHUB_TOKEN`, `NOTION_TOKEN`). Only the credentials that are set reach the sandbox, through the provider's
   private env file. 411 of the 500 ground-truth trajectories call at least one key-gated server; five servers
   (Airtable, Google Calendar, Notion, MongoDB, Slack) also need the upstream `data_exports/` seeded into the
   accounts. Management, model and judge credentials must not be forwarded as workload credentials.
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

Set `sandbox_image` to that file. `--cleanenv` keeps host credentials out of the workload; the provider
shares host networking, so separate filesystems/ports are not a scoring boundary. Apptainer does not
enforce provider `ttl_s`; an exited host process can leave instances requiring operator cleanup.

For development, `sandbox_base_url` points at one running agent-environment
(`docker run -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.7`). Every rollout then shares it, so
stateful servers leak between tasks. This also requires `sandbox_development_mode: true`. Neither
development transport may be used for benchmark scores.

## Differences from upstream

- The upstream harness stops at the 100th tool call; here the model gets an error and may still answer.
- The upstream harness stops a task after 1800 s; this server sets no task timeout.
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
