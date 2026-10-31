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
  with caching off. A tool error reaches the model as the sandbox's error text, an empty result as `success`,
  and a call that takes more than 60 s as `Tool call timed out after 60s`, as in the upstream harness.
- **Limits.** 256 model turns (`max_steps`) and 100 tool calls; calls past the 100th return an error that asks
  for the final answer.
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

1. Build a SIF that a non-root user can run. The image keeps its pre-installed tools under `/root`, mode 700:
   ```bash
   apptainer build --sandbox mcp-atlas docker://ghcr.io/scaleapi/mcp-atlas:1.2.7
   chmod -R a+rX mcp-atlas && chmod -R a+rwX mcp-atlas/root mcp-atlas/data mcp-atlas/agent-environment
   apptainer build mcp-atlas-1.2.7-nonroot.sif mcp-atlas
   ```
2. Set `sandbox_image` to the SIF. The default provider runs it with
   `--userns --writable-tmpfs --no-home --cleanenv`, and `--cleanenv` on every exec, so this server's own
   environment never reaches the sandbox. Each sandbox needs about 1 to 4 GB of memory.
3. Export the service credentials under the upstream names (`service_env_names`, for example `EXA_API_KEY`,
   `GITHUB_TOKEN`, `NOTION_TOKEN`). Only the credentials that are set reach the sandbox, through the provider's
   private env file. 411 of the 500 ground-truth trajectories call at least one key-gated server; five servers
   (Airtable, Google Calendar, Notion, MongoDB, Slack) also need the upstream `data_exports/` seeded into the
   accounts.
4. The sandbox needs internet access to the services and to PyPI, npm and GitHub, because some servers install
   packages when they start. `sandbox_files` pins that resolution to the image date (2026-07-17).

For development, `sandbox_base_url` points at one running agent-environment
(`docker run -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.7`). Every rollout then shares it, so
stateful servers leak between tasks: do not use it for scores.

## Differences from upstream

- The upstream harness stops at the 100th tool call; here the model gets an error and may still answer.
- The upstream harness stops a task after 1800 s; this server sets no task timeout.
- Upstream counts a claim whose judge call fails 8 times as not fulfilled; here the rollout is masked.

## Licensing

Code: MIT (upstream). Data: CC-BY-4.0 (Scale AI).
