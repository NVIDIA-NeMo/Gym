# BrowseComp Sandboxed Agent

`browsecomp_sandboxed_agent` is `browsecomp_agent` with one change: `bash_command`
runs in a per-rollout sandbox instead of on the resources server host.

| Part | Where it runs |
|---|---|
| Agent loop (context reset, progress board, nudges, `update_progress`) | Agent server (host), unchanged |
| Model calls | Agent server to the Gym model server, unchanged |
| `search`, `browse`, contamination filter, provider keys | Resources server (host), unchanged |
| `bash_command` (read-only pocketshell) | Sandbox |
| `/verify` and judge | Resources server (host), unchanged |

## How bash_command runs

1. On the first `bash_command` of a rollout, the agent creates a sandbox and
   unpacks the runtime into `<sandbox_root>/rt`: `pod_bash.py`, `bash_guard.py`
   and pocketshell from `resources_servers/browsecomp_advanced_harness`. They
   need only the Python standard library, so a stock `python:3.12-slim` image works.
2. Before each `bash_command`, the agent calls the resources server's
   `/workspace_files` route for the pages written since the last call and
   unpacks them into `<sandbox_root>/ws`.
3. The agent runs `pod_bash.py` there. The guards, pocketshell and output format
   are the same code as the host path, so the model sees the same output.
4. When the rollout ends, the agent stops the sandbox. Rollouts without a
   `bash_command` create no sandbox.

The sandbox holds no credentials and needs no network. The shipped config denies
all egress.

## Run

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` (see
[OpenSandbox setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)),
then use `benchmarks/browsecomp/sandboxed.yaml` in place of `benchmarks/browsecomp/config.yaml`:

```bash
gym env start \
    --config benchmarks/browsecomp/sandboxed.yaml \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml
```

## Known difference from the host path

pocketshell uses the optional `regex` package for `grep -P` lookaround and
possessive syntax. The stock image does not have it, so in the sandbox `grep -P`
uses the standard-library `re` engine. Other commands are unaffected.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
