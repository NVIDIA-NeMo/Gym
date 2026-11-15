# OpenThoughts-TBLite

[OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite) (Terminal-Bench Lite) is 100
Harbor-format terminal tasks that track Terminal-Bench 2 performance but run faster. Each task has its own agent
and verifier time limits, and its `tests/test.sh` grades the attempt.

| Config | Agent | Time limits |
| --- | --- | --- |
| `openthoughts_tblite` | [`harbor_agent_general`](../../responses_api_agents/harbor_agent_general/README.md): Harbor's Terminus-2 | each task's own (`task.toml`) |
| `openthoughts_tblite/terminus_2` | [`terminus_2_sandboxed_agent`](../../responses_api_agents/terminus_2_sandboxed_agent/README.md): Gym's Terminus-2 on the Terminal-Bench 2.1 resources server | 3600 s for every task (TB-Lite's longest) |

Use `openthoughts_tblite` for scores comparable with Harbor's and the
[OpenThoughts leaderboard](https://ot-agent-leaderboard.replit.app/).

> [!important]
> Harbor grades an attempt that hits its agent time limit with the task's tests. `harbor_agent_general` on main
> drops such attempts as failures instead, so `openthoughts_tblite` scores don't match Harbor's until
> [#4818](https://github.com/NVIDIA-NeMo/Gym/pull/4818) merges. Many TB-Lite attempts hit their limit.

## Prepare data

```bash
gym eval prepare --benchmark openthoughts_tblite
```

[`prepare.py`](prepare.py) fetches the tasks at commit `f075e46` (Harbor registry `openthoughts-tblite@2.0`)
into `data/tasks/` and writes `data/benchmark.jsonl`. The tasks ship only Dockerfiles, so it adds each task's
prebuilt image as `[environment] docker_image`. These are `docker.io/kbuchanan31150/tblite:<task>`,
byte-identical copies of NousResearch's `nousresearch/tblite-<task>:latest` under fixed tags. Nothing else in the
tasks changes. Run the prepare command above before either config; `gym eval run` uses the prepared data but does
not fetch missing tasks.

## Run

Install Gym with its Ray extra (`uv sync --extra ray`); `harbor_agent_general` runs each attempt as a Ray task.
Serve the model on an OpenAI-compatible endpoint, then:

```bash
gym eval run --benchmark openthoughts_tblite --split benchmark \
  --model <served-model-name> --model-url http://<host>:<port>/v1 --model-api-key EMPTY \
  --output results/openthoughts_tblite/rollouts.jsonl --resume
```

Harbor's Terminus-2 calls the model itself (`hosted_vllm/<model>` at `--model-url`, with `--model-api-key`). It
writes every attempt's trajectory and test output to its own folder under `results/openthoughts_tblite/harbor/`.
Each task runs 3 times; change this with `--num-repeats`.

Harbor accepts only model names of letters, digits, `.`, `-` and `_`, up to 64 characters. Serve a model such as
`nvidia/Nemotron-3-Super-120B-A12B` under a name without the `/` (vLLM: `--served-model-name`) and pass that name
as `--model`.

Set the agent's context limit and sampling for your model. Save this as `model.yaml` and add `--config model.yaml`:

```yaml
openthoughts_tblite:
  responses_api_agents:
    harbor_agent_general:
      harbor_agent:
        kwargs:
          temperature: 1.0
          interleaved_thinking: true  # keep earlier turns' reasoning in the history
          model_info:
            max_input_tokens: 230000  # Terminus-2 summarizes before this; keep it below the server's context
          llm_call_kwargs:
            top_p: 0.95
```

### Sandboxes

Tasks run in Docker by default. For OpenSandbox, add this to `model.yaml` and export `OPENSANDBOX_DOMAIN` and
`OPENSANDBOX_API_KEY` (Harbor's OpenSandbox client, `harbor[opensandbox]`):

```yaml
openthoughts_tblite:
  responses_api_agents:
    harbor_agent_general:
      harbor_environment:
        type: opensandbox
        kwargs:
          protocol: http
          use_server_proxy: true
```

### Gym's Terminus-2

```bash
gym eval run --benchmark openthoughts_tblite/terminus_2 --split benchmark --model-type vllm_model \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model <served-model-name> --model-url http://<host>:<port>/v1 --model-api-key EMPTY \
  --output results/openthoughts_tblite_terminus_2/rollouts.jsonl --resume
```

This runs the same rows on the Terminal-Bench 2.1 resources server, which applies one agent limit
(`sandbox_timeout`) and one verifier limit (`evaluation_timeout`) to every task. Its scores are not comparable
with Harbor's. For OpenSandbox, use `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml` instead of
the Docker config. Model-server settings are as in [Terminal-Bench 2.1](../terminal_bench_2_1/README.md).
