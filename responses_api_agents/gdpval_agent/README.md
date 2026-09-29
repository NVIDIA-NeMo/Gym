# GDPVal agent

Runs the [GDPVal](https://huggingface.co/datasets/openai/gdpval) benchmark with a pluggable
harness executing inside the GDPVal container, and scores the files the harness produced.

The benchmark's own agent, `stirrup_agent`, drives the model itself. This agent instead
provisions the container, stages the task's reference files, runs another Gym agent inside
it, and collects the deliverables. That makes GDPVal usable as a harness comparison.

## How it differs from stirrup_agent

`stirrup_agent` runs its loop on the host and sends only its shell tool into the container.
Here the whole harness process runs inside the container, so the harness needs no sandbox of
its own and its file tools write directly into the workspace.

The harness has no tool for declaring which files are its deliverables. Instead the prompt
names an output directory and only that directory is collected. If it is empty, files under
the workspace that are newer than the start marker are collected instead.

## Setup

Build the container:

```bash
apptainer build gdpval.sif responses_api_agents/stirrup_agent/containers/gdpval.def
```

Environment:

| Variable | Purpose |
| --- | --- |
| `GDPVAL_CONTAINER_PATH` | Absolute path to the `.sif`. |
| `PERSIST_DELIVERABLES_DIR` | Absolute path where deliverables are written for scoring. |
| `HF_TOKEN` | Downloading the dataset and its reference files. |
| `JUDGE_API_KEY` | Judge endpoint key. Use an `sk-` key; `nvapi-` keys reject multimodal payloads. |
| `JUDGE_BASE_URL`, `JUDGE_GPT_MODEL`, `JUDGE_GEMINI_MODEL`, `JUDGE_CLAUDE_MODEL` | Judge overrides. |

## Run

```bash
gym eval prepare --benchmark gdpval/hermes
gym eval run \
    --benchmark gdpval/hermes \
    --model-type vllm_model \
    --split benchmark \
    --output results/gdpval_hermes.jsonl
```

## Swapping the harness

Change `agent_server_module`, `agent_server_class` and `agent_config_class` in
`configs/gdpval_agent.yaml`. The dependency prefix is built from
`responses_api_agents/<harness>/scripts/<harness>_deps.sh`, so a harness needs that script to
exist. Set `agent_kwargs` to whatever that harness's config class accepts.

## Checking a run

The deliverables that reached the judge are on disk at
`$PERSIST_DELIVERABLES_DIR/task_<task_id>/repeat_<n>/`. They should be real documents, which
`file` will confirm, and they should be flat: the scorer does not descend into
subdirectories. A reward with no files there means the judge scored the model's closing
message rather than its work.
