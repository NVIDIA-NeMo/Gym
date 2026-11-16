# Unified video environment

One Gym configuration that serves four video environments at once, and one mixed rows file that routes each row
to its environment, following [Multi-Environment Training](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/multi-environment-training).
It adds no agent, verifier or server: it composes members that exist, so a model can be trained or evaluated on
video captioning, video question answering and two families of video agentic tasks in the same run instead of one
run per task.

## Members

| Task | Environment | Rows | Routed by | Reward |
| --- | --- | --- | --- | --- |
| Video captioning with an LLM judge | [`resources_servers/video_caption`](../../resources_servers/video_caption/README.md) | frames + `atomic_events` | `agent_ref` = `video_caption_simple_agent` | 0..1 |
| Video-MME multiple choice | [`resources_servers/videomme_v1`](../../resources_servers/videomme_v1/README.md) | `input_video` + question | `agent_ref` = `videomme_v1_single_turn` | 0/1 |
| Agentic-VBench video editing (generated training tasks) | [`environments/harbor/agentic_vbench_train`](../harbor/agentic_vbench_train/README.md) | Harbor task folders, `opencode_agent` in the task image | taskset `agentic_vbench_train` | task verifier, 0..1 |
| ClawEval video agent tasks | [`environments/harbor/claweval_train`](../harbor/claweval_train/README.md) | Harbor task folders, `opencode_agent` in the task image | taskset `claweval_train` | native grader, 0..1 |

The first two are JSONL environments: `config.yaml` lists their configs and `prepare.py` stamps `agent_ref` on
their rows. The last two are task-folder datasets run by the generic
[`harbor` environment server](../../environment_servers/harbor/README.md): their rows are materialized by a run of
the entry and carry their taskset, which is what routes them, in every routing mode. `config.yaml` sets
`environment_routing_mode: agent` so one batch holds both kinds of rows.

## Run the JSONL members

```sh
# Terminal 1: both members' servers plus one policy model. The caption judge and tokenizer come from the
# VIDEO_CAPTION_* variables documented by that environment.
gym env start --config environments/unified_video_agentic/config.yaml --model-type vllm_model

# Terminal 2: one run over the mixed rows; each row's agent_ref picks the member.
gym eval run --no-serve \
    --input environments/unified_video_agentic/data/example.jsonl \
    --output results/unified_video_example.jsonl
```

`data/example.jsonl` holds five rows (three Video-MME, two captioning) built from the members' own example rows:

```sh
python environments/unified_video_agentic/prepare.py --example
```

## Run all four

The Harbor members are prepared once from their task roots (NVIDIA-internal for both), and each entry is run once
with `--limit 1`, which writes its rows and its run config:

```sh
python -m benchmarks.agentic_vbench.prepare_harbor_env --tasks-root <agentic-vbench root>
python -m benchmarks.claweval.prepare_harbor_env --tasks-root <claweval root> --claw-eval-source <claw_eval checkout>
for entry in agentic_vbench_train claweval_train; do
  gym eval run "$entry" --agent opencode_agent --sandbox docker \
    --config "environments/harbor/$entry/harbor_train.yaml" --limit 1 \
    --model-type vllm_model --model <model> --model-url <url> --model-api-key <key>
done
# results/harbor/{agentic_vbench_train,claweval_train}/{tasks.jsonl,run_config_opencode_agent.yaml} exist now.
```

A training run then lists, in this order, the model server, the two run configs, the two overlays and this entry's
config (last, so its routing mode wins), and the mixed rows file from the next section as its data:

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - results/harbor/agentic_vbench_train/run_config_opencode_agent.yaml
      - environments/harbor/agentic_vbench_train/harbor_train.yaml
      - results/harbor/claweval_train/run_config_opencode_agent.yaml
      - environments/harbor/claweval_train/harbor_train.yaml
      # optional, both or neither: the two entries' compaction_plugins.yaml
      - environments/unified_video_agentic/config.yaml
    sandbox: docker   # or the provider the run configs name
```

The two overlays set the shared `opencode_agent` instance (image input, token capture, the time ceiling, the
context and output budgets) to the same values, so their order does not matter; a value for one member only goes
under that taskset's agent instance, `harbor_agentic_vbench_train_agent` or `harbor_claweval_train_agent`, which
the run configs derive from the base. The harbor server resolves each row's task folder through the run config's
taskset mapping, so the entries must stay where they were prepared. The judge key for both verifiers is
`NVINFERENCE_API_KEY` in the server's environment (the overlays' `secrets`).

## Build a training mixture

```sh
python environments/unified_video_agentic/prepare.py \
    --member video_caption_simple_agent=/data/video_caption_train.jsonl \
    --member videomme_v1_single_turn=/data/videomme_v1_train.jsonl:0.5 \
    --taskset results/harbor/agentic_vbench_train/tasks.jsonl \
    --taskset results/harbor/claweval_train/tasks.jsonl:0.6 \
    --output /data/unified_video_train.jsonl
```

Each `--member` is `NAME=PATH[:WEIGHT]`: `NAME` is the agent instance in the config, `PATH` a JSONL the member's
own pipeline produced (rows already carry `responses_create_params` and `verifier_metadata`), `WEIGHT` scales the
member's share (`round(weight * rows)`, a weight above 1 repeats rows). Each `--taskset` is `PATH[:WEIGHT]` for a
Harbor member's `tasks.jsonl`; its rows already name their taskset and are taken as they are. The script stamps
`agent_ref` on the JSONL members' rows, refuses a row that already routes elsewhere, shuffles with `--seed`
(default 0) and writes `<output>_mix.json` with the per-member counts. Media is never read or rewritten: a row's
`file://` or `data:` URLs must resolve where the policy model runs, as each member documents.

Batch sizing follows the multi-environment tutorial: enough prompts per step that every member appears several
times, or the sparser member's reward estimate is noise. For RL frameworks that route by `task_source` (NeMo RL's
Gym integration), every row keeps it: the JSONL members' as their pipelines wrote it, the Harbor members' as the
entry name.

## Scope

Rewards are each member's own; nothing here normalizes across members. There is no manifest: this entry declares
no verifier or dataset of its own, so `gym env validate --config` on the config, the members' own validation and
`tests/unit_tests/benchmarks/test_unified_video_agentic_prepare.py` are the checks that apply.
