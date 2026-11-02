---
name: gym-add-model-config
description: >
  Onboard a model's serving config for NeMo Gym evaluations: write the vLLM policy service
  (model path, served name, tensor/pipeline/data/expert parallelism, max model length, max
  sequences, tool-call and reasoning parsers, chat template, generation config, environment),
  wire it as the policy with the right model type, set the reasoning toggle through chat-template
  kwargs, validate the config without GPUs, smoke it with a small `--limit`, and read the resolved
  config. Also says what must stay fixed for a fair comparison and what deployment tuning may
  change. Use when a new model or checkpoint has to be evaluated in Gym. Not for running the
  evaluation itself (use gym-eval) or adding a benchmark (use add-benchmark). Triggered by:
  "add a model", "new checkpoint", "serve this model in Gym", "model config", "vllm config",
  "enable thinking", "reasoning parser", "tool call parser", "compare two models fairly".
---

# Add a Model Config to Gym

The Gym pages are the reference; when they and this skill disagree, the pages win:
[Configuration](../../../fern/versions/latest/pages/orchestration/configuration.mdx) (the `services`
and `driver` fields),
[Multi-node vLLM](../../../fern/versions/latest/pages/orchestration/how-to/multi-node-vllm.mdx) and
[vLLM model server](../../../fern/versions/latest/pages/model-server/vllm.mdx) (the `vllm_model`
fields, `chat_template_kwargs`, `extra_body`).

A model in a Gym evaluation has two parts:

- **The deployment**: a `vllm` service in the submit config, started with `vllm serve`, and named
  as the policy with `driver.policy_model`.
- **The client**: the model-type config Gym talks to it with (`driver.policy_model_type`, the
  `--model-type` of `gym eval run`). `openai_model` is the default; `vllm_model` adds reasoning
  extraction, `chat_template_kwargs`, `extra_body` and token information.

## 1. Read the model card first

Collect before writing anything: the context length, the recommended sampling settings, whether
the model reasons (and which vLLM reasoning parser its family uses), its tool-call format (and the
vLLM tool-call parser), whether its chat template has a reasoning switch and what it is called, and
whether its repository ships a `generation_config.json`.

Without a model card, read the checkpoint itself: `chat_template.jinja` (or `chat_template` in
`tokenizer_config.json`) for the reasoning switch and the tool-call format, `generation_config.json`
for sampling, and `config.json` or `tokenizer_config.json` for the context length. The parser names
come from the serving image: `vllm serve --help=all` lists the choices of `--tool-call-parser` and
`--reasoning-parser`. Pick the tool-call parser whose format the template emits, and say in your
notes which parser you chose and why.

## 2. Write the policy service

Write everything site-specific (images, paths, account, partition, hosts) as `${oc.env:VAR}` and
export the variables before you submit; an angle-bracket placeholder does not validate. This is the
smallest config that resolves and runs:

```yaml
services:
  policy:
    type: vllm
    container: ${oc.env:SERVING_IMAGE}
    model: /checkpoint # or <org>/<model> from the Hub
    served_model_name: <model name>
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    number_of_instances: 1 # data-parallel replicas
    trust_remote_code: false
    node_pool: gpu
    health_check:
      timeout_seconds: 1800 # the default 60 s is shorter than most models take to load
    extra_args: >-
      --max-model-len 65536
      --max-num-seqs 256
      --enable-auto-tool-choice --tool-call-parser <parser>
      --reasoning-parser <parser>
      --generation-config auto
    env:
      HF_HOME: lit:/cache/huggingface
      HF_TOKEN: runtime:HF_TOKEN
    mounts:
      - ${oc.env:CHECKPOINT_DIR}:/checkpoint:ro
      - ${oc.env:CACHE_DIR}:/cache

driver:
  container: ${oc.env:GYM_DRIVER_IMAGE} # Python 3.13.14 or newer; the default python:3.12 is older
  policy_model: policy
  policy_model_type: vllm_model
  gym_install:
    repo: https://github.com/NVIDIA-NeMo/gym
    ref: <full Gym commit>
  benchmarks:
    <benchmark>:
      run:
        config_paths: [benchmarks/<benchmark>/config.yaml]

job:
  output_path: ${oc.env:OUTPUT_DIR}

compute:
  cluster:
    type: slurm
    account: ${oc.env:SLURM_ACCOUNT}
    hostname: ${oc.env:SLURM_LOGIN_HOST} # null when you submit on the login node itself
    forward_env: [HF_TOKEN] # runtime: credentials, forwarded when submitting over SSH
    walltime: "04:00:00"
    node_pools:
      gpu:
        partition: ${oc.env:SLURM_PARTITION}
        nodes: 1
        gpus_per_node: 4

otel:
  enabled: false # on by default; it needs the collector in the job's environment
```

`driver.gym_install.ref` is the Gym the job installs and scores the run with: a full commit of
public Gym that has your benchmark. Recovery (`gym eval resubmit --recover`) needs a commit at or
after the one that ships this skill. The CLI you submit with may be newer than the pinned commit;
the job still installs the pinned one. Site `#SBATCH` directives (a QoS, a node to avoid) go in
`compute.cluster.extra_args`, without leading dashes: `{qos: <qos>, exclude: <node>}`.

| What | Where |
| --- | --- |
| Model path or name | `model`. A local checkpoint also needs a `mounts` entry. |
| Served name | `served_model_name`; Gym passes it as `policy_model_name`. |
| Tensor, pipeline parallel | `tensor_parallel_size`, `pipeline_parallel_size` (one instance). |
| Data parallel | `number_of_instances`. Gym picks the vLLM data-parallel, multi-node or Ray Serve layout from it and the node count (see Multi-node vLLM). |
| Expert parallel | vLLM's own flag in `extra_args`, for example `--enable-expert-parallel`. |
| Max model length, max sequences | `--max-model-len`, `--max-num-seqs` in `extra_args`. |
| Tool-call and reasoning parsers | `--enable-auto-tool-choice --tool-call-parser <p>`, `--reasoning-parser <p>` in `extra_args`. |
| Chat template | The model's own by default; `--chat-template <path>` in `extra_args` to replace it. |
| Generation config | `--generation-config auto` takes the model's `generation_config.json` defaults for any sampling value a request leaves unset; `vllm` takes vLLM's own. |
| Environment | `env`, each value prefixed `lit:`, `host:VAR` or `runtime:VAR`; shell statements that `env` cannot express go in `pre_command`. |

`extra_args` is passed to `vllm serve` verbatim: check each flag against the vLLM version in the
image. Gym does not validate them.

Two readiness limits: the job waits `health_check.timeout_seconds` for the policy service, and
the driver then waits `server_spinup_timeout_seconds` (under the benchmark's `run`, 600 s by
default) for the Gym servers it starts for the benchmark. A benchmark whose servers start slowly
ends the chain typed `server_spinup_timeout`; gym-eval says how to raise the limit on a saved run.

## 3. The reasoning toggle

A model whose chat template has a reasoning switch takes it through `chat_template_kwargs` on the
`vllm_model` client. The key name is model-specific; check the template.

```yaml
driver:
  benchmarks:
    <benchmark>:
      run:
        policy_model:
          responses_api_models:
            vllm_model:
              uses_reasoning_parser: true # true when vLLM runs a reasoning parser for this model
              chat_template_kwargs:
                enable_thinking: false
```

A benchmark's `run` mapping becomes `+key=value` overrides, applied after every config file, so
these values win over the benchmark's and the model type's own. The same keys work on the command
line: `+policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=false`.

- Set `uses_reasoning_parser` on the client to match the server: true when vLLM runs a reasoning
  parser for this model.
- Some benchmark configs set `chat_template_kwargs` (or `extra_body`) themselves. That is a
  benchmark setting: leave it as the benchmark sets it, because an override would replace it. The
  shipped `vllm_model` configs leave these keys unset, so a benchmark's value survives
  `--model-type vllm_model`, which merges after the benchmark config.
- A request's own `chat_template_kwargs` is used only with `forward_request_chat_template_kwargs:
  true` (see the vLLM model server page).

## 4. Validate without GPUs

```bash
# The submit config: schema, policy wiring, services and compute.
gym eval submit --config submit.yaml --resolve-only

# The job scripts, including the exact `vllm serve` command line.
gym eval submit --config submit.yaml --dry-run

# Each Gym server block against its server's config class, composed as `gym eval run` composes it.
gym env validate --server-types responses_api_models,responses_api_agents \
  --benchmark <benchmark> --model-type vllm_model --allow-job-time-values
```

`--allow-job-time-values` stands in for values that only exist at job time, such as the policy URL
that submit injects. `gym env validate` exits 0 with no invalid finding, 1 with one, and 2 when the
config does not compose; success prints nothing. It checks the benchmark's and the model type's own
server configs, not your submit config: the `run` overrides of sections 2 and 3 appear only in the
`--dry-run` job script, as `+key=value` tokens. Read them there. A key rendered twice (a dotted key
beside a nested block at the same path) fails the job on the cluster, after the GPUs are allocated.

## 5. Smoke it

Submit the model with one benchmark and a small task count before a full evaluation:

```yaml
driver:
  benchmarks:
    <benchmark>:
      run:
        limit: 5
        num_repeats: 1
```

`limit` counts tasks; the run log prints the tasks and the rows they become. Bound the smoke to
one allocation with the walltime and `resumable: {max_retries: 0, max_walltime: <the walltime>}`;
when it completes, it cancels the chain's pending follow-up and sentinel itself (see gym-eval, step 10).
Then follow the [gym-eval](../gym-eval/SKILL.md) skill: `gym eval resubmit "$GYM_JOB_ID"
<benchmark> --recover --inspect` and `gym eval health-check` (gym-eval, step 9). Look for answers that end mid-thought (the output token
budget or `--max-model-len` is too small), reasoning that lands in the answer (the reasoning parser
or `uses_reasoning_parser` is wrong), and tool calls that arrive as plain text (the tool-call
parser is wrong). In the inspection's output, `gate_task.evaluation_health.model.kinds` and
`gate_task.failures.response_failures.by_cause` (`truncated`, `empty`, `malformed`) are the model's
own faults: they count in the score, never fail health, and point at the deployment (output
budget, `--max-model-len`, parsers). `UNVERIFIED` with `trial_counts.unobserved_reason` naming the
benchmark's server is about that server's grading record, not about your deployment: report it and
change nothing for it.

## 6. Read the resolved config

`gym eval submit --resolve-only` prints the config as it will run; after submission, the run
directory's `resolved-config.yaml` is the record. Diff two of them to compare two runs: the config
file may have changed since, and overrides mean nothing without it.

## 7. A fair comparison

| Keep identical between the models you compare | May change with the deployment |
| --- | --- |
| The benchmark config: data, split, prompts, task limits. | Tensor, pipeline, data and expert parallelism; `number_of_instances`. |
| Sampling settings the benchmark sets, and `num_repeats`. | `--max-num-seqs`, GPU memory settings, client concurrency. |
| The judge model and the grading config. | Ports, container image, node layout. |
| The benchmark's token budgets and timeouts. | `--max-model-len`, as long as it holds the benchmark's longest prompt plus its output budget. |
| A reasoning switch the benchmark sets, and the Gym commit that scores the run. | |

A deployment that is too slow shows up as timeouts and truncated answers, which count as model
failures under the benchmark's rules. Make the deployment faster; do not raise the evaluation's
limits.

A model's own chat template, parsers and generation-config defaults are part of the model config,
not the deployment: record them with the results, because they change the outputs. Gym's resume
check treats the same split: a setting named as operational (timeouts, retries, ports,
concurrency, paths, tracking) resumes with a warning, and any other changed setting refuses the
resume and is named.

## Exit checks

- [ ] `--resolve-only` and `gym env validate --server-types` pass, and the `--dry-run` job script
  shows each `run` override once, the partition and `--gpus-per-node`.
- [ ] The smoke run completes, and its health and answers look right.
- [ ] The resolved config shows the intended served name, parallelism, parsers and reasoning switch.
- [ ] Nothing in the "keep identical" column differs from the run you compare against.
