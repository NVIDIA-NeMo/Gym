---
name: add-model-eval-config
description: >
  Onboard a model's serving config for NeMo Gym evaluations: write the vLLM policy service
  (model path, served name, tensor/pipeline/data/expert parallelism, max model length, max
  sequences, tool-call and reasoning parsers, chat template, generation config, environment),
  wire it as the policy with the right model type, set the reasoning toggle through chat-template
  kwargs, compose the whole eval config by naming each benchmark's own config from Gym's
  `benchmarks/<name>/` directory, validate the config without GPUs, smoke it with a small `--limit`,
  and read the resolved config. Also says what must stay fixed for a fair comparison and what deployment tuning may
  change. Use when a new model or checkpoint has to be evaluated in Gym. Not for running the
  evaluation itself (use run-and-babysit-eval) or adding a benchmark (use add-benchmark). Triggered by:
  "add a model", "new checkpoint", "serve this model in Gym", "model config", "vllm config",
  "enable thinking", "reasoning parser", "tool call parser", "compare two models fairly",
  "eval config", "where does the benchmark config come from", "run evaluations on model Y on
  benchmark Z with serving params X". To run the composed config, continue with
  run-and-babysit-eval.
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

If the benchmarks use tools, also check that the served template renders them. It needs a `tools` branch
(`grep -c tools chat_template.jinja` is above 0) that renders the request's `tools`, the assistant's tool calls and the
tool results. A template without one drops the tools without any error: the model answers in free text, the agent
sees no tool call, and every rollout ends in a format error and scores 0.0.

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
public Gym that has your benchmark, never a branch. Recovery (`gym eval resubmit --recover`) reads
the run with that commit. The CLI you submit with may be newer than the pinned commit; the job
still installs the pinned one. Install the CLI as run-and-babysit-eval (step 1) describes. Site `#SBATCH`
directives (a QoS, a node to avoid) go in `compute.cluster.extra_args`, without leading dashes:
`{qos: <qos>, exclude: <node>}`.

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

Each instance uses `tensor_parallel_size × pipeline_parallel_size` GPUs, and all instances together
must fit the node pool (`nodes × gpus_per_node`). More instances raise throughput for many
rollouts; they do not change the outputs. A model with no reasoning output runs without
`--reasoning-parser` and sets `uses_reasoning_parser: false` on the client (section 3). Weight
quantization (`--quantization`, a quantized checkpoint), `--dtype` and `--kv-cache-dtype` change
the outputs and the score: they are part of the model config, not a memory setting (section 8).

Two readiness limits: the job waits `health_check.timeout_seconds` for the policy service, and
the driver then waits `server_spinup_timeout_seconds` (under the benchmark's `run`, 600 s by
default) for the Gym servers it starts for the benchmark. A benchmark whose servers start slowly
ends the chain typed `server_spinup_timeout`; run-and-babysit-eval says how to raise the limit on a saved run.

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
  parser for this model. The shipped `vllm_model` config sets it to true, so a model served without
  `--reasoning-parser` needs `false` here.
- A benchmark whose own config already declares the policy model fails validation with "at most 1
  item" when another model type is composed beside it: set `driver.policy_model_type: ""` for it.
- Some benchmark configs set `chat_template_kwargs` (or `extra_body`) themselves. That is a
  benchmark setting: leave it as the benchmark sets it, because an override would replace it. The
  shipped `vllm_model` configs leave these keys unset, so a benchmark's value survives
  `--model-type vllm_model`, which merges after the benchmark config.
- A request's own `chat_template_kwargs` is used only with `forward_request_chat_template_kwargs:
  true` (see the vLLM model server page).

What the client sends, and what it leaves to the server:

- Gym's agents speak the Responses API; `vllm_model` turns each call into a chat-completions
  request (`is_responses_native: false` by default) and the tool calls back into Responses items,
  so the served template and tool-call parser decide whether tools work.
- Sampling (`temperature`, `top_p`, the output budget `max_output_tokens`) comes from the
  benchmark and its data. Any value they leave unset falls to the server: the model's
  `generation_config.json` with `--generation-config auto`, vLLM's defaults otherwise. A product
  deployment's sampling defaults are not evaluation settings. Leave the client's
  `sampling_overrides` unset for evaluation: it replaces the benchmark's sampling on every request.
- When the output budget alone exceeds `--max-model-len`, the client lowers it to fit the window
  (`refit_output_budget_to_window`, on by default) and records it in the model call's capture. A
  window that is too short therefore shows up as truncated answers, not as an error.
- System prompts come from the benchmark. A template that rejects the `developer` role needs
  `replace_developer_role_with_system: true` on the client; check the template before setting it.

## 4. Compose the whole evaluation config

Sections 2 and 3 write the model's half of the submit config. The other half, what is evaluated and
how it is graded, comes from Gym: every benchmark ships its own evaluation config, and the submit
config points at it instead of restating it.

| Part of the submit config | What it holds | Where it comes from |
| --- | --- | --- |
| `services.<name>` | The vLLM deployment of the model. | Section 2. |
| `driver.policy_model`, `driver.policy_model_type` | Which service is the policy, and the client Gym talks to it with. | Sections 2 and 3. Submit fills in `policy_base_url`, `policy_model_name` and `policy_api_key` in every benchmark's `run`. |
| `driver.gym_install.ref` | The Gym commit the job installs, which holds the benchmark and scores the run. | A full commit of public Gym that has the benchmark. |
| `driver.benchmarks.<benchmark>.prepare` | Overrides for `gym eval prepare`, which builds the benchmark's data in the job before it runs. | The benchmark's config path. |
| `driver.benchmarks.<benchmark>.run` | Overrides for `gym eval run`: the benchmark's config, the split, and the model settings of section 3. | The benchmark's config path and split; your section 3 keys. |
| `driver.benchmarks.<benchmark>.resumable` | Resuming across allocations (`max_retries`, `max_walltime`). | run-and-babysit-eval. |
| `job`, `compute`, `otel` | The output directory, the cluster and its node pools, telemetry. | Your site (section 2). |

### Where a benchmark's evaluation config lives

Each benchmark is a directory `benchmarks/<benchmark>/` in the Gym commit that `driver.gym_install.ref`
pins (a benchmark with variants has one directory per variant, `benchmarks/<benchmark>/<variant>/`):

- `config.yaml` is the evaluation config. Through its own `config_paths` it chains to the resources
  server's config, `resources_servers/<server>/configs/<server>.yaml`, and it declares the benchmark's
  agent, its dataset (`type: benchmark`, the prepared `jsonl_fpath`, the `prepare_script`, `num_repeats`,
  and a `prompt_config` where the benchmark has one) and the grading settings it changes on the server.
- `manifest.yaml` describes the benchmark: its servers, its datasets, the split to evaluate
  (`canonical_split`) and the reward range.
- `prepare.py` builds the dataset file that `config.yaml` reads. Prepared data is not checked in, so a
  benchmark without a `prepare` block finds no data in the job.
- `README.md` names the source data, how it is graded, and the benchmark's own commands.

Read them at the pinned commit, not on a newer branch: the benchmark's data, prompts and grading are
part of what the run measures.

Do not copy the benchmark's config into the submit config; name it. `run.config_paths:
[benchmarks/<benchmark>/config.yaml]` loads what `gym eval run --benchmark <benchmark>` loads, the same
path under `prepare.config_paths` prepares its data, and `run.split` is the manifest's
`canonical_split`. The job composes them as `gym eval run` does: the benchmark's config, then the
model type's config, then the `run` keys as `+key=value` overrides. So the model's settings from
section 3 go in the same `run` mapping as the benchmark keys and win over both files, while the
benchmark's own settings (prompts, sampling, `num_repeats`, grading) stay as Gym ships them. Overriding
one of those changes what is measured; section 8 says which ones a fair comparison keeps.

### Example: IFEval

`benchmarks/ifeval/` needs no judge and no credentials for its data. Its `config.yaml` chains to
`resources_servers/instruction_following/configs/instruction_following.yaml` and reads
`benchmarks/ifeval/data/ifeval_benchmark.jsonl` (541 prompts, one repeat each). Its `prepare.py`
downloads them from the upstream repository, and `manifest.yaml` gives `canonical_split: benchmark`.
With a small reasoning model from the Hub as the policy, the whole config is:

```yaml
services:
  policy:
    type: vllm
    container: ${oc.env:SERVING_IMAGE}
    model: Qwen/Qwen3-8B
    served_model_name: qwen3-8b
    tensor_parallel_size: 1
    number_of_instances: 4 # one replica per GPU
    node_pool: gpu
    health_check:
      timeout_seconds: 1800
    extra_args: >-
      --max-model-len 40960
      --max-num-seqs 256
      --reasoning-parser qwen3
      --generation-config auto
    env:
      HF_HOME: lit:/cache/huggingface
    mounts:
      - ${oc.env:CACHE_DIR}:/cache

driver:
  container: ${oc.env:GYM_DRIVER_IMAGE}
  policy_model: policy
  policy_model_type: vllm_model
  gym_install:
    repo: https://github.com/NVIDIA-NeMo/gym
    ref: <full Gym commit>
  benchmarks:
    ifeval:
      prepare:
        config_paths: [benchmarks/ifeval/config.yaml]
      run:
        config_paths: [benchmarks/ifeval/config.yaml]
        split: benchmark
      resumable:
        max_retries: 3
        max_walltime: "08:00:00"

job:
  output_path: ${oc.env:OUTPUT_DIR}

compute:
  cluster:
    type: slurm
    account: ${oc.env:SLURM_ACCOUNT}
    hostname: ${oc.env:SLURM_LOGIN_HOST}
    walltime: "04:00:00"
    node_pools:
      gpu:
        partition: ${oc.env:SLURM_PARTITION}
        nodes: 1
        gpus_per_node: 4

otel:
  enabled: false
```

The model's half is the `policy` service and the three `policy_*` driver keys; everything under
`benchmarks.ifeval` comes from the benchmark's directory. Qwen3 reasons by default and the shipped
`vllm_model` client already expects a reasoning parser, so this model needs no section 3 keys; a model
that does adds them to the same `run` mapping. Validate it as section 5 describes. In the `--dry-run`
script the job runs `gym eval prepare '+config_paths=[benchmarks/ifeval/config.yaml]'` and then
`gym eval run --model-type vllm_model` with `+config_paths=[benchmarks/ifeval/config.yaml]`,
`+split=benchmark` and the three policy keys submit filled in.

Then run it with run-and-babysit-eval.

## 5. Validate without GPUs

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

## 6. Smoke it

Submit the model with one benchmark and a small task count before a full evaluation
(`gym eval submit --config submit.yaml`; it prints the run's `GYM_JOB_ID`):

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
when it completes, it cancels the chain's pending follow-up and sentinel itself (see run-and-babysit-eval, step 2).
Then follow the [run-and-babysit-eval](../run-and-babysit-eval/SKILL.md) skill: `gym eval resubmit "$GYM_JOB_ID"
<benchmark> --recover --inspect` and `gym eval health-check` (run-and-babysit-eval, step 6). Look for answers that end mid-thought (the output token
budget or `--max-model-len` is too small), reasoning that lands in the answer (the reasoning parser
or `uses_reasoning_parser` is wrong), and tool calls that arrive as plain text (the tool-call
parser is wrong). For a tool-using benchmark also count the tool calls: `ng_perf.num_tool_calls` in `rollouts.jsonl`
must be above 0 on the rows that need tools, and a request's `prompt_tokens` must exceed the token count of its
messages alone (equal means the template dropped the tools). In the inspection's output,
`gate_task.evaluation_health.model.kinds` and
`gate_task.failures.response_failures.by_cause` (`truncated`, `empty`, `malformed`) are the model's
own faults: they count in the score, never fail health, and point at the deployment (output
budget, `--max-model-len`, parsers). Answers that never stop and run to the output budget point
at an end-of-turn token that the template and the model's generation config disagree on.
`UNVERIFIED` with `trial_counts.unobserved_reason` naming the benchmark's server is about that server's grading record, not about your deployment: report it and
change nothing for it.

## 7. Read the resolved config

`gym eval submit --resolve-only` prints the config as it will run; after submission, the run
directory's `resolved-config.yaml` is the record. Diff two of them to compare two runs: the config
file may have changed since, and overrides mean nothing without it.

## 8. A fair comparison

| Keep identical between the models you compare | May change with the deployment |
| --- | --- |
| The benchmark config: data, split, prompts, task limits. | Tensor, pipeline, data and expert parallelism; `number_of_instances`. |
| Sampling settings the benchmark sets, and `num_repeats`. | `--max-num-seqs`, `--gpu-memory-utilization`, client concurrency. |
| Weight quantization, `--dtype` and `--kv-cache-dtype` (or record each difference as part of the model). | |
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

## What this skill does not do

- It does not submit, watch, resume or recover a run, or judge its health. That is
  [run-and-babysit-eval](../run-and-babysit-eval/SKILL.md), which takes the config this skill writes.
- It does not add a benchmark to Gym or change one; that is add-benchmark. A benchmark that has no
  directory under `benchmarks/` at the pinned commit cannot be composed here.

## Exit checks

- [ ] Each benchmark's `run.config_paths` and `prepare.config_paths` name its `config.yaml` at the
  pinned commit, and `run.split` is its manifest's `canonical_split`.
- [ ] `--resolve-only` and `gym env validate --server-types` pass, and the `--dry-run` job script
  shows each `run` override once, the partition and `--gpus-per-node`.
- [ ] The smoke run completes, and its health and answers look right.
- [ ] The resolved config shows the intended served name, parallelism, parsers and reasoning switch.
- [ ] Nothing in the "keep identical" column differs from the run you compare against.
