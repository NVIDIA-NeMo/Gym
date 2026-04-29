# SWE Agents

A unified Responses-API wrapper that runs LLM-driven agents against real-world software-engineering benchmarks (SWE-bench and friends), executes the proposed patch inside the dataset's evaluation harness, and returns trajectories + a binary "resolved" reward suitable for both evaluation and RL training.

The entrypoint is [`app.py`](app.py), which exposes a `SWEBenchWrapper` (a `SimpleResponsesAPIAgent`) over HTTP. Each `responses` request takes one dataset instance, runs an agent inside an Apptainer container, runs the matching evaluation harness in a second container, and returns the trajectory plus reward.

---

## Table of Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [Agent flow per instance](#agent-flow-per-instance)
- [Supported datasets and harnesses](#supported-datasets-and-harnesses)
- [OpenHands integration](#openhands-integration)
- [Prompt and agent-class diversity](#prompt-and-agent-class-diversity)
- [Tool-name diversity](#tool-name-diversity)
- [Configuration reference](#configuration-reference)
- [Quick Start](#quick-start)
- [Batch evaluation / data collection](#batch-evaluation--data-collection)
- [Output format](#output-format)
- [GRPO masking and failure modes](#grpo-masking-and-failure-modes)
- [Debug / profiling](#debug--profiling)

---

## Architecture at a glance

```
                 ┌──────────────────────────────────────────────┐
   client ──▶    │  SWEBenchWrapper  (Responses API server)     │
   (one          │   • _setup_params: build per-instance config │
    instance)    │   • _build_apptainer_command: bind mounts    │
                 │   • runner_ray_remote: runs on Ray worker    │
                 └────────────────┬─────────────────────────────┘
                                  │ Ray
                                  ▼
                 ┌──────────────────────────────────────────────┐
                 │ RunOpenHandsAgent.process_single_datapoint   │
                 │                                              │
                 │   spawn agent container ──┐                  │
                 │                           │ in parallel      │
                 │   spawn eval container ───┘ (waits for       │
                 │                              prediction.jsonl)│
                 │                                              │
                 │   wait agent → copy patch to /trajectories   │
                 │   wait eval  → produce report.json           │
                 │   postprocess (per-dataset)                  │
                 └──────────────────────────────────────────────┘
```

Two Apptainer containers are launched concurrently per instance:

1. **Agent container** — runs OpenHands inside the dataset's task SIF (the one that contains the repo at the right base commit). The agent edits files and writes a unified diff.
2. **Eval container** — also a SIF for the same instance. It busy-waits on the predictions file written by the agent, then runs the dataset's local evaluation harness against the patch.

The two containers are launched at the same time so the eval container's spin-up cost (often tens of seconds for SWE-bench's harness) is hidden behind the agent's run time. The eval container blocks on `until [ -f <predictions> ]; do sleep 5; done` until the agent finishes.

Concurrency across instances is bounded by `concurrency` (default 256) via an asyncio semaphore on the server, and Ray's `SPREAD` scheduling distributes per-instance workers across the cluster.

---

## Agent flow per instance

Implemented in `RunOpenHandsAgent.process_single_datapoint` (and `_run_golden_patch_verification` for the verify-only path):

1. **Setup params** (`SWEBenchWrapper._setup_params`)
   - Pick a per-instance `persistent_dir` under `swebench_results_<run_session_id>/<instance_id>_<timestamp>_<uuid>`. This is bind-mounted into both containers as `/trajectories_mount`.
   - Resolve the SIF for this instance (`_find_container`) — supports exact match, `__` → `_1776_` / `_s_` rewrites, and fuzzy `*<id>*.sif` glob, plus dataset-specific rules for `SWE-rebench` and `R2E-Gym`.
   - Write the single-row dataset JSONL (the original `instance_dict`) to a per-instance file and mount it as `/root/dataset/data.jsonl` so OpenHands does not call the HF dataset API at run time.
   - Pick the dataset-specific `BaseDatasetHarnessProcessor` (see below).
   - Resolve any prompt/agent-class override (see [Prompt and agent-class diversity](#prompt-and-agent-class-diversity)).
   - Build both Apptainer command strings.
2. **Spawn agent + eval containers** in parallel via `asyncio.create_subprocess_shell`, each streaming logs to `<persistent_dir>/apptainer_logs/<instance_id>_{agent,eval}.log`.
3. **Wait for the agent** to finish or hit `swebench_agent_timeout` (default 45 min). Copy the produced `output.jsonl` and the most recent `llm_completions/*.json` back from OpenHands' eval-output dir into the per-instance trajectories root.
4. **Extract the patch** from `out_dict["test_result"]["git_patch"]` and rewrite it in SWE-bench-prediction format at `output_for_eval.jsonl`. This file is what the eval container is waiting for.
5. **Wait for the eval container** to finish or hit `swebench_tests_timeout` (default 30 min). It produces a `report.json` whose location is dataset-specific.
6. **Postprocess** the report (per-dataset; e.g. SWE-rebench / NV-internal / SWE-bench-Ext do their parsing host-side because the eval images may not have python3).
7. **Decide `mask_sample`** (GRPO) — see [GRPO masking and failure modes](#grpo-masking-and-failure-modes).
8. **Build the response** — convert the OpenHands chat-completions trajectory to Responses-API items via `VLLMConverter`, attach the tool list, return reward = `1.0 if resolved else 0.0` and metrics.

If `verify_golden_patch=true` (currently only for `swe-bench-ext`), step 2–4 are skipped: the dataset's golden patch is written directly as the prediction and the eval container is the only thing that runs. This is a sanity check that a dataset sample's golden patch actually resolves under our local eval.

---

## Supported datasets and harnesses

Selection is driven by `problem_info["dataset_name"]` (set in the input JSONL). Each dataset is paired with a `BaseDatasetHarnessProcessor` subclass that knows how to (a) install the harness once, (b) build the in-container eval command, (c) postprocess the resulting report.

| `dataset_name` (substring match)         | Processor class                          | Setup script (one-time)                         | Eval harness                                                                                              |
|------------------------------------------|------------------------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `princeton-nlp/SWE-bench*` (default)     | `SweBenchDatasetProcessor`               | `setup_scripts/swebench.sh` (HeyyyyyyG fork)   | `swebench.harness.run_local_evaluation` against the mounted JSONL                                          |
| contains `SWE-bench_Multilingual`        | `SweBenchMultilingualDatasetProcessor`   | `setup_scripts/swebench_multilingual.sh` (Kipok fork) | Same harness as SWE-bench but built from the multilingual fork                                       |
| contains `R2E-Gym`                       | `R2EGymDatasetProcessor`                 | `setup_scripts/r2e_gym.sh` (sdevare-nv fork)   | `r2egym.agenthub.run.run_local_evaluation`                                                                 |
| contains `SWE-rebench`                   | `SWERebenchDatasetProcessor`             | `setup_scripts/swe_rebench.sh` (V2)            | In-container `git apply` + `test_cmd`; **host-side** parsing via SWE-rebench's `log_parsers`               |
| `swe-bench-ext`                          | `SweBenchExtDatasetProcessor`            | none (uses `swe_bench_ext` helper module)      | Run framework-specific test command via `lighthouse`-style flags; host-side parsing via `parse_and_check_tests` |
| `nv-internal-1`                          | `NVInternalDatasetProcessor`             | none                                            | Synthesizes an env+`run_script.sh`+`parsing_script.py` from the instance's docker `ENV` lines and `before_repo_set_cmd`; tests gated by `f2p ⊆ passed ∧ p2p ⊆ passed` |

All harnesses are installed lazily on first use and locked across nodes with `mkdir`-based cross-node locks (`_setup_directory_lock`) — atomic on Lustre/NFS where `fcntl.flock` is not. Stale locks older than 1h are auto-broken.

The eval container is built from the same SIF as the agent for some datasets, but for SWE-bench / R2E-Gym / SWE-rebench / SWE-bench-Multilingual the harness venv is mounted at both `/{dataset}_setup` *and* its host absolute path (because `uv venv` bakes absolute paths into its wrappers).

---

## OpenHands integration

The agent always runs inside [OpenHands](https://github.com/All-Hands-AI/OpenHands), via `OpenHandsHarnessProcessor`. The fork pinned by `agent_framework_repo` / `agent_framework_commit` adds the prompt/tool-name diversity hooks documented below.

Key details:

- **Setup is one-time per workspace.** `setup_scripts/openhands.sh` clones the configured fork at the configured commit and bootstraps a miniforge3 + poetry venv at `swe_openhands_setup/OpenHands/`. The setup directory is mounted read-only into every agent container.
- **Inside the container** the agent driver is `OpenHands/evaluation/benchmarks/swe_bench/scripts/run_infer.sh`, parametrized by `agent_cls`, `agent_max_turns`, dataset name + split, and the per-instance dataset JSONL.
- **LLM config** is generated per-run by reading `configs/oh_config.toml`, overriding `llm.model.{model,base_url,temperature,top_p}` from the request, dumping it back as a TOML string, and writing it to `/tmp/config_<run_id>.toml` inside the container.
- **NeMo-Gym wiring** — `NEMO_GYM_METRICS_FPATH`, `NEMO_GYM_CONFIG_DICT`, and `NEMO_GYM_MODEL_SERVER_NAME` are exported into the agent container so the OpenHands fork can call back into the model server registered with this NeMo-Gym instance instead of using a raw HTTP base URL.
- **Workspace safety** — for datasets where the SIF does *not* bake `/workspace`, the agent script aborts if `/workspace` is mounted, because OpenHands' default behaviour deletes everything in `/workspace`. This check is intentionally skipped for `SWE-rebench*`, `nv-internal-1`, and `swe-bench-ext` whose images legitimately use `/workspace` or the agent works in `/{repo_name}` / `/app`.
- **`cryptography<43` shim** — for `nv-internal-1` and `swe-bench-ext`, a `cryptography<43` wheel is installed into a temp dir and prepended to `PYTHONPATH`. This works around openssl/cryptography ABI mismatches in older base images.
- **R2E-Gym test hiding** — when running the *agent* (not eval) under R2E-Gym, the wrapper deletes `/r2e_tests` and `/run_tests.sh` from `/`, `/root`, and `/testbed` so the agent can't peek at the held-out tests.
- **Trajectories** — after the agent finishes, the wrapper copies `output.jsonl` and the latest `llm_completions/*/*.json` out of OpenHands' per-run eval output directory and into `<persistent_dir>/trajectories/<instance_id>/`, then deletes the OpenHands-side dir to keep the shared setup tree clean.

---

## Prompt and agent-class diversity

OpenHands ships several agent classes; this wrapper supports four:

| `agent_cls`       | Notes                                                                |
|-------------------|----------------------------------------------------------------------|
| `CodeActAgent`    | Default. CodeAct-style react-loop with bash/edit tools.              |
| `OpenCodeAgent`   | OpenCode-style agent.                                                |
| `CodexAgent`      | Codex-style agent.                                                   |
| `Terminus2Agent`  | Terminus 2 agent.                                                    |

On top of agent class, each instance can be run with a different **system / user prompt** chosen from a list of overrides. The `prompts/` directory ships 15 prompt families:

```
prompts/
├── breadth_first/        ├── incremental/         ├── plan_and_execute/
├── codex/                ├── minimalist/          ├── root_cause/
├── divide_and_conquer/   ├── opencode/            ├── surgical/
├── explore_plan_execute/ ├── openhands/           ├── terminus/
├── hypothesis_driven/    ├── test_driven/         ├── verify_first/
```

Each prompt family contains a `system_prompt.j2` and `user_prompt.j2` and is paired with an `agent_cls` in the config. See `configs/swebench_multi_tools.yaml` for the canonical 15-way bundle.

**How selection works** (`SWEBenchWrapper._setup_params`):

```yaml
agent_prompt_overrides:
  - user_prompt_template: prompts/codex/user_prompt.j2
    system_prompt_template: prompts/codex/system_prompt.j2
    agent_cls: CodexAgent
    diversify_tool_names: false
    camel_case_tool_names: false
  - ...
agent_prompt_override_random: false   # default
```

- If `agent_prompt_override_random=false` (default), one override is picked **deterministically per `instance_id`** (`random.Random(instance_id).choice(overrides)`). This means the same instance always gets the same prompt across runs — useful for evaluation reproducibility.
- If `agent_prompt_override_random=true`, one is picked uniformly at random per run — useful for RL training where you want every rollout of the same instance to potentially see a different prompt.
- The selected `user_prompt_template` and `system_prompt_template` are bind-mounted over OpenHands' default Jinja templates at `OpenHands/{user_prompt,system_prompt,system_prompt_long_horizon}.j2` (the same system template is mounted for both the regular and long-horizon variants).
- Paths in the override may be absolute or relative; relative paths are resolved against `nemo_gym.PARENT_DIR`.

If `agent_prompt_overrides` is unset, the OpenHands defaults are used and `agent_cls` defaults to `CodeActAgent`.

---

## Tool-name diversity

Two independent knobs control how OpenHands surfaces tools to the model. They are off by default and enabled by setting them on the override that gets selected.

| Override field            | Env var exported into the agent container | What the OpenHands fork does                                                |
|---------------------------|-------------------------------------------|------------------------------------------------------------------------------|
| `diversify_tool_names`    | `DIVERSIFY_TOOL_NAMES=true`               | Randomly samples from a pool of synonym tool names per run instead of using the canonical name (e.g. `bash` ↔ `execute_command` ↔ `shell`). |
| `camel_case_tool_names`   | `CAMEL_CASE_TOOL_NAMES=true`              | Re-encodes whatever name was chosen above into `camelCase` (e.g. `execute_command` → `executeCommand`). |

The two stack: with both enabled, the model sees a sampled-then-camelCased name. Use this to break the model's reliance on memorising specific tool names during training, and to test robustness during eval. Both are part of `AgentPromptOverride`, so the same per-instance / per-run selection logic that picks the prompt also picks these flags.

---

## Configuration reference

The full schema lives in `SWEBenchWrapperConfig` (and the per-override `AgentPromptOverride`) in `app.py`. Highlights:

| Field                              | Default                                           | Purpose                                                                 |
|------------------------------------|---------------------------------------------------|-------------------------------------------------------------------------|
| `agent_framework_repo`             | OpenHands official                                | Fork to clone for the agent runtime.                                    |
| `agent_framework_commit`           | `HEAD`                                            | Commit to pin.                                                          |
| `agent_max_turns`                  | `100`                                             | Max OpenHands iterations.                                               |
| `agent_config`                     | `null`                                            | Path to the OpenHands TOML (`configs/oh_config.toml`).                  |
| `agent_tools_file`                 | `null`                                            | (SWE-agent only) JSON tool list in OpenAI format.                       |
| `container_formatter`              | `docker://swebench/sweb.eval.x86_64.{instance_id}`| Path template (or list of templates) for SIFs. Supports the `_1776_` / `_s_` rewrites and fuzzy glob fallbacks. |
| `swebench_agent_timeout`           | `2700` (45 min)                                   | Per-instance agent wall-clock budget.                                   |
| `swebench_tests_timeout`           | `1800` (30 min)                                   | Per-instance eval wall-clock budget.                                    |
| `apptainer_memory_limit_mb`        | `32768`                                           | `ulimit -v` applied before `apptainer exec`.                            |
| `command_exec_timeout`             | `300`                                             | OpenHands per-command timeout inside the agent container.               |
| `concurrency`                      | `256`                                             | Server-side asyncio semaphore for concurrent instances.                 |
| `dataset_path`                     | `null`                                            | Optional default dataset JSONL.                                         |
| `verify_golden_patch`              | `false`                                           | Skip the agent and eval the dataset's own golden patch (currently `swe-bench-ext` only). |
| `agent_prompt_overrides`           | `null`                                            | List of `AgentPromptOverride` entries. See above.                       |
| `agent_prompt_override_random`     | `false`                                           | `false` = deterministic per `instance_id`; `true` = random per run.     |
| `openhands_should_log`             | `false`                                           | If true, sets `LOG_LEVEL=DEBUG`, `LOG_TO_FILE=true`, etc.               |
| `debug`                            | `false`                                           | Enables Profiler around the agent run + dumps callgrind/dot/png.        |

Bundled YAML configs:

- `configs/swebench_openhands.yaml` — single OpenHands `CodeActAgent`, the simplest setup.
- `configs/swebench_openhands_training.yaml` — same shape as above but tuned for training.
- `configs/swebench_swe_agent.yaml` — alternative SWE-agent path (uses `agent_tools_file`).
- `configs/swebench_multi_tools.yaml` — full 15-way prompt × agent-class × tool-name bundle.

---

## Quick Start

### Prerequisites — install Apptainer

```bash
apt install -y wget && cd /tmp && \
    wget https://github.com/apptainer/apptainer/releases/download/v1.4.1/apptainer_1.4.1_amd64.deb && \
    apt install -y ./apptainer_1.4.1_amd64.deb
apptainer --version
```

### Step 1 — configure the model

In `env.yaml` at the NeMo-Gym root:

```yaml
# OpenAI
policy_base_url: https://api.openai.com/v1
policy_api_key: <your OpenAI API key>
policy_model_name: gpt-4.1-2025-04-14
```

Or run a local vLLM:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-model-len 131072 --enable-expert-parallel \
  --tensor-parallel-size 4 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --port 8000 --enforce-eager
```

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: dummy
policy_model_name: Qwen/Qwen3-Coder-30B-A3B-Instruct
```

### Step 2 — start the SWE-agents server

```bash
# OpenHands single-prompt
config_paths="responses_api_agents/swe_agents/configs/swebench_openhands.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

# Or full prompt × agent-class × tool-name diversity
config_paths="responses_api_agents/swe_agents/configs/swebench_multi_tools.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_run "+config_paths=[$config_paths]" \
    +swe_agents.responses_api_agents.swe_agents.container_formatter=/lustre/xxx/images/swe-bench/swebench_sweb.eval.x86_64.\{instance_id\}.sif \
    +swe_agents.responses_api_agents.swe_agents.model_server.name=vllm_model
```

For converting docker images into the `.sif` files referenced by `container_formatter`, see [`nemo_skills/dataset/swe-bench/dump_images.py`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/swe-bench/dump_images.py).

You should see something like:

```
INFO:     Started server process [1815588]
INFO:     Uvicorn running on http://127.0.0.1:25347 (Press CTRL+C to quit)
```

### Step 3 — query the agent

```bash
python responses_api_agents/swe_agents/client.py
```

---

## Batch evaluation / data collection

```bash
ng_collect_rollouts +agent_name=swe_agents \
    +input_jsonl_fpath=swebench-verified-converted.jsonl \
    +output_jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl \
    +model=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    +temperature=0.7 +top_p=0.8
```

`ng_collect_rollouts` defaults to a concurrency of 100; tune to your hardware. View the results with:

```bash
ng_viewer +jsonl_fpath=swebench-verified.openhands.qwen3-30b-coder.jsonl
```

---

## Output format

Each `responses` call returns a `NeMoGymResponse` whose `output` is a Responses-API conversion of the OpenHands chat-completion trajectory, whose `tools` is the function-tool list the agent saw, and whose `metadata` carries `metrics` (a `SWEBenchMetrics` JSON) and the full `instance_config`.

`run` wraps that in `SWEBenchVerifyResponse`:

```jsonc
{
  "responses_create_params": { /* full input incl. system+user prompts the agent saw */ },
  "response":               { /* output messages + tool calls */ },
  "reward": 1.0,            // 1.0 iff resolved, else 0.0
  "resolved": true,
  "patch_exists": true,
  "model_patch": "diff --git ...",
  "agent_error_kind": null, // "max_iteration" | "context_window" | "stuck_in_loop" | "other" | null
  "agent_timed_out": false,
  "eval_timed_out": false,
  "ray_queue_time": 0.12,
  "openhands_run_time": 412.3,
  "generation_apptainer_spinup_time": 11.4,
  "final_eval_apptainer_spinup_time":  9.7,
  "final_eval_time": 87.2,
  "instance_config": { /* the full per-instance SWEBenchWrapperInstanceConfig */ }
}
```

---

## GRPO masking and failure modes

`SWEBenchWrapperInstanceConfig.mask_sample` is set to `True` (so downstream RL drops the gradient for this rollout) when:

1. The patch resolved the tests **but** the agent terminated in a `max_iteration` or `context_window` error — the reward is accidental.
2. The eval container hit `swebench_tests_timeout` — reward is unreliable.
3. The agent hit `swebench_agent_timeout` (wall-clock) regardless of `resolved`.

Agent error strings are bucketed by `_classify_agent_error`:

| Substring (case-insensitive)        | Bucket           |
|-------------------------------------|------------------|
| `maximum iteration`                 | `max_iteration`  |
| `ContextWindow` / `context window`  | `context_window` |
| `stuck in a loop`                   | `stuck_in_loop`  |
| anything else                       | `other`          |

---

## Debug / profiling

Set `debug=true` to wrap the agent run in a `Profiler` (callgrind output), then auto-render `.dot` and `.png` graphs via `gprof2dot` + `pydot` after the run. Profiling output lands under `<persistent_dir>/profiling/`. Apptainer also exports `NG_PROFILING_DIR` into the agent container so the OpenHands fork can dump matching profiles.

Set `openhands_should_log=true` to flip OpenHands to `LOG_LEVEL=DEBUG`, `LOG_TO_FILE=true`, and write per-event logs. Otherwise the wrapper aggressively quiets OpenHands (`LOG_LEVEL=CRITICAL`, all `DEBUG_*` flags off).

Per-instance Apptainer stdout/stderr is always streamed to `<persistent_dir>/apptainer_logs/<instance_id>_{agent,eval}.log` regardless of these flags.
