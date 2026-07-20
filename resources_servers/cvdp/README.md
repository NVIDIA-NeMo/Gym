# CVDP Benchmark

This resources server is for model evaluation purposes. It is reproducing [CVDP](https://github.com/NVlabs/cvdp_benchmark).

Models are given a hardware design specification and asked to produce a SystemVerilog
implementation. Verification is done by running the per-task cocotb test harness inside
Apptainer (using the `ghcr.io/hdl/sim/osvb` simulator image with Icarus Verilog). 

## Verification Flow

Native resources server — verification runs as an Apptainer subprocess invoked
from `verify()`. The harness files (compose config, test scripts) are embedded in each
JSONL entry so the server is self-contained.

Mirrors `repository.py` in the [CVDP source](https://github.com/NVlabs/cvdp_benchmark):

1. Obtain the candidate RTL: grade the files the harness wrote on disk (`rtl_files` in the verify request, sandboxed-harness flow) when present, otherwise parse the model's text response via `ModelHelpers.parse_model_response()`
2. Write harness files to temp workspace — applies image placeholder substitutions
3. Write extracted RTL to `workdir/rtl/`
4. For each service in `docker-compose.yml`, pull the Docker image as a cached SIF file and run it through the configured sandbox provider (Apptainer by default; `instance start` + `exec`) with `--bind` mounts for `rtl/`, `verif/`, `docs/`, `src/`, `rundir/`
5. Exit code `0` across all services → reward `1.0`; any failure → reward `0.0`

## Configuration


| Field                     | Default                 | Description                                                                                                                                                                                                                                                            |
| ------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `oss_sim_image`           | `ghcr.io/hdl/sim/osvb`  | Container image for open-source simulation (Icarus)                                                                                                                                                                                                                    |
| `oss_pnr_image`           | `""`                    | Container image for place-and-route problems                                                                                                                                                                                                                           |
| `eda_sim_image`           | `""`                    | Commercial EDA image (Cadence Xcelium etc.)                                                                                                                                                                                                                            |
| `container_timeout`       | `600`                   | Seconds before an Apptainer run is killed                                                                                                                                                                                                                              |
| `num_processes`           | `4`                     | Max concurrent Apptainer jobs                                                                                                                                                                                                                                          |
| `sif_cache_dir`           | `~/.cache/nemo-gym/sif` | Directory for cached SIF images pulled from Docker registries                                                                                                                                                                                                          |
| `sandbox_provider`        | `{apptainer: {}}`       | Sandbox backend used for grading, as a single-key provider config (e.g. `{apptainer: {}}` or `{opensandbox: {}}`). Selects the backend via config instead of hard-coding it; apptainer-specific knobs (writable rootfs, timeouts) are filled in automatically          |
| `harness_workspace_dir`   | `""`                    | Optional host directory where per-rollout temp workspaces are created (default: system temp)                                                                                                                                                                           |
| `container_tmp_bind_path` | `""`                    | If set, redirects in-container temp (e.g. `/tmp`) to per-rollout host storage and forces temp env vars (`TMPDIR`, `XCELIUM_TMPDIR`, `CDS_LOCK`, `JAVA_TOOL_OPTIONS`) — useful when default `/tmp` is too small or tools (Cadence/Java) write large temp/lock artifacts |


**Note**: To run the commercial subset, pass the EDA image name in the yaml config file (`resources_servers/cvdp/configs/cvdp.yaml`).

```
eda_sim_image: cvdp-cadence-verif:latest
```

## Agents

Both flavors are the **same agent class — `CVDPAgent`** (`responses_api_agents/cvdp_agent/app.py`); you pick the flavor with the **`simple_agent`** config flag. They differ only in **how the candidate RTL is produced and graded**:

- **`simple_agent: true` — direct generation** (`cvdp_agent`, config `configs/cvdp_agent.yaml`): the model emits the RTL inline in its response; the server parses it out (`_parse_model_response`) and runs the harness. A single completion — no tools, no file editing.
- **`simple_agent: false` — sandboxed harness** (`cvdp_agent_generic_*`, configs `configs/cvdp_agent_generic_*.yaml`): a coding **harness** runs **inside** the EDA sim container, editing files on disk and self-testing with the in-container EDA tools; the server grades the files it wrote back (`rtl_files`). The harness is a config-selected unit — `agent_server_module`/`class`/`config_class` name any Gym `responses()` agent. Claude Code is the default; Hermes ships as a second example (`configs/cvdp_agent_generic_hermes.yaml`), and any other harness drops in via config + a deps script. See [`responses_api_agents/cvdp_agent/`](../../responses_api_agents/cvdp_agent/).

### Sandboxed harness settings (`configs/cvdp_agent_generic_claude.yaml`)

| Field                 | Default                                       | Description                                                                                          |
| --------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `agent_server_module` | `responses_api_agents.claude_code_agent.app`  | Python module of the Gym harness agent booted inside the sandbox                                     |
| `agent_server_class`  | `ClaudeCodeAgent`                             | Harness agent class                                                                                  |
| `agent_config_class`  | `ClaudeCodeAgentConfig`                       | Harness agent config class                                                                           |
| `agent_kwargs`        | `{}`                                          | Passed straight into the harness config class; keys must be valid fields on it (Claude: `model`, `anthropic_base_url`, `anthropic_api_key`, `max_turns`) |
| `image`               | `${cvdp_sim_image}`                           | Sandbox image the harness runs inside. Any container ref works: a bare registry ref (`ghcr.io/hdl/sim/osvb`, `nvidia/cvdp-sim:v1.0.0`, …), an explicit `.sif` path, or a `docker://`/`oras://` uri. A bare ref is resolved to a cached `.sif` under `~/.cache/nemo-gym/sif/`, pulling it on first use; only prebuild it manually if the image exists locally-only — see below |
| `sandbox_provider`    | `{apptainer: {create: {mount_point: /code, extra_start_args: [--writable-tmpfs]}}}` | Sandbox backend (single-key provider config). `mount_point` must match `container_workdir` (writable bind); `--writable-tmpfs` makes the rest of the container writable |
| `deps_provision`      | `bind`                                        | `bind` = mount a portable deps prefix into the sandbox; `baked` = deps already in the image          |
| `container_workdir`   | `/code`                                       | Workspace mount point + cwd inside the container (keep equal to `mount_point`)                       |
| `model_server`        | `policy_model`                                | Gym model server reference                                                                           |
| `timeout`             | `900`                                         | Per-task wall-clock budget (seconds)                                                                 |
| `concurrency`         | `4`                                           | Max concurrent agent runs                                                                            |

The `agent_server_*` block selects **which** harness to boot; `agent_kwargs` are the knobs for **that specific harness** (so the valid keys depend on the harness's config class). To swap harnesses, change the `agent_server_*` block + `agent_kwargs` and provide a matching `setup_scripts/<key>_deps.sh`. A Hermes example is shipped at `configs/cvdp_agent_generic_hermes.yaml`.

The first run for a harness downloads and caches its software (portable Python/Node + the harness CLI) into a deps prefix, so it needs network once; later runs are offline-fast.

**Sandbox image:** `cvdp_sim_image` is any container ref — a bare registry ref, an explicit
`.sif` path, or a `docker://`/`oras://` uri. Both the sandboxed-harness agent and this verifier
resolve a bare ref the same way: to a cached `.sif` under `~/.cache/nemo-gym/sif/` (same
naming convention, `<ref with / and : replaced by _>.sif`), so one value works on both sides.

- **Registry-hosted image (the common case):** point it at anything pullable — the CVDP default
  `nvidia/cvdp-sim:v1.0.0`, the open-source simulator `ghcr.io/hdl/sim/osvb`, an `nvcr.io` image,
  etc. On first use the resolver runs `apptainer pull docker://<ref>` into the cache automatically;
  later runs are offline-fast. No manual prebuild needed.
- **Local-only image (built yourself, never pushed):** `apptainer pull docker://…` can't reach an
  image that only exists in your local Docker daemon, so prebuild it into the cache once with the
  conventional name (this is only required for `nvidia/cvdp-sim:v1.0.0` if you built it locally per
  [Build the Open-Source Simulation Image](#build-the-open-source-simulation-image) rather than
  pulling it):

  ```bash
  apptainer build ~/.cache/nemo-gym/sif/nvidia_cvdp-sim_v1.0.0.sif docker-daemon://nvidia/cvdp-sim:v1.0.0
  ```

Add the harness settings to your repo-root `env.yaml` (Claude example). Set `cvdp_sim_image` to
whatever image you want the harness to run inside — e.g. `ghcr.io/hdl/sim/osvb` to use the
open-source simulator with no prebuild:

```yaml
anthropic_model_name: <claude-model>
anthropic_api_key: <your-api-key>
anthropic_base_url: https://api.anthropic.com
cvdp_sim_image: ghcr.io/hdl/sim/osvb   # or nvidia/cvdp-sim:v1.0.0, an .sif path, a docker:// uri, …
```

### Run the sandboxed-harness variant

**Step 1 — Start servers** (loads the harness agent config; `apptainer` must be installed):

```bash
gym env start \
    --config responses_api_agents/cvdp_agent/configs/cvdp_agent_generic_claude.yaml \
    --resources-server cvdp \
    --model-type vllm_model
```

**Step 2 — Collect rollouts** (`--agent` is the config's top-level name):

```bash
gym eval run --no-serve \
    --agent cvdp_agent_generic_claude \
    --config responses_api_agents/cvdp_agent/configs/cvdp_agent_generic_claude.yaml \
    --input resources_servers/cvdp/data/<dataset>.jsonl \
    --output results/rollouts.jsonl \
    --num-repeats 5 \
    --concurrency 4 \
    --resources-server cvdp \
    --model-type vllm_model \
    --resume
```

**Step 3 — Generate report:**

```bash
python resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py \
    --rollouts  results/rollouts.jsonl \
    --output    results/report/ \
    --model     cvdp_agent_generic_claude \
    --dataset   <original_cvdp_dataset>.jsonl \
    --k         1
```

To run a different harness, swap `cvdp_agent_generic_claude` for another config (e.g. `cvdp_agent_generic_hermes`) in the commands above.

## Build the Open-Source Simulation Image

If you're using the CVDP v1.1.0 data (e.g. `data/example_agentic.jsonl`), build the open-source
simulation image **once** before collecting rollouts. CVDP v1.1.0 uses a dedicated open-source
simulation image for non-commercial simulation tasks:

```bash
cd /path/to/cvdp_benchmark
docker build -f docker/Dockerfile.sim -t nvidia/cvdp-sim:v1.0.0 .
```

This image provides the default `OSS_SIM_IMAGE` environment used by dataset harnesses via
`__OSS_SIM_IMAGE__`. CVDP v1.1.0 no longer uses the legacy third-party simulation images for this
default open-source simulation flow. The build includes cocotb 2.0.1, pytest 8.3.2, Icarus Verilog
v13_0, Yosys yosys-0.40, and Verilator v5.038.

If you tag the image differently, set the matching value in `.env`:

```bash
OSS_SIM_IMAGE=nvidia/cvdp-sim:v1.0.0
```

Open-source place-and-route tasks still use the separate `OSS_PNR_IMAGE` setting, but in CVDP
v1.1.0 its default points at the same `nvidia/cvdp-sim:v1.0.0` image:

```bash
OSS_PNR_IMAGE=nvidia/cvdp-sim:v1.0.0
```

## Download Dataset

The data can be found [on Hugging Face](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset). 

## Step 1 — Download dataset from HF

```bash
hf download nvidia/cvdp-benchmark-dataset \
  --include "cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl" \
  --repo-type dataset \
  --local-dir resources_servers/cvdp/data/cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl
```

## Step 2 — Export prompts from CVDP

Use CVDP's built-in `local_export` mode to generate the exact prompts CVDP would send to a model:

```bash
cd /path/to/cvdp_benchmark
python run_benchmark.py \
    -f <dataset>.jsonl \
    --model local_export \
    --prompts-responses-file <output_prompts>.jsonl \
    --llm \
    --prefix export_run
```

This produces a JSONL with `{id, prompt, system, user}` per entry.

## Step 3 — Convert to NeMo-Gym format

```bash
python resources_servers/cvdp/scripts/convert_to_gym.py \
    --prompts  <prompts_from_step1>.jsonl \
    --dataset  <original_cvdp_dataset>.jsonl \
    --output   resources_servers/cvdp/data/<output>.jsonl
```

Each output row has `responses_create_params` (system + user prompts) and `verifier_metadata`
(harness files, target files, category, difficulty) needed by the resources server.

## Setup

If you do not have one, create an env.yaml at the repo root and configure your inference endpoint.

This example is for using the **NV inference endpoint** with the vLLM backend. This can also be run with any OpenAI-compatible endpoint with any Gym-supported backend.

**NVIDIA hosted API**

```yaml
policy_base_url: https://integrate.api.nvidia.com/v1
policy_api_key: <your-api-key>
policy_model_name: <your-model-name>
```

**Self-hosted vLLM**(start `vllm serve <model> --port 8000` first):

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: token-abc123
policy_model_name: <your-model-name>
```

Install dependencies:

```bash
uv venv && source .venv/bin/activate
uv sync --extra dev
cd resources_servers/cvdp 
uv pip install -r requirements.txt
pre-commit install
```

To install apptainer:

```bash
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb                                               
apt install -y ./apptainer_1.3.1_amd64.deb 
```

## Collect Rollouts

### Step 1 — Start servers

```bash
gym env start \
    --resources-server cvdp \
    --model-type vllm_model
```

### Step 2 — Run rollout collection

`num_repeats` controls how many times each problem is run (for pass@k metrics):

```bash
gym eval run --no-serve \
    --agent cvdp_agent \
    --input resources_servers/cvdp/data/<dataset>.jsonl \
    --output results/rollouts.jsonl \
    --num-repeats 5 \
    --concurrency 4 \
    --max-output-tokens 4096 \
    --temperature 0.2 \
    --top-p 0.7 \
    --resources-server cvdp \
    --model-type vllm_model \
    --resume
```

At the end of this step, you should have rollouts.jsonl, rollouts_agent_metrics.json, rollouts_materialized_inputs.jsonl, rollouts_reward_profiling.jsonl.

### Step 3 — Generate report

```bash
python resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py \
    --rollouts  results/rollouts.jsonl \
    --output    results/report/ \
    --model     cvdp_agent \
    --dataset   <original_cvdp_dataset>.jsonl \
    --k         1
```


| Arg          | Required | Default    | Description                                               |
| ------------ | -------- | ---------- | --------------------------------------------------------- |
| `--rollouts` | Yes      | —          | Rollout JSONL from `gym eval run --no-serve`              |
| `--output`   | Yes      | —          | Output directory for reports                              |
| `--model`    | No       | `nemo-gym` | Model name shown in report metadata                       |
| `--dataset`  | No       | —          | Path to original CVDP dataset JSONL (for report metadata) |
| `--k`        | No       | `1`        | Pass@k threshold for composite report                     |


This produces the following report structure:

```
results/report/
├── composite_report.json      # Aggregate pass@k metrics across all samples
├── composite_report.txt       # Human-readable composite (per-category breakdown, pass rates)
├── sample_1/
│   ├── report.json            # Per-sample metrics (pass rate, category breakdown)
│   ├── report.txt             # Human-readable per-sample report
│   ├── raw_result.json        # Raw pass/fail results for every task
│   └── cvdp_copilot_<task>/   # Per-task directories
│       ├── prompts/
│       │   └── <issue>.md     # System + user prompt sent to the model
│       └── reports/
│           └── <issue>.txt    # Apptainer test harness stdout/stderr
├── sample_2/
│   └── ...
└── ...
```

- `**composite_report.txt**` — the main result: per-sample pass rates, mean/stddev, overall pass@k, and per-category breakdown.
- `**report.txt**` (per-sample) — individual sample results with the same category breakdown.
- `**raw_result.json**` — machine-readable pass/fail for each task, used by CVDP's `combine_reports()`.
- **Per-task `reports/<issue>.txt`** — full cocotb test output (from Apptainer container) for debugging individual failures.

#### Beginning of composite_report.txt

```
=== Benchmark Report ===
Dataset: /Users/artij/Projects/cvdp/cvdp_benchmark/cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl
Generated: 2026-03-05 13:09:04

=== Run Configuration ===
Model/Agent: nemotron_super_checkpoint_3

=== Composite Report ===
Number of samples: 5
Pass@1, n=5 threshold: A problem passes if it passes in at least 1 out of 5 samples
=======================


=== Sample Statistics ===
+------------+------------------+-------------------+-------------+----------------------------------------------+
|   Sample # |   Total Problems |   Passed Problems | Pass Rate   | Prefix                                       |
+============+==================+===================+=============+==============================================+
|          1 |              302 |                63 | 20.86%      | results/302_checkpoint3_run2/report/sample_1 |
+------------+------------------+-------------------+-------------+----------------------------------------------+
|          2 |              302 |                72 | 23.84%      | results/302_checkpoint3_run2/report/sample_2 |
+------------+------------------+-------------------+-------------+----------------------------------------------+
|          3 |              302 |                60 | 19.87%      | results/302_checkpoint3_run2/report/sample_3 |
+------------+------------------+-------------------+-------------+----------------------------------------------+
|          4 |              302 |                60 | 19.87%      | results/302_checkpoint3_run2/report/sample_4 |
+------------+------------------+-------------------+-------------+----------------------------------------------+
|          5 |              302 |                55 | 18.21%      | results/302_checkpoint3_run2/report/sample_5 |
+------------+------------------+-------------------+-------------+----------------------------------------------+

Pass Rate Statistics: Mean = 20.53%, StdDev = 2.08%
```

