# CVDP Benchmark

Models are given a hardware design specification and asked to produce a SystemVerilog
implementation. Verification is done by running the per-task cocotb test harness inside
Apptainer (using the `ghcr.io/hdl/sim/osvb` simulator image with Icarus Verilog).

## Verification Flow

Native resources server — verification runs as an Apptainer subprocess invoked
from `verify()`. The harness files (compose config, test scripts) are embedded in each
JSONL entry so the server is self-contained.

Mirrors `repository.py` in the [CVDP source](https://github.com/NVlabs/cvdp_benchmark):

1. Parse model response via `ModelHelpers.parse_model_response()`
2. Write harness files to temp workspace — applies image placeholder substitutions
3. Write extracted RTL to `workdir/rtl/`
4. For each service in `docker-compose.yml`, pull the Docker image as a cached SIF file and run via `apptainer exec` with `--bind` mounts for `rtl/`, `verif/`, `docs/`, `src/`, `rundir/`
5. Exit code `0` across all services → reward `1.0`; any failure → reward `0.0`

## Configuration


| Field               | Default                 | Description                                                   |
| ------------------- | ----------------------- | ------------------------------------------------------------- |
| `oss_sim_image`     | `ghcr.io/hdl/sim/osvb`  | Container image for open-source simulation (Icarus)           |
| `oss_pnr_image`     | `""`                    | Container image for place-and-route problems                  |
| `eda_sim_image`     | `""`                    | Commercial EDA image (Cadence Xcelium etc.)                   |
| `container_timeout` | `600`                   | Seconds before an Apptainer run is killed                     |
| `num_processes`     | `4`                     | Max concurrent Apptainer jobs                                 |
| `sif_cache_dir`     | `~/.cache/nemo-gym/sif` | Directory for cached SIF images pulled from Docker registries |


## Download Dataset

The data can be found [on Hugging Face](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset). 

## Step 1 — Download dataset from HF

```bash
huggingface-cli download nvidia/cvdp-benchmark-dataset \
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
policy_base_url: https://inference-api.nvidia.com/v1
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
ng_run "+config_paths=[resources_servers/cvdp/configs/cvdp.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

### Step 2 — Run rollout collection

`num_repeats` controls how many times each problem is run (for pass@k metrics):

```bash
ng_collect_rollouts \
    +agent_name=cvdp_simple_agent \
    +input_jsonl_fpath=resources_servers/cvdp/data/<dataset>.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +num_repeats=5 \
    +num_samples_in_parallel=4 \
    "+responses_create_params={max_output_tokens: 4096, temperature: 0.2, top_p: 0.7}" \
    "+config_paths=[resources_servers/cvdp/configs/cvdp.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
    "+resume_from_cache=True"
```

At the end of this step, you should have rollouts.jsonl, rollouts_agent_metrics.json, rollouts_materialized_inputs.jsonl, rollouts_reward_profiling.jsonl.

### Step 3 — Generate report

```bash
python resources_servers/cvdp/scripts/cvdp_pass_at_k_report.py \
    --rollouts  results/rollouts.jsonl \
    --output    results/report/ \
    --model     cvdp_simple_agent \
    --dataset   <original_cvdp_dataset>.jsonl \
    --k         1
```


| Arg          | Required | Default    | Description                                               |
| ------------ | -------- | ---------- | --------------------------------------------------------- |
| `--rollouts` | Yes      | —          | Rollout JSONL from `ng_collect_rollouts`                  |
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

