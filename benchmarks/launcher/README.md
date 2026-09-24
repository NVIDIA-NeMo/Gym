# Getting Started

On your Slurm cluster with Pyxis/Enroot, put these exports in `benchmarks/launcher/.env`, replacing the placeholders:

```bash
# Slurm
export SBATCH_ACCOUNT="<your-slurm-account>"
export SBATCH_PARTITION="<your-gpu-partition>"
export SBATCH_QOS="<your-gpu-qos>"

# Cluster
export REPLICAS_PER_NODE=4

# Images
export GYM_CONTAINER="/path/to/gym.sqsh"
export ROUTER_CONTAINER="/path/to/vllm-router.sqsh"
export VLLM_CONTAINER="registry-1.docker.io#vllm/vllm-openai:v0.29.0-aarch64"

# Sandbox
export OPENSANDBOX_DOMAIN="http://<your-opensandbox-endpoint>"
export OPENSANDBOX_API_KEY="<your-opensandbox-key>"

# Weight and Biases
export WANDB_API_KEY="<your-wandb-key>"
export WANDB_PROJ="<your-project>"
export WANDB_ENTITY="<your-team-or-user>"
export WANDB_MODE=online
```

`eval.sh` automatically loads the gitignored `.env`; no manual sourcing is needed.
Keep credentials private with `chmod 600 benchmarks/launcher/.env`.
Images can be shared `.sqsh` files or Pyxis registry references (`registry#namespace/image:tag`).

From the Gym repository root, launch TB 2.1 with Laguna S and DFlash:

```bash
bash benchmarks/launcher/eval.sh \
  --profile benchmarks/launcher/laguna_s/terminal_bench_2_1.sh \
  --checkpoint /path/to/checkpoints/Laguna-S-2.1-FP8
```

For Laguna XS 2.1 BF16, use the matching XS profile and checkpoint:

```bash
bash benchmarks/launcher/eval.sh \
  --profile benchmarks/launcher/laguna_xs/terminal_bench_2_1.sh \
  --checkpoint /path/to/checkpoints/Laguna-XS-2.1
```

Both profiles use TP1, a 262144-token context limit, and seven DFlash draft tokens.
The XS profile uses `poolside/Laguna-XS-2.1-DFlash`; the S profile retains its
existing `poolside/Laguna-S-2.1-DFlash-FP8` draft.
See the [Laguna XS model card](https://huggingface.co/poolside/Laguna-XS-2.1)
for serving requirements.

Replace `terminal_bench_2_1.sh` with `swe_verified.sh`, `swe_multilingual.sh`,
or `swe_pro.sh` for SWE evaluations. Add `--smoke` for one task and one rollout.
Logs and results go to `runs/<job-id>-<benchmark>/` beside the Gym checkout. Set `RUNS_DIR` to override that location.

# Mounting a Development Gym Checkout

Add this to `benchmarks/launcher/.env`:

```bash
export GYM_DEV_CHECKOUT="/path/to/dev-gym"
```

The checkout must be accessible on compute nodes. It is mounted read-only at
`/mnt/gym-dev` and copied into the container at `/opt/nemo-gym`, excluding runtime
logs, caches, environments and credentials. Installation and evaluation modify only
this private copy. Source edits after startup apply to future jobs; logs and results
still go to the shared run directory.

Set `GYM_INFERENCE_METRICS_ENABLED=true` to scrape vLLM replicas and the router
and publish their counters and gauges through the configured exporters (including W&B).
Router metrics appear under `router/main/`; vLLM metrics remain under `vllm/`.


# GPU Idle-Time Exemption

GPU evaluation jobs request a 240-minute idle-time exemption through the Slurm
job comment, using the `benchmarking` reason. This matches the launcher's
four-hour wall time and allows task tools and verifiers to finish while inference
GPUs are idle. CPU cleanup jobs do not receive this comment.

To supply a different job comment, set `SLURM_COMMENT` in the launcher’s
`.env`. An override replaces the entire default JSON comment, including the
exemption. The exemption is interpreted by the cluster's job reaper; the Slurm
wall-time limit still applies. This setting affects newly submitted jobs.
