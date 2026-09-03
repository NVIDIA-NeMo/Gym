# Nemotron 3.5 Super Evaluation setup
- [Nemotron 3.5 Super Evaluation setup](#nemotron-35-super-evaluation-setup)
  - [Run production evals](#run-production-evals)
  - [Development commands](#development-commands)
    - [vllm-router patch (decode-node cache imbalance)](#vllm-router-patch-decode-node-cache-imbalance)
      - [Measured effect](#measured-effect)
    - [Build eval container](#build-eval-container)
    - [Launch vLLM](#launch-vllm)
    - [Interactive development on GPUs with Ray cluster](#interactive-development-on-gpus-with-ray-cluster)
    - [Run eval against external vLLM endpoint](#run-eval-against-external-vllm-endpoint)


## Run production evals
TODO @bxyu-nvidia: Will publish these by Thu Jul 30

Results will appear in that checkpoint folder.


## Development commands

### vllm-router patch (decode-node cache imbalance)
`build_eval_container.sh` requires `VLLM_ROUTER_WHEEL` and does **not** fall back to
installing the released `vllm-router` wheel.

The released router resets every worker's in-flight load counter from the registry
health checker, every 10 health-check cycles -- 10 minutes at the default 60s
interval. The `cache_aware` policy reads those counters to decide when to abandon
prefix affinity in favour of shortest-queue routing, so the reset makes an
already-saturated worker look idle. Under P/D disaggregation
(`--vllm-pd-disaggregation --decode-policy cache_aware`, as used by
`sbatch_external_vllm.sh`) that closes a feedback loop: the worker holding the hot
prefixes keeps attracting requests, and shortest-queue never triggers to break it.

Note the reset is *unconditional*. There is a second, dead copy of the same logic in
`src/core/worker.rs` guarded by `max_load <= 2`; reading only that one leads to the
wrong conclusion that the reset fired just when workers were idle and was therefore
harmless. The one that actually ran, in `src/core/worker_registry.rs`, zeroed every
worker every 10 cycles regardless of load.

- bug: https://github.com/vllm-project/router/issues/197
- fix: https://github.com/vllm-project/router/pull/216 (unmerged upstream)

Build the wheel once with `build_vllm_router_wheel.sh`, then pass it to the container
build. The wheel is built inside the eval base image, so its extension module matches
the Python that runs `vllm-router` at eval time:

```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
SBATCH_GRES=gpu:4 \
CONTAINER=/path/to/vllm/container \
sbatch benchmarks/nemotron_3.5_super/build_vllm_router_wheel.sh
# -> results/vllm_router/wheels/vllm_router-*.whl
```

The wheel's directory is mounted into the build automatically; it only has to live on
storage the compute node can read.

#### Measured effect

Two SWE-bench Multilingual runs on the released router exhibited the runaway. Both
are 2 prefill / 2 decode with 450 rollouts in parallel. Ratio is decode max/min
running requests sampled per minute, over the minutes where the busiest decode node
held at least 100 running; `starved` counts minutes where one node sat at ~0 running
while its peer was busy:

| run | router | loaded | starved | median | max | early -> late | max KV | max queued |
|-----|--------|--------|---------|--------|-----|---------------|--------|------------|
| 6800138 | stock 0.1.15 | 153m | 3 | 9.84 | 333.00 | 2.62 -> 103.50 | 100% | 224 |
| 6794553 | stock 0.1.15 | 104m | 0 | 4.72 | 183.60 | 2.28 -> 37.92 | 99.8% | 207 |
| 6802686 | #216 | 32m | 0 | 1.03 | 1.11 | 1.03 -> 1.02 | 27.8% | 0 |
| 6803266 | #216 | 34m | 0 | 1.04 | 1.14 | 1.03 -> 1.04 | 27.5% | 0 |
| 6803267 | #216 | 31m | 0 | 1.04 | 1.21 | 1.04 -> 1.02 | 27.7% | 0 |
| 6803268 | #216 | 31m | 0 | 1.04 | 1.30 | 1.06 -> 1.04 | 28.7% | 0 |
| 6803269 | #216 | 34m | 0 | 1.04 | 1.34 | 1.04 -> 1.06 | 27.9% | 0 |

Both bad runs show the same signature: an even start that diverges monotonically
(`early -> late`), ending with one decode node pinned near 100% KV cache with 200+
requests queued while its peer drains toward idle. 6800138 stalled at 583/900
rollouts. The patched runs stay flat, never queue, and hold KV below 29%.

Not every run on the released router hits this -- it needs a sustained saturated
decode regime -- so a clean run does not tell you which router you are on. Use the
fingerprint above instead.


### Build eval container
Example run:
```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
INPUT_CONTAINER=/path/to/vllm/container \
OUTPUT_CONTAINER=/path/to/vllm/container___with_gym.sqsh \
VLLM_ROUTER_WHEEL=/path/to/vllm_router/whl \
MOUNTS=/path/to/env.yaml:/opt/Gym/env.yaml:x-create=file,/path/to/config.yaml:/opt/Gym/config.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/eval_container_config.yaml \
sbatch --gres=gpu:4 \
  benchmarks/nemotron_3.5_super/build_eval_container.sh
```


### Launch vLLM
This script assumes:
- GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
- Nemotron 3 Ultra configs e.g. with the parser configs.

Example run:
```bash
MODEL=/path/to/model \
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh
```


### Interactive development on GPUs with Ray cluster
Example run:
```bash
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
SBATCH_GRES=gpu:4 \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash scripts/sbatch_interactive.sh
```


### Run eval against external vLLM endpoint
This script assumes:
- The container is one built via benchmarks/nemotron_3.5_super/build_eval_container.sh
- GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
- Nemotron 3 Ultra configs e.g. with the parser configs.

If you want to use your own custom local Gym, please mount:
```bash
MOUNTS=/shared/fs:/shared/fs,/path/to/custom/local/Gym:/opt/Gym
```
The existing Gym venv and individual server venvs will still use the ones baked into the container.

Example run:
```bash
MODEL=/path/to/model \
EXPERIMENT_NAME=my-experiment-name \
NUM_NODES=4 \
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
CONTAINER=/path/to/vllm/container \
MOUNTS=/shared/fs:/shared/fs \
bash benchmarks/nemotron_3.5_super/sbatch_eval_with_external_vllm.sh \
--config benchmarks/my-benchmark/config.yaml
```
