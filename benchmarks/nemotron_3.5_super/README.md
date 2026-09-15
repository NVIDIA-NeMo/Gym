# Nemotron 3.5 Super Evaluation setup
- [Nemotron 3.5 Super Evaluation setup](#nemotron-35-super-evaluation-setup)
  - [Run production evals](#run-production-evals)
    - [Typical job shapes](#typical-job-shapes)
    - [Open problems](#open-problems)
    - [Tuning and debug protocol](#tuning-and-debug-protocol)
      - [Current vLLM decode speeds across engine batch sizes](#current-vllm-decode-speeds-across-engine-batch-sizes)
  - [Development commands](#development-commands)
    - [vllm-router patch (decode-node cache imbalance)](#vllm-router-patch-decode-node-cache-imbalance)
      - [Measured effect](#measured-effect)
    - [Build eval container](#build-eval-container)
    - [Launch vLLM](#launch-vllm)
    - [Interactive development on GPUs with Ray cluster](#interactive-development-on-gpus-with-ray-cluster)
    - [Run eval against external vLLM endpoint](#run-eval-against-external-vllm-endpoint)


## Run production evals
Results will appear in that checkpoint folder.

### Typical job shapes
These job shapes have been tuned to finish evaluation on Nemotron 3.5 Super checkpoints within a 4 hour Slurm timeout window. All compute numbers assume GB200 NVL72.

|Name|Argument|
|---|---|
|Prefill nodes|`NUM_PREFILL_NODES=<>`|
|Decode nodes|`NUM_DECODE_NODES=<>`|
|Concurrency|`++num_samples_in_parallel=<>`|

|Benchmark|Harness|Prefill nodes|Decode nodes|Concurrency|
|---|---|---|---|
|SWE Bench Verified + Multilingual|OpenCode|2|2|1024|
|SWE Bench Pro|OpenCode|4|6|1024|
|DeepSWE (1 repeat)|OpenCode|2|2|1024|
|Terminal Bench 2.1|Terminus 2|2|8|512|
|Terminal Bench 2.1|OpenCode|?|?|?|

### Open problems
1. We can't reduce the number of prefill nodes because the TRT LLM kernel isn't large enough to support higher max_num_batched_tokens
2. Once MTP is functional with PD-disagg / async scheduling / prefix caching / etc, we should be able to reduce the decode nodes as well.

### Tuning and debug protocol

Prerequisites for tuning and debugging:
1. Gym Slurm log containing the vLLM engine prints.
2. Final Gym output aggregate metrics including the harness finish rate.

1. Check if the harness finish rate is expected or not.
   1. For example, as of Mon Sep 07, the expected finish rate for TerminalBench 2.1 + Terminus 2 harness is around 90% Terminus 2 harness finish rate.
   2. If the finish rate is within the expected range, then usually things are fine from an infra perspective.
2. Inspect the vLLM engine logs in the Gym Slurm logs.
   1. Identify the prefill and decode vLLM engine logs by looking at the "Prompt throughput" and "Generation throughput". The engines that have non-zero "Prompt throughput" are the prefill engines, and the ones with non-zero "Generation throughput" are decode engines.
3. Do I need to increase compute because of waiting requests?
   1. Check if there are any "Waiting requests" on any of the engine types.
      1. The typical number of waiting requests against an engine is 0 or close to 0.
      2. If there are waiting requests built up, the number will typically be 100s or 1000s.
   2. If there are waiting requests on any of the engines, rerun the same config with an increase in the number of that engine type.
      1. For example, if the current shape is p2d2 and the decode engines have a lot of waiting requests, try increasing to p2d4.
      2. Typical shapes are powers of 2 up to whatever the max NVLink shape supported is e.g. p2d8, p2d14 (segment 16), p2d16 (segment 18), etc.
4. Do I need to increase compute because of decode speed?
   1. Check if the finish rate is non-zero and lower than you expect. It could be 3% lower or 40% lower depending on the verbosity of the checkpoint.
   2. Please refer to the decode speeds table below to see what compute shape you need to satisfy your latency requirement.
5. Did something weird happen on the vLLM engine side?
   1. If the progress rollouts/min reported in W&B is very different than usual, that may indicate a transient failure on the vLLM engine side. Try rerunning with the same config and see if the same behavior persists.
6. Is there something else wrong?
   1. Message @bxyu-nvidia @sdevare in Slack and share your Slurm logs and W&B.

#### Current vLLM decode speeds across engine batch sizes
Definitions
1. Batch size: The number of requests that the engine is currently running.
2. Engine throughput: The total tokens/s throughput for all requests, logged by vLLM every 10s interval.
3. Effective tokens/second/request (tok/s/req): Engine throughput divided by the instantaneous batch size reported by vLLM.

Written as of Mon Sep 07, 2026 using this [Super 3.5 config](https://github.com/NVIDIA-NeMo/Gym/blob/ae8d388dda62f40fe8b8105bf079be132383fe4d/benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh).

|Batch size|Engine throughput (tok/s)|Tok/s/req|
|---|---|---|
|<=16|2000|130|
|32|2500|80|
|64|3600|60|
|128|6000|45|
|256|8000|30|
|512|9000|15|


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

#216 on its own is not enough, and for prefill-heavy benchmarks it is worse than not
applying it. It makes the worker load counters honest, which switches on a second
latent bug: `cache_aware` decides whether to use prefix affinity from the
*fleet-wide* load spread, so one hot worker discards affinity for every request --
including requests whose own worker is idle. Under P/D that gate is open almost
permanently, because prefill worker load counts queued requests as well as running
ones. Routing degenerates to shortest-queue, already-cached prompts get recomputed,
prefill saturates and decode starves behind it.

The pin therefore points at a branch carrying #216 plus a fix that applies the same
load check per request, against the worker that request wants:

- prefill fix: https://github.com/vllm-project/router/pull/238

Both are plain commit SHAs fetched from `vllm-project/router`; a PR head is a ref
there even when the branch lives on a contributor's fork. Repin to a released commit
once these land upstream.

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
