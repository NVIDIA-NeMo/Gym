# Nemotron 3.5 Super Evaluation setup
- [Nemotron 3.5 Super Evaluation setup](#nemotron-35-super-evaluation-setup)
  - [Run production evals](#run-production-evals)
  - [Development commands](#development-commands)
    - [Build eval container](#build-eval-container)
    - [vllm-router patch (decode-node cache imbalance)](#vllm-router-patch-decode-node-cache-imbalance)
    - [Launch vLLM](#launch-vllm)
    - [Interactive development on GPUs with Ray cluster](#interactive-development-on-gpus-with-ray-cluster)
    - [Run eval against external vLLM endpoint](#run-eval-against-external-vllm-endpoint)


## Run production evals
TODO @bxyu-nvidia: Will publish these by Thu Jul 30

Results will appear in that checkpoint folder.


## Development commands
### Build eval container
Example run:
```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=batch \
INPUT_CONTAINER=/path/to/vllm/container \
OUTPUT_CONTAINER=/path/to/vllm/container___with_gym.sqsh \
MOUNTS=/path/to/env.yaml:/opt/Gym/env.yaml:x-create=file,/path/to/config.yaml:/opt/Gym/config.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/eval_container_config.yaml \
sbatch --gres=gpu:4 \
  benchmarks/nemotron_3.5_super/build_eval_container.sh
```


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
SBATCH_PARTITION=cpu \
SBATCH_QOS=cpu-normal \
CONTAINER=/path/to/vllm/container \
sbatch --nodes=1 --ntasks=1 --cpus-per-task=96 --mem=0 --time=03:00:00 --gpus-per-node=0 \
  benchmarks/nemotron_3.5_super/build_vllm_router_wheel.sh
# -> results/vllm_router/wheels/vllm_router-*.whl

VLLM_ROUTER_WHEEL=results/vllm_router/wheels/vllm_router-0.1.15-cp38-abi3-linux_aarch64.whl \
INPUT_CONTAINER=... OUTPUT_CONTAINER=... MOUNTS=... GYM_CONFIG=... \
sbatch benchmarks/nemotron_3.5_super/build_eval_container.sh
```

The wheel's directory is mounted into the build automatically; it only has to live on
storage the compute node can read.

PR #216 does not bump the version -- a fixed router and the stock 0.1.15 wheel both
report `0.1.15` -- so the container build asserts on the compiled extension instead:
stock carries the string `Resetting worker loads (cycle`, and only the fix carries
`Attempted to decrement load counter that is already at 0`. A wheel built from the
wrong commit fails the build rather than shipping silently.

To check an image after the fact, boot it and grep the extension:

```bash
srun --container-image=/path/to/image.sqsh --no-container-mount-home bash -c '
  so=$(python3 -c "import vllm_router_rs; print(vllm_router_rs.__file__)")
  grep -aqF "Resetting worker loads (cycle" "$so" && { echo "STOCK - unpatched"; exit 1; }
  grep -aqF "Attempted to decrement load counter that is already at 0" "$so" || { echo "missing #216"; exit 1; }
  echo "OK: carries vllm-project/router#216"'
```

The same one-liner works against a *running* job with
`srun --overlap --jobid=<id> --container-name=container-on-node`. From a finished
job's Slurm log, the router's own source line identifies the build:
`vllm_pd_router.rs:1935` for stock 0.1.15, `:1999` for the #216 build.

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

#### What the fix does not cover

Neither of these is reachable with the flags `sbatch_external_vllm.sh` uses today,
and neither is a regression -- v0.1.15 behaved the same way. They matter only if the
deployment changes.

- **Streaming.** `process_vllm_two_stage_request` releases the decode load guard as
  soon as the decode response *headers* arrive, before the body is forwarded. For a
  streaming request Starlette emits `http.response.start` before iterating the body
  generator, so the decode counter returns to 0 at request acceptance and the whole
  generation runs uncounted -- reproducing #197 at full strength. NeMo Gym pins
  `stream: Literal[False]`, so we never hit it. Applies to `/v1/responses` too, which
  funnels into the same handler via `route_transparent`.
- **Service-discovery mode.** `process_vllm_two_stage_request_discovered` performs no
  load accounting at all, so `cache_aware` sees `min_load == max_load == 0` forever
  and degenerates to pure prefix affinity with no load ceiling. Only reachable via
  `--vllm-discovery-address`; we pass explicit `--prefill`/`--decode` URLs.

Also worth knowing: with the CLI defaults `--balance-abs-threshold 64` and
`--balance-rel-threshold 1.5`, cache affinity can hold a sustained ~64-request or
1.5x decode split indefinitely without ever tripping shortest-queue. That is the
ceiling the fix leaves in place.

Once upstream merges #216 and cuts a release, drop `build_vllm_router_wheel.sh` and
the `VLLM_ROUTER_WHEEL` requirement and go back to `uv pip install --system
vllm-router`.


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
