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
`build_eval_container.sh` does **not** install the released `vllm-router` wheel.

The released router resets every worker's in-flight load counter from the registry
health checker, every 10 health-check cycles (10 minutes at the default 60s
interval). The `cache_aware` policy reads those counters to decide when to abandon
prefix affinity in favour of shortest-queue routing, so the reset makes an
already-saturated worker look idle. Under P/D disaggregation
(`--vllm-pd-disaggregation --decode-policy cache_aware`, as used by
`sbatch_external_vllm.sh`) that is a feedback loop: the worker holding the hot
prefixes keeps attracting requests, and shortest-queue never triggers to break it.

Note the reset is *unconditional*. There is a second, dead copy of the same logic in
`src/core/worker.rs` guarded by `max_load <= 2`; reading only that one leads to the
wrong conclusion that the reset fired just when workers were idle and was therefore
harmless. The one that actually ran, in `src/core/worker_registry.rs`, zeroed every
worker every 10 cycles regardless of load.

- bug: https://github.com/vllm-project/router/issues/197
- fix: https://github.com/vllm-project/router/pull/216 (unmerged upstream)

The container build pins and builds the PR head
(`VLLM_ROUTER_COMMIT`, defaults to the #216 head) and asserts the periodic reset is
really gone before building. Set `VLLM_ROUTER_WHEEL` to a wheel path reachable
inside the container to install a prebuilt artifact instead of paying for the Rust
build again. `build_vllm_router_wheel.sh` produces such a wheel, and is the place to
validate a router change before committing to a full container rebuild:

```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=cpu \
SBATCH_QOS=cpu-normal \
CONTAINER=/path/to/vllm/container \
sbatch --nodes=1 --ntasks=1 --cpus-per-task=96 --mem=0 --time=03:00:00 --gpus-per-node=0 \
  benchmarks/nemotron_3.5_super/build_vllm_router_wheel.sh
# -> results/vllm_router/wheels/vllm_router-*.whl
```

Once upstream merges #216 and cuts a release, drop both build paths and go back to
`uv pip install --system vllm-router`.

#### Measured effect

Five patched SWE-bench Multilingual runs (2 prefill / 2 decode, 450 in parallel)
against one unpatched run of the same checkpoint, on the same cluster in the same
window. Ratio is decode max/min running requests, sampled per minute and counted
only while combined decode load >= 200 -- a run whose queue drains early looks
balanced whatever router it used, which is exactly how a second unpatched run
(6792467, saturated only 32 minutes) managed to look innocent.

| run | router | saturated | median | max | early -> late | max KV | max queued |
|-----|--------|-----------|--------|-----|---------------|--------|------------|
| 6800138 | stock 0.1.15 | 113m | 4.99 | 70.06 | 2.11 -> 19.39 | 100% | 224 |
| 6802686 | #216 | 30m | 1.03 | 1.11 | 1.03 -> 1.02 | 27.8% | 0 |
| 6803266 | #216 | 33m | 1.04 | 1.14 | 1.03 -> 1.04 | 27.5% | 0 |
| 6803267 | #216 | 31m | 1.04 | 1.21 | 1.04 -> 1.02 | 27.7% | 0 |
| 6803268 | #216 | 30m | 1.04 | 1.16 | 1.06 -> 1.04 | 28.7% | 0 |
| 6803269 | #216 | 33m | 1.04 | 1.16 | 1.04 -> 1.06 | 27.9% | 0 |

The cost is throughput, not just tidiness. The unpatched run decayed to 6% of its
opening rollout rate (11.5/min -> 0.7/min) and stalled at 583/900 with one decode
node down to 1.4 concurrent requests while the other held 171; its own ETA went to
9h32m at 108 s/it. All five patched runs held 89-134% of their opening rate and
finished, at 1.75-14 s/it.

Reproduce either table with `tools/decode_balance.py`, `tools/compare_decode_balance.py`
and `tools/rollout_progress.py` against the job's slurm log.

#### What the fix does not cover

Both verified against the shipped source; neither is reachable with the flags
`sbatch_external_vllm.sh` uses today, and neither is a regression -- v0.1.15 behaved
the same way. They matter only if the deployment changes.

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
ceiling the fix leaves in place. Runs above sit at 1.03-1.04, well inside it.

The router exports `vllm_router_worker_load` and `vllm_router_worker_health` on a
Prometheus endpoint that is always enabled, and #216 is what wires the load gauge on
the P/D path. Scraping it would replace the vLLM-engine-log reconstruction the tools
above do. Health-state transitions are logged nowhere, so that endpoint (or the
router's own `/health`, which lists unhealthy servers) is the only way to see them.


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
