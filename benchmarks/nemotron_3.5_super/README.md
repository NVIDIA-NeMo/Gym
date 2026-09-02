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

If an eval image already exists and the router is the only thing that needs to
change, `patch_container_vllm_router.sh` swaps it in place and leaves everything
else byte-identical -- a full rebuild re-runs `gym eval prepare`, which needs live
credentials for the gated benchmarks. It is also what you want for a controlled A/B:

```bash
SBATCH_ACCOUNT=my-slurm-account \
SBATCH_PARTITION=cpu \
SBATCH_QOS=cpu-normal \
INPUT_CONTAINER=/path/to/with_gym.sqsh \
OUTPUT_CONTAINER=/path/to/with_gym_patched.sqsh \
VLLM_ROUTER_WHEEL=results/vllm_router/wheels/vllm_router-0.1.15-cp38-abi3-linux_aarch64.whl \
sbatch benchmarks/nemotron_3.5_super/patch_container_vllm_router.sh
```

Both container-producing scripts stage to `$OUTPUT_CONTAINER.partial` and publish
only on success. This matters more than it looks: pyxis `--container-save` exports
the image when the step tears down *regardless of the inner script's exit status*,
and overwrites whatever already sits at the target -- so without the staging step a
failed build silently replaces a verified image with an unverified one. The
provenance marker is likewise written only after the binary assertions pass, so a
shipped image cannot claim a commit its `.so` does not carry. To check an image
after the fact, boot it and grep the extension:

```bash
srun --container-image=/path/to/image.sqsh --no-container-mount-home bash -c '
  so=$(python3 -c "import vllm_router_rs; print(vllm_router_rs.__file__)")
  grep -aqF "Resetting worker loads (cycle" "$so" && { echo "STOCK - unpatched"; exit 1; }
  grep -aqF "Attempted to decrement load counter that is already at 0" "$so" || { echo "missing #216"; exit 1; }
  echo "OK: carries vllm-project/router#216"'
```

The same one-liner works against a *running* job with
`srun --overlap --jobid=<id> --container-name=container-on-node`, which is how to
confirm which router an in-flight eval is actually using.

Once upstream merges #216 and cuts a release, drop both build paths and go back to
`uv pip install --system vllm-router`.

#### Measured effect

The failure is intermittent, and that governs how strong a claim the data supports.
Across the 23 SWE-bench Multilingual runs in `slurm-logs/` that carry a router
fingerprint (18 stock 0.1.15, 5 built from #216 -- the router logs its own source
line, `vllm_pd_router.rs:1935` vs `:1999`, which identifies the build), the runaway
appears in **2 of 18** stock runs and **0 of 5** patched ones:

| run | router | loaded | starved | median | max | early -> late | max KV | max queued |
|-----|--------|--------|---------|--------|-----|---------------|--------|------------|
| 6800138 | stock | 153m | 3 | 9.84 | 333.00 | 2.62 -> 103.50 | 100% | 224 |
| 6794553 | stock | 104m | 0 | 4.72 | 183.60 | 2.28 -> 37.92 | 99.8% | 207 |
| 16 other stock runs | stock | 15-46m | 0 | 1.03-1.11 | <= 1.87 | flat | <= 37% | 0 |
| 6802686 | #216 | 32m | 0 | 1.03 | 1.11 | 1.03 -> 1.02 | 27.8% | 0 |
| 6803266 | #216 | 34m | 0 | 1.04 | 1.14 | 1.03 -> 1.04 | 27.5% | 0 |
| 6803267 | #216 | 31m | 0 | 1.04 | 1.21 | 1.04 -> 1.02 | 27.7% | 0 |
| 6803268 | #216 | 31m | 0 | 1.04 | 1.30 | 1.06 -> 1.04 | 28.7% | 0 |
| 6803269 | #216 | 34m | 0 | 1.04 | 1.34 | 1.04 -> 1.06 | 27.9% | 0 |

**What this does and does not establish.** The mechanism is established
independently of these runs: the reset is unconditional in the source, it is what
gates `cache_aware`'s rebalance, and the shipped container was verified at the
binary level. When the runaway does fire it is unmistakable -- 6800138 ends with one
decode node at 100% KV and 224 requests queued while its peer sits at ~0 running.

But 5 clean patched runs is **not** statistically meaningful against a ~11% base
rate: 16 of 18 stock runs are also clean, and `0.89^5 = 0.56`, so five clean runs
are more likely than not even with no fix at all. Fisher's exact on 2/16 vs 0/5
gives p = 1.00; roughly 26 clean patched runs would be needed for p < 0.05. Treat
the patched rows as *consistent with* the fix, not as a demonstration of it.

**Do not read a throughput claim into this.** Wall-clock comparison across these
runs is confounded. Completed *balanced* stock runs take 206-216 minutes for 900
rollouts, while the patched runs are pacing at roughly half that -- and since both
groups are equally balanced, decode balance cannot be the cause; cluster and sandbox
provisioning conditions differed between the two windows. Note also that 6794553
exhibited a severe imbalance (median 4.72, peak 183:1) yet still completed 900/900
in 234 minutes, only ~10% slower than a balanced run. Only 6800138 actually stalled.
So imbalance is not reliably a throughput catastrophe, and the throughput difference
observed here is not attributable to the router.

**Why the gate is on the busiest node, not the pair.** `compare --min-load` gates on
`max(loads)`. Gating on the pair's combined running count is post-treatment: when the
router hot-spots, the cold node's slots go unused and work accumulates in the hot
node's *waiting* queue rather than the pair's *running* count, so combined load falls
precisely when imbalance is worst (Spearman -0.77 on 6800138). An earlier version of
this table used a combined-load gate and reported 6800138 as median 4.99 / max 70,
because the gate had silently discarded its most extreme minutes. The `starved`
column exists for the same reason -- minutes where one node sits at ~0 have no finite
ratio and would otherwise vanish from the summary entirely.

Reproduce with `decode_balance.py` in this directory:

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
