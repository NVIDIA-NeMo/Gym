# Measured rates, for compute planning

Per-environment throughput, measured lane by lane against `manifests/nemotron_3_ultra.yaml`
(now archived; `nemotron_3_5_super.yaml` has the identical entry list, so the rates carry) on
1 prefill + 2 decode = **12 GPUs**, `num_repeats: 8`.

**These are shares of one fleet, not independent capacities.** Every environment in a lane shared
one endpoint, so a fast environment's rate reflects queueing behind slow ones. Use them to rank
which environments dominate; do not sum them for GPU-hours — that overcounts by roughly the number
of environments. Size from the aggregate at the bottom.

| environment | lane | rollouts/hr | notes |
|---|---|---|---|
| tau_pivot | plain | 755 | shares an agent with `tau_pivot_aq_mms` |
| tau_pivot_aq_mms | plain | 755 | shared rate |
| comp_coding | plain | 494 | shares an agent with `comp_coding_nemotronx` |
| comp_coding_nemotronx | plain | 494 | shared rate, but 688 KB/row vs 306 — true rate likely lower |
| ds1_augmented | plain | 387 | shares an agent with `ds1_basic` |
| ds1_basic | plain | 387 | shared rate |
| structured_outputs | plain | 385 | shares an agent with `structured_outputs_v2` |
| structured_outputs_v2 | plain | 385 | shared rate |
| equivalence_rule | plain | 288 | |
| structured_outputs_v4 | plain | 281 | |
| gdpval_pivot_v1 | plain | 278 | |
| toolcall_schema | plain | 269 | |
| workbench | plain | 264 | |
| ds2_freeform | plain | 261 | |
| droid_pivot_v1 | plain | 261 | |
| search_pivot | plain | 235 | |
| stem_mcqa_ultra_0 | plain | 232 | |
| ds3_citation | plain | 206 | |
| calendar_v2 | plain | 172 | |
| nvarc_transductive | plain | 167 | |
| instruction_following | plain | 152 | |
| swe_pivot | plain | 126 | |
| nvarc_inductive | plain | 85 | |
| ether0 | plain | 80 | |
| reasoning_gym | plain | 56 | slowest plain env; a tail driver |
| terminal_pivot | plain | — | zero rollouts in the measured run; unsized |
| multichallenge | judge | 285 | |
| long_context | judge | 278 | |
| abstention | judge | 271 | post-fix. The pre-fix 544/hr is void — truncated judge returning early |
| inverse_if | judge | 258 | |
| stem_openqa_ultra_0 | judge | 231 | |
| math_cot | judge | 75 | slowest judge env by 4x; n=9-11, rate and reward both unresolved |
| math_tir | sandbox | 421 | all three `ns_tools` entries share one agent and one rate |
| stem_mcqa_tools_ultra_0 | sandbox | 421 | shared rate |
| stem_openqa_tools_ultra_0 | sandbox | 421 | shared rate |
| lean | sandbox | 61 | n=7, all reward 0.000. Too few to read; worst env in the pre-sweep too |

Lane aggregates: **plain 5,621/hr** (concurrency 128), **judge 1,398-2,139/hr** (48),
**sandbox 483/hr** (32, one sandbox node with 32 uWSGI workers).

## Sizing

Size from the mixed-workload run, not from the lanes. Running all 36 together on
2 shards x (1 prefill + 1 decode) = 16 GPUs gave **~5,620 rollouts/hr** (job 6602112/6602113),
and a launcher run at the same shape sustained a **~550 rollouts/hr/GPU peak** (job 6608099/6608100).

```
5,808,968 rollouts / 5,620 per hr = 1,034 h on 16 GPUs = 16,540 GPU-hours
```

| nodes | GPUs | wall |
|---|---|---|
| 24 | 96 | 7.2 days |
| 48 | 192 | 3.6 days |
| 96 | 384 | 1.8 days |

Summing the lanes instead gives 23,178-24,409 GPU-hours, and the gap is the point of running them
together: a judge rollout blocked on the gateway and a sandbox rollout blocked on uWSGI hold no GPU
while they wait, so they fill time the GPU-bound environments would leave idle.

Disk, one-time per (manifest, checkpoint): `01_materialize.sh` writes **271.5 GB** in ~27.5 min
(5,808,968 rows from 726,121 source rows, 36 workers). **Peak during the run is ~541 GB, roughly
2x the final size**, because `_parts/` is not removed until concatenation finishes — budget for the
peak, not the result. Sharding copies the file again, so a sharded run peaks near 815 GB.

## P2D8 on full data (job 6706202, 2026-08-29)

First run of the whole 5,808,968-rollout input at a production shape: 2 prefill + 8 decode = 10
nodes, 40 GPUs (TP=4 per node), `nemotron_n4_post` on the `normal` queue, 4 h walltime. The job hit
the walltime; it did not fail.

### Where the four hours went

| phase | wall | notes |
|---|---|---|
| `01_materialize.sh` | 27.5 min | 726,121 source rows -> 5,808,968 rollouts, 271.5 GiB |
| queue wait | ~1 h | `normal` qos behind ~33k pending node-requests |
| vLLM + 63 Gym servers up | 2.5 min | from job start to `All 63 / 63 servers ready!` |
| driver preflight | ~20 min | scans the whole 271.5 GiB input before the first dispatch |
| collecting | 3 h 26 min | 25,494 rollouts |

**The ~20 min preflight is per job, not per sweep** — every resubmission pays it again. It is a
linear scan of the input, so it scales with the file the job is given: a 1/16 shard is ~17 GiB and
pays about a minute. This is a concrete argument for sharding beyond parallelism.

### Throughput

| | |
|---|---|
| collection window | 3.43 h (01:15:11 -> 04:41:01 UTC) |
| rollouts | 25,494 of 5,808,968 (**0.44%**) |
| average | **7,431 rollouts/hr** = **186 per GPU-hr** |
| first hour | 9,429/hr |
| remaining 2.4 h | 6,614/hr |
| driver concurrency | 4,096 (512 x 8 decode nodes) |
| in flight at vLLM | ~33 per engine, so ~264 |
| GPU KV cache usage | **4.5%**, `Waiting: 0 reqs` |

Throughput decays as fast tasks drain, the same tail effect the whole-manifest runs show.

### Only 6 of 36 environments were reached

At 0.44% complete the run touched six environments. Everything else has **zero** rollouts, so it is
unmeasured, not slow:

| environment | rollouts | rollouts/hr | of its target | KB/rollout |
|---|---|---|---|---|
| tau_pivot | 8,996 | 2,622 | 0.66% | 30 |
| swe_pivot | 4,479 | 1,306 | 0.84% | 117 |
| structured_outputs_v2 | 4,363 | 1,272 | 1.94% | 26 |
| nvarc_transductive | 4,253 | 1,240 | 5.32% | 111 |
| math_tir | 2,481 | 723 | 7.95% | 128 |
| inverse_if | 922 | 269 | 11.53% | 43 |

These are shares of one contended fleet, not independent capacities, and each environment's
completed tasks sit in a narrow contiguous band of the input rather than spread across its range.
**Do not treat this table as per-domain rates for the blend** — six of thirty-six, at half a
percent, is not a sample worth sizing from. The 12-GPU lane table above is still the better basis.

### What is actually starving the GPUs

4.5% KV cache with `Waiting: 0` means vLLM is never the constraint. The log says where the other
~93% of the concurrency window went:

- **The `vllm-router` is a single process on one node** (`nodes[0]`, port 8000). It absorbed
  **12,495 `ClientOSError` retries**, and individual requests reached **`retry=954`** — a request
  retrying ~950 times holds a concurrency slot indefinitely without ever reaching a GPU.
- **The sandbox is one node** serving all 63 Gym servers: 1,168 IPython session timeouts, 190x502,
  137x504. Its two consumers, `ns_tools` and `lean`, returned **zero** rollouts.
- **Agentic rollouts are mostly not on the GPU.** `tau_pivot` and `swe_pivot` spend their wall time
  in tool calls, env stepping and sandbox execution. Low KV usage is the expected shape for this
  blend, not purely a defect.

**The judge is not the bottleneck.** Correcting the earlier reading of this run: the judge is the
hosted endpoint (`https://inference-api.nvidia.com/v1`) and it took **4 retries in the whole run**,
against 12,495 to the local router. Note the judge-heavy environments produced no rollouts, so the
judge is untested at load here rather than proven healthy.

### Sizing at this shape

```
5,808,968 / 6,614 per hr (steady) = 878 h on 40 GPUs = 35,132 GPU-hours   -> 36.6 days on one P2D8
                                                                          -> 2.3 days on 16 shards
```

That is roughly twice the 16,540 GPU-hours estimated from the 16-GPU run (186 vs 351 per GPU-hr):
2.5x the GPUs bought about 1.25x the throughput. **Fix the router and the sandbox tier before
adding decode nodes**; until those move, extra decode nodes are idle capacity.

## Caveats

- **Every rate is a floor.** Each decode engine schedules up to `max_num_seqs=512`; 128 concurrency
  against 2 decode nodes ran at ~12.5% of capacity. The client semaphore was the limit, not the GPUs.
- **Plan around the tail, not the aggregate.** A whole-manifest run went 151 -> 55 -> 18 -> 14 -> 12
  -> 6 -> 3 rollouts per 30 s window — ~18,000/hr instantaneous down to ~360, a 50x collapse. Fast
  environments drain immediately and the slow ones dominate the end. Expect a residue job.
- **Sandbox capacity is a separate axis from GPU count.** It measured lowest of any lane at the
  lowest concurrency, on a single node. Size the sandbox tier before trusting the sandbox figures.
- **The judge lane ran at concurrency 48 against the plain lane's 128**, so the 4x aggregate gap
  overstates the judge penalty. It also measured 1,398 and 2,139/hr on two runs at identical
  concurrency — a 53% spread, probably prefix-cache warmth. Re-measure on a cold endpoint.
- **Two config fixes live in the manifest overlays, not upstream.** `ns_tools` registers only
  `math_with_judge` on main, so rows carrying `verifier_type: mcqa` fail with a bare 500; and
  `abstention` caps its judge at `max_output_tokens: 64`, truncating a reasoning judge mid-thought.
  The sandbox and judge rates here depend on those overlays being present.
