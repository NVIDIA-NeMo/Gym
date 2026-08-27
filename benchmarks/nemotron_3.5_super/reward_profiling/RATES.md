# Measured throughput, for compute planning

Rates for sizing a run of `manifests/nemotron_3_ultra.yaml`. Measured on the P/D stack described
under Conditions.

**Every number below was measured lane-by-lane through a bench harness, not through
`scripts/03_run.sh`.** The launcher's first successful end-to-end run (job
6563900, 2026-08-27) collected 259/288 rollouts across all 36 environments with zero failures, but
at `num_samples_in_parallel=1024` against a 288-row input -- every row in flight at once, so it
produced a burst rather than a rate and is not comparable. Its shape is still informative and is
recorded under Tail behaviour.

## Conditions

| | no-judge lane | judge lane | sandbox lane |
|---|---|---|---|
| job | 6553550 | 6559594 | 6559594 |
| topology | 1 prefill + 2 decode = **12 GPUs** | same | same |
| `num_samples_in_parallel` | 128 | 48 | 32 |
| result | 2,245 rollouts / 1,438 s | 244 rollouts / 411 s | 55 rollouts / 410 s |
| aggregate | **5,621 rollouts/hr** | **2,139 rollouts/hr** | **483 rollouts/hr** |
| sandbox tier | n/a | n/a | 1 node, 32 uWSGI workers |

## Sizing (use these)

Aggregate rate is the number to plan with. Per-environment rates below are *shares of one fleet*,
not independent capacities -- summing per-environment GPU-hours overcounts by roughly the number
of environments.

```
GPU-hours = total_rollouts / (aggregate_rate / GPUs_measured)
```

| lane | rows | rollouts (x8) | aggregate | GPU-hours |
|---|---|---|---|---|
| no-judge (26 envs) | 623,006 | 4,984,048 | 5,621/hr | **10,640** |
| judge (6 envs) | 51,707 | 413,656 | 1,398-2,139/hr | **2,320-3,551** |
| sandbox (4 envs) | 51,408 | 411,264 | 483/hr | **10,218** |
| **total (all 3 lanes)** | | | | **23,178-24,409** |

Superseded by the mixed-workload measurement below: **16,540 GPU-hours**. Summing lanes
measured separately double-counts, because judge and sandbox rollouts hold no GPU while blocked.

| nodes | GPUs | wall (all 3 lanes) |
|---|---|---|
| 12 | 48 | 20.1-21.2 days |
| 24 | 96 | **10.1-10.6 days** |
| 48 | 192 | 5.0-5.3 days |
| 96 | 384 | 2.5-2.6 days |

**Sandbox is 42% of the total from 4 of 36 environments.** It measured 483/hr against a single
sandbox node running 32 uWSGI workers, at driver concurrency 32 -- the lowest of any lane. This is
the softest number here and the most improvable: sandbox capacity is a separate axis from GPU
count, so adding sandbox nodes should move it without touching the fleet. Size the sandbox tier
before committing to these figures.

**Every figure is a floor.** Each decode engine schedules up to `max_num_seqs=512`, so 128
concurrency against 2 decode nodes ran at ~12.5% of capacity -- the client semaphore, not the GPUs,
was the limit. Raising concurrency should improve all of it.

## Per-environment rates (relative ranking only)

Contended: every environment in a lane shared one endpoint, so a fast environment's rate reflects
queueing behind slow ones. Useful for spotting which environments dominate, not for per-environment
GPU-hours.

| no-judge | /hr | | judge | /hr |
|---|---|---|---|---|
| tau_pivot | 755 | | multichallenge | 285 |
| code_gen | 494 | | lc_judge | 278 |
| structured_outputs_v3 | 387 | | abstention | 271 |
| structured_outputs | 385 | | inverse_if | 258 |
| lc_equivalence_rule | 288 | | equivalence_llm_judge | 231 |
| structured_outputs_v4 | 281 | | math_with_judge | 75 |
| gdpval_pivot | 278 | | | |
| toolcall_schema | 269 | | | |
| workplace_assistant | 264 | | | |
| freeform_formatting | 261 | | | |
| droid_pivot | 261 | | | |
| search_pivot | 235 | | | |
| mcqa | 232 | | | |
| citation_format | 206 | | | |
| calendar | 172 | | | |
| nvarc_transductive | 167 | | | |
| instruction_following | 152 | | | |
| swe_pivot | 126 | | | |
| nvarc_inductive | 85 | | | |
| ether0 | 80 | | | |
| reasoning_gym | 56 | | | |

Sandbox lane, same caveat:

| sandbox | /hr | mean reward | note |
|---|---|---|---|
| ns_tools (3 entries, 1 agent) | 421 | 0.792 | math_tir + stem_mcqa + stem_openqa share `ns_tools_simple_agent`, so one rate covers all three |
| math_formal_lean | 61 | 0.000 | 7 rollouts, all zero -- too few to read, see caveats |

## Caveats

- **The judge lane ran at concurrency 48 vs the no-judge lane's 128**, so the 4x aggregate gap
  overstates the true judge penalty. Re-measure at matched concurrency before trusting the ratio.
- **The judge lane measured 1,398/hr and 2,139/hr on two runs at identical concurrency (48)** --
  a 53% spread with no deliberate change between them, most likely prefix-cache warmth from the
  earlier run against the same endpoint. Neither number is solid; the range above reflects that.
  Anything that depends on the judge figure needs a repeat measurement on a cold endpoint.
- **`math_with_judge` is the slowest environment in the lane by 4x** (79/hr, vs 333-544/hr for its
  peers) and returned mean reward 0.000 on one run and 0.222 on the next, on 11 and 9 rollouts.
  Both samples are too small to read. Its rate and its reward are both unresolved.
- **The sandbox lane produced 0 rollouts and 0 failures.** No sidecar exists, so `ns_tools` finds
  nothing at `127.0.0.1:6000` and connection errors retry forever without being recorded. Its four
  environments (~411k rollouts) are entirely unsized, and in the pre-sweep runs they were the most
  expensive of the whole set.
- **`terminus_judge_string_only`** produced zero rollouts in the no-judge run; no rate.
- **`math_formal_lean` returned 0.000 on all 7 rollouts.** Not read as a finding: n=7, and lean was
  the worst env in the pre-sweep too (25.8% completion). It needs a longer run before its reward or
  its 61/hr means anything.
- **Two config fixes are carried in the manifests, not upstream.** `ns_tools` registers only
  `math_with_judge` on main and nv-internal-main, so rows carrying `verifier_type: mcqa` fail with
  a bare 500; and `abstention` caps its judge at `max_output_tokens: 64`, which truncates a
  reasoning judge mid-thought. Both are fixed on `Gym-ultra-3-rebased` and in the 260603 bundle,
  neither is on main. The sandbox and judge rates here depend on the manifest overlays supplying
  them. **The pre-fix abstention numbers (544/hr, reward 0.685) are void** -- that rate was a
  truncated judge returning early.
- Environments sharing an agent share a rate: `comp_coding`/`comp_coding_nemotronx`,
  `ds1_augmented`/`ds1_basic`, `tau_pivot`/`tau_pivot_aq_mms`. `comp_coding_nemotronx` averages
  688 KB/row against `comp_coding`'s 306, so its true rate is likely lower.


## All 36 environments together, sharded (job 6602112/6602113, 2026-08-27)

The first measurement of the whole manifest running as one mixed workload rather than lane by lane.

| | |
|---|---|
| topology | 2 shards x (1 prefill + 1 decode) = **16 GPUs** |
| `num_samples_in_parallel` | 512 per shard |
| collected | 2,249 of 2,304 (97.6%), 0 failures |
| collection window | ~24 min per shard, after ~16 min of vLLM load and server startup |
| **aggregate** | **~5,620 rollouts/hr** (2,800 + 2,822) |

Per GPU that is 351/hr, against 468/hr for the no-judge lane alone. The mixed run is slower per GPU
because it includes the sandbox and judge environments, and the figure spans the tail -- the early
rate was far higher and decayed as the slow environments came to dominate.

**Sizing the full sweep from this**, which is the number to prefer over summing the lanes:

```
5,808,968 rollouts / 5,620 per hr = 1,034 h on 16 GPUs = 16,540 GPU-hours
```

| nodes | GPUs | wall |
|---|---|---|
| 24 | 96 | 7.2 days |
| 48 | 192 | 3.6 days |
| 96 | 384 | 1.8 days |

That is well under the 23,178-24,409 GPU-hours the per-lane numbers implied, and the gap is the
point of running them together: a judge rollout blocked on the gateway and a sandbox rollout
blocked on uWSGI occupy no GPU while they wait, so they fill time the GPU-bound environments would
have left idle. Summing lanes measured separately double-counts that.

Caveats. One measurement, at one topology, and 512 concurrency against a single decode engine is
still short of `max_num_seqs`. It includes the tail, so it understates steady-state and overstates
what you get if you stop at 95%. The sandbox ran one node per shard with 32 uWSGI workers; sandbox
capacity scales independently of GPUs and was not varied here.

## Tail behaviour (job 6563900, all 36 environments in one job)

Rollouts per 30s window, from a standing start:

```
   0  ->  151  ->  206  ->  224  ->  238  ->  250  ->  256  ->  259
      +151     + 55     + 18     + 14     + 12     +  6     +  3
```

The first window ran at roughly 18,000 rollouts/hr instantaneous, the last at ~360 -- a 50x
collapse. Fast environments (`code_gen`, `calendar`, `mcqa`, `structured_outputs_v4`, all 8/8)
drain immediately; the tail is `reasoning_gym` 7/8, `ether0` 4/8, `ns_tools` 14/24,
`math_formal_lean` 2/8.

This is the straggler problem the pre-sweep runs hit: first pass 74.4% completion, rising to 89.8%
only after hand-splitting into `final_fast` / `final_slow`. Plan the sweep around the tail, not the
aggregate. Mixing lanes is still correct -- judge rollouts block on the gateway and sandbox
rollouts on uWSGI, so they occupy no GPU while waiting -- but a residue job for the slow entries is
worth expecting.
