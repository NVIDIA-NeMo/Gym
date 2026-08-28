# Measured rates, for compute planning

Per-environment throughput for `manifests/nemotron_3_ultra.yaml`, measured lane by lane on
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

Disk, one-time per (manifest, checkpoint): `01_materialize.sh` writes **291.5 GB** in ~2,087 s
(5,808,968 rows from 726,121 source rows, 36 workers). Sharding copies it, so peak is **~583 GB**.

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
