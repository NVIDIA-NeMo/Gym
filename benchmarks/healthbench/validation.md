# HealthBench implementation validation

Checked on 2026-10-14 against rebased Gym commit `04a2d461e` with the implementation changes uncommitted. These are development checks, not certification or published-score parity evidence.

## Offline checks

- Both `gym env validate` commands pass, streaming 5,000 HealthBench and 525 Professional rows.
- Both `gym env test` commands pass the scorer's full-reward, zero-reward and malformed-request fixtures.
- 388 focused tests pass: benchmark preparation, verifier behavior, core judge handling, manifest validation, onboarding and aggregation.
- Verifier coverage: 96.72%, above the repository's configured 95% threshold.
- Ruff, formatting, shell syntax and whitespace checks pass.
- Scoped pre-commit hooks pass. The private guardword catalog is unavailable, so that hook reports a skip.
- All prepared rows preserve the source conversations in `responses_create_params.input`.

## Live runs

Both runs used policy `nvidia/nvidia/nemotron-3.5-lightning` on `https://inference-api.nvidia.com/v1`, temperature 0.7, 8,192 output tokens, eight tasks, one repeat and two parallel rollouts. Judges ran as separate instances through the same gateway. Those deployment aliases are not pinned reference snapshots.

| Run | Outcome | Artifacts relative to Gym |
| --- | --- | --- |
| HealthBench, judge `azure/openai/gpt-4.1` | Eight scored rows; health: 8 healthy, 0 unhealthy, 0 unobserved; no failed/masked rows. All 101 rubric grades have judgement ids and usage. | `results/healthbench-compliance-smoke-v4/` |
| Professional, judge `azure/openai/gpt-5.4`, low reasoning effort | Seven healthy scored rows and one masked judge timeout. Scored rows preserve raw negative length-adjusted values. | `results/healthbench-professional-compliance-smoke/` |
| Professional failed-only reverification | The same saved policy answer encountered a judge-server error again. It remains masked with `evaluation_completed: false`; the seven prior successes are carried through. | `results/healthbench-professional-compliance-smoke/recovered.jsonl` and its failure sidecar |
| Final HealthBench verifier replay | A saved answer was successfully regraded after the completion/reverification updates; `evaluation_completed: true`, unmasked. No new policy inference. | `results/healthbench-final-verifier-replay/` |

Resolved configs, engine logs, model-call captures and aggregate artifacts are retained locally. Full API-key values were checked for in readable run artifacts and were not found. Initial smoke servers started before the completion-flag changes; the final replay and Professional recovery exercise the revised verifier. Worktree fingerprints record this distinction. Run artifacts and full dataset contents are excluded from Git.

Earlier diagnostic attempts are also retained: a stale generated-metrics conflict, duplicate environment routing, and a 1,024-token/direct-OpenAI-judge smoke. The last of those produced two reasoning-only answers and six masked judge failures. The successful smoke uses the larger cap and working gateway judge route.

## Open exit checks

`gym-p0/v1` inspection of the HealthBench bundle fails. `--no-tools` is applicable to this pair and removes tool requirements. The verifier's originally missing `evaluation_completed` field is now fixed and covered by a real replay. The remaining upstream checks `ownership.call_owner` and `steps.attempt_accounting` count captured judge calls as policy calls without policy invocation/turn ownership. Keep those records; do not suppress verifier or step requirements to force a pass. The retained original-run conformance report explains the failures. The final verification replay passes the TE-6 completion checks but is not a new agent trajectory; its other evidence failures do not constitute a fresh harness qualification.

The workload stays experimental and `verified: false`. Before declaring compliance:

1. Resolve the judge-call evidence gap with the Harness PIC/Frontier Eval and pass EC-2 on a fresh run.
2. Recover the Professional judge failure and rerun both benchmarks on the final merged source with the agreed run settings.
3. Establish reference-score parity and tolerance on a full-size current-frontier-model run, and measure runtime with the Deployment PIC.
4. Supply the reused harness's capability-matrix evidence.
5. Add the Config Management Repo bindings, registered recipes, leaves and explicit health policies; run composition and Evergreen checks, then follow the SWQA process through Frontier Eval.

No score band or certificate was created from these smoke scores. See [the onboarding checklist](onboarding.md) for the commands and division of responsibilities.


## Subset interface

Full, Consensus and Hard remain one `healthbench` catalog entry. `configs/consensus.yaml` and `configs/hard.yaml` select their protocols for both `gym eval prepare` and `gym eval run`. Preparation and Gym's evaluation-data collation were checked on the pinned datasets: 5,000 Full, 3,671 Consensus and 1,000 Hard rows. Each has a distinct prepared path and `metadata.healthbench_subset`. Tests cover config-order independence, typed dataset paths, source selection and prevention of cross-subset overwrites. These interface checks require no new model inference; grading is unchanged.

Both HealthBench and Professional default to one repeat in their configs and manifest mirrors. Explicit run settings can still increase repeats for variance measurement.
