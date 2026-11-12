# HealthBench Onboarding

Shared preparation, smoke-run guidance and onboarding requirements for HealthBench and HealthBench Professional.

Preparation keeps the original message roles, conversation contents, rubric points, and metadata. The shared verifier uses the grader prompt from [OpenAI simple-evals](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py), pinned at `652c89d0ca9df547706735883097e9537d40dc47`. Its MIT attribution is retained in `resources_servers/healthbench_rubric/prompts/LICENSE`.

Judge transport failures, timeouts, and exhausted malformed-grade retries raise `JudgeError`; Gym's endpoint wrapper reports `mask_sample`, `failure_kind`, `failure_reason`, and `judge_failed` for recovery with `gym eval reverify --judge-failed-only`. A valid negative criterion is graded as written. Never replace a missing judgement with a false criterion.

Each successful rubric grade retains its judgement id, explanation, and provider token usage (unknown when unavailable). Raw scores remain on the row. The bounded Gym reward supports the manifest contract; official-style headline metrics average raw scores and then clamp the mean to [0, 1]. The benchmark does not report bootstrap confidence intervals or all reference tag-level metrics yet.

## Before review

Follow the [Benchmark and Harness Onboarding Guide](https://rfc.frontier-evals.nvidia.com/m/frontier-eval-rfcs/r/benchmark-onboarding-journey). The implemented contracts and offline fixtures do not establish full compliance. Record these remaining checks in the PR:

| Check | Evidence |
| --- | --- |
| Static and verifier checks | `gym env validate healthbench`, `gym env validate healthbench_professional`, and `gym env test` for both names; shared unit tests and pre-commit. |
| EC-1 | Real smoke and full runs; inspect healthy, unhealthy, and unobserved counts in `quality_summary.json`. The RFC proposes at most 0.1% unhealthy. Masked judge failures must also be inspected; they reduce scored coverage. |
| EC-2 | `python scripts/inspect_harness_conformance.py --bundle <run_dir>/rollouts.jsonl --output <new_dir> --no-tools`. Reuse the existing simple-agent harness evidence; do not claim success without checking the real bundle. |
| EC-3 | Full-size run on a current frontier model against a trusted reference score, with matching data, prompts, judge, sampling, and repeats. `gym eval compare` compares Gym aggregate artifacts; attach the external reference separately. Specify a tolerance based on repeat-run variance. Lightning smoke runs alone cannot establish this. |
| EC-4 | `resolved-config.yaml` with each run. The local smoke script saves a redacted configuration; `gym eval submit` saves it automatically. Record the tested Git commit and uncommitted diff when applicable. |
| EC-5 | Applicable only when onboarding multiple harnesses: reference-harness parity and a finished second-harness run. These benchmarks currently declare only simple-agent. |
| EC-6 | Harness PIC supplies the committed capability-matrix evidence for the reused simple-agent harness. |
| EC-7 | Config Management Repo binding and leaf resolve; Evergreen `certify-runs --dry-run` agrees with the full run's metric key. This is a proposed exit check in the RFC. |

Also inspect judge usage/id coverage, timed-out tasks, and masked outcomes. No benchmark tools or sandbox are used. Agree the measured runtime with the Deployment PIC against the four-hour target.

Start the Config Management Repo draft MR while building. Use a gym-native benchmark binding that imports `run_common` and `resumable`, declares config paths for preparation and execution, and records all outside inputs. The leaf's policy must explicitly set `artifact_health.max_unhealthy_fraction`; register its recipe in the binding manifest. Escape judge-key environment interpolation there as `\${oc.env:VAR}`. Before the MR leaves draft, merge Gym and pin a full SHA on main or a tag. Frontier Eval handles SWQA coordination; score bands come from the recorded SWQA run, and Evergreen writes certificates.

## Smoke settings

`configs/nemotron-lightning-smoke.yaml` targets `https://inference-api.nvidia.com/v1` with policy model `nvidia/nvidia/nemotron-3.5-lightning`. Its 8192-token cap, eight tasks and single repeat are for smoke testing only. The smoke routes independent judges through NVIDIA as `azure/openai/gpt-4.1` and `azure/openai/gpt-5.4`; these deployment aliases are not pinned snapshots. The benchmark configs keep independent reference judges: GPT-4.1 for HealthBench, and the GPT-5.4 low-effort option for Professional. Professional is an unofficial external implementation, as described by its [dataset card](https://huggingface.co/datasets/openai/healthbench-professional).

```bash
bash benchmarks/healthbench/scripts/run-smoke.sh healthbench
bash benchmarks/healthbench/scripts/run-smoke.sh healthbench_professional
```

For locally started servers and `--no-serve`, enable observability on both commands and share one absolute capture directory per evaluation. Choose sidecar routing deliberately; the smoke leaves Gym's default of stopping on a failed `/run` call in place. Inspect the aggregate file and engine log as well as health counts.


The verifier sets `evaluation_completed` explicitly and clears obsolete failure flags on successful reverification. The current upstream `gym-p0/v1` checker counts every captured model call except compaction as a policy call; judged runs therefore fail `ownership.call_owner` and `steps.attempt_accounting` for the additional judge calls. Retain that evidence and route the gap to the Harness PIC/Frontier Eval. `--no-tools` is applicable because these tasks dispatch no tools; `--no-steps` and `--no-verifier` are not applicable workarounds.
