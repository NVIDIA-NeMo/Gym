# Variance

Use this to decide whether a gap is larger than the run-to-run spread, and how many repeats a decision needs.

## Where Gym Reports Spread

- With two or more repeats, `*_aggregate_metrics.json` already holds `repeat_level_metrics` (per repeat, across that repeat's tasks: mean, SD, SEM and a 95% Student-t interval) and, in each agent's metrics, `std_across_repeats/mean/reward`, `se_across_repeats/mean/reward` and `ci_low_95_across_repeats/mean/reward` / `ci_high_95_across_repeats/mean/reward`. `gym eval profile` writes the same to `*_repeat_level_metrics.json` and `*_agent_metrics.json`.
- Servers that compute pass@k also write `pass@1[avg-of-k]/<score>/std_dev_across_runs` and `pass@1[avg-of-k]/<score>/std_err_across_runs`, on a 0–100 scale. Not every server does; when they are missing, use the `*_across_repeats` keys.
- `gym eval compare --baseline <rollouts A> --candidates <rollouts B>` reports a Welch 95% confidence interval on the difference of per-repeat means (two or more repeats on each side) and, when every reward is 0 or 1, per-task pass/fail flips. It reads each run's aggregate metrics file, so both runs must use the same failure-accounting flags. It joins tasks on `_ng_task_index` alone, which assumes the same dataset, split, limit and order.

## Definitions

- `std_across_repeats/mean/reward`: the sample standard deviation of the per-repeat mean rewards, on 0–1. This is `sd1` below: how much a single repeat's score varies. A repeat's own `std/reward` is the spread across tasks and is much larger.
- `std_dev_across_runs`: the sample standard deviation of the k per-repeat scores.
- `std_err_across_runs`: `std_dev_across_runs / sqrt(k)`, the standard error of the reported mean. It is not a binomial standard error over all samples.
- `avg_sample_std_dev`: the mean, over tasks, of each task's standard deviation across repeats. It is on the 0–1 score scale, unlike the two keys above.
- Standard error of the gap between two independent numbers: `sqrt(se_a^2 + se_b^2)`, with both on the same scale. `se_across_repeats/mean/reward` is on 0–1; the `*_across_runs` keys are on 0–100.

## Rules of Thumb

- On a fixed task set with independent tasks, one run's standard error is at most `sqrt(p * (1 - p) / n)` for n tasks at pass rate p: about 5 points for 100 tasks at p = 0.5, 2.2 for 500 and 1.1 for 2,000. It is lower when most tasks always pass or always fail. The same formula is the extra error a reference picks up from scoring a different sample of tasks. Shared failures, such as an outage that hits many tasks in one run, push the spread above it, so measure it with repeats.
- To reach a standard error s, you need about `k = (sd1 / s)^2` repeats, where `sd1` is the standard deviation of a single repeat's score (`std_across_repeats/mean/reward`). A benchmark whose single repeat varies by 4 points needs about 16 repeats for a 1-point standard error.
- A gap smaller than about `2 * sqrt(se_a^2 + se_b^2)`, two standard errors of the difference, is not evidence of a difference. Report the interval, not only the means.
- Agentic benchmarks with tens of tasks move several points between single runs. Average three or more runs before comparing; the Terminal-Bench 4.0 README describes Artificial Analysis averaging three.

## Spread That More Repeats Won't Remove

- Different clusters or load: wall-clock timeouts bite harder on slower serving, and different parallelism or GPU types change numerics.
- Resumes and retries: `--resume` re-dispatches retryable failures and lost rollouts up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3), and a later success counts. A reference that allows one attempt per rollout is not counting the same thing. Restarted rollouts can also behave differently from uninterrupted ones.
- Infrastructure nondeterminism: environment setup, patch application and sandbox restarts. For deterministic verifiers, identical outputs must get identical rewards; check that across runs.
- Judges: an unpinned judge model, or judge sampling at a nonzero temperature, adds its own noise. Re-judge a sample with `gym eval reverify` and measure agreement.
