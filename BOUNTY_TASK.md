# Bounty Objective: feat(eval): Add opt-in Bayes@N summaries to reward profiling
Target: NVIDIA-NeMo/Gym Issue #4810
URL: https://github.com/NVIDIA-NeMo/Gym/issues/4810

## Issue Description
**Use cases, pain points, and background**

Repeated rollouts retain per-task outcomes, but the existing across-run standard deviations answer a different question from posterior uncertainty in the fixed benchmark's mean success probability. [Bayes@N](https://arxiv.org/abs/2510.04265) supplies closed-form posterior mean and standard deviation for that quantity. The [Scorio reference](https://github.com/mohsenhariri/scorio/blob/7278f677c30396d99a0454bcbb08f5a9a53c4efb/scorio/eval/bayes.py) supports categorical outcomes; this proposal covers its binary, uniform-prior case.

**Description**:

Add an optional `gym eval profile --bayes-at-n` flag. It would read the existing materialized-input manifest and rollout file, then add `bayes@N/reward` and `std/bayes@N/reward`, replacing N with the observed trial count (for example, `bayes@2/reward`) to each agent's saved metrics. Both values use reward units on [0, 1]. The second is posterior standard deviation, separate from the existing empirical standard deviations and standard errors. Existing profile output would remain unchanged without the flag.

**Design**:

The implementation would use `RewardProfiler` in `nemo_gym/reward_profile.py`, configuration and dispatch in `nemo_gym/cli/{main,eval}.py`, focused unit tests, and the existing evaluation documentation. It would have no new runtime dependency. The materialized manifest is needed to detect an entirely missing task; the grouped results received by `compute_pass_majority_metrics` alone cannot establish that completeness.

For each agent/environment run, a question is one `_ng_task_index`, and its N trials are separately collected rollouts. Require every scheduled rollout in the supplied manifest, unique task/rollout keys, identical positive N for every question in that run, and finite rewards exactly 0 or 1. Reject masked, partial or ragged data. Verify that repeated materialized question inputs agree, allowing the native per-rollout seed field to vary. Keep separate run identities separate, including fan-out. Users must keep the model, agent, verifier and environment fixed and collect independent fresh generations; rerunning a verifier over the same generation does not create a trial.

With s successes in a question, use Beta(s+1, N-s+1). Average the question posterior means, and compute the corpus standard deviation as the square root of summed question variances divided by the question count. Do not pool the outcomes into one Beta posterior or average question standard deviations.

**Out of scope**:

Continuous-reward thresholding, categorical weights, historical observations, credible intervals, adaptive allocation, training-checkpoint comparisons and changes to generation or verification. GraphWalks uses continuous F1 rewards, so binary-looking outcomes from a continuous grader do not establish eligibility; this option is intended for verifiers with a binary correctness contract.

**Acceptance Criteria**:

- [x] Opt-in CLI profiling writes both posterior moments; existing default output is unchanged.
- [x] Missing, duplicate, masked, nonbinary and unequal-count outcomes raise clear errors before a posterior is emitted.
- [x] Independent exact examples distinguish per-question posteriors from pooled counts; tests cover fan-out, input identity and JSON export.
- [x] Documentation states the repeated-generation assumptions, output units and supported binary subset, with an executable profiling example.
- [x] Focused existing tests and the applicable local style/documentation checks pass.

I am a coauthor of Bayes@N and maintain Scorio. This source investigation and proposal were prepared with Codex.


## Instructions for Kuro (Autonomous Solver):
1. Use skills: `systematic-debugging` and `code-review-and-quality`.
2. Inspect the repository structure, locate relevant files, and understand the bug/feature.
3. Apply the minimal, robust fix or implementation.
4. Run any existing test suites or linters in the workspace.
5. Ensure the fix adheres to the project's existing code style.
6. When done, output a summary of files modified and verification results.
