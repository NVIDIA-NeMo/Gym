# Benchmark notes for gym-eval

Facts about single benchmarks that the decision steps of [the skill](../SKILL.md) rely on. Read the one
that matches the cell; none of them changes an action in the decision table.

## Agent time limit: Terminal-Bench 4.0

A clean agent timeout is the model's result under the benchmark's own time limit, and a harness that
records its agent deadline as a graded zero has settled it. For Terminal-Bench 4.0, 8 h per task is the
official agent budget (TB 4.0 release notes, "Calibrating resources"), and Gym's default
`agent_timeout_sec` of 28800 matches it. An `agent_timeout` that kept no answer is a saved failure the
recovery policy decides, retried by default.

## GDP-PDF: incomplete answers and input limits

A policy answer that comes back incomplete because its output budget was spent is the row's result,
graded as a truncated answer and not regenerated. The HTTP 4xx input-limit negotiation (the DPI step-down
to 72, the image cap, the text-only fallback) is the benchmark's delivery protocol, counted per row
(`document_delivery.input_limit_rejections`, `output_truncation_regenerations`) and in the run facts.
`regenerate_on_incomplete_output=true` reproduces the old main behaviour, for comparison only, so its cells
differ from main by design.

## Output cut inside an agent episode: Stirrup

A benchmark agent may recover from an output cut inside the same episode, as part of its own protocol: the
Stirrup agent's `truncation_recovery` setting follows a turn that spent its whole completion budget
without a valid tool call with one more turn that has thinking off and a one-time instruction to act now,
not recorded in the trajectory. That is the agent's protocol, not a retry of a model failure by recovery;
regenerating an incomplete answer from scratch after a model-side failure stays excluded.

## Benchmarks that read `UNVERIFIED` by construction

A verifier that calls a judge without recording it and saves no recognised grading artifact (`judge_record`,
`rubric_verdict`, `symbolic_match`, `metric_record`, `test_execution`, `environment_reward`,
`repository_test_run`, `substep_tests`) is unobserved by construction, and its runs read `UNVERIFIED` with
`trial_counts.unobserved_reason` naming the server; so does a run whose saved config does not name its
resources server. At the head this note was written for, that is the case for `aegis_v4_safety`,
`agentif`, `beam`, `deepsearchqa`, `finance_agent_v2`, `finance_sec_search`, `gdp_pdf`, `imo_proofbench`,
`longmemeval`, `simpleqa`, `widesearch`, `litmus-bench`, the `abstention`, `inverse_if` and
`multichallenge` parts of `nemotron_3.5_super`, `frontierscience_olympiad` and `automationbench` (the
benchmarks graded by the verifiers agent, AutomationBench among them). `UNVERIFIED` means the answers were
scored, but the grading evidence was not recorded on this stack version. A resubmit cannot change it, and
no flag does: it needs a run on a stack version that records grading. Report it with the reason and do not
present the run as verified. The fix is that server's grading record, in Gym.
