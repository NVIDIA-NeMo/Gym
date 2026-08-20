# Web Agent

`web_agent` is one multimodal rollout loop with benchmark profiles rather than
three copied agents. It reads a normalized `web_task`, renders a11y or SoM
observations, validates model-generated actions without executing arbitrary
Python, and drives the stateful BrowserGym resource server.

- WebArena: a11y observation, BrowserGym high-level action syntax, colocated evaluator.
- VisualWebArena: screenshot + SoM + a11y text, colocated evaluator.
- WebVoyager: compact labelled-interactive-element text plus SoM screenshot,
  legacy action syntax translated to BrowserGym, and an external
  screenshot-and-answer judge.

Only the most recent configured number of screenshots is retained in model
context. An invalid action gets bounded format-repair turns and does not step
the browser. A policy failure remains a valid zero-reward sample; runtime or
judge failures set `mask_sample`.

Verification is episode-scoped rather than dataset-scoped. WebArena and
VisualWebArena run their colocated native evaluator as soon as the rollout
terminates. WebVoyager retains its final answer and last configured screenshots,
closes the browser session, and then calls the external judge before returning
that rollout to the collector. Transient judge failures are retried against the
same immutable evidence; they never replay browser actions. Gym can run many of
these independently verified episodes concurrently, while a learner consumes a
rollout only after its reward is resolved.

Bounded network, timeout, capacity, and session failures are written as
retryable infrastructure sidecars. Structured invalid-task and benchmark
precondition responses are written as terminal masked sidecars. Rollout
collection also writes `<output_stem>_population_status.json`; scores are only
complete when every materialized rollout has a main result and terminal,
exhausted, retryable, and missing counts are all zero.

Every seeded rollout returns `artifact_session_id`, including bounded failures
that happen after seeding. When browser recording is enabled, finalized video
references are returned in `recording_artifacts`; consumers do not need to
infer a failed rollout's artifact directory from evaluator evidence or step
counts.

Visual profiles configure their page text independently with
`visual_observation_text`: `full_axtree`, `som_only`, or `none`. They may also
replace both the image and page text of old visual turns with
`redact_old_visual_observations`. The shared defaults preserve full visual
AXTree history; the WebVoyager benchmark selects `som_only`, one current policy
screenshot, and whole-observation redaction to match upstream behavior. Judge
screenshot history is configured separately.
