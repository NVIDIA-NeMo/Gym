# WebVoyager Judge

This resource server preserves WebVoyager's automated evaluation boundary:
the task instruction, final agent answer, and the final *k* screenshots are
sent to a vision-capable judge model. A parsed `SUCCESS`/`FAILURE` verdict
becomes binary reward.

An empty final answer can still be judged from the screenshots. A parsed task
failure is a valid reward zero. Missing required screenshots are inconclusive
and return `mask_sample: true`; judge transport errors are verifier failures.
Text-only judging is supported when `require_screenshot` is false. A response
without a definitive verdict follows `unparseable_verdict_policy`: `zero` keeps
the historical valid-zero behavior, while `mask` marks it inconclusive.

The prompt semantics are adapted from WebVoyager's Apache-2.0
`evaluation/auto_eval.py`; `NOT SUCCESS` is checked before `SUCCESS`, matching
the upstream scorer.

The five records under `data/` exercise Gym's resource-server data contract.
Their stored rollouts have no screenshots, so all five are masked for missing
required judge evidence and none are scored. They are fixtures, not WebVoyager
benchmark scores.
