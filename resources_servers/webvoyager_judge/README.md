# WebVoyager Judge

This resource server preserves WebVoyager's automated evaluation boundary:
the task instruction, final agent answer, and the final *k* screenshots are
sent to a vision-capable judge model. A parsed `SUCCESS`/`NOT SUCCESS` verdict
becomes binary reward.

No final answer is a valid policy failure with reward zero. Missing required
screenshots, judge transport errors, or a response without a definitive verdict
are verifier failures and must be masked by the caller.

The prompt semantics are adapted from WebVoyager's Apache-2.0
`evaluation/auto_eval.py`; `NOT SUCCESS` is checked before `SUCCESS`, matching
the upstream scorer.

The five records under `data/` exercise Gym's resource-server data contract.
Their stored rollouts use the deterministic empty-answer path and are fixtures,
not WebVoyager benchmark scores.
