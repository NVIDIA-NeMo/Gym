# Simple agent with context compaction

This Responses API agent follows the same model/tool rollout loop as
`simple_agent`, while maintaining a semantic history that can be materialized
through a configured context-compaction policy before each model call.

Context compaction is opt-in through this agent. The existing `simple_agent`
remains unchanged.

## Rollout trace evidence

The `/run` response includes a `rollout_trace_contract`. Canonical prompt,
generation, and log-probability arrays remain in the ordinary response output.
One `model_call_metadata` record accompanies each trainable model call, and its
digest binds the metadata to those canonical arrays. Boundary events describe
intentional history rewrites so training consumers can construct physical
traces without retokenizing the rollout. The same bounded trace representation
is returned whether resource verification runs or is explicitly skipped;
verification status affects reward provenance, not trace encoding.

# Licensing information

Code: Apache 2.0

Data: N/A

# Dependencies

- nemo_gym: Apache 2.0
