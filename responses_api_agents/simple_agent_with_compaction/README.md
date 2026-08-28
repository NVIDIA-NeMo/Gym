# Simple agent with context compaction

This Responses API agent follows the same model/tool rollout loop as
`simple_agent`, while maintaining a semantic history that can be materialized
through a configured context-compaction policy before each model call.

Context compaction is opt-in through this agent. The existing `simple_agent`
remains unchanged.

# Licensing information

Code: Apache 2.0

Data: N/A

# Dependencies

- nemo_gym: Apache 2.0
