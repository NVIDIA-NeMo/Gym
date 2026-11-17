# Multimodal coding with Opencode

This experimental pairing uses Gym's existing `opencode_agent`, a resource-owned
remote task sandbox, and the same independent execution-gated quality reward
as the single-answer environment. The five examples are synthetic interface
rows and contain no benchmark test inputs or training data.

See the [resources server README](../../resources_servers/mm_coding_training/README.md)
for runtime settings, private data requirements and remaining smoke gates.

Each catalog manifest describes one harness. The NeMo RL multi and mixed
recipes compose both pairings through `environment_server_routes`.
Prepared training rows freeze the harness route before GRPO sibling expansion.
