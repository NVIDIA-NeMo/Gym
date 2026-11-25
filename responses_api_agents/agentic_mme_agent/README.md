# Agentic-MME agent

See [environment documentation](../../resources_servers/agentic_mme/README.md)
for tools, configuration, data conversion, reward semantics and limitations.
This is a native Gym atomic-tool harness, not a wrapper around the reference
runner. State is local to each rollout; prior model outputs, generated images
and token metadata remain in the conversation.

Each run reports its policy-call references, conversation and measured image/retrieval
tool attempts. With model-call observability enabled, Gym joins those observations to
captured calls for trajectory health; judge calls remain separate from policy turns.
