# Simple agent with context compaction

This sequential Responses agent applies a semantic history policy before each model call. Original image content stays in that history, so retained images are sent through the ordinary model request again. No MediaArena or training-segment definitions are exposed by the harness.

Use the regular `vllm_model` server with framework-owned external capture. NeMo-RL enables `token_id_capture.framework_owned_context` when its `token_capture.context_compaction` option is active. The paired capture protocol uses schema v3; both repositories must be upgraded together.

The result contains one logical attempt ID, ordered selected response IDs and an outcome. NeMo-RL owns serving-prefix decisions, captured model inputs, training-row planning, logical advantages, and processed media. This agent keeps policy decisions, tool execution and verification. Its reasoning-only/empty-output termination follows the ordinary simple agent.

Configure `context_history.enabled`, the recency policy, optional turn-chunk schedule and guards. Token-budget guards call `/context/{attempt}/measure`; that endpoint uses the worker's generation preprocessing without staging a call. The initial path rejects streaming, concurrent calls, provider session state and resampling. A lost response acknowledgement terminates the attempt instead of automatically replaying a model or tool call.

CPU tests cover policy, original-image retention, adapter parity and HTTP custody. Real model and verifier smoke rollouts remain required for release.

## Licensing information

Code: Apache 2.0

Data: N/A

## Dependencies

- nemo_gym: Apache 2.0
