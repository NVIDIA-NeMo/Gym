# Simple agent with context compaction

This sequential Responses agent applies a semantic history policy before each model call. Original image content stays in that history, so retained images are sent through the ordinary model request again. No MediaArena or training-segment definitions are exposed by the harness.

Use the regular `vllm_model` server with framework-owned external capture. This agent requires training token capture (`token_id_capture.enabled: true`) and a framework rollout ID; capture-disabled evaluation is not supported. A policy that rewrites history and `guards.max_total_tokens` require `token_id_capture.parent_resolution: source_history`. NeMo-RL enables that mode with `token_capture.forest_training`. Source-history mode requires external vLLM staging and the paired schema-v3 protocol; upgrade Gym and the worker together.

The result uses ordinary Gym response history. Its final response ID names the last accepted action, and output retains accepted generations before and after compaction. The verifier receives the complete image history; `/run` removes raw image parts from `response.output` and `responses_create_params.input`. NeMo-RL owns serving-prefix decisions, captured model inputs, training rows, logical advantages and processed media. The agent owns its policy, tool execution and verification. Reasoning-only/empty-output termination follows `simple_agent`.

Configure `context_history.enabled`, the recency policy, an optional turn-chunk schedule and guards. The default `max_steps` is 256; it must not exceed `context_history.max_model_calls` (also 256 by default). Increase both for longer episodes. Token-budget guards call `/ng-rollout/{attempt}/training-token-capture/context/measure`, which uses the worker's generation preprocessing without staging a call. The client rejects streaming, provider session state and concurrent use of one client. This agent accepts the first response of each call, so `context_history.max_response_retries` has no effect. It does not retry failed model calls or replay mutating tools after an ambiguous acknowledgement.

## Migration from the previous agent

Gym 0.6.0 and 0.7.0 shipped `responses_api_agents.simple_agent_with_compaction.compaction` and the `vllm_model_with_compaction` server (`policy_model_for_compaction`). This PR removes both. Use `ContextManagedResponsesClient` from `nemo_gym.agents.context_management` and point `model_server.name` at a regular `vllm_model`, such as `policy_model` in `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`. Reasoning recency requires `uses_reasoning_parser: true` on that server, so reasoning is represented separately from assistant text. The earlier unlimited step default is replaced by the explicit limits above.

## Licensing information

Code: Apache 2.0

Data: N/A

## Dependencies

- nemo_gym: Apache 2.0
