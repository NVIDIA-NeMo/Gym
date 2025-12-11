(generation-backend-and-openai-compatible-http-server)=

# Generation Backend

Gym requires an OpenAI-compatible HTTP server to handle model generations during training. This page covers the server requirements and existing implementations across popular RL frameworks.

## OpenAI-Compatible Server Requirements

Gym communicates with generation backends using the OpenAI HTTP API specification. Your generation server must implement endpoints compatible with one of these reference implementations:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Provider
  - Documentation
* - OpenAI API
  - [Responses API Reference](https://platform.openai.com/docs/api-reference/responses/create)
* - Gemini
  - [OpenAI Compatibility](https://ai.google.dev/gemini-api/docs/openai)
* - vLLM
  - [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
* - SGLang
  - [OpenAI-Compatible APIs](https://docs.sglang.io/basic_usage/openai_api.html)
* - TGI
  - [OpenAI Messages API](https://huggingface.co/docs/text-generation-inference/en/reference/api_reference#openai-messages-api)
```

## Generation in RL Training

Most RL frameworks that support policy optimization algorithms (PPO, GRPO) require online on-policy model generations. Integrating generation backends into the RL training loop introduces several challenges:

- **Refit**: Synchronizing model weights between training and generation
- **Off-policyness**: Ensuring generations reflect the current policy state
- **Latency**: Minimizing generation overhead during training iterations

## Existing Framework Implementations

The following table shows how popular RL frameworks implement generation backends.

:::{tip}
If your framework uses vLLM or SGLang, you can reference these implementations when adding OpenAI HTTP server support.
:::

```{list-table}
:header-rows: 1
:widths: 25 25 50

* - Framework
  - Generation Backend
  - Reference Implementation
* - NeMo RL
  - vLLM
  - [vllm_generation.py](https://github.com/NVIDIA-NeMo/RL/blob/a99bc262e5cde92575538c31ccacde27c60c3681/nemo_rl/models/generation/vllm/vllm_generation.py)
* - VeRL
  - HF, vLLM, SGLang
  - [hf_rollout.py](https://github.com/volcengine/verl/blob/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/hf_rollout.py), [vLLM rollout](https://github.com/volcengine/verl/tree/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/vllm_rollout), [SGLang rollout](https://github.com/volcengine/verl/tree/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/sglang_rollout)
* - TRL
  - vLLM, HF
  - [grpo_trainer.py (vLLM)](https://github.com/huggingface/trl/blob/cbd90d4297a877587a07bdcd82f8fc87338efe5b/trl/trainer/grpo_trainer.py#L557), [grpo_trainer.py (HF)](https://github.com/huggingface/trl/blob/cbd90d4297a877587a07bdcd82f8fc87338efe5b/trl/trainer/grpo_trainer.py#L661)
* - Slime
  - SGLang
  - [sglang_engine.py](https://github.com/THUDM/slime/blob/0612652a8e6ed7fd670ecc29101d4ca877490bf6/slime/backends/sglang_utils/sglang_engine.py#L87)
* - OpenPIPE ART
  - vLLM
  - [vLLM module](https://github.com/OpenPipe/ART/tree/6273a6fa5457e87e696b1c3a5820292826684370/src/art/vllm)
```

NeMo RL, VeRL, Slime, and OpenPIPE ART all expose OpenAI-compatible HTTP server endpoints.

## Integration Guidelines

### Frameworks Using vLLM or SGLang

If your training framework already uses vLLM or SGLang but does not expose an OpenAI-compatible HTTP server:

1. Reference the implementations listed above
2. Add server endpoints that follow the OpenAI API specification
3. Test your implementation using the [vLLM HTTP server tests from NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/a99bc262e5cde92575538c31ccacde27c60c3681/tests/unit/models/generation/test_vllm_generation.py#L1079-L1247)

### Frameworks Using Other Backends

If your training framework does not use vLLM or SGLang as a generation backend, you may need significant refactoring to achieve proper Gym integration. Consider:

- Migrating to vLLM or SGLang for generation
- Implementing an adapter layer that exposes OpenAI-compatible endpoints
- Evaluating the complexity of maintaining a custom generation backend

## Related Topics

After setting up your generation backend, proceed to:

- {doc}`openai-compatible-http-server-on-policy-correction` - Required fixes for multi-step and multi-turn scenarios
- {doc}`gym-integration-footprint-and-form-factor` - Full integration component breakdown
