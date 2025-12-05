(generation-backend-and-openai-compatible-http-server)=

# Generation backend and OpenAI-compatible HTTP server
Gym requires an OpenAI compatible HTTP server, similar to those provided by
1. [OpenAI API](https://platform.openai.com/docs/api-reference/responses/create)
2. [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
3. [VLLM OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
4. [SGLang OpenAI compatible APIs](https://docs.sglang.io/basic_usage/openai_api.html)
5. [TGI OpenAI Messages API](https://huggingface.co/docs/text-generation-inference/en/reference/api_reference#openai-messages-api)

Most RL frameworks today that support policy optimization algorithms like PPO, GRPO, etc which require online on-policy model generations will have existing ways of performing generation. There are quite a few nuances to supporting generation backends in the RL training loop in addition to the training policy, notably issues like refit, off-policyness, etc.

As of Dec 04, 2025:
1. [NeMo RL uses vLLM](https://github.com/NVIDIA-NeMo/RL/blob/a99bc262e5cde92575538c31ccacde27c60c3681/nemo_rl/models/generation/vllm/vllm_generation.py)
2. VeRL uses [HF rollout](https://github.com/volcengine/verl/blob/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/hf_rollout.py), [vLLM](https://github.com/volcengine/verl/tree/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/vllm_rollout), or [SGLang](https://github.com/volcengine/verl/tree/fd893c788dbdb967c6eb62845b09a02e38819ac1/verl/workers/rollout/sglang_rollout)
3. TRL uses [vLLM](https://github.com/huggingface/trl/blob/cbd90d4297a877587a07bdcd82f8fc87338efe5b/trl/trainer/grpo_trainer.py#L557) or [HF native generation](https://github.com/huggingface/trl/blob/cbd90d4297a877587a07bdcd82f8fc87338efe5b/trl/trainer/grpo_trainer.py#L661)
4. [Slime uses SGLang](https://github.com/THUDM/slime/blob/0612652a8e6ed7fd670ecc29101d4ca877490bf6/slime/backends/sglang_utils/sglang_engine.py#L87)
5. [OpenPIPE ART uses vLLM](https://github.com/OpenPipe/ART/tree/6273a6fa5457e87e696b1c3a5820292826684370/src/art/vllm)

where NeMo RL, VeRL, Slime, and OpenPIPE ART all expose corresponding OpenAI-compatible HTTP server endpoints.

---

1. If your training framework uses vLLM or SGLang as a generation backend but does not already expose an OpenAI-compatible HTTP server, please refer to one of the above examples to integrate your own server. The server implementation can be tested similar to these [vLLM HTTP server tests in NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/a99bc262e5cde92575538c31ccacde27c60c3681/tests/unit/models/generation/test_vllm_generation.py#L1079-L1247).
2. If your training framework does not use vLLM or SGLang as a generation backend, you may need to do significant refactoring in order to get it to work properly.
