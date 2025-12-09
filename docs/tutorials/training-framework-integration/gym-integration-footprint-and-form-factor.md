(gym-integration-footprint-and-form-factor)=

# Gym integration footprint and form factor
1. [Prerequisite] vLLM OpenAI-compatible HTTP server [NeMo RL logic](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L264)
   1. [NeMo RL tests](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/models/generation/test_vllm_generation.py#L1107)
2. [Prerequisite] vLLM OpenAI-compatible HTTP server on-policy fixes [NeMo RL logic](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40)
   1. [NeMo RL tests](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/models/generation/test_vllm_generation.py#L1250)
3. Gym spinup and integration [NeMo RL logic](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/environments/nemo_gym.py)
   1. [NeMo RL tests](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/environments/test_nemo_gym.py)
4. Gym rollout orchestration [NeMo RL logic](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/experience/rollouts.py#L975)
   1. [NeMo RL tests](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/experience/test_rollouts.py#L754)
5. Gym rollout orchestration integration into GRPO train loop [NeMo RL logic](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/algorithms/grpo.py#L1157)
   1. As of Dec 08, 2025, end-to-end tests still to be implemented in the NeMo RL repo!
