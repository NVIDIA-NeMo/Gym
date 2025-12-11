(gym-integration-footprint-and-form-factor)=

# Integration Footprint

This page provides a reference for the components required to integrate Gym into your training framework. Each component includes links to the NeMo RL reference implementation and corresponding tests.

## Integration Components

A complete Gym integration consists of five components, implemented in sequence:

```{list-table}
:header-rows: 1
:widths: 5 25 35 35

* - 
  - Component
  - Implementation
  - Tests
* - 1
  - **OpenAI-Compatible HTTP Server**
  - [vllm_worker_async.py:264](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L264)
  - [test_vllm_generation.py:1107](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/models/generation/test_vllm_generation.py#L1107)
* - 2
  - **On-Policy Token ID Fixes**
  - [vllm_worker_async.py:40](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/models/generation/vllm/vllm_worker_async.py#L40)
  - [test_vllm_generation.py:1250](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/models/generation/test_vllm_generation.py#L1250)
* - 3
  - **Gym Spinup and Integration**
  - [nemo_gym.py](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/environments/nemo_gym.py)
  - [test_nemo_gym.py](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/environments/test_nemo_gym.py)
* - 4
  - **Rollout Orchestration**
  - [rollouts.py:975](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/experience/rollouts.py#L975)
  - [test_rollouts.py:754](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/tests/unit/experience/test_rollouts.py#L754)
* - 5
  - **GRPO Train Loop Integration**
  - [grpo.py:1157](https://github.com/NVIDIA-NeMo/RL/blob/64ab08df3edf25131959fc474b44ed5e36a1600b/nemo_rl/algorithms/grpo.py#L1157)
  - End-to-end tests in progress
```

:::{note}
As of December 8, 2025, end-to-end tests for GRPO train loop integration are still being implemented in the NeMo RL repository.
:::

## Component Details

### 1. OpenAI-Compatible HTTP Server

**Purpose**: Expose your generation backend as an OpenAI-compatible endpoint.

**Prerequisites**: vLLM or SGLang generation backend.

**Reference**: Refer to {doc}`generation-backend-and-openai-compatible-http-server` for implementation guidance.

### 2. On-Policy Token ID Fixes

**Purpose**: Prevent train-generation mismatch in multi-step and multi-turn scenarios.

**Prerequisites**: OpenAI-compatible HTTP server.

**Reference**: Refer to {doc}`openai-compatible-http-server-on-policy-correction` for technical details.

### 3. Gym Spinup and Integration

**Purpose**: Initialize and connect to Gym training environments.

**Key responsibilities**:

- Environment configuration loading
- Connection management
- State synchronization

### 4. Rollout Orchestration

**Purpose**: Coordinate rollout collection between the policy and Gym environments.

**Key responsibilities**:

- Batch rollout management
- Multi-step and multi-turn handling
- Token ID tracking for on-policy corrections

### 5. GRPO Train Loop Integration

**Purpose**: Integrate Gym rollouts into the policy optimization training loop.

**Key responsibilities**:

- Rollout scheduling within training iterations
- Loss calculation with Gym-generated experiences
- Weight synchronization between training and generation

## Implementation Checklist

Use this checklist to track your integration progress:

- [ ] OpenAI-compatible HTTP server implemented and tested
- [ ] On-policy token ID fixes implemented and tested
- [ ] Gym spinup and environment connection working
- [ ] Rollout orchestration handling multi-step/multi-turn scenarios
- [ ] GRPO (or equivalent) train loop integration complete

## Related Topics

- {doc}`gym-rl-framework-integration-success-criteria` - Validate your integration
- {doc}`generation-backend-and-openai-compatible-http-server` - Generation backend setup
