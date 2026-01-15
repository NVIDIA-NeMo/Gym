(training-verl)=
# VeRL Training

:::{admonition} 🚧 Integration Not Yet Implemented
:class: warning

**No working VeRL integration exists.** This page documents the conceptual integration pattern for contributors.

**Status**: Pattern defined, implementation needed. See [GitHub Issue #549](https://github.com/NVIDIA-NeMo/Gym/issues/549).

**Train now?** Use {doc}`NeMo RL <../tutorials/nemo-rl-grpo/index>` (production-ready) or {doc}`Unsloth <../tutorials/unsloth-training>` (single GPU).
:::

Train models with NeMo Gym using [VeRL](https://github.com/volcengine/verl), a distributed reinforcement learning framework for large language model training.

---

## Quick Reference

| Component | Status | Notes |
|-----------|--------|-------|
| Resource server rewards | ✅ Available | NeMo Gym verifiers work with any HTTP client |
| Rollout collection | ✅ Available | Use `ng_collect_rollouts` CLI |
| VeRL reward wrapper | ❌ Not implemented | Pattern defined; see below |
| End-to-end example | ❌ Not implemented | Blocked on wrapper |

:::{tip}
**Glossary**
- **Resource server**: HTTP service that provides tools and verifies model outputs for RL training
- **Rollout**: One complete interaction sequence (prompt → model response → tool calls → verification)
- **Verifier**: Logic that evaluates model output and returns a reward score (0.0–1.0)

See {doc}`Core Concepts <../about/concepts/index>` for details.
:::

---

## Why VeRL + NeMo Gym?

[VeRL](https://github.com/volcengine/verl) provides:

- **Hybrid parallelism**: Data, tensor, and pipeline parallelism
- **Memory efficiency**: Activation checkpointing for large models
- **Ray-native**: Built on Ray for distributed orchestration

NeMo Gym adds:

- Domain-specific verifiers (math, code, tool calling)
- HTTP-based reward API compatible with any framework
- Pre-built training environments

---

## Integration Pattern (Conceptual)

NeMo Gym exposes HTTP endpoints. VeRL integration requires a wrapper:

```text
┌─────────────┐     HTTP      ┌──────────────────────┐
│  VeRL       │ ──────────▶   │  NeMo Gym Resource   │
│  Trainer    │               │  Server (Verifier)   │
│             │ ◀──────────   │                      │
│             │   reward      │                      │
└─────────────┘               └──────────────────────┘
```

### Conceptual Reward Wrapper

:::{warning}
This code is **conceptual only**—it demonstrates the pattern but requires adaptation to VeRL's actual reward function interface. See [VeRL documentation](https://verl.readthedocs.io/) for the expected interface.
:::

```python
"""
Conceptual pattern for VeRL reward wrapper.
NOT RUNNABLE - adapt to VeRL's actual interface.

Reference: nemo_rl/environments/nemo_gym.py in NeMo RL repo
"""
import asyncio
import aiohttp
from typing import List, Dict, Any

class NeMoGymRewardWrapper:
    """Wrap NeMo Gym /verify endpoint for VeRL."""
    
    def __init__(self, server_url: str, timeout_seconds: float = 30.0):
        self.server_url = server_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Reuse session for connection pooling."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def compute_reward(
        self,
        response: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Call NeMo Gym /verify endpoint.
        
        Args:
            response: Model output from VeRL rollout
            ground_truth: Expected answer from dataset
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        session = await self._get_session()
        payload = {
            "response": {"output": response},
            "ground_truth": ground_truth,
        }
        try:
            async with session.post(
                f"{self.server_url}/verify",
                json=payload
            ) as resp:
                result = await resp.json()
                return result.get("reward", 0.0)
        except asyncio.TimeoutError:
            return 0.0  # Or raise, depending on your error handling
    
    async def close(self):
        """Clean up session."""
        if self._session:
            await self._session.close()
```

### Resource Server Endpoints

Every NeMo Gym resource server implements:

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/seed_session` | Initialize session state | `{"session_id": "..."}` |
| `/verify` | Compute reward | `{"reward": 0.0-1.0, ...}` |

**Source**: `nemo_gym/base_resources_server.py:63-64`

---

## Available Now: Collect Training Data

You can collect rollouts today for offline training or VeRL integration development:

### Prerequisites

```bash
pip install nemo-gym
```

### Collect Rollouts

```bash
# Terminal 1: Start resource server
ng_run "+config_paths=[resources_servers/math/configs/math.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Terminal 2: Collect rollouts
ng_collect_rollouts +agent_name=math_simple_agent \
    +input_jsonl_fpath=resources_servers/math/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +limit=100
```

**Output format** (JSONL):
```json
{
  "reward": 1.0,
  "output": [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

---

## Production Alternative: NeMo RL

NeMo RL provides **working** NeMo Gym integration today:

```bash
# Install NeMo RL (separate from NeMo Gym)
pip install nemo-rl

# Run GRPO training with NeMo Gym environment
python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
```

NeMo RL includes:

- ✅ Full rollout orchestration
- ✅ On-policy token ID corrections
- ✅ GRPO (Group Relative Policy Optimization) and DAPO (Diversity-Aware Policy Optimization)
- ✅ Multi-node distributed training

**Reference**: `nemo_rl/environments/nemo_gym.py`

See {doc}`NeMo RL GRPO Tutorial <../tutorials/nemo-rl-grpo/index>` for instructions.

---

## Contributing: Implement VeRL Integration

To add VeRL integration, follow the {doc}`Training Framework Integration Guide <../contribute/rl-framework-integration/index>`:

1. **HTTP Server**: Expose VeRL's generation as OpenAI-compatible endpoint
2. **On-Policy Corrections**: Implement token ID fixes for multi-step scenarios  
3. **Reward Wrapper**: Adapt the pattern above to VeRL's interface
4. **Training Loop**: Connect NeMo Gym rollouts to VeRL optimization

**See also**: [VeRL reward function docs](https://verl.readthedocs.io/)

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Production-ready training with NeMo Gym. Works today.
+++
{bdg-primary}`recommended`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Browse Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Find resource servers on GitHub.
+++
{bdg-secondary}`github`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc

Fast fine-tuning on a single GPU.
+++
{bdg-secondary}`single-gpu`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Environment
:link: ../tutorials/creating-resource-server
:link-type: doc

Create your own resource server.
+++
{bdg-secondary}`tutorial`
:::

::::
