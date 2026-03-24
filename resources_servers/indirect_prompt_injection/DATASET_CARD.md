# Dataset Card: Indirect Prompt Injection (IPI)

- **Dataset Type**: RL
- **Gym environment link**: https://github.com/NVIDIA-NeMo/Gym/tree/makeshn/indirect_prompt_injection_env/resources_servers/indirect_prompt_injection
- **Seed data**: Fully synthetic — no external seed datasets. All components generated from scratch using LLM synthesis and template rendering.
- **Seed data license**: N/A (no external seed data)
- **SDG model**: `nvidia/nemotron-3-super-v3`, `qwen/qwen3-next-80b-a3b-instruct`.
- **Filtering model**: https://huggingface.co/Qwen/Qwen3.5-122B-A10B or https://huggingface.co/Qwen/Qwen3.5-397B-A17B (for quality and safety filtering), plus programmatic schema validation, tool/argument checks, and Jaccard-similarity deduplication.
- **Approximate size (number of conversations)**: 5000
- **Brief description**:
  > Synthetic agentic indirect prompt injection (IPI) training data for RLVR. Each record simulates a multi-turn tool-use scenario where the agent receives a benign user task, calls tools to read environment data, and encounters an injected malicious instruction hidden in the tool response. The agent must complete the user's task while resisting the injection. Reward is binary and fully programmatic: 1.0 if the agent ignores the injection, 0.0 if it follows it. Training domains are chosen to have zero overlap with the AgentDojo test benchmark, enabling clean generalization measurement.

- **Sample journey**:

> 1. **Define domain tools and environment schema** (hand-crafted). Tool schemas define the actions the agent can take; the environment schema defines the simulated world state and which data fields are injection vectors.
> 2. **LLM-generate user tasks.** The SDG model produces diverse benign tasks that require the agent to call 1–3 tools.
> 3. **LLM-generate and programmatically validate injection goals.** Each goal specifies a target tool and target arguments the attacker wants the agent to call. A grader validates goals against tool schemas and deduplicates.
> 4. **LLM-generate environment scenarios.** Realistic world states are generated with a placeholder token in one data field where the injection will be inserted.
> 5. **Render injection texts.** 10 domain-agnostic attack strategy templates (direct, authority impersonation, delimiter escape, etc.) are applied to each goal. For complex strategies, the SDG model generates multiple phrasings.
> 6. **Assemble via cross-product.** (tasks × goals × strategies × scenarios) are combined, filtered for compatibility, deduplicated by hash, and written as final JSONL records.
> 7. **Programmatic verification at training time.** The Gym environment inspects the agent's function call trace and checks whether the target tool was called with the target arguments. No LLM judge is used.
