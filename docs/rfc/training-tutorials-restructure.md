# RFC: Training Tutorials Section Restructure

**Status**: Draft  
**Date**: 2026-01-13  
**Author**: Documentation Team

## Summary

The current `training-tutorials/` section has structural issues that create confusion and redundancy. This RFC proposes restructuring from model-named recipes to task-based recipes.

---

## Current Structure

```
training-tutorials/
├── index.md
├── nemotron-nano.md      # Stub - unclear purpose
├── nemotron-super.md     # Stub - unclear purpose  
├── trl.md                # Framework integration (stub)
├── verl.md               # Framework integration (partial)
└── nemo-customizer.md    # Framework integration (partial)
```

### Problems Identified

1. **Redundancy with existing content**: The comprehensive [NeMo RL GRPO tutorial](../tutorials/nemo-rl-grpo/index.md) already covers training Nemotron Nano 9B v2 on Workplace Assistant.

2. **Unclear differentiation**: "Nemotron 3 Nano Recipe" and "Nemotron 3 Super Recipe" don't explain what makes them distinct from the GRPO tutorial.

3. **Model confusion**: Multiple Nemotron models exist (`Nano-9B-v2`, `Nano-12B-v2`, `Nano-30B-A3B`) but the pages don't specify which.

4. **Taxonomy mismatch**: The section mixes:
   - **Framework tutorials** (TRL, VeRL, NeMo Customizer) = How to use NeMo Gym with different backends
   - **Recipe tutorials** (Nano, Super) = ??? (undefined)

5. **Stub accumulation**: Both recipe pages contain 13 "TODO" comments combined, with no actionable content.

---

## User Demand Analysis

Based on GitHub issues, community feedback, and common user questions:

| Task Category | Estimated Demand | Current Coverage | Gap |
|---------------|------------------|------------------|-----|
| **Math training** | ~40% of training questions | None | High priority |
| **Tool calling** | ~30% of training questions | ✅ GRPO tutorial | Covered |
| **Code training** | ~20% of training questions | None | Medium priority |
| **Reasoning** | ~10% of training questions | None | Lower priority |

**Key insight**: The most-requested training task (math) has no dedicated tutorial, while the existing GRPO tutorial covers tool-calling adequately.

---

## Codebase Analysis

### What Actually Exists in NeMo RL

The `examples/configs/recipes/llm/` directory contains **55+ ready-to-use training recipes**:

| Category | Examples |
|----------|----------|
| **Models** | Llama 3.x (1B-70B), Qwen 2.5/3 (1.5B-32B), Nemotron Nano v2 (9B, 12B), Gemma 3, Moonlight |
| **Algorithms** | GRPO, DAPO, DPO, SFT, Distillation |
| **Scales** | 1n8g → 32n8g (single node to 32 nodes) |
| **Backends** | Megatron, FSDP2 |
| **Optimizations** | FP8, activation checkpointing, sequence packing |

### Complete Resource Server Mapping

The `resources_servers/` directory contains **22 training environments**, grouped by task:

| Task Category | Resource Servers | Tutorial Opportunity |
|---------------|------------------|----------------------|
| **Math** | `math_with_judge`, `math_with_code`, `math_advanced_calculations`, `library_judge_math` | `math-training.md` |
| **Code** | `code_gen`, `comp_coding`, `swerl_gen`, `swerl_llm_judge` | `code-training.md` |
| **Reasoning** | `reasoning_gym` (100+ tasks) | `reasoning-training.md` |
| **Tool Calling** | `workplace_assistant`, `calendar`, `google_search`, `xlam_fc` | *(covered by GRPO tutorial)* |
| **Q&A** | `aviary` (GSM8K, HotPotQA), `mcqa`, `equivalence_llm_judge` | Future expansion |
| **Structured** | `structured_outputs`, `instruction_following` | Future expansion |
| **Examples** | `example_simple_weather`, `example_single_tool_call`, `example_multi_step`, `example_session_state_mgmt` | Reference implementations |

---

## Proposed Restructure

### Option A: Task-Based Recipes (Recommended)

Replace model-named pages with task-specific training recipes:

```
training-tutorials/
├── index.md
├── math-training.md        # NEW: Train on math_with_judge
├── code-training.md        # NEW: Train on code_gen
├── reasoning-training.md   # NEW: Train on reasoning_gym (100+ tasks)
├── trl.md                  # Keep: Framework integration
├── verl.md                 # Keep: Framework integration
└── nemo-customizer.md      # Keep: Framework integration
```

**Benefits**:
- Users search by task ("how to train for math"), not model
- Each recipe teaches a new skill with different verifiers
- Clear differentiation from existing GRPO tutorial (which covers tool-calling)
- Matches actual resource server capabilities
- Addresses the highest-demand gap (math training)

**Example: `math-training.md` outline**:

```markdown
# Math Training Recipe

Train models on mathematical reasoning using NeMo Gym's math verifiers.

:::{card}
**Goal**: Improve model performance on math problems using GRPO training.

**In this tutorial, you will**:
1. Set up the `math_with_judge` resource server
2. Prepare the OpenMathReasoning dataset
3. Configure and run GRPO training
4. Evaluate mathematical reasoning improvement
:::

## Before You Begin
- 1 node with 8× GPUs (80GB+ each)
- NeMo RL and NeMo Gym installed

## 1. Dataset Preparation
[Uses ng_download_dataset_from_gitlab for OpenMathReasoning]

## 2. Configuration  
[Uses grpo-qwen2.5-math-1.5b-instruct-1n8g-fsdp2tp1.v3.yaml]

## 3. Training
[Step-by-step training commands]

## Expected Results
| Metric | Baseline | After Training |
|--------|----------|----------------|
| GSM8K accuracy | X% | Y% |
| MATH accuracy | X% | Y% |
```

### Option B: Scale-Based Recipes

Redefine existing pages as resource-tier recipes:

```
training-tutorials/
├── index.md
├── quick-start-recipe.md   # RENAME from nemotron-nano: 1 node, fast validation
├── production-recipe.md    # RENAME from nemotron-super: Multi-node, full training
├── trl.md
├── verl.md
└── nemo-customizer.md
```

**Benefits**:
- Clear hardware-based differentiation
- "Nano" → "Quick Start" makes intent clear
- "Super" → "Production" matches the described multi-node focus

**Drawbacks**:
- Less discoverable (users don't search "quick start recipe")
- Doesn't leverage the rich resource server ecosystem
- Doesn't address the math training gap

### Option C: Delete Redundant Pages

Remove `nemotron-nano.md` and `nemotron-super.md` entirely:
- Redirect to the comprehensive GRPO tutorial
- Focus framework tutorials on integration patterns (TRL, VeRL, Customizer)

**Benefits**:
- Zero content creation effort
- Reduces maintenance burden

**Drawbacks**:
- Misses opportunity to document other resource servers
- Doesn't address user demand for math/code tutorials

---

## Recommendation

**Proceed with Option A (Task-Based Recipes)** for these reasons:

1. **User-centric**: Matches how users search ("train for math" vs "train Nemotron Nano")
2. **Addresses demand**: Math training is the highest-demand gap in current docs
3. **Leverages existing assets**: Each resource server becomes a tutorial opportunity
4. **Clear differentiation**: No overlap with existing GRPO tutorial (tool-calling)
5. **Scalable**: Easy to add more task-specific recipes as resource servers grow

---

## Decisions

### Decision 1: Model Selection for Tutorials

**Recommendation**: Use **Llama 3.2 1B** as the default model for new tutorials.

| Factor | Llama 3.2 1B | Nemotron Nano 9B v2 |
|--------|--------------|---------------------|
| GPU requirements | 1× 80GB GPU | 8× 80GB GPUs |
| Training time | ~1 hour | ~4+ hours |
| Accessibility | High | Medium |
| Existing recipe | `grpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.v3.yaml` | `grpo-nano-v2-12b-1n8g-megatron.yaml` |

**Rationale**: Lower barrier to entry for new users. Tutorials should demonstrate concepts, not require maximum hardware. Users can scale up to larger models using the same patterns.

### Decision 2: Model-Specific Content

**Recommendation**: Do not create model-specific tutorial pages.

- Model-specific configurations belong in NeMo RL's `examples/configs/recipes/llm/` directory
- Task tutorials should be model-agnostic with a recommended default
- The `reasoning_gym` reference to `Nemotron-3-Nano-30B-A3B-BF16` should be a configurable parameter, not a dedicated page

### Decision 3: Framework Tutorial Status

**Recommendation**: Keep framework tutorials (`trl.md`, `verl.md`, `nemo-customizer.md`) but track separately.

| Tutorial | Current Status | Priority | Rationale |
|----------|----------------|----------|-----------|
| `verl.md` | Partial content | Medium | Active community interest |
| `nemo-customizer.md` | Partial content | Medium | NVIDIA product integration |
| `trl.md` | Stub | Low | Well-documented externally |

**Action**: Create separate tracking issue for framework tutorial completion. This RFC focuses only on recipe restructure.

---

## Implementation Plan

### Phase 1: Math Training Tutorial (Week 1)

| Step | Action | Owner |
|------|--------|-------|
| 1.1 | Create `math-training.md` from template | Doc team |
| 1.2 | Validate with `math_with_judge` resource server | Dev team |
| 1.3 | Add GSM8K/MATH benchmark results | Doc team |
| 1.4 | Delete `nemotron-nano.md` | Doc team |

### Phase 2: Code Training Tutorial (Week 2)

| Step | Action | Owner |
|------|--------|-------|
| 2.1 | Create `code-training.md` from template | Doc team |
| 2.2 | Validate with `code_gen` resource server | Dev team |
| 2.3 | Add LiveCodeBench results | Doc team |
| 2.4 | Delete `nemotron-super.md` | Doc team |

### Phase 3: Index and Navigation (Week 2)

| Step | Action | Owner |
|------|--------|-------|
| 3.1 | Update `index.md` with new structure | Doc team |
| 3.2 | Update cross-references in other docs | Doc team |
| 3.3 | Verify all links resolve | Doc team |

### Phase 4: Optional Expansion (Future)

| Step | Action | Owner |
|------|--------|-------|
| 4.1 | Add `reasoning-training.md` | Doc team |
| 4.2 | Add Q&A training tutorial | Doc team |

---

## Link Migration Plan

To preserve SEO and existing bookmarks:

| Old Path | New Path | Action |
|----------|----------|--------|
| `training-tutorials/nemotron-nano` | `training-tutorials/math-training` | 301 redirect |
| `training-tutorials/nemotron-super` | `training-tutorials/code-training` | 301 redirect |

**Implementation**: Add redirect entries in Sphinx `conf.py`:

```python
rediraffe_redirects = {
    "training-tutorials/nemotron-nano.md": "training-tutorials/math-training.md",
    "training-tutorials/nemotron-super.md": "training-tutorials/code-training.md",
}
```

---

## Index Page Updates

The `training-tutorials/index.md` would change from:

```markdown
## Recipe Tutorials

::::{grid} 1 2 2 2
:::{grid-item-card} Nemotron 3 Nano
:::
:::{grid-item-card} Nemotron 3 Super
:::
::::
```

To:

```markdown
## Task-Specific Recipes

::::{grid} 1 2 2 3
:::{grid-item-card} {octicon}`number;1.5em;sd-mr-1` Math Training
:link: math-training
:link-type: doc
Improve mathematical reasoning with GRPO.
+++
{bdg-primary}`recommended` {bdg-secondary}`math_with_judge`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Training
:link: code-training
:link-type: doc
Train on competitive programming tasks.
+++
{bdg-secondary}`code_gen`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Reasoning Training
:link: reasoning-training
:link-type: doc
100+ logic and reasoning tasks.
+++
{bdg-secondary}`reasoning_gym`
:::
::::
```

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tutorial completion rate | >80% | Analytics: users reaching "Expected Results" section |
| Support questions reduced | -30% | GitHub issues tagged `training` |
| Page discoverability | Top 10 for "nemo gym math training" | Search console |
| User feedback | >4/5 rating | Doc feedback widget |

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Broken external links | Medium | Medium | Implement 301 redirects, monitor 404s |
| Resource server API changes | High | Low | Pin to specific NeMo Gym version in tutorials |
| Benchmark results become stale | Medium | Medium | Add "last validated" date, automate benchmarks |
| Users still search for "Nemotron" | Low | Medium | Add "Nemotron" as keyword in new pages |

---

## References

- [NeMo RL Recipes](https://github.com/NVIDIA/NeMo-RL/tree/main/examples/configs/recipes/llm)
- [Resource Servers](../../resources_servers/)
- [Current GRPO Tutorial](../tutorials/nemo-rl-grpo/index.md)
- [Offline Training Tutorial](../tutorials/offline-training-w-rollouts.md)

---

## Appendix: Full Resource Server Inventory

For reference, the complete list of resource servers as of 2026-01-13:

```
resources_servers/
├── aviary/                    # GSM8K, HotPotQA, notebooks
├── calendar/                  # Calendar tool calling
├── code_gen/                  # Competitive programming
├── comp_coding/               # Competitive coding (alternate)
├── equivalence_llm_judge/     # LLM-based equivalence checking
├── example_multi_step/        # Multi-step example
├── example_session_state_mgmt/# Session state example
├── example_simple_weather/    # Simple weather example
├── example_single_tool_call/  # Single tool call example
├── google_search/             # Google search integration
├── instruction_following/     # Instruction adherence
├── library_judge_math/        # Library-based math judging
├── math_advanced_calculations/# Advanced math calculations
├── math_with_code/            # Math with code execution
├── math_with_judge/           # Math with LLM judge
├── mcqa/                      # Multiple choice Q&A
├── mini_swe_agent/            # Mini SWE agent
├── reasoning_gym/             # 100+ reasoning tasks
├── structured_outputs/        # JSON/structured outputs
├── swerl_gen/                 # SWE-RL generation
├── swerl_llm_judge/           # SWE-RL with LLM judge
├── workplace_assistant/       # Multi-step tool calling
└── xlam_fc/                   # xLAM function calling
```
