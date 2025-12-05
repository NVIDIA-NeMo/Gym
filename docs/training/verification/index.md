(training-verification-overview)=

# Verification

Design and implement reward signals that measure task correctness, quality, and efficiency.

---

## How It Works

Verification is how NeMo Gym scores agent performance. Every resource server implements a `verify()` function that evaluates rollouts and returns reward signals for training.

```{mermaid}
flowchart LR
    subgraph Rollout["Completed Rollout"]
        I["Input"]
        O["Output"]
        T["Tool Calls"]
    end
    
    subgraph Verify["verify()"]
        C["Correctness"]
        Q["Quality"]
        E["Efficiency"]
    end
    
    subgraph Reward["Reward Signal"]
        S["Score<br/>(0.0 - 1.0)"]
    end
    
    Rollout --> Verify
    Verify --> Reward
    
    classDef rollout fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef verify fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef reward fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000
    
    class I,O,T rollout
    class C,Q,E verify
    class S reward
```

Verification can be:

- **Deterministic** — Unit tests, exact match, regex patterns
- **Heuristic** — Rule-based scoring, partial credit
- **Model-based** — LLM judges for subjective evaluation

---

## How-To Guides

Implement and test verification logic for your training environments.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`repo-forked;1.5em;sd-mr-1` Multi-Verifier Training
:link: multi-verifier
:link-type: doc

Use multiple verification environments in a single training run.
+++
{bdg-secondary}`advanced`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Validate Verification
:class-card: sd-card-placeholder

Unit test `verify()` functions and analyze reward distributions.
+++
{bdg-light}`planned`
:::

::::

---

## Patterns & Reference

Explore common verification patterns and scoring strategies.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Custom Patterns Cookbook
:class-card: sd-card-placeholder

Patterns for tool usage, answer correctness, code execution, and LLM judges.
+++
{bdg-light}`planned`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Multi-Objective Scoring
:class-card: sd-card-placeholder

Combine correctness, efficiency, and style into weighted composite rewards.
+++
{bdg-light}`planned`
:::

::::

---

```{toctree}
:maxdepth: 1
:hidden:

multi-verifier
```

