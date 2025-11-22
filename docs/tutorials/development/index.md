(development-index)=

# Development Patterns

Learn foundational patterns and best practices for building resource servers and agents in NeMo Gym.

---

## Core Patterns

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Interactions
:link: multi-step-patterns
:link-type: doc

Master session-based state management for agents that make multiple tool calls within a single trajectory. Learn when to use stateful vs stateless patterns.
+++
{bdg-primary}`foundational` {bdg-secondary}`state-management` {bdg-secondary}`sessions`
:::

::::

---

## What You'll Learn

These tutorials cover essential patterns that most resource servers and agents need:

**Multi-Step Interactions**
- Understanding sessions and state management
- When to use stateful vs stateless resource servers
- State storage patterns (in-memory, database, external services)
- The `seed_session` pattern for initialization
- Best practices and common pitfalls

---

## Prerequisites

Before diving into these patterns, we recommend:
1. Complete the {doc}`Get Started <../../get-started/index>` tutorials
2. Understand {doc}`Core Abstractions <../../about/concepts/core-abstractions>`
3. Familiarity with FastAPI and async Python

---

```{toctree}
:hidden:
:maxdepth: 2

multi-step-patterns
```

