---
orphan: true
---

# Development Patterns and Tutorials

This directory contains tutorials and guides for developing resource servers and agents in NeMo Gym.

## Contents

### Core Patterns

- **[multi-step-patterns.md](multi-step-patterns.md)**: Comprehensive guide to session-based state management for multi-step agent interactions. Essential reading for building stateful resource servers.

- **[PATTERNS-QUICK-REFERENCE.md](PATTERNS-QUICK-REFERENCE.md)**: Quick reference card with decision trees, code snippets, and common patterns.

## Learning Path

For developers new to NeMo Gym:

1. Complete the [Get Started](../../get-started/index.md) tutorials first
2. Read [Multi-Step Patterns](multi-step-patterns.md) to understand state management
3. Review [PATTERNS-QUICK-REFERENCE.md](PATTERNS-QUICK-REFERENCE.md) for quick lookup
4. Explore the `example_stateful_counter` reference implementation

## Key Concepts

- **Multi-Step**: Multiple sequential tool calls within one trajectory
- **Sessions**: Unique identifiers that maintain state across tool calls
- **Stateful vs Stateless**: Choosing the right pattern for your resource server
- **State Storage**: In-memory, database, or external service patterns

## Reference Implementations

- **example_stateful_counter**: Complete example of session-based state management (`resources_servers/example_stateful_counter/`)
- **example_simple_weather**: Example of stateless resource server (`resources_servers/example_simple_weather/`)

## Related Documentation

- [Core Abstractions](../../about/concepts/core-abstractions.md)
- [Sessions Explained](../../about/concepts/sessions-explained.md)
- [Key Terminology](../../about/concepts/key-terminology.md)

