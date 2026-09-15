---
name: simplicity-first
description: >
  Apply a simplicity-first gate to implementation and code review. Use when
  designing, coding, refactoring, or reviewing changes to prefer the smallest
  clear solution and reject speculative abstractions or unrelated scope.
---

# Simplicity First

Prefer the smallest change that is correct, readable, and easy to verify.
Simplicity means obvious code, not merely fewer lines.

Adapted from the
[`simplicity-first`](https://github.com/NVIDIA-NeMo/labs-molt/tree/main/.claude/skills/simplicity-first)
skill in NVIDIA NeMo labs-molt.

## Working Rules

- Solve the stated requirement and current constraints. Do not design for
  hypothetical future needs.
- Reuse established repository patterns. Add a new abstraction, option,
  dependency, or layer only when the present change clearly needs it.
- Keep control flow and data flow easy to trace. Prefer direct code over clever
  indirection, but extract a helper when it genuinely improves clarity or reuse.
- Preserve intentional behavior, compatibility, performance, and observability.
  Simplification is not permission to remove required capability.
- Keep the diff focused. Record worthwhile unrelated improvements separately.

Before finishing, ask:

1. Can any added concept or code be removed without weakening the solution?
2. Is each remaining addition required now and understandable in one pass?
3. Do the tests demonstrate the requested behavior rather than the
   implementation details?

Stop when the requirement is met and the relevant checks pass.
