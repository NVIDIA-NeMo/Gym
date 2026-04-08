# Terminal Multi-Harness Docs

This directory documents how to extend `terminal_multi_harness` to a new agent
harness without guessing the verifier contract or polluting existing harness
behavior.

## Docs

- [Development Workflow](./development_workflow.md)
  - end-to-end workflow from raw harness collection to rollout validation and
    reward-profile sanity checks
- [Harness Extension Architecture](./harness_extension_architecture.md)
  - codebase structure, extension boundaries, and rules for keeping harnesses
    isolated and maintainable

## Rulebooks

Every new harness should land a rulebook in `docs/rulebooks/` before shared
implementation work starts.

- Codex:
  - [codex_match_rules.md](./rulebooks/codex_match_rules.md)

## Current packaged harness envs

- `codex`
  - config:
    - `resources_servers/terminal_multi_harness/configs/terminal_multi_harness_codex.yaml`
  - agent identity:
    - `terminal_multi_harness_codex_agent`

Future harnesses should usually follow the same pattern:

- add a harness-specific packaged config
- give it a harness-specific agent name
- keep shared comparison logic in `common/`
- keep harness-specific policy in docs, configs, tests, and normalizers first

## In-repo references

- package overview:
  - `resources_servers/terminal_multi_harness/README.md`
- packaged Codex rulebook:
  - `resources_servers/terminal_multi_harness/docs/rulebooks/codex_match_rules.md`
