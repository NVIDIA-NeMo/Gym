# Docs Scope Instructions

This `AGENTS.md` applies to everything under:

- `resources_servers/terminal_multi_harness/docs/`

## Purpose

These docs exist to make multi-harness env extension repeatable and safe.

When a coding agent is asked to support a new harness, the default expectation
is:

1. write or update the harness rulebook in `docs/` first
2. review the rulebook until the verifier contract is explicit
3. only then patch implementation code

Do not treat the docs as a retrospective after the code change. For new harness
support, the docs should lead the implementation.

## Required doc-first workflow

Before implementing a new harness in `terminal_multi_harness`, add or update:

- a harness rulebook under `docs/rulebooks/`
- any reusable workflow or architecture notes in `docs/` if a new general
  lesson was learned

The rulebook should be the normative source of truth for:

- completion behavior
- action classification
- tool-call gate order
- tool-specific checks
- multiple-tool-call comparison

If the rulebook is still vague, do not patch shared verifier code yet.

## Rulebook standards

Rulebooks should:

- be short
- be normative
- focus on match semantics only
- avoid mixing in profiling history or implementation notes

Put evidence elsewhere, for example:

- run plans
- profiling notes
- stats reports

## Architecture standards

When documenting a new harness, bias toward:

- one packaged RL env config per harness
- one harness-specific agent identity
- shared code only for truly shared behavior

Do not document or encourage:

- one generic agent name shared by all harnesses
- speculative abstractions before the harness contract is stable
- silent changes to token-cap semantics

## Editing guidance

- Keep docs concise and operational.
- Prefer adding a focused new file over bloating an existing one.
- When a new harness is added, update this directory's `README.md`.
- When a new rulebook is added, link it from this directory's `README.md`.
- Preserve existing harness docs unless they are actually obsolete.
