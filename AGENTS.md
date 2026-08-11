# Agent guidance for NeMo Gym

Instructions for AI coding assistants (Cursor, Claude, Codex, OpenCode, and similar) working in this repository. This file is also the shared quality bar for human contributors — the docs site links here instead of duplicating the list.

Humans: see [Development Setup → Use of AI and LLM Tools](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup#use-of-ai-and-llm-tools) (maintainer response policy) and [Agent Skills](https://docs.nvidia.com/nemo/gym/latest/contribute/agent-skills).

## Quality bar

- Prefer focused changes. Do not make unrelated "drive-by" edits. If a drive-by fix is worth keeping, open a separate issue or PR.
- Intentional synthetic scaling of environments is fine when scoped via an issue or focused PR; do not dump unreviewed bulk diffs.
- You (the human author) own every line submitted. Treat model output as untrusted until reviewed.
- For environment or agent changes: run real rollouts with a model and inspect agent and verifier behavior. Green unit tests alone are not enough.
- Before opening a PR, run the local checks that mirror CI: tests (skip or N/A for docs-only), `pre-commit run --all-files`, and DCO sign-off (`git commit -s`). Cryptographic `-S` signing is not required.
- AI-generated tests must assert real behavior; avoid vacuous pass-through tests.

## Repo conventions

- Follow `CLAUDE.md` for architecture and coding patterns.
- Prefer the vetted skills under `.agents/skills/` (see [Agent Skills](https://docs.nvidia.com/nemo/gym/latest/contribute/agent-skills)).
- Docs live under `fern/versions/latest/pages/`. Bleeding-edge nav is `fern/versions/main.yml`. See `fern/README.md` and the `nemo-gym-docs` skill.
- Do not introduce licenses incompatible with Apache-2.0. New source files need the standard NVIDIA SPDX header.
