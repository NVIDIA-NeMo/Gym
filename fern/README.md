# NeMo Gym — Fern Docs

This directory holds the source for the NeMo Gym documentation site, [docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym).

This README covers only setting up your machine to preview the docs locally. For how to write, check and publish pages, follow the [`nemo-gym-docs` skill](../.agents/skills/nemo-gym-docs/SKILL.md), which people and coding agents both use.

You don't need local setup to contribute. Edit pages under `fern/versions/latest/pages/` and open a PR: CI checks your changes and comments a link to a preview of the site.

## One-time local setup

From the repo root:

```bash
# 1. Install Node.js 22+, then install the Fern CLI globally (one-time)
npm install -g fern-api
# or use it ad-hoc via:  npx -y fern-api@latest <subcommand>

# 2. Provision your Fern account and CLI auth (once per machine)
make docs-login   # or `make docs-login-remote` on a headless remote machine

# 3. Start the local preview at http://localhost:3000
make docs
```

Local preview builds the Python API reference, which needs a Fern account in the `nvidia` org. Run `make docs-login` first. It signs you in to the Fern dashboard before running `fern login`; `fern login` alone is not enough, and `fern docs md generate` then fails with `HTTP 403: User does not belong to organization`.

## Maintainer secrets

The preview and publish workflows use the `DOCS_FERN_TOKEN` repository or organization secret, issued with `fern token` from a privileged Fern dashboard account.

## Links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/gym |
| Fern dashboard | https://dashboard.buildwithfern.com (NVIDIA org) |
| Docs skill | [`.agents/skills/nemo-gym-docs/SKILL.md`](../.agents/skills/nemo-gym-docs/SKILL.md) |
| CI workflows | [`.github/workflows/`](../.github/workflows/) (`fern-docs-*.yml`, `publish-fern-docs.yml`) |
| Fern docs | https://buildwithfern.com/learn/docs/getting-started/overview |
