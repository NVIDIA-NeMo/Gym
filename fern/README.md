# NeMo Gym — Fern Docs

This directory holds the Fern MDX source for the NeMo Gym documentation site at **[docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym)**.

**How to write, preview, check and publish pages is in the [`nemo-gym-docs` skill](../.agents/skills/nemo-gym-docs/SKILL.md).** It is the single source for docs conventions and commands, for people and coding agents alike. This README only covers the one-time setup a person has to do.

## Do you need setup at all?

No, to contribute a page. Edit MDX under `fern/versions/latest/pages/` and open a PR; CI checks it and comments a preview URL.

Yes, to preview locally with the Python API reference (`make docs`). That needs a Fern account in the `nvidia` org.

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

`make docs-login` is required. Without it, `fern docs md generate` returns `HTTP 403: User does not belong to organization`: `fern login` alone is not enough, because Fern needs you to sign in to the dashboard first so your account exists.

## Getting `nvidia` org access

External contributors do not need it; use the CI preview. If you need local library generation and hit the 403, [open a GitHub issue](https://github.com/NVIDIA-NeMo/Gym/issues/new/choose) titled `Fern nvidia org access request` (label `community-request` if available). NVIDIA employees can also ask in the `#fern` Slack channel.

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
