# NeMo Gym — Fern Docs

This directory holds the Fern MDX source for the NeMo Gym documentation site at **[docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym)**.

All new pages and edits should land here. The Sphinx tree under `../docs/` is preserved for reference but is no longer the source of truth for the published site.

## Quick links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/gym |
| Fern dashboard | https://dashboard.buildwithfern.com (NVIDIA org) |
| Skill for agents | [`../.claude/skills/nemo-gym-docs/SKILL.md`](../.claude/skills/nemo-gym-docs/SKILL.md) |
| CI workflows | [`../.github/workflows/fern-docs-*.yml`](../.github/workflows/) |
| Make targets | [`../Makefile`](../Makefile) |

## Quickstart

First time on this machine (run from the repo root):

```bash
# 1. Install the Fern CLI globally (one-time)
npm install -g fern-api
# or use it ad-hoc via:  npx -y fern-api@latest <subcommand>

# 2. Provision your Fern account + CLI auth (one-time per machine).
#    Walks you through the dashboard sign-in step before running `fern login`.
make docs-login   # or `make docs-login-remote` when working on a headless remote machine

# 3. Build the API library reference and start the local dev server
make docs           # http://localhost:3000

# 4. (Optional) validate config + MDX without booting the server
make docs-check
```

**`make docs-login` is load-bearing.** Skip it and `fern docs md generate` returns `HTTP 403: User does not belong to organization` — the CLI's `fern login` flow alone is *not* enough; Fern requires that you sign in to the dashboard first so your account record exists in Fern's user DB.

### Fern CLI + docs reference

| Resource | Link |
|---|---|
| Fern docs (overview, writing, configuration) | https://buildwithfern.com/learn/docs/getting-started/overview |
| Fern CLI reference | https://buildwithfern.com/learn/cli-api-reference/cli-reference/overview |
| MDX components (Cards, Callouts, Tabs, …) | https://buildwithfern.com/learn/docs/content/components/overview |
| Frontmatter fields | https://buildwithfern.com/learn/docs/content/frontmatter |
| Versioning | https://buildwithfern.com/learn/docs/configuration/versions |
| Redirects | https://buildwithfern.com/learn/docs/seo/redirects |
| `libraries:` (Python autodoc) | https://buildwithfern.com/learn/docs |

## Layout

```
fern/
├── fern.config.json          # Fern CLI org slug + version pin
├── package.json              # `npm run check|dev|generate|generate:library`
├── docs.yml                  # Site config: instances, versions, redirects, libraries (`global-theme: nvidia`)
├── assets/                   # Page images (logos/favicon/fonts come from the `nvidia` global theme)
├── components/               # NavButton.tsx + co-located NavButton.css
├── versions/
│   ├── main.yml              # Nav for the bleeding-edge train — paths point at ./latest/pages/
│   ├── latest/pages/         # Bleeding-edge MDX content (edited on every PR; published at /main/...)
│   ├── v<release>.yml        # Nav for each frozen GA snapshot — paths point at ./v<release>/pages/
│   └── v<release>/pages/     # Frozen GA content (back-ports only)
└── product-docs/             # GENERATED Python API reference (gitignored — `npm run generate:library` rebuilds)
```

```
File path                                              Published URL
─────────────────────────────────────────────────────  ─────────────────────────────────────────────────
fern/versions/latest/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/main/get-started/quickstart
fern/versions/v<release>/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/v<release>/get-started/quickstart
docs.nvidia.com/nemo/gym/latest/get-started/quickstart      redirects to docs.nvidia.com/nemo/gym/main/get-started/quickstart
```

The folder name `latest/` is historical — it holds the **bleeding-edge** tree and is mounted under the `main` slug via `main.yml`. Versioned folders are frozen GA snapshots, only changed via deliberate back-port. There is no `latest` Fern version; legacy `/latest/...` URLs are handled by redirects in `docs.yml`.

## Local development

From the repo root:

```bash
make docs                   # generate library reference, then `fern docs dev` → http://localhost:3000
make docs-check             # `fern check` (config + MDX validation)
make docs-preview           # shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)
make docs-publish           # trigger the `Publish Fern Docs` workflow on origin/main
make docs-generate-library  # standalone library regeneration (rarely needed; `make docs` runs it)
```

For first-time-on-this-machine setup, see the [Quickstart](#quickstart) above — `make docs-login` / `make docs-login-remote` walks through dashboard provisioning + `fern login` together.

`make docs` first runs `fern docs md generate`, which populates `fern/product-docs/` from the `nemo_gym` package source declared in the `libraries:` block of `docs.yml`. Without it, a cold `fern docs dev` will fail with `Folder not found: ./product-docs/...`. Re-run only when the upstream Python source changes — for prose-only iteration after the first generation, `cd fern && npm run dev` is enough.

Underlying npm scripts (run from `fern/`) are also available if you want to bypass Make:

```bash
npm run check               # `fern check`
npm run dev                 # `fern docs dev`
npm run generate:library    # `fern docs md generate`
```

## Authoring conventions

### Frontmatter

```yaml
---
title: "<Page Title>"        # required — used by Fern as the page title and breadcrumb
description: ""              # required (may be empty string) — SEO
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

The MDX body should still open with `# <Page Title>` matching the frontmatter title. Folders using `title-source: frontmatter` in the version YAML pull the nav label from `title:`.

### Components

Use the bundled custom components in `components/`:

| Component | Purpose |
|---|---|
| `<NavButton ... />` | Inline wayfinding pill button for tutorial back/prev/next links |

Component-scoped CSS lives next to its TSX (e.g. `NavButton.css` next to `NavButton.tsx`) and is loaded via a sibling `import "./NavButton.css"` — keep new component styles co-located the same way.

The footer, logos, favicon, fonts, brand colors, base CSS, and OneTrust JS are all inherited from the `nvidia` global theme published from [NVIDIA/fern-components](https://github.com/NVIDIA/fern-components) via `global-theme: nvidia` in `docs.yml`. Don't re-add `footer:`, `logo: { dark, light, height }`, `favicon:`, `css:`, `js:`, `colors:`, `layout:`, `theme:`, or `navbar-links:` here — change them upstream and re-upload the theme. The one exception is the `logo.right-text: NeMo Gym` override (the theme hardcodes "Documentation").

Standard Fern components are also available — `<Note>`, `<Tip>`, `<Info>`, `<Warning>`, `<Cards>` / `<Card>`, `<Badge>`, etc. Don't use GitHub `> [!NOTE]` syntax — it does not render in MDX.

### Internal links

Use **version-prefixed paths** matching the slug of the tree the page lives in:

```mdx
[Quickstart](/main/get-started/quickstart)        // links inside versions/latest/pages/
[Quickstart](/v<release>/get-started/quickstart)  // links inside versions/v<release>/pages/
```

Cross-version links (for example, from a `main/` page to a versioned GA page) trigger broken-link warnings in `fern docs dev`; those are **false positives** — Fern's local validator does not resolve cross-version slugs from `docs.yml`. The published site renders them correctly.

### Cross-repo references (yaml configs, source files)

Repository source paths like `resources_servers/example_single_tool_call/...` or `responses_api_models/...` are not part of the docs site. Link to them as **absolute GitHub URLs**:

```mdx
[example_single_tool_call.yaml](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml)
```

## Versioning

`docs.yml` `versions:` is the source of truth for the current published version list. It follows this pattern:

| display-name | slug | availability | path |
|---|---|---|---|
| `Main` | `main` | beta or default | `./versions/main.yml` |
| current GA | `v<release>` | `stable` | `./versions/v<release>.yml` |
| supported older GA | `v<older-release>` | `stable` | `./versions/v<older-release>.yml` |

**`main` is the bleeding-edge tree** — ordinary docs PRs land in `versions/latest/pages/` and publish under the `main` slug. **Versioned `v<release>/` folders are frozen GA snapshots** with their own copy of every page; change them only by deliberate back-port. Legacy `/latest/...` URLs are redirects in `docs.yml`, not a mounted version.

When the next GA cuts (for example, `v0.4.0`):

1. `cp -r versions/latest versions/v0.4.0` — fresh frozen snapshot of the bleeding-edge tree
2. `cp versions/main.yml versions/v0.4.0.yml`, then rewrite `./latest/` path prefixes to `./v0.4.0/`
3. Add the new frozen-pin entry to `docs.yml` `versions:` (`display-name: "0.4.0"`, `slug: v0.4.0`, `availability: stable`); demote/remove previous GA snapshots per the support policy
4. Keep the `/latest` redirect rules in `docs.yml` pointed at the intended legacy target, currently `/main`
5. `versions/latest/pages/` keeps moving forward as the bleeding-edge tree

See [`../.claude/skills/nemo-gym-docs/SKILL.md`](../.claude/skills/nemo-gym-docs/SKILL.md) for the same procedure framed for an agent.

## CI and publishing

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | `push: pull-request/[0-9]+` (FW-CI mirror) | `fern check` on PRs |
| `fern-docs-preview-build.yml` | `pull_request` | Untrusted half: collect `fern/` artifact (no secrets) |
| `fern-docs-preview-comment.yml` | `workflow_run` after build | Trusted half: build preview with `DOCS_FERN_TOKEN`, post 🌿 comment |
| `publish-fern-docs.yml` | push to `main` (`fern/**` or `docs/**`), `docs/v*` tag, or manual | Publish to docs.nvidia.com/nemo/gym |

Required org secret: **`DOCS_FERN_TOKEN`** (issued via `fern token` on a privileged dashboard account).

PRs that touch `fern/**` get an automatic preview URL posted as a 🌿 comment.

## Commits

DCO sign-off is required:

```bash
git commit -s -m "docs: <add|update|remove> <page-title>"
```

PR titles follow Conventional Commits (e.g. `docs(fern): add rollout collection guide`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 403: User does not belong to organization` on `fern docs md generate` | Sign in to https://dashboard.buildwithfern.com first, then re-run `npx -y fern-api@latest login` ([#1185](https://github.com/NVIDIA-NeMo/Gym/issues/1185)) |
| `Folder not found: ./product-docs/...` in `fern docs dev` | Run `npm run generate:library` once; library generation populates `product-docs/` |
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML |
| Page 404 in preview | `slug:` missing/duplicated in the same section, or `position:` collision in an auto-discovered folder |
| Broken-link warning for cross-version path | False positive in `fern docs dev`; the published site resolves it correctly |
| `JSX expressions must have one parent element` | Wrap multi-element MDX content in `<>...</>` or a `<div>` |
| Card badges have no spacing | Don't add inline styles — the `nvidia` global theme handles `.fern-card .fern-docs-badge` spacing |
| Old URL breaks | Add a `redirects:` entry in `docs.yml` |
| Library reference missing after generation | Re-run `npm run generate:library`; check `libraries:` block in `docs.yml` matches the package source path |

## Reference

- [Fern docs (upstream)](https://buildwithfern.com/learn/docs/getting-started/overview)
