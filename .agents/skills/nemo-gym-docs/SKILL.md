---
name: nemo-gym-docs
description: >
  Maintain the NeMo Gym Fern docs site — add, update, move, or remove pages
  under fern/. Use for any documentation change. Triggered by: "edit docs",
  "add doc page", "update docs", "rename page", "fix broken link", "add
  redirect", "preview docs", "publish docs", any request that touches `fern/`.
---

# NeMo Gym Docs Maintenance

How to add, update, move, and remove pages on the NeMo Gym docs site ([docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym)), and how to preview, check, and publish them. This file is the single source for docs conventions and commands. [`fern/README.md`](../../../fern/README.md) only covers the one-time account setup a person has to do.

## Scope Rule

**All docs edits happen under `fern/`.** Ordinary edits land in `fern/versions/latest/pages/`, the bleeding-edge tree, published under the `main` slug. The folder name `latest/` is historical.

Each `fern/versions/v<release>/` folder is a frozen copy of a shipped release. Change one only when the user explicitly asks for a back-port. `fern/docs.yml` `versions:` is the source of truth for which releases are published. There is no `latest` Fern version: legacy `/latest/...` URLs are redirects in `docs.yml`.

## Layout

```
fern/
├── fern.config.json          # Fern org slug + CLI version
├── package.json              # `npm run check|dev|generate|generate:library`
├── docs.yml                  # Site config: instances, versions, redirects, libraries (`global-theme: nvidia`)
├── assets/                   # Page images
├── components/               # Custom TSX components (NavButton.tsx)
├── versions/
│   ├── main.yml              # Nav for the bleeding-edge tree; paths point at ./latest/pages/
│   ├── latest/pages/         # Bleeding-edge MDX (published at /main/...)
│   ├── v<release>.yml        # Nav for each frozen release; paths point at ./v<release>/pages/
│   └── v<release>/pages/     # Frozen release MDX (back-ports only)
└── product-docs/             # GENERATED Python API reference (gitignored)
```

```
File path                                              Published URL
fern/versions/latest/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/main/get-started/quickstart
fern/versions/v<release>/pages/get-started/quickstart.mdx  docs.nvidia.com/nemo/gym/v<release>/get-started/quickstart
docs.nvidia.com/nemo/gym/latest/get-started/quickstart      redirects to /main/get-started/quickstart
```

## Operations

### Add a Page

1. Gather: page title, target section, filename (kebab-case `.mdx`), subdirectory under `fern/versions/latest/pages/`.
2. Create `fern/versions/latest/pages/<subdirectory>/<filename>.mdx`:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
   position: 3
   ---

   <content>
   ```

3. If the parent folder is mounted in `main.yml` with `title-source: frontmatter`, the page is auto-discovered and needs no nav edit. Otherwise add a `- page:` entry under the right `section:` in `fern/versions/main.yml`. `position:` orders pages in an auto-discovered folder; without it, pages sort alphabetically by filename.
4. Run `make docs-check` and preview the page at `/main/<subdirectory>/<filename>`.
5. Do not copy the page into a `v<release>/` folder unless the user asks for a back-port.

### Update a Page

1. Locate it by path, title, or keyword (`grep -rn` in `fern/versions/latest/pages/`).
2. Content change: edit the MDX in `fern/versions/latest/pages/` only.
3. Title change: update the frontmatter `title:`. If the parent folder does not use `title-source: frontmatter`, update the nav `- page:` entry too. Update link text that names the page.
4. Section move: `git mv` the file, update its `path:` in `main.yml` if it has a manual entry, fix incoming links, and add a redirect.
5. Slug change: the filename sets the slug, so renaming the file changes the URL. Add a redirect in `fern/docs.yml` so the old URL keeps working.

### Remove a Page

1. Find incoming links: `grep -rn "<filename>" fern/versions/latest/pages --include="*.mdx"`.
2. `git rm` the file.
3. Remove its `- page:` entry from `main.yml` if it had one.
4. Fix or remove the incoming links.
5. Add a redirect in `fern/docs.yml` if the URL was published.

### Back-port to a Release

Only when the user asks. Apply the same change under `fern/versions/v<release>/pages/` and update `fern/versions/v<release>.yml` if the nav changes.

### Worked Example: Renaming a Slug

Request: *"Rename `/main/get-started/setup` to `/main/get-started/detailed-setup`."*

1. `git mv fern/versions/latest/pages/get-started/setup.mdx fern/versions/latest/pages/get-started/detailed-setup.mdx`, and update its `path:` in `main.yml` if it has a manual entry.
2. Add a redirect in `fern/docs.yml`. Both `source` and `destination` need the full `/nemo/gym/` basepath with a leading slash:

   ```yaml
   redirects:
     - source: "/nemo/gym/main/get-started/setup"
       destination: "/nemo/gym/main/get-started/detailed-setup"
   ```

3. `grep -rn "/get-started/setup" fern/versions/latest/` and update incoming links.

## Content Conventions

### Frontmatter

```yaml
---
title: "<Page Title>"        # required: the page heading, breadcrumb, and (with title-source: frontmatter) nav label
description: ""              # required (may be empty): SEO
position: 1                  # optional: orders auto-discovered pages within a folder
---
```

Fern renders `title:` as the page heading, so do not repeat it as a `# <Page Title>` line in the body; the page would show the title twice. `tests/unit_tests/test_fern_docs_links.py` fails on a repeated title.

### Product Name

Call the product NeMo Gym in prose, not Gym. Commands, package names and other identifiers, such as `gym eval run` or `nemo_gym`, stay as written.

### Components

Use Fern's MDX components. GitHub `> [!NOTE]` syntax does not render.

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid on index pages | `<Cards>` with `<Card title="..." href="...">` children |
| Status/scope tag inside a Card | `<Badge minimal outlined>tag</Badge>` |
| Tutorial back/next links | `<NavButton ... />` (custom, in `fern/components/`) |

Badges go on the last line inside a `<Card>`, separated from the body by a blank line. Valid intents: `success`, `note`, `tip`, `warning`, `error`, `info`, `launch`, `check`; omit the intent for a neutral tag. Do not add inline `style=` props; the `nvidia` global theme handles spacing.

```mdx
<Cards>
  <Card title="Quickstart" href="/main/get-started/quickstart">
    Install, start servers, and collect your first rollouts in one page.

    <Badge intent="success" minimal outlined>start here</Badge> <Badge minimal outlined>5 min</Badge>
  </Card>
</Cards>
```

A new custom component goes in `fern/components/`. Put its styles in a `<style>` block inside the TSX, as `NavButton.tsx` does: Fern's bundler does not load `import "./*.css"`, and the global theme owns `docs.yml` `css:`.

The footer, logos, favicon, fonts, colors, base CSS and OneTrust JS come from the `nvidia` global theme (`global-theme: nvidia` in `docs.yml`). Do not add `footer:`, `logo:`, `favicon:`, `css:`, `js:`, `colors:`, `layout:`, `theme:`, or `navbar-links:` to `docs.yml`; change them upstream in NVIDIA/fern-components. The one exception is the `logo.right-text: NeMo Gym` override.

### Images

Put shared images in `fern/assets/`, or version-scoped images under that version's `pages/`, and reference them with root-relative paths.

### Links

Link inside the docs with version-prefixed paths that match the tree the page lives in:

```mdx
[Quickstart](/main/get-started/quickstart)        // pages in versions/latest/pages/
[Quickstart](/v<release>/get-started/quickstart)  // pages in versions/v<release>/pages/
```

Link text names the target page's title, not its slug.

Link to repository files (configs, source) with absolute GitHub URLs; they are not part of the docs site:

```mdx
[example_single_tool_call.yaml](https://github.com/NVIDIA-NeMo/Gym/blob/main/resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml)
```

## Preview and Check

From the repo root:

```bash
make docs                   # generate the library reference, then `fern docs dev` at http://localhost:3000
make docs-check             # `fern check`: config + MDX validation
make docs-preview           # shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)
make docs-generate-library  # regenerate the library reference only
make docs-publish           # trigger the publish workflow on main (only when the user asks)
```

Or from `fern/`: `npm run check`, `npm run dev`, `npm run generate:library`.

`make docs` runs `fern docs md generate` first, which fills `fern/product-docs/` from the `libraries:` block in `docs.yml`. After that, `cd fern && npm run dev` is enough for prose edits. Library generation needs Fern `nvidia` org access; see [`fern/README.md`](../../../fern/README.md). Without it, preview through CI instead.

Before committing, run `make docs-check` and `pytest tests/unit_tests/test_fern_docs_links.py`.

## Commit, CI and Publishing

```bash
git commit -s -m "docs: <add|update|remove> <page-title>"
```

DCO sign-off is required. PR titles follow Conventional Commits (for example, `docs(fern): add rollout collection guide`).

| Workflow | Trigger | Purpose |
|---|---|---|
| `fern-docs-ci.yml` | PR mirror push touching `fern/**` or the docs test | `fern check` and `tests/unit_tests/test_fern_docs_links.py` |
| `fern-docs-preview.yml` | Approved PR mirror push touching `fern/**` | Publishes a preview and comments its URL on the PR. Fork PRs need mirror approval first. |
| `publish-fern-docs.yml` | Push to `main` touching `fern/**`, a `docs/v*` tag, or manual dispatch | Publishes to docs.nvidia.com/nemo/gym |

Preview and publish use the `DOCS_FERN_TOKEN` secret (setup in [`fern/README.md`](../../../fern/README.md)). Release tags use the form `docs/v<MAJOR>.<MINOR>.<PATCH>`, for example `git tag docs/v0.8.0 && git push origin docs/v0.8.0`. Do not push a tag or run `make docs-publish` unless the user asks.

## Cutting a New Release

When a release ships (for example, `v0.8.0`):

1. `cp -r fern/versions/latest fern/versions/v0.8.0`
2. `cp fern/versions/main.yml fern/versions/v0.8.0.yml`, then rewrite the `./latest/` path prefixes to `./v0.8.0/`.
3. Add the release to `fern/docs.yml` `versions:` (`display-name: "0.8.0"`, `slug: v0.8.0`, `availability: stable`), and demote or remove older releases per the support policy.
4. Keep the `/latest` redirects in `docs.yml` pointed at their intended target, currently `/main`.
5. `fern/versions/latest/pages/` keeps moving forward as the bleeding-edge tree.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 403: User does not belong to organization` on `fern docs md generate` | The account setup in [`fern/README.md`](../../../fern/README.md) is incomplete; ask the user to run `make docs-login` |
| `Folder not found: ./product-docs/...` in `fern docs dev` | Run `make docs-generate-library` once |
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to the version YAML |
| Page 404 in preview | `slug:` missing or duplicated in the same section, or a `position:` collision in an auto-discovered folder |
| Broken-link warning for a cross-version link | False positive in `fern docs dev`; the published site resolves it |
| `JSX expressions must have one parent element` | Wrap multi-element MDX in `<>...</>` or a `<div>` |
| Card badges have no spacing | Remove inline styles; the global theme handles it |
| Old URL breaks | Add a `redirects:` entry in `fern/docs.yml` |
| Library reference missing after generation | Re-run `make docs-generate-library`; check the `libraries:` block in `docs.yml` |

## References

- [Fern docs](https://buildwithfern.com/learn/docs/getting-started/overview), [CLI reference](https://buildwithfern.com/learn/cli-api-reference/cli-reference/overview), [components](https://buildwithfern.com/learn/docs/content/components/overview), [frontmatter](https://buildwithfern.com/learn/docs/content/frontmatter), [versions](https://buildwithfern.com/learn/docs/configuration/versions), [redirects](https://buildwithfern.com/learn/docs/seo/redirects)
