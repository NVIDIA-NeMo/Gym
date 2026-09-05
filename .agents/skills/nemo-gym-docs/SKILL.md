---
name: nemo-gym-docs
description: >
  Maintain the NeMo Gym Fern documentation under fern/. Use when adding,
  updating, moving, removing, previewing, or publishing documentation pages,
  changing navigation or redirects, or cutting a versioned docs snapshot.
---

# Maintain NeMo Gym Docs

## Establish the source of truth

Read `fern/README.md` before changing docs. Treat it, `fern/docs.yml`, and the relevant version YAML as authoritative
for repository layout, versioning, navigation, validation, and release commands.

Stable routing rules:

- Author ordinary documentation in `fern/versions/latest/pages/` and edit navigation in
  `fern/versions/main.yml`. The editable source publishes as `/main`.
- Frozen `fern/versions/v<release>/` trees and their YAML files are snapshots. Change one only for an explicit
  release operation or targeted backport.
- There is no `fern/versions/latest.yml`. `/latest` URLs are redirects configured in `fern/docs.yml`.
- Do not update every frozen version when editing current docs.

## Choose the operation

### Add a page

Create the MDX file under the matching `latest/pages/` section, add it to `main.yml` only when its parent is not
auto-discovered, and link it from relevant nearby pages. Prefer existing patterns over new one-off structure.

### Update a page

Make the smallest current-source edit that fixes the contract. Check inbound and outbound docs links and update
navigation only when the title or placement changes.

### Move, rename, or remove a page

Update `main.yml`, repository references, and `fern/docs.yml` redirects when a published URL changes. Use an explicit
redirect target; do not leave duplicate pages to preserve old routes.

### Cut or backport a release

Only do this when explicitly requested. Follow the exact release procedure in `fern/README.md`; do not infer a version
number, retarget a nonexistent `latest.yml`, publish, or create a tag without authorization.

## Authoring contracts

- Use version-prefixed docs paths that match the tree (`/main/...` for `latest/pages/` and `/v<release>/...` for a
  frozen snapshot). Follow the cross-version caveat in `fern/README.md`; use full URLs only for external references.
- Follow the frontmatter and component conventions in `fern/README.md` and adjacent pages. Put shared images in
  `fern/assets/`.
- Do not hand-edit generated API reference pages. Update their source or generator and regenerate them.
- Preserve the terminology and heading style of adjacent pages. Avoid duplicating mutable rules that already have an
  authoritative home; link to that contract instead.

## Validate

Run the repository docs checks after edits:

```bash
make docs-check
```

For structural or visual changes, preview locally:

```bash
make docs
```

Inspect changed navigation and redirects as data, not only rendered prose. Report any check that could not run. Publish
or tag docs only when the user explicitly requests that external action.
