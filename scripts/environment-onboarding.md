# Environment onboarding maintenance scripts

These maintainer tools update migration inventory and generate catalog output. They are not shipped `gym` CLI commands and do not participate in runtime workload loading.

## Migration inventory

The checked-in inventory is a census, not a queue of manifests to publish. Refresh it without creating draft manifests:

```bash
uv run python scripts/migrate_environment_manifests.py --inventory-only
uv run python scripts/migrate_environment_manifests.py --check
```

Follow [Migrate an existing environment](../fern/versions/latest/pages/contribute/environments/migrating-an-environment.mdx) to select one owner-backed canonical recipe, distinguish component-only Resources Servers, and complete its manifest from evidence.

`--write` requires the exact config path for one reviewed unit and creates one non-overwriting placeholder manifest for local migration work:

```bash
uv run python scripts/migrate_environment_manifests.py \
  --write \
  --config resources_servers/mcqa/configs/mcqa.yaml
```

Treat the file as a temporary authoring aid, complete every `TODO_REQUIRED` field from owner-reviewed evidence, and do not commit an untouched generated draft.

The inventory retains legacy descriptive, domain, and value metadata. It deliberately excludes the old `verified` and `verified_url` claims because they are not a publication trust root. Runnable configs under `resources_servers/` use collision-free inventory names; groups that share one component directory remain explicit exceptions until an owner selects, splits, or classifies them.

## Catalog generation

Generate catalog JSON and accessible static HTML from the same manifest and legacy union used by `gym list catalog --json`:

```bash
uv run python scripts/generate_environment_catalog.py \
  --json-output /tmp/nemo-gym-catalog/catalog.json \
  --html-output /tmp/nemo-gym-catalog/index.html
```

Catalog files are deployment artifacts, not source-controlled migration state. The `Publish Environment Catalog` workflow regenerates both files under the runner's temporary directory after every push to `main`, rejects invalid manifests or publication records with `--fail-on-issues`, and publishes the result through GitHub Pages. Its deployment summary provides the canonical site URL; append `/catalog.json` for the machine-readable view.

Before the first deployment, configure the repository's Pages source as **GitHub Actions** and restrict the `github-pages` environment's deployment branch policy to `main`. Manual workflow dispatches are also limited to `main`.

## Schema and onboarding checks

Verify the checked-in manifest schema and the onboarding policy:

```bash
uv run python scripts/generate_environment_manifest_schema.py --check
uv run python scripts/check_environment_onboarding.py
```

For a pull request, pass its Git revisions and enable scoped enforcement:

```bash
uv run python scripts/check_environment_onboarding.py \
  --base-ref "$BASE_SHA" \
  --head-ref "$HEAD_SHA" \
  --enforce-changes
```

The scoped check enforces new and directly changed recipes, reports affected shared-component dependents, verifies generated schema and publication records, and leaves unchanged legacy units runnable. Add `--json` for the complete inventory and per-unit decisions.
