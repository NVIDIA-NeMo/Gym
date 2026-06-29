# NeMo Gym docs

The Sphinx tree that used to live here has been retired. NeMo Gym's documentation is now authored in [Fern](https://buildwithfern.com/) MDX under [`../fern/`](../fern/) and published to **[docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym)**.

- **Read the docs:** https://docs.nvidia.com/nemo/gym
- **Edit pages:** see [`../fern/README.md`](../fern/README.md) for layout, local dev, and authoring conventions.
- **Add or edit a page:** update `fern/versions/latest/pages/` and, when needed, `fern/versions/main.yml`. The `latest/pages/` directory name is historical: it is the bleeding-edge source tree published at `/main/...`.
- **Release snapshots:** do not copy ordinary Main docs fixes into GA folders. Create a new frozen snapshot under `fern/versions/v<release>/` only when that release cuts, then add it to `fern/docs.yml` `versions:`. Back-port to an existing GA snapshot only when the fix explicitly needs to change already-published release docs.
- **Preview a PR:** PRs touching `fern/**` get an automatic 🌿 preview URL posted as a comment by `.github/workflows/fern-docs-preview-comment.yml`.

For the agent-facing version of the same workflow, see [`../.claude/skills/nemo-gym-docs/SKILL.md`](../.claude/skills/nemo-gym-docs/SKILL.md).

Old `/nemo/gym/...` URLs from the Sphinx build are redirected to their Fern equivalents via `redirects:` in [`../fern/docs.yml`](../fern/docs.yml). Legacy `/nemo/gym/latest/...` URLs are redirects too; `latest` is not a separate Fern version. If you find a broken link to the published site, add a redirect there.
