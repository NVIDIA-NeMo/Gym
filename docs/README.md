# NeMo Gym docs

The Sphinx tree that used to live here has been retired. NeMo Gym's documentation is now authored in [Fern](https://buildwithfern.com/) MDX under [`../fern/`](../fern/) and published to **[docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym)**.

- **Read the docs:** https://docs.nvidia.com/nemo/gym
- **Edit pages:** see [`../fern/README.md`](../fern/README.md) for layout, local dev, and authoring conventions.
- **Add a page:** drop an MDX file under `fern/versions/latest/pages/` (the bleeding-edge tree, published at `/main/...` with `availability: beta`) and wire it into `fern/versions/main.yml`. Back-port to the current GA snapshot under `fern/versions/<ga>/` (e.g. `v0.2.1/` at time of writing — check `fern/docs.yml` `versions:` for the current GA) only when the fix needs to ship to that release.
- **Preview a PR:** PRs touching `fern/**` get an automatic 🌿 preview URL posted as a comment by `.github/workflows/fern-docs-preview-comment.yml`.

For the agent-facing version of the same workflow, see [`../.claude/skills/nemo-gym-docs/SKILL.md`](../.claude/skills/nemo-gym-docs/SKILL.md).

Old `/nemo/gym/...` URLs from the Sphinx build are redirected to their Fern equivalents via `redirects:` in [`../fern/docs.yml`](../fern/docs.yml). If you find a broken link to the published site, add a redirect there.
