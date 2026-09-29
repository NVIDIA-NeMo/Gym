# Fern preview operations

Hosted previews run only for successful `pull_request` collector runs whose
head repository is NVIDIA-NeMo/Gym. The publisher also verifies the
current open PR and head through GitHub's API. Forks retain the token-free
`Fern docs (check)` workflow and collection; they do not receive hosted previews.

The publisher installs Fern **5.50.4** outside the downloaded artifact, with npm
lifecycle scripts disabled. It uses the workflow revision's docs configuration,
components, and organization (`nvidia`). Only Markdown/MDX and document assets
are overlaid from the PR. Changes to navigation, components, CLI configuration,
or tooling require a trusted-branch update before they affect hosted previews.
New pages must already be reachable through the trusted navigation to render.
Library generation uses the fixed CLI and trusted library declarations. Any
post-processing runs a trusted-checkout helper in a separate token-free step.

Artifacts are downloaded by the triggering run ID into an isolated directory.
PR number, head SHA, run ID, and attempt metadata must match the GitHub run and
current PR. Legacy artifacts without these fields fail closed: rerun the updated
collector after deployment. A final head check prevents stale preview comments.

`DOCS_FERN_TOKEN` is passed only to the fixed CLI's library-generation and publish invocations. Direct
page links are intentionally omitted: no HTTP request to the emitted preview
URL receives this token. The displayed URL must be HTTPS on a Fern docs host,
without credentials, port overrides, query, or fragment. GitHub comment access
is limited to the run/PR lookup and comment steps; only the bot's own marked
comment is updated.

There is no `PUBLISH_FERN` gate in this repository's preview workflow. Live
repository/organization variable values could not be inspected by the authoring
identity (API access denied); do not infer secret availability from source.
Docs owners must confirm configuration and a successful same-repository docs PR
preview/comment after deployment. Operational owner follow-up is tracked separately.

Offline helper regression command:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_fern_preview.py'
```

The fixtures cover content updates, non-executable package/script inputs,
trusted configuration, link rejection, run/attempt binding, fork/stale/closed PR
rejection, and display URL validation. They do not publish or use credentials.

The hosted CLI is pinned to 5.50.4 instead of the mutable local `latest` selector.
Node 22 satisfies the docs package engine requirement. CLI/render compatibility
requires a successful deployment smoke check before rollout is complete.
