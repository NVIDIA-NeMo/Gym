# SWE-Together

Run the 109 pinned repository tasks through the interactive Environment Server, SWE-Together Resources, and an independently selected candidate harness. The default composition selects OpenCode 1.15.13; the fixed evaluator uses Claude Code 2.1.108. This integration remains experimental until reference baselines are reviewed.

Prepare a checkout of [the pinned official source](https://github.com/Togetherbench/SWE-Together/tree/891d19eb4b3a64a47c3d49bbd066a311e0133254) and a JSON image inventory mapping each task ID to `image`, immutable `image_digest`, and `workdir`. Other inventory metadata is allowed. Assets are hash-checked against `manifest.json` and remain on the Resources host.

```bash
export SWE_TOGETHER_SOURCE=/path/to/SWE-Together
export SWE_TOGETHER_IMAGES=/path/to/image-pins.json
python -m benchmarks.swe_together.prepare
```

The same importer supports `gym eval prepare --benchmark swe_together` and explicit `--source`, `--images`, `--output`, and repeatable `--task` arguments. `config.yaml` selects `recipes/reference.yaml`; the recipe supplies model/provider definitions, runtime paths, and qualification settings. See the [Resources documentation](../../resources_servers/swe_together/README.md) for simulator behavior, scoring profiles, and retained artifacts.

The benchmark source and adapted protocol are Apache-2.0; task containers retain their component licenses. The Resources verifier fixture exercises frozen-goal score derivation and malformed verdict handling. Agentic judge completion, image health, and interactive continuation require separate live qualification.
