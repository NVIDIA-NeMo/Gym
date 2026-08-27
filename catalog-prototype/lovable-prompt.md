# Lovable build prompt

Build a responsive NeMo Gym Environments Hub using `catalog-prototype/catalog.json` as the only data source.

Create a dark, research-oriented interface inspired by an environment registry, while using original NeMo Gym
visual styling rather than copying Prime Intellect branding or assets. The main page should include:

- tabs for All, Benchmarks, and Environments;
- full-text search over name, description, domain, and dataset names;
- filters for domain, modality, integration profile, licensing, lifecycle, and catalog status;
- featured cards for the six entries where `metadata_complete` is true;
- a compact grid for the remaining entries;
- visible `experimental` and `no-manifest` badges, with explanatory tooltips;
- an explicit “Metadata pending” treatment for fields absent from legacy entries;
- responsive layouts for desktop and mobile.

Each entry should open a detail route. The detail page should show description, version, source paths, authors,
composition, datasets, reward range and direction, determinism, canonical split, standard prompt configuration,
licensing, and lifecycle. Hide sections that have no meaningful data. Include copy buttons for these commands:

```bash
gym env validate <name> --kind <kind> --json
gym env start --<kind> <name> --model-type vllm_model
```

Add a client-side Star action backed by local storage. Do not add authentication, a database, hosted evaluations,
or write operations in this prototype. Treat the JSON as immutable and show a helpful empty state for every filter.
