---
name: add-benchmark
description: >
  Guide for adding a catalog benchmark overlay to NeMo-Gym. Copy the closest
  existing overlay, reuse its shipped scorer, adapt prepare.py, and run gym eval.
  Use when the user asks to add, create, or integrate a benchmark or eval
  that an existing scorer can grade. Triggered by: "add benchmark",
  "integrate benchmark", "add eval", "copy gsm8k". Do not use when
  implementing verify() or running gym env init.
---

# Add Benchmark to NeMo-Gym

## Determine integration type

**Overlay (this skill)** — new dataset on a shipped scorer. Copy the closest folder under `benchmarks/` that already uses that scorer and agent. No `app.py`.

**New `verify()` / tools / state** — stop. `fern/versions/latest/pages/contribute/environments/new-environment.mdx`.

Do not run `gym env init --benchmark` or `--reuse-verifier`. Those write `manifest.yaml`. Catalog overlays do not use that layout.

## Workflow

From the repo root. Gitignored `env.yaml` needs `policy_base_url`, `policy_api_key`, `policy_model_name`. If the copied `prepare.py` reads `hf_token`, add that too. Never commit `env.yaml`. Commits: `git commit -s` (DCO; `-S` optional). Copy SPDX from the overlay you cloned.

### Step 1: Find a scorer and copy an overlay

Bare `gym search "…"` lists environments, not scorers or benchmarks:

```bash
gym search resources-servers "math word problems"
gym list resources-servers
gym search benchmarks math
```

Start from a working overlay that already uses the scorer you need. Common defaults:

| Task | Copy | Scorer |
| --- | --- | --- |
| Math / short answer | `benchmarks/gsm8k` | `math_with_judge` |
| Multiple choice | `benchmarks/gpqa` | `mcqa` |
| Unit-test code | `benchmarks/livecodebench/v5_2408_2502` | `code_gen` |

If none of those match, copy the closest overlay from `gym list benchmarks` (instruction following, ASR, SWE, search, …). Copy that agent and scorer pair. Do not copy GSM8K and then swap in a different agent.

Copy the folder that contains the `config.yaml` you want, into `benchmarks/<new_name>/` (so `--benchmark <new_name>`). For nested sources, copy the nested folder (`v5_2408_2502`), not the parent grouping dir. Extra YAML in the copy (`cascade.yaml`, flavors) are extra overlays — rename them the same way, or delete them if unused.

```bash
cp -r benchmarks/gsm8k benchmarks/my_bench
```

Edit those files in place. Do not scaffold a resources server. Do not replace `config.yaml` / `prepare.py` with a blank template.

```text
benchmarks/my_bench/
├── config.yaml
├── prepare.py       # return Path == jsonl_fpath
├── README.md        # add one if the source had none
└── data/.gitignore  # keep the copied patterns
```

`__init__.py` is optional.

### Step 2: Rename identity

Rename overlay identity only. Do not paste a math YAML over a different scorer's overlay.

**Rewrite** (paths and names inside the copied folder):

- YAML block names and `resources_server.name` (the new block, not the scorer folder)
- Dataset `name`, `jsonl_fpath`, `prepare_script`
- `OUTPUT_FPATH` in `prepare.py`
- README title and run commands

**Leave alone** (they point outside the copied folder, or at upstream data):

- `config_paths` and `_inherit_from` (the scorer you copied)
- Scorer overrides that came with the copy (`grading_mode`, `allowed_agents`, …)
- Shared `prompt_config` paths under `benchmarks/prompts/`
- Imports of sibling packages, e.g. `from benchmarks.livecodebench.prepare_utils import`
- HuggingFace / URL source ids in `prepare.py` (e.g. `Idavidrein/gpqa`)
- The copied `data/.gitignore`

```bash
git grep -n <source_overlay_name> -- benchmarks/my_bench
```

Hits on out-of-folder paths and upstream ids stay. Hits on YAML blocks, dataset paths, and README get renamed.

### Step 3: Prepare data

Keep the copied fetch/transform logic; change output path and your data source. `gym eval prepare` imports `benchmarks.my_bench.prepare` (path `/` → `.`) and calls `prepare()`. The returned Path must equal `jsonl_fpath`.

Row shape follows the overlay you copied — do not invent a schema:

- If `prompt_config` is set, JSONL rows are raw fields. Do not bake `responses_create_params.input`.
- If `prompt_config` is missing or `null`, keep that overlay's baked `responses_create_params` / `verifier_metadata` shape.
- Field names come from the copied `prepare.py` and `resources_servers/<scorer>/` (README or `task_data.py`). Examples: `math_with_judge` uses `question` + `expected_answer`; `mcqa` uses GPQA-shaped `question` / `problem` / `options` / `expected_answer`.

### Step 4: Smoke-run

```bash
gym env validate --benchmark my_bench
gym eval prepare --benchmark my_bench
gym eval run --benchmark my_bench \
  --model-type openai_model \
  --split benchmark \
  --output results/my_bench_rollouts.jsonl \
  --limit 2
```

`--output` and `--split benchmark` are required. `--limit` is tasks; copied `num_repeats` still applies. Do not run `gym env validate my_bench` (positional looks up a manifest). Success prints `Key metrics` with `mean/reward`.

Prefer this `--benchmark` run: it starts servers and applies dataset `prompt_config` when one is set. `--no-serve` does not apply `prompt_config`. Overlay PRs skip `gym env test --resources-server` and do not add `example_rollouts.jsonl` under `benchmarks/`.

### Step 5: Pre-commit and PR

```bash
pre-commit run --files benchmarks/my_bench/**/*
```

If hooks touch other trees, `git checkout --` those paths. README: Apache-2.0 code, data license, upstream URL, and the Step 4 `gym eval run --benchmark` commands. Do not copy the source overlay's `--no-serve` snippet.

Score stability (`gym eval profile`) is optional for the first PR. See `.agents/skills/nemo-gym-reward-profiling`. Profile `--inputs` is the materialized JSONL, not `prepare.py` output.

## Definition of done

- [ ] Closest overlay copied and edited in place; in-folder YAML / README / output paths have no leftover source-overlay name
- [ ] Out-of-folder refs still resolve (scorer inherit, shared prompts, sibling imports such as `prepare_utils`)
- [ ] `prepare()` return path equals `jsonl_fpath`; generated JSONL gitignored
- [ ] `gym eval prepare --benchmark my_bench` and a short `gym eval run` succeed
- [ ] README licenses + Step 4 run commands; SPDX on new Python; `git commit -s`
- [ ] No `app.py`, no new `resources_servers/` tree
