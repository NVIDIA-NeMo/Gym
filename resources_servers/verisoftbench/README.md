# VeriSoftBench benchmark environment

[Dataset](https://huggingface.co/datasets/maxRyeery/VeriSoftBench) (MIT),
[code](https://github.com/utopia-group/VeriSoftBench) (MIT) and
[paper](https://arxiv.org/abs/2602.18307) (Xin, Chen, Durrett and Dillig, 2026).

500 Lean 4 proof obligations drawn from 23 public formal-methods repositories
(compiler verification, type systems, zero-knowledge proofs, smart contracts,
semantic frameworks) — 22 under open-source licences and one, `IntroEffects`,
redistributed by written consent (see Licensing). The model receives curated
context — the definitions and lemmas static analysis found relevant, plus the file
content before the theorem —
and must return the proof body of the target theorem. The proof is inserted into
the repository, compiled with that repository's own Lean toolchain, and accepted only
if the kernel accepts it. A candidate that fails gets upstream's error-feedback
repair prompt and up to three repair rounds.

Pinned revisions:

| Artifact | Pin |
| --- | --- |
| `utopia-group/VeriSoftBench` (harness, prompts, Lean image build, dataset) | `d066523f2b087f28b793c05eacb29bf2e42c0504` |
| `data/verisoftbench.jsonl` at that commit | sha256 `ea584250021fbfaddc060834be2e05b9b2fae4f720c770a9932c71d039a50961` |
| `maxRyeery/VeriSoftBench` (Hugging Face mirror, byte-identical) | `741c2a14e68a5293b70ef874ce8b7876ce053173` |

The 23 Lean repositories and their commits are pinned by upstream in `repos.json` and
`build_config/`; the image build below uses those pins unchanged.

## Scope

| Item | Implemented |
| --- | --- |
| VeriSoftBench-Full, 500 tasks | yes |
| Curated context (`filtered_context`, upstream's primary mode) | yes |
| Full-repository context (`full_context`, `full_context_limited`) | no; `context_mode` records what built a row |
| Repair loop, r = 3 rounds per failed candidate | yes (`verisoftbench_agent`) |
| k candidates per task | `--num-repeats k` at collection time; upstream's headline is k = 8 |
| VeriSoftBench-Aristotle (100-task subset) | rows carry `subset_aristotle`; no separate config |

## Prompting

Both turns are upstream's, built by upstream's own `PromptBuilder` from the pinned
checkout at preparation time:

- System turn: `prompts/templates/init.txt`, one sentence. Upstream sends the same
  system prompt on the initial call and on every repair call.
- User turn: `prompts/templates/user.txt` from `### Input Information` onward, with
  the task's curated context in `<used_lib_defs>`, `<used_repo_defs>`, `<lib_lemmas>`,
  `<repo_lemmas>`, `<local_ctx>` and `<target_theorem>` tags. The template embeds one
  worked example (`prompts/templates/examples/e1.txt`), so the protocol is one-shot.
- Repair turn: `prompts/templates/fix.txt` with `<potential_broken_lemmas>`,
  `<potential_broken_theorem>` and upstream's formatted `<err_msg>`. Its
  `<code_before_error>` section is empty: upstream fills it from `prompts/generated/`,
  a directory the release does not ship, and the released harness behaves the same.

Prompts are long: median 15k characters, maximum 76k. The paper publishes no
decoding parameters; upstream's example configs use temperature 0.7 and 8192 output
tokens, and a run should declare what it used.

## Dataset format

Each prepared row carries:

- `responses_create_params.input` — the system turn and the user turn
- `verifier_metadata` — every field of the upstream row (see `task_data.py` for
  per-field notes), plus `context_mode`. `verify()` reads `id`, `thm_name`, `thm_stmt`,
  `lean_root`, `rel_path`, `imports`, `local_ctx` and `ground_truth_proof` (the last
  only to choose the statement/proof separator, as upstream does); the rest is
  provenance or prompt-only and is carried so a stored rollout can be re-verified
  without the dataset
- `agent_ref` — `{"type": "responses_api_agents", "name": "verisoftbench_agent"}`

## Scoring

A rollout is one of upstream's k candidates and its repair chain. Round 0 verifies
the text inside `<lean4_proof>` (plus optional `<lean4_invented_lemmas>`); a repair
round verifies `<fixed_lean4_thm_proof>` / `<fixed_lean4_lemmas>`, falling back to the
previous round's text when a tag is missing, as upstream does. Verification runs
upstream's code unchanged (`upstream.py`): the verification context is rebuilt from
the source file inside the container, invented-lemma names that clash with the file
are renamed, references to theorems defined *earlier* in the same file (the ones the
prompt's `<local_ctx>` stops before) are renamed so they cannot be used, the file is
assembled with `format_generated_lean`, compiled with `lake env lean` in the
repository's toolchain, and "has already been declared" clashes are resolved once on
round 0 and up to ten times on repair rounds. Upstream detects `sorry`/`admit`
before compiling, still runs the compilation, and combines both into the rejection.

`reward` is 1.0 if the round verified. The agent's final response adds
`solved_without_repair` and `repair_rounds_used`. Every verify response carries a
`status`; a zero reward that is not a judgement on the policy also carries
`failure_reason`, `harness_failure: 1.0` and `mask_sample: true`:

| Status | Meaning | Harness fault |
| --- | --- | :---: |
| `verified` | kernel accepted the proof | no |
| `lean_error` | compiled and rejected; includes `sorry`/`admit` and compile timeouts | no |
| `no_proof_tag` | no proof tag in the reply; round 0 compiles an empty proof, a repair round recompiles the previous one | no |
| `empty_output` | empty reply; compiled as an empty proof | no |
| `backend_error` | Lean container or docker unavailable, or upstream's code raised | yes |
| `bad_metadata` | a required `verifier_metadata` field is missing or mistyped | yes |
| `bad_repair_state` | repair round without the previous round's state | yes |
| `rounds_exhausted` | a round beyond `max_repair_rounds` was requested | yes |

Headline selection: `compute_metrics` groups the rollouts of a task and emits
`any_of_k/solved` — the fraction of tasks with at least one verified candidate or
repair, which is upstream's "pass@k with r repair rounds" — and
`any_of_k/solved_without_repair` (upstream's `proof_success_rate_wo_fix`), in percent.
These are not the HumanEval unbiased pass@k; with exactly k rollouts per task Gym's
`pass@k` coincides with `any_of_k/solved`. Per-category values are emitted as
`any_of_k/solved/<category>` but not promoted. `mean/reward` is the per-rollout
mean, a quantity upstream never reports; it stays in `agent_metrics` and
`key_metrics` as the per-candidate rate.

### Departures from upstream

- **Concurrency.** Upstream verifies a task's k candidates in one process with
  8 threads and then repairs the failed ones in parallel. Here each candidate is an
  independent rollout; `max_concurrent_compilations` (default 8) bounds compilations
  server-wide. Temporary Lean files carry a per-rollout id so concurrent candidates of
  one theorem do not collide.
- **Compile timeout.** Upstream's paper states 300 s. Its released Docker path passes
  600 s to `docker exec` and reports the expiry as "Compilation timeout (>300s)". The
  released value is used unchanged; a timeout counts against the candidate.
- **Leaked-identifier check needs `data/repo_index/`.** Upstream's
  `clean_leaked_identifiers` reads `data/repo_index/<repo>.json` from the checkout,
  which the release does not ship; it is generated inside the image at
  `/workspace/repo_index`. A stock upstream run degrades to a silent no-op with a
  warning when that copy is skipped; here the harness refuses to start unless the
  index covers every repository the dataset names (Setup step 4) and logs a sha256
  over the required index files, so each run records which index it scored under.
  The synthetic example tasks place their theorems in `GymExample.lean`, a file no
  repository contains: they stay deliberately unindexed inside those validated
  files, and no benchmark check is weakened for them.

## Harness validation

No model is involved. `scripts/validate_model_free.py` drives `verify()` in process
against the running container and writes per-task records with denominators:

- `gold`: every shipped ground-truth proof inside `<lean4_proof>`;
- `empty`, `sorry`, `wrong_theorem` (another task's proof), `unparseable`: must not verify;
- `repair`: `sorry` on round 0, then the ground-truth proof as a round-1 repair; must
  verify on round 1 through the repair path.

The results of that sweep are recorded outside this repository with the run they
qualify; this README states the procedure only.

## Setup

```bash
cd resources_servers/verisoftbench
uv venv --python 3.13
uv pip install -r requirements.txt
cd ../..
source resources_servers/verisoftbench/.venv/bin/activate
```

**Every command below runs from the repository root with that environment
active.** The Lean image is about 110 GB and takes over an hour to build; it needs
Docker and network access to GitHub and Mathlib's cache server.

```bash
# 1. pinned upstream checkout + dataset rows (clones on first use)
python resources_servers/verisoftbench/scripts/prepare_verisoftbench.py

# 2. Lean image, from the checkout the preparer created
bash resources_servers/verisoftbench/docker/build_image.sh \
    cache/verisoftbench/VeriSoftBench-d066523 verisoftbench/lean:d066523

# 3. start the container (no ports; the server uses `docker exec`)
docker run -d --name verisoftbench-lean verisoftbench/lean:d066523

# 4. repo index for upstream's leaked-identifier check
docker cp verisoftbench-lean:/workspace/repo_index cache/verisoftbench/VeriSoftBench-d066523/data/repo_index
```

`docker/build_image.sh` assembles upstream's build context and applies three changes to
the *tooling* (not to what is built): one image layer per repository; Lake packages
pre-seeded by SHA before `lake build`, because at least one revision upstream pins
(`quote4` `9371548b`, needed by `iris-lean`) is no longer reachable from any branch and
a stock `lake build` fails on it; and, after `lake build`, every manifest revision
fetched and checked out, because `lake build` left `Cli` in `lean-mlir` at a revision
other than the pinned one and `lake env lean` then failed with git exit 128 on every
lean-mlir task. On an image built before that third step, the same fix can be applied
to a running container:

```bash
docker exec verisoftbench-lean bash -c 'cd /workspace/lean_repos/lean-mlir/.lake/packages/Cli \
  && git fetch --quiet origin d9fc8ae23024be37424a189982c92356e37935c8 \
  && git checkout --quiet --detach d9fc8ae23024be37424a189982c92356e37935c8'
```

Five repositories report partial build failures upstream expects (`formal-snarks-project`,
`loom`, `clean`, `veil`, `lean-mlir`); the model-free sweep records which tasks that costs.

## Quickstart

```bash
# model-free validation (all six sweeps over all 500 tasks)
python resources_servers/verisoftbench/scripts/validate_model_free.py \
    --output results/verisoftbench_validation/records.jsonl

# 1. start servers (leave running)
gym env start \
    --resources-server verisoftbench \
    --model-type openai_model \
    --model "<model id>" \
    --model-url "<openai-compatible base url>" \
    --model-api-key "$API_KEY"

# 2. collect k = 8 candidates per task with three repair rounds each
gym eval run --no-serve \
    --agent verisoftbench_agent \
    --input resources_servers/verisoftbench/data/verisoftbench_full.jsonl \
    --output results/verisoftbench_full.jsonl \
    --num-repeats 8 --temperature 0.7 --max-output-tokens 8192
```

Regenerate the committed example artifacts. `example.jsonl` is five synthetic rows
written for this repository (elementary `Nat`, `Bool` and `List` facts placed at the
root of five repositories the image contains); the benchmark's own theorems are
third-party content and are not redistributed here. Because upstream's prompt
template embeds its one-shot worked example (`prompts/templates/examples/e1.txt`,
MIT) in every user prompt, that upstream text does appear verbatim inside the
tracked example files; it is attributed in the repository-level `ATTRIBUTIONS.md`.

One script writes all five tracked files under `data/` — `example.jsonl`,
`example_metrics.json`, `example_rollouts.jsonl`,
`example_rollouts_materialized_inputs.jsonl` and
`example_rollouts_aggregate_metrics.json` — deterministically: no model calls, no
servers, no Lean container. The rollout files rebuild one recorded smoke run from
the hand-checked outcomes embedded in the script, with every service identifier,
provider route and timestamp a fixed placeholder; the aggregate file goes through
the server's own `compute_metrics`, so it shows the `any_of_k/*` headline. The
tests assert the committed copies match the generator byte for byte. The collate
gate below only validates `example.jsonl` against its metrics sidecar; it rewrites
nothing new.

```bash
python resources_servers/verisoftbench/scripts/make_example_data.py
gym dataset collate \
    "+config_paths=[resources_servers/verisoftbench/configs/verisoftbench.yaml]" \
    +output_dirpath=resources_servers/verisoftbench/data \
    +mode=example_validation
```

## Tests

```bash
ng_test +entrypoint=resources_servers/verisoftbench
ng_test +entrypoint=responses_api_agents/verisoftbench_agent
```

## Licensing

Code: Apache 2.0.

VeriSoftBench harness and dataset: MIT (`LICENSE` of utopia-group/VeriSoftBench;
the Hugging Face dataset card also declares MIT). The harness is not vendored; it is
cloned at the pinned commit at preparation time. `docker/Dockerfile` and
`docker/build_repo.sh` are derived from upstream's `Dockerfile` and
`scripts/build_repo.sh` (MIT) with the changes described above; each derived file
reproduces the complete upstream MIT permission notice at the top.

The 23 Lean repositories compiled into the image retain their own licences
(Apache-2.0, MIT, BSD and one "written consent" repository, `IntroEffects`, which is
not under an open-source licence); upstream lists them in
`THIRD_PARTY_NOTICES.txt`, which the image carries together with the licence files
under `/workspace/THIRD_PARTY_LICENSES/`. Task and corpus content (theorem
statements, file context, reference proofs) is drawn from those repositories and is
downloaded at run time, not committed here. The one piece of upstream content that
is committed: the prompt template embeds upstream's one-shot worked example
(`prompts/templates/examples/e1.txt`, MIT) in every user prompt, so it appears
verbatim inside the tracked synthetic example files under `data/`; it is attributed
in the repository-level `ATTRIBUTIONS.md`.
