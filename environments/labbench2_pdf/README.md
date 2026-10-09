# LabBench2 direct-PDF retrieval environment

This environment runs LitQA3, FigQA2, and TableQA2 through NeMo Gym's
[general Harbor agent](../../responses_api_agents/harbor_agent_general) with
OpenCode as the Harbor agent. Each task gives OpenCode a read-only,
benchmark-relevant PDF corpus and asks it to write a short answer to
`/app/answer.txt`. Harbor owns the model call, sandbox, and task-local verifier;
the Gym bridge converts Harbor's ATIF trajectory and reward to Gym output.

This is the direct-access condition from
`litqa3-rag-agent-native`, intentionally narrowed to one experiment:

- corpus scope is always `relevant` (the union of source papers for one benchmark);
- no agent skill is mounted or mentioned;
- no page-text cache or retrieval helper is mounted;
- source URLs, source filenames, and gold answers are absent from agent-visible files;
- the runtime provides ordinary PDF, OCR, image, and data-analysis packages.

The source checkout currently contains `tableqa2`, not `tableqa3`, so this
integration preserves the source benchmark name.

For the cataloged workflow (`gym eval prepare --benchmark labbench2_pdf` and
`gym eval run --benchmark labbench2_pdf`), start with the
[benchmark README](../../benchmarks/labbench2_pdf/README.md). The commands below
remain useful when developing or debugging the environment directly.

## Prepare data and Harbor tasks

Run the preparation command from the Gym repository root:

```bash
python environments/labbench2_pdf/prepare_data.py \
  --environment-type docker \
  --build-docker-image \
  --overwrite
```

This is the reproducible default path. It:

1. downloads the three question configs from
   `EdisonScientific/labbench2` at the pinned revision recorded in
   `download_questions.py`;
2. resolves and downloads every distinct source DOI it can find into the
   gitignored `data/source/papers` cache;
3. audits PDF coverage and skips questions that do not have every listed
   source PDF; and
4. materializes the remaining questions as a Harbor registry and Gym input
   JSONL files under `data/tasks_docker`.

Both download caches are validated and reused on subsequent runs. Task names
contain the original position in each upstream benchmark, so a newly available
PDF cannot silently renumber existing task IDs.

### Use an existing local snapshot

Pass the standalone question and PDF directories to bypass downloading completely:

```bash
python environments/labbench2_pdf/prepare_data.py \
  --questions-dir /path/to/litqa3-rag-agent-native/data/questions \
  --papers-dir /path/to/litqa3-rag-agent-native/data/all_papers \
  --environment-type docker \
  --build-docker-image \
  --overwrite
```

The supplied directories are read-only inputs: the preparation command neither
writes to them nor invokes a question or DOI downloader. To supply only the
paper snapshot, omit `--questions-dir`; preparation then downloads or reuses the
pinned question snapshot before checking the PDFs and filtering the generated
tasks.

The strict default is `--source-policy all`: every DOI listed for a question
must resolve to a local PDF. `--source-policy any` retains the standalone
materializer's permissive behavior for multi-source questions. Use
`--missing-paper error` when incomplete coverage should fail preparation rather
than skip rows. The exact decisions are recorded in `coverage_audit.json`.

### Public PDF acquisition

The environment includes a DOI-based downloader so PDFs do not need to be
redistributed with Gym. It checks the current PMC Open Access Cloud dataset,
Europe PMC, OpenAlex open-access locations, Crossref-declared publisher links,
and public publisher/DOI landing pages. It accepts only responses that validate
as PDFs. It does not send credentials or cookies, bypass paywalls, or scrape
search-engine results.

`prepare_data.py` invokes the downloader automatically when `--papers-dir` is
omitted. The individual stages can also be run for inspection or recovery:

```bash
python environments/labbench2_pdf/download_questions.py

python environments/labbench2_pdf/download_papers.py \
  --questions-dir environments/labbench2_pdf/data/source/questions
```

PDFs and `download_manifest.json` are written by default to the gitignored
`environments/labbench2_pdf/data/source/papers` directory. The manifest records
the successful resolver, checksum, failed attempts, and question-level coverage.
Pass `--unpaywall-email you@example.com` to add Unpaywall's public OA locations;
an email is required by that service. Re-running the command validates and
reuses cached PDFs and recorded failures. Use `--retry-failed` to retry only
the failures, or `--overwrite-papers` with `prepare_data.py` (`--overwrite`
with `download_papers.py`) to redownload everything. OpenAlex also
offers a cached full-text service for records with `content_urls.pdf`; opt in
with `OPENALEX_API_KEY=...` and `--openalex-content`. Only works marked open
access are eligible. OpenAlex content downloads may be metered, so that route
is never enabled implicitly.

## Generated layout

`prepare_data.py` obtains or selects the inputs and delegates task packaging to
`prepare.py`:

```text
Hugging Face questions or --questions-dir
public DOI downloads or --papers-dir
                    |
                    v
              prepare_data.py
                    |
                    +-- preparation_manifest.json
                    +-- registry.json
                    +-- example_input.jsonl
                    +-- *_input.jsonl
                    +-- _corpus_views/<benchmark>/papers/*.pdf
                    `-- <task>/
                          +-- instruction.md
                          +-- task.toml
                          +-- environment/
                          +-- tests/
                          `-- solution/
```

By default, each unique selected PDF is copied once into a generated corpus
store. Benchmark views and Singularity task views are hard-linked to that store.
Docker binds the benchmark view directly at `/papers`; the existing Harbor
Singularity bootstrap exposes its task view at the same path. Neither profile
copies the corpus once per task.

For a smaller packaging smoke test with one task from each benchmark, add:

```bash
python environments/labbench2_pdf/prepare_data.py \
  --limit-per-benchmark 1 \
  --overwrite
```

To prepare exactly one FigQA2 task, select that benchmark as well:

```bash
python environments/labbench2_pdf/prepare_data.py \
  --benchmarks figqa2 \
  --limit-per-benchmark 1 \
  --overwrite
```

The default output is `environments/labbench2_pdf/data/tasks_docker`. Omit
`--limit-per-benchmark` to materialize all currently answerable rows. The
standalone snapshot has 168 LitQA3, 101 FigQA2, and 100 TableQA2 rows; its
strict `all` policy produces 158, 100, and 99 tasks respectively. It skips the
12 incomplete rows, including two that have only some of their source PDFs, and
records each missing DOI in `coverage_audit.json`. Those counts apply to the
supplied standalone paper snapshot. A public DOI download usually produces a
smaller set and can change as open-access locations appear or disappear; use
`preparation_manifest.json` and `coverage_audit.json` as the authoritative
counts for each run.

The safe default, `--corpus-view-mode copy`, works across filesystems and keeps
the external corpus independent from generated artifacts. For a trusted source
on the same filesystem, `--corpus-view-mode hardlink` enables external
zero-copy storage. Each unique selected PDF is stored only once even when it is
relevant to multiple benchmarks.

The generated files include:

- `rollout_input.jsonl`: all materialized tasks;
- `example_input.jsonl`: up to five deterministic, benchmark-balanced smoke tasks;
- `litqa3_input.jsonl`, `figqa2_input.jsonl`, and `tableqa2_input.jsonl`;
- `preparation_manifest.json`: whether questions and papers were downloaded or supplied;
- `materialization_manifest.json`: immutable setup metadata and task list;
- `coverage_audit.json`: developer-only source-to-paper resolution details.

The manifests and coverage audit remain outside every task container. Gold
answers exist only inside each task's hidden verifier and Oracle files.

## Configure policy and judge endpoints

Configure the OpenAI-compatible policy endpoint and judge in the root,
gitignored `env.yaml`:

```yaml
policy_base_url: https://your-policy-endpoint/v1
policy_api_key: your-policy-key
policy_model_name: your-policy-model

judge_base_url: https://your-judge-endpoint/v1
judge_api_key: your-judge-key
judge_model_name: your-judge-model
```

Keep `policy_model_name` equal to the complete model ID expected by the policy
endpoint. The environment adds OpenCode's `openai/` provider prefix around that
value; OpenCode consumes only that added segment and sends the existing Gym
model name to the endpoint unchanged.

The task-local verifier first checks normalized exact and numeric equivalence,
then calls the configured judge for non-exact answers. The environment passes
the policy and judge settings only to Harbor's Ray worker. Harbor resolves the
`${OPENAI_*}` and `${JUDGE_*}` references there, so the values are not embedded
in a Harbor task or job config and do not need to be exported separately before
`gym env start`. Harbor-compatible numeric rewards are written to
`verifier/reward.json`; diagnostic fields are written separately to
`verifier/details.json`. An unavailable judge is marked there as an
infrastructure error unless the deterministic check already proved the answer
correct.

## Run a Docker evaluation

The committed Docker configuration is
[`config.yaml`](./config.yaml). It selects OpenCode, the Docker environment, and
the generated `data/tasks_docker` Harbor registry. A single-task rollout does
not need another YAML config: the input JSONL selects which task names from that
registry Gym dispatches.

Start Gym with the Docker profile:

```bash
gym env start --environment labbench2_pdf
```

This profile does not start a Gym model server. OpenCode talks to the configured
policy endpoint from inside the Harbor trial.

Then collect the committed three-task smoke input (one task per benchmark):

```bash
gym eval run --no-serve \
  --agent harbor_agent_general \
  --input environments/labbench2_pdf/data/example.jsonl \
  --output results/labbench2_pdf/example_rollouts.jsonl \
  --concurrency 1
```

For a complete single-benchmark run, replace the input with the corresponding
generated file, for example:

```bash
gym eval run --no-serve \
  --agent harbor_agent_general \
  --input environments/labbench2_pdf/data/tasks_docker/litqa3_input.jsonl \
  --output results/labbench2_pdf/litqa3_rollouts.jsonl
```

For exactly one real FigQA2 rollout, derive a temporary one-row input from the
generated registry and keep using the same running environment:

```bash
sed -n '1p' \
  environments/labbench2_pdf/data/tasks_docker/figqa2_input.jsonl \
  > /tmp/labbench2_figqa2_single.jsonl

gym eval run --no-serve \
  --agent harbor_agent_general \
  --input /tmp/labbench2_figqa2_single.jsonl \
  --output results/labbench2_pdf/figqa2_opencode_single.jsonl \
  --concurrency 1
```

Gym writes the converted rollout to the requested output. Harbor's lossless
ATIF trajectory remains under the configured
`results/labbench2_pdf/harbor_jobs/<job>/<trial>/agent/trajectory.json`. Inspect
that file to understand how the agent used the supplied PDFs. The agent may use
text extraction, page rendering, image crops, OCR, or any combination it chooses.

## Prepare and run Singularity tasks

The shared Harbor Singularity environment accepts either an absolute `.sif`
path or a Docker registry reference in each task's `docker_image` field. Build
the same Dockerfile, publish it to a registry reachable from the cluster (or
convert it to a `.sif`), then materialize with that reference:

```bash
python environments/labbench2_pdf/prepare_data.py \
  --papers-dir /path/to/litqa3-rag-agent-native/data/all_papers \
  --environment-type singularity \
  --image registry.example.com/team/nemo-gym-labbench2-pdf:2.0 \
  --limit-per-benchmark 1 \
  --overwrite
```

Use an absolute `.sif` path for `--image` when conversion has already happened.
Then start the explicit Singularity config:

```bash
gym env start \
  --config environments/labbench2_pdf/config_singularity.yaml
```

The generated input files live under
`environments/labbench2_pdf/data/tasks_singularity`.

## Validation

Run the materializer and verifier tests without a container runtime:

```bash
pytest environments/labbench2_pdf/tests -q
```

For the required end-to-end environment validation, run the three-task smoke
through Docker (or Singularity), inspect Harbor's raw trial artifacts under
`results/labbench2_pdf/harbor_jobs`, and compare the resulting scores with the
same direct/relevant/no-skill condition in the standalone checkout.

The task package itself can be checked without a model or judge request:

```bash
JUDGE_API_KEY= JUDGE_BASE_URL= JUDGE_MODEL= \
  harbor run \
  --path environments/labbench2_pdf/data/tasks_docker \
  --include-task-name 'litqa3-*' \
  --n-tasks 1 \
  --agent oracle \
  --env docker \
  --no-delete \
  --yes
```

## Licensing

Integration code is Apache-2.0. The question snapshots and papers are external,
generated inputs and are not committed here. Downloading an unauthenticated PDF
does not grant redistribution rights; retain the per-paper source and license
metadata and review those rights before publishing any corpus or derived dataset.
