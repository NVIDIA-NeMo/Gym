# Apex Agents resources server

This server owns the held-out side of the [APEX–Agents](https://huggingface.co/datasets/mercor/apex-agents)
evaluation: rubric metadata, artifact preprocessing, criterion-level LLM judging, and reward aggregation.
The public dataset is CC-BY-4.0 and is restricted by its owner to evaluation use.

The paired agent is `responses_api_agents/apex_agent`. The data and runtime contract is intentionally
agent-independent: any future agent can submit a final Responses API response plus an artifact snapshot
with paths rooted at `filesystem/` or `.apps_data/`.

## Data preparation

Download `tasks_and_rubrics.json` and `world_descriptions.json` after accepting the gated dataset terms,
then convert them:

```bash
python benchmarks/apex_agents/prepare.py \
  --tasks /path/to/tasks_and_rubrics.json \
  --worlds /path/to/world_descriptions.json \
  --output resources_servers/apex_agents/data/apex_agents_validation.jsonl
```

The converter puts only the prompt, task/world identity, domain, and required optional services in top-level
agent fields. `expected_output`, `rubric`, `gold_response`, and `gold_response_type` remain under
`verifier_metadata`. The Gym agent never serializes that metadata into its task sandbox. During conversion,
`expected_output` is mapped to a held-out grading target on every criterion. That target determines whether the
criterion is graded from the console response, a required changed-file type, or both.

World ZIPs are environment state rather than JSONL payloads. Benchmark preprocessing downloads every unique
`world_files_zipped/<world_id>.zip` into `benchmarks/apex_agents/data/world_cache`. During rollout execution, the
resources server calls `hf_hub_download(..., local_files_only=True)` against that cache. It never contacts Hugging
Face. The agent server receives the cached ZIP from `/world`, places it in a temporary host directory, then uploads
it into the per-task sandbox.

The dataset is gated. Accept its terms with the Hugging Face account that will run Gym, create a read token, put it
in the root `env.yaml` as `hf_token`, and run:

```bash
gym eval prepare --benchmark apex_agents
```

That one preprocessing command downloads the two index files, converts the 480 tasks into Gym rows, and downloads
all world ZIPs. The token is used only by the preprocessing process; it is never passed to the resources server,
agent, or sandbox. Environment startup fails immediately if the offline cache is missing, and a missing individual
world reports the same preparation command.

The lower-level converter performs the same complete preparation when invoked directly:

```bash
python benchmarks/apex_agents/prepare.py \
  --tasks /path/to/tasks_and_rubrics.json \
  --worlds /path/to/world_descriptions.json \
  --output benchmarks/apex_agents/data/apex_agents_benchmark.jsonl \
  --world-cache-dir benchmarks/apex_agents/data/world_cache
```

Set `apex_world_cache_dir` to the same path when workers should use a shared cache.

## Verification

`verify()` compares the safely extracted initial and final snapshots, renders the changed deliverables, and grades
each held-out APEX criterion with Gym's configured judge model. The grading system prompt and
`is_criteria_true` response schema follow the output grader pinned by Apex harness commit
`1fd94befbb570eb6effe76b1895e5d599e820227`. The implementation is intentionally Apex-specific: there is no generic
Archipelago runner, evaluator registry, metrics layer, LiteLLM dependency, external document API, or Reducto call.

Text is extracted locally from common documents, spreadsheets, presentations, PDFs, text files, and application
SQLite state. Supported deliverables are also rendered for multimodal judging when LibreOffice is available.
`judge_model_server`, `judge_model`, and `judge_create_params_overrides` select the LLM-as-judge and its request
parameters. `apex_judge_context_window_size`, defaulting to `32768`, bounds the artifact text included in each prompt.
No judge receives the agent trajectory.

The verify response follows the same three-part contract as the upstream wrapper:

- `reward`: the fraction of rubric criteria that passed.
- `rubric_scores`: the criterion-level score/status/message/value mapping keyed by verifier ID.
- `judge_response`: grading metadata including `ok`, `grading_run_id`, `status`, `scoring`, `verifier_count`, and
  `document_extraction`.

Final Pass@1 conversion, if required, is deliberately outside this judging adapter.

The snapshot is excluded from the verify response so large Office artifacts are not embedded in rollout
JSONL files. Consequently this initial integration declares re-verification unsupported.

Human-inspectable outputs are retained on the trusted host by default under `results/apex_agents_artifacts`. Set
`apex_artifact_output_dir` to choose another location, or to `null` to disable persistence. Each submission is written
under `<root>/<task_id>/rollout_<index>_attempt_<index>_<id>/` with:

- `initial_snapshot.zip`: the rubric-free state before the agent runs.
- `final_snapshot.zip`: the rubric-free state after the agent runs.
- `artifacts/`: the safely extracted changed files under `filesystem/` and `.apps_data/`.
- `submission.json`: final response, trajectory (including any model-emitted reasoning), tool calls/results, token
  usage, and task identity. Held-out `verifier_metadata` and the inline snapshot payload are explicitly omitted.
- `grading.json`: reward, rubric scores, and judge run metadata, or the grading error.

The response returns `artifact_output_dir`, `initial_snapshot_path`, and `final_snapshot_path`; `artifact_paths`
separately lists changed files found inside the snapshot. The ZIP bytes themselves are excluded from the response so
large Office artifacts are not embedded in rollout JSONL files.

# Licensing information

Code: Apache 2.0

Data: CC-BY-4.0, evaluation-only use restriction from the dataset owner
