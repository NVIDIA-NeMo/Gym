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
`is_criteria_true` JSON response follow raw Archipelago commit `b581241ed67deed520f8d7854d20e39569e259f5`,
without the Apex harness's compatibility patches. The implementation is intentionally Apex-specific: there is no generic
Archipelago runner, evaluator registry, metrics layer, LiteLLM dependency, external document API, or Reducto call.

Text is extracted locally from common documents, spreadsheets, presentations, PDFs, text files, and application
SQLite state. A workbook's chartsheets have no cells; each is kept as a sheet so that a change to its chart counts. Supported deliverables are also rendered for multimodal judging when a document converter is configured.
Presentations, spreadsheets, and PDFs are flattened into changed slides, sheets, and pages, and the visual evidence
is the changed ones: the changed PDF pages, the changed slides of a `.pptx`, and the changed sheets of a workbook
(exported one sheet per page), up to 10 images per file. A changed sheet is shown that way only while its single
page stays readable, at most 1200 pt on its longer side (about 60 rows); a workbook with a longer or wider changed
sheet is shown by the first 10 pages of the ordinary paginated export. A file with no known changed part, such as a
`.docx`, is also shown by its first 10 pages. A file whose bytes changed while every slide, sheet and page reads the same (speaker
notes, formatting, comments) is kept as one file-level change; byte-identical files are not changes.
For spreadsheets with uncached formulas, the verifier re-saves the workbook through LibreOffice before extraction;
workbooks containing charts are rendered to PDF pages and attached with Archipelago-compatible `[CHART_N]` metadata.
`.doc`, `.odt`, `.ods`, `.ppt` and `.odp` deliverables have no extractor of their own: LibreOffice converts them to
`.docx`/`.xlsx`/`.pptx`, which are then read as above. Without a converter they are described to the judge by name
and size only.

LibreOffice and Poppler open files the model wrote, so they run inside a sandbox image, never on the Resources host
by default. `document_converter_image` defaults to the agent's image (`apex_agents_image`, else
`responses_api_agents/apex_agent/images/archipelago.sif`), and each conversion runs as
`apptainer exec --cleanenv --containall` with a single scratch directory bound and no host `/tmp`, home or working
directory. The container still shares the host network. The image may be a local image file of any format apptainer
runs (`.sif`, `.sqsh`; a relative path is taken from the repository root), a `docker://`, `oras://` or `library://`
URI, which is passed to apptainer unchanged, or a bare OCI reference such as `registry/name:tag`, which is passed as
`docker://registry/name:tag`.

The server checks the converter when it starts: if an image is configured and `apptainer` is not on `PATH`, or a
local image file does not exist, startup fails with the reason. There are three ways out:

- point `document_converter_image` at an image this host can run with apptainer;
- set `allow_host_document_tools: true` (`apex_allow_host_document_tools=true`) to use LibreOffice/Poppler installed
  in the outer Gym container. They are then preferred over the image, and model-written documents are processed
  outside any sandbox;
- set `document_converter_image: null`. No conversion is then attempted: Office and PDF files are graded from their
  extracted text only, with no page images and no formula recalculation, and a verify is never masked for a missing
  converter.

A converter that passed the startup check and still cannot start at verify time (apptainer fails to enter the image)
makes that verify a masked verifier fault (`verifier_error`), not a text-only grading. A file LibreOffice itself
cannot convert is graded on its extracted text. So is a file whose conversion times out (120 s), since a file the
model wrote can cause that; the timeout is logged and listed in `judge_response.document_conversion_timeouts` with
the tool and file name.

The verifier bounds what one submission can cost it. Each changed file is read once per verify, not once per
criterion, and its rendering is shared by the criteria. At most 4,000,000 characters of a file's text are kept, a
PDF, legacy workbook or notebook over 256 MiB is not parsed, and an Office package is opened only if no part of it
expands past 256 MiB and the whole stays under 1 GiB; a file over a bound is described to the judge instead of
loaded. Extraction, conversion and persistence run on a pool of `apex_verifier_concurrency` worker threads, so the
server keeps answering other requests meanwhile.

The grading prompt is budgeted at two characters per token against `apex_judge_context_window_size`, and images are
budgeted with it (a fixed token cost each, at most a quarter of the window, 60 images and 16 MiB encoded). If the
judge still refuses a request as too large (HTTP 413, or HTTP 400 naming a context-length or payload limit) and the
request really was large, that criterion scores 0: the size is the policy's output and the refusal would repeat.
"Large" means the request's estimated prompt tokens, text plus the image budget, exceed half of
`apex_judge_context_window_size`, or, for an HTTP 413, its encoded body exceeds 1 MiB. A small request refused in the
same words (an output cap above the judge's window, a proxy's payload limit) is a masked judge failure, as are
connection errors, timeouts and 5xx replies. The judge model server must forward those statuses for this to apply; the
shipped `apex_judge_model` config sets `propagate_upstream_http_status_codes` accordingly.
`judge_model_server`, `judge_model`, and `judge_create_params_overrides` select the LLM-as-judge and its request
parameters. `apex_judge_context_window_size` defaults to Gemini's 1,000,000-token model context and is shared by
artifact selection and final grading; there is no separate 32K grading-prompt cap.
No judge receives the agent trajectory.

The verify response follows the same three-part contract as the upstream wrapper:

- `reward`: `1` only when every rubric criterion passed; otherwise `0`. Rows whose `verifier_metadata.scoring` is
  `weighted_average` instead get the weighted mean of their criterion scores, using each rubric item's optional
  `weight` (default 1; zero-weight criteria are skipped), as in Archipelago's weighted-average scoring methods.
- `rubric_scores`: the criterion-level score/status/message/value mapping keyed by verifier ID. Every criterion
  counts toward the reward even when two share an ID; the repeat is keyed `<id>#2`.
- `judge_response`: grading metadata including `ok`, `grading_run_id`, `status`, `scoring`, `verifier_count`, and
  `document_extraction`.

The fractional criterion pass rate, the all-pass result, and the weighted score are retained in grading metadata for
analysis whichever scoring method sets the reward.

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
