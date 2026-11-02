# GDPVal resources server

Scores deliverables produced by the Stirrup agent on the GDPVal benchmark.

Two modes via `reward_mode` config:

- `rubric` (default) — LLM judge scores each deliverable against a per-task
  rubric, reward in `[0.0, 1.0]`.
- `comparison` — pairwise judge compares eval deliverable vs. one or more
  reference rollouts (`reference_deliverables_dir`, or `reference_models` for
  multi-reference), reward in `{0.0, 0.5, 1.0}`. `aggregate_metrics` reduces to
  an ELO rating.

Comparison mode also supports **multi-stage adaptive ELO** — a sequence of
stages that judge sampled tasks against an adaptively-chosen reference subset,
enabled with `++multistage.enabled=true`. It is implemented in
`multistage_orchestrator.py` (pure logic in `multistage_elo.py`) and runs through
the standard `gym eval run` pipeline. See the "Run multi-stage adaptive ELO"
section of `benchmarks/gdpval/README.md`.

Canonical entry point is the benchmark at `benchmarks/gdpval/`:

```bash
gym eval prepare --benchmark gdpval
gym eval run \
  --model-type vllm_model \
  --benchmark gdpval \
  --split benchmark
```

See `benchmarks/gdpval/README.md` for the full run recipe.

## Task tools

The server serves the agent's task tools as `POST` routes. `finish` and `abandon_task_finish` record how the
agent ended the task. `web_search` searches with Tavily and `fetch_web_page` extracts a page as markdown.
Set `tavily_api_key` (the shipped configs read `TAVILY_API_KEY`).

## Sandbox image

Each task runs in its own sandbox, started from `sandbox_config.image` through the provider named by
`sandbox_provider` (e.g. `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml`). The image is the
GDPval-AA v2 environment that `containers/Dockerfile` defines. Gym ships the Dockerfile, not an image, so build and
push one before running:

```bash
IMAGE=<registry>/gdpval-sandbox:<tag>
docker build -t "$IMAGE" resources_servers/gdpval/containers
docker run --rm "$IMAGE" python /opt/gdpval/verify_gdpval_sandbox.py
docker push "$IMAGE"
export GDPVAL_SANDBOX_IMAGE="$IMAGE"
```

The verifier audits the image against the vendored GDPval-AA v2 manifests and exercises the toolchain. Build for the
architecture your sandboxes run on; an arm64 build leaves out the pins in
`containers/gdpval_aa_v2_arm64_exclusions.txt`. For a private registry, also set
`sandbox_config.provider_options.image_auth: {username: ..., password: ...}`, and select the architecture with
`sandbox_config.provider_options.platform: {os: linux, arch: arm64}` when it is not the provider's default.

## Office to PDF conversion

The judge reads Office deliverables as PDFs that LibreOffice renders. Comparison mode preconverts them beside the
source files (`preconvert_office_to_pdf`), and rubric mode renders the ones without a PDF into a temp directory. By
default the server runs `libreoffice` from `PATH`, and with `preconvert_office_to_pdf: true` startup installs it and a
Java runtime with `apt-get` on Linux when they are missing or broken. Startup then converts a small workbook with a
formula and a chart, and fails in comparison mode if that conversion fails.

To run LibreOffice from a container or a wrapper script instead, set `libreoffice_command` to the command that replaces
`libreoffice`. For example, to use Apptainer, set it like this in `configs/gdpval.yaml`, next to
`persist_deliverables_dir`:

```yaml
libreoffice_command: [apptainer, exec, --bind, /path/to/data, --bind, /tmp, /path/to/libreoffice.sif, libreoffice]
```

`/path/to/libreoffice.sif` can be any Apptainer image that has LibreOffice. To make one from a Docker image, run
`apptainer build libreoffice.sif docker://<image>`.

The command must see these directories at the same absolute paths:

- `persist_deliverables_dir`;
- the reference deliverables;
- the temp directory (`TMPDIR`, or `/tmp`);
- any other deliverables directory that `/verify` gets, such as an earlier run's when `gym eval reverify` re-judges it.

Startup runs the command with `--version` and converts a small workbook in the temp directory, under
`persist_deliverables_dir` and under each reference `deliverables_dir`. It fails if any of these fails, if the
conversion leaves files that the server cannot delete, or if a reference `deliverables_dir` does not exist. A reference
directory that the server cannot write is not checked, so its Office files need their PDFs already, and neither are
other deliverables directories. If the command cannot see one of them, `.docx`, `.xlsx` and `.pptx` files
there can still convert, without a warning, from a copy that a Python library re-saves and that can lose content such
as shapes in a workbook; `.doc`, `.xls` and `.ppt` files can fail. The `apt-get` install is skipped when the command is
set.

The PDFs depend on the command's LibreOffice version and fonts, so they can look different from a host install's.
Comparison mode keeps each PDF next to its source file and reuses it in later runs.

A Docker command also needs:

- `--user` with the server's uid and gid, so that LibreOffice does not run as root and leave files that the server
  cannot delete. Leave it out with rootless Docker or Podman, which already run containers as your user;
- `--rm`, so that each conversion's container is deleted;
- a time limit inside the container, such as `timeout -k 5 100`, and `--init`, without which the container may not stop
  when that limit fires. Gym gives each LibreOffice run 120 s, but when that runs out it stops only the `docker`
  client, and the container keeps running.

For example, with Docker:

```yaml
libreoffice_command: [docker, run, --rm, --init, --user, "<uid>:<gid>", -v, /path/to/data:/path/to/data, -v, /tmp:/tmp,
                      libreoffice-image, timeout, -k, "5", "100", libreoffice]
```

## Judge failure telemetry

Comparison judging can optionally write structured request and failure metadata:

```yaml
judge_telemetry_output_dir: /path/to/run/judge_telemetry
```

Each resources-server process writes a separate `judge-events.<pid>.jsonl` file.
Events correlate preflight exclusions, provider attempts, retries, recovery, and
invalid responses with the task, reference, judge, request size, file counts,
relative file labels, media types, and attachment sizes. Prompts, attachment
contents, base64 payloads, credentials, and absolute paths are never recorded.
Telemetry is best-effort and does not alter judge routing, retries, scores, or
ELO calculation.

Summarize one file or a directory of process-local files with:

```bash
python -m resources_servers.gdpval.judge_telemetry /path/to/run/judge_telemetry
```
