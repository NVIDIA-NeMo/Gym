# GeoBenchX source and data attestation

## Approval

- Evaluation and training: approved by legal dataset review.
- Redistribution of the task records and JSONL files in this repo: under legal
  review. Until it is approved, nothing from the task set is committed (see
  "Benchmark records").
- Conditions for redistribution:
  - Go through the Release Process before publishing.
  - Flag to the legal release PIC that this data includes output of OpenAI and
    Anthropic models, which may be subject to their respective terms of use (see
    "Upstream methodology" below).
- The upstream data snapshot is **not** covered for redistribution; see
  "Not included".

## Benchmark records

The task records are **not committed** until the Release Process is complete.
`task_data.py` downloads them from the pinned upstream commit on first use and checks
the file's SHA-256 before writing it to `upstream/` (gitignored).

- Source: `https://github.com/Solirinai/GeoBenchX`
- Pinned commit: `0edf6109ea5dca3e70fdfe081d39c9ccd0f832fb` (`upstream/COMMIT`)
- Downloaded file: `benchmark_set/tasks_and_reference_solutions.json`, SHA-256
  `ec21a16df85279093e3a599b9b4ecb6906510c549972955eca9b92f2dac64ae3`
- Committed: `upstream/COMMIT` and `upstream/LICENSE` only.
- Upstream license: MIT, copyright 2025 Varvara Krechetova.
- Upstream methodology: seed tasks came from practitioners and publications; additional
  tasks were generated using Claude 3.5 Sonnet, then manually reviewed. Reference solutions
  were drafted with GPT-4o or Claude 3.5 Sonnet and manually reviewed/rewritten. This is
  the OpenAI and Anthropic model output the approval conditions refer to.

## Not included

The 18 statistical, 21 vector, and 11 raster snapshots used by the original executable
tools are **not redistributed**. Upstream publishes them as a separate Google Drive archive
(about 507 MB) and each underlying source has its own terms. The real-tools backend
(`configs/geobenchx_real_tools.yaml`) reads a locally mounted copy; the snapshot, and tool
outputs derived from it, must not be committed to this repo or published with it.

The default Gym tool backend is a deterministic catalog/dataflow simulator, so the
environment runs without the snapshot.

## Generated outputs

`data/benchmark.jsonl`, `data/train.jsonl`, `data/validation.jsonl`, and
`data/example.jsonl` are built locally and not committed. They are mechanical conversions
of the 202 official task records into NemoGym Responses API rows. They add the paper's
system rules, a fixed benchmark date, and tool JSON schemas; they do not add synthetic task
or answer content. `train.jsonl` and
`validation.jsonl` partition the 202 tasks (149 / 53) with the fixed split in
`task_data.split_of`.

Rollout files written by local runs (for example `data/*rollouts*.jsonl`) contain model
outputs and, in real-tools mode, data derived from the snapshot. Keep them out of the
release.

## Paper

Krechetova, V. and Kochedykov, D. (2025), “GeoBenchX: Benchmarking LLMs in Agent Solving
Multistep Geospatial Tasks,” arXiv:2503.18129v2 / ACM GeoGenAgent 2025.
