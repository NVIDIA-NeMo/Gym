# GeoBenchX NemoGym

NemoGym implementation of
[GeoBenchX](https://github.com/Solirinai/GeoBenchX), the benchmark from
“GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks”
([arXiv:2503.18129](https://arxiv.org/abs/2503.18129)).

## Scope

- All 202 official tasks and all annotated alternative reference solutions.
- Four paper groups: Merge/Visualize; Process/Merge/Visualize; Spatial Operations;
  Heatmaps/Contour Lines.
- The released interface's 23 GIS tools plus the additional `reject_task` tool
  (24 callable tools total).
- At most 25 tool calls and the paper's solvability rules.
- Match score `0` (no match), `1` (partial), or `2` (full), exposed as reward
  `0`, `0.5`, or `1`.
- Solution-length metrics for regression and efficiency analysis.

The paper used a panel of three LLM judges. This Gym defaults to a deterministic
reference matcher so repeated release-regression runs are reproducible. It ignores
temporary dataframe handle names, tolerates extra steps after a complete solution,
and treats colormap, NoData, and spatial-predicate differences as partial matches.
The official references remain vendored, so a panel-judge verifier can be added
without changing the dataset.

## Build the datasets

```bash
.venv/bin/python -m resources_servers.geobenchx.task_data
```

The task records are not committed. On first use, `task_data` downloads
`tasks_and_reference_solutions.json` from the upstream commit pinned in
`upstream/COMMIT`, checks its SHA-256, and saves it to `upstream/`. This needs network
access once; later runs, the server and the tests reuse the saved file.

This writes four files to `data/`:

| File | Rows | Use |
| --- | --- | --- |
| `benchmark.jsonl` | 202 | Full benchmark, for baselines |
| `train.jsonl` | 149 | GRPO training only |
| `validation.jsonl` | 53 | Held-out evaluation before and after training |
| `example.jsonl` | 5 | First five training rows, for smoke tests |

The split is fixed (`task_data.split_of`) and keeps near-duplicates on one side.
Many tasks are templates of each other, and many unsolvable tasks are edits of
solvable ones, so `task_data.task_families` first groups tasks into families:
prompts that are nearly the same, reference solutions that load the same
datasets with the same tool sequence, or an unsolvable task close to a solvable
one. Whole families then go to validation, stratified by group and task kind,
with every group and kind represented. The system prompt carries a fixed
benchmark date, so rebuilding does not change the rows.

## Run

```bash
.venv/bin/gym env start \
  "+config_paths=[responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/geobenchx/configs/geobenchx.yaml]"

.venv/bin/gym eval run --no-serve \
  --agent geobenchx_simple_agent \
  --input resources_servers/geobenchx/data/benchmark.jsonl \
  --output resources_servers/geobenchx/data/rollouts.jsonl \
  --limit 202 --num-repeats 1
```

## Tool backends

### Simulator (default)

The default tools simulate catalog and dataflow state. That is enough for the
matching-based score, needs no external data, and is what CI uses. Tool outputs
are canned, so the model never sees real column names, values, or errors. Scores
from the simulator are not comparable with real-tools scores, and the simulator
is not suitable for training.

### Real tools

`configs/geobenchx_real_tools.yaml` switches to the upstream GeoBenchX tool
implementations running on the upstream data snapshot. Each session's
dataframes live in one worker process, and tool errors are returned to the model
in the same form as upstream's langgraph agent.

1. Clone upstream at the pinned commit in `upstream/COMMIT`.
2. Download and unzip the upstream data snapshot (`Data/StatData`, `Data/GeoData`).
   Keep it local: it is not redistributed (see `DATA-ATTESTATION.md`).
3. Create the tool interpreter. GDAL's Python bindings are not pip-installable,
   so this is a conda env kept separate from the server's venv:

   ```bash
   mamba env create -p /path/to/envs/geobenchx -f resources_servers/geobenchx/tools_env/environment.yml
   ```

4. Add to `env.yaml`:

   ```yaml
   geobenchx_upstream_python: /path/to/envs/geobenchx/bin/python
   geobenchx_upstream_src: /path/to/GeoBenchX
   geobenchx_data_root: /path/to/GeoBenchX/Data
   geobenchx_scratch: /path/to/scratch
   ```

5. List the overlay after `geobenchx.yaml`:

   ```bash
   .venv/bin/gym env start \
     "+config_paths=[<model config>,resources_servers/geobenchx/configs/geobenchx.yaml,resources_servers/geobenchx/configs/geobenchx_real_tools.yaml]"
   ```

Differences from calling the upstream tools directly:

- Unknown arguments are dropped, as upstream's langchain tool schemas do.
- State runs with `visualize: True`, as upstream's agent does; figure display
  is a no-op. Some tools only build their figure on that path.
- Each worker writes files in its own scratch directory. `output_folder` is
  always that directory and `output_filename` is reduced to a bare file name,
  so parallel rollouts cannot overwrite each other and the model cannot write
  elsewhere.
- Two upstream defects that make documented calls fail are patched before the
  call (`PRECALL_FIXES` in `upstream_worker.py`):
  - `generate_contours_display`: an omitted `min_value`/`max_value` now uses
    the raster's range, as documented, instead of comparing data to `None`;
    `plot_result=False` no longer fails on a missing figure.
  - `select_features_by_spatial_relationship`: `index_left`/`index_right`
    columns from an earlier join are dropped, so selections can be chained.

Replaying all 222 upstream reference solutions through the real tools, 201 run
with no tool error. The remaining 21 fail on mistakes in the hand-written
references: dataframes that were never created, raster files that are not in
the catalog, a column title over the tools' 10-character limit, and selections
over upstream's feature-count limit.

## Nemotron 3.5 Super VL GA candidates

Four GA release candidates are evaluated; only GA BF16 is trained.

| Variant | Checkpoint | Role |
| --- | --- | --- |
| `bf16` | `Nemotron-3.5-Super-VL-120B-A12B-GA-candidate-MTP-boosted-20261001` | Training start |
| `nvfp4` | `Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP` | Quantization reference |
| `lilicorr` | NVFP4 target + `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr` draft | Speculative-decoding reference |
| `base` | `NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained` | Pretrained lower bound |

GA BF16 is the only candidate that is both full precision and post-trained.
NVFP4 is quantized and meant for deployment. LiLiCorr is a draft model that does
not generate on its own; its accuracy should match NVFP4, and the numbers worth
recording are throughput, latency, and draft acceptance. The base model has no
instruction or RL tuning.

Evaluate them one at a time, in this order, then train:

1. `bf16`: `benchmark.jsonl` and `validation.jsonl`, real tools.
2. `nvfp4`: the same evals.
3. `lilicorr`: the same evals, with nothing else on the node.
4. `base`: the same evals.
5. GRPO on `bf16` with `train.jsonl` only, using all eight GPUs.
6. The trained model on `validation.jsonl`, with the same settings as step 1.

Each candidate takes the whole 8x B200 node as data-parallel replicas of its
model card's layout (TP 2 x DP 4; LiLiCorr runs one low-concurrency replica):

```bash
resources_servers/geobenchx/local_model/download_super_vl.sh      # about 575 GB
resources_servers/geobenchx/local_model/serve_super_vl.sh bf16
curl -s http://127.0.0.1:8300/v1/models
```

The serve script refuses to start while another candidate is served or the GPUs
hold memory. Stop a candidate with `docker rm -f super-vl-<variant>`.

Point `env.yaml` at the server, changing the model name with the variant:

```yaml
policy_base_url: http://127.0.0.1:8300/v1
policy_api_key: EMPTY
policy_model_name: nvidia/super-vl-ga-bf16
```

```bash
.venv/bin/gym env start \
  "+config_paths=[resources_servers/geobenchx/configs/super_vl_local.yaml,resources_servers/geobenchx/configs/geobenchx.yaml,resources_servers/geobenchx/configs/geobenchx_real_tools.yaml]"

.venv/bin/gym eval run --no-serve \
  --agent geobenchx_simple_agent \
  --input resources_servers/geobenchx/data/validation.jsonl \
  --output resources_servers/geobenchx/data/super_vl_bf16_validation.jsonl \
  --num-repeats 4 --concurrency 32
```

`super_vl_local.yaml` uses the model card's sampling (temperature 1.0, top-p
0.95) in place of the rows' temperature 0, so evals and GRPO rollouts sample the
same way. Report the mean over repeats.

## Regression metrics

Each rollout returns:

- `geobenchx_match`
- normalized reward
- solvable/rejection behavior
- candidate and mean reference step counts
- relative solution-length delta
- matched-reference index and step coverage

Aggregate these by `group` and `task_kind` to reproduce the paper's main regression
cuts: task complexity, solvable tasks, and intentionally unsolvable tasks.
