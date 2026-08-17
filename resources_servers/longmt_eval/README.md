# longmt_eval

Document-level machine translation verifier using the SEGALE pipeline. Scores
a model's translation of an entire document (book chapter, news article, or
structured text) with reference-free COMETKiwi, without requiring a human
reference translation.

Scoring runs in three phases inside a persistent Ray GPU actor pool:

1. **Segment** — split source and MT text into sentences with ersatz
2. **Align** — embed sentence overlaps with LASER2, align with vecalign
3. **Score** — run COMETKiwi over aligned (source, MT) span pairs

The mean COMETKiwi score across all valid aligned spans is returned as the RL
reward. Each actor holds LASER2 + COMETKiwi resident in GPU memory across
calls — no per-request cold load.

Set `compute_segale: false` for local smoke tests. `verify()` returns
`reward=0.0` without touching the actor pool, so the server starts without a
GPU.

## Benchmarks

Two benchmark configs ship with this server:

| Config | Dataset | Doc type | Actor layout |
|--------|---------|----------|--------------|
| `longmt_wmt24pp.yaml` | WMT24++ (55 lang pairs) | Short news docs | `comet_num_shards=4`, `actors_per_gpu=4` → 16 actors on 4 H100s |
| `longmt_pg19.yaml` | PG-19 books | Long book chapters | same layout, `use_extra_gpu: false` |

## Scoring

### Per-sample reward

`verify()` returns `comet_qe` as the `reward` field — the mean COMETKiwi score
over all valid (non-sentinel) aligned spans, roughly in `[0, 1]`. Deleted
source spans (no matching MT) and hallucinated MT spans (no matching source)
receive `comet_qe=0.0` and are flagged in the `spans` list.

An empty generation (after reasoning-preamble stripping) returns `reward=0.0`
immediately without touching the actor pool.

Each rollout in `rollouts.jsonl` carries:

| Field | Type | Description |
|-------|------|-------------|
| `reward` | float | Mean COMETKiwi (same as `comet_qe`) |
| `comet_qe` | float \| null | Mean COMETKiwi over valid spans |
| `lang_fidelity` | float \| null | Fraction of 500-char chunks detected in the target language |
| `total_seg` | int | Total aligned spans scored |
| `misaligned_seg` | int | Spans flagged as deleted or hallucinated |
| `generation` | str | Model output after reasoning stripping |
| `segale_error` | str \| null | Error message if the SEGALE pipeline failed |
| `spans` | list \| null | Per-segment scores — each entry has `src`, `tgt`, `comet_qe`, `hallucinated`, `deleted` |

The `spans` list gives per-segment visibility into alignment quality. For example:

```json
{
  "src": "A DOCTOR OF THE OLD SCHOOL",
  "tgt": "老派医生",
  "comet_qe": 0.7085,
  "hallucinated": false,
  "deleted": false
}
```

Spans where `deleted: true` have a source sentence with no matching MT output;
`hallucinated: true` spans have MT output with no matching source sentence. Both
types receive `comet_qe: 0.0` and are counted in `misaligned_seg`.

### Aggregate metrics (`compute_metrics`)

Groups rollouts by `target_language` and reports per-language and overall
aggregates:

```json
{
  "de_DE": {
    "comet_qe": 0.842,
    "lang_fidelity": 0.97,
    "total_seg": 1240,
    "misaligned_seg": 38,
    "misaligned_rate": 0.031,
    "n_docs": 12
  },
  "overall_comet_qe": 0.831
}
```

`get_key_metrics()` returns a flat `{target_language: comet_qe}` dict suitable
for W&B logging.

## SEGALE actor pool

`_ensure_actors()` is called lazily on the first `verify()` request. It spawns
`comet_num_shards × actors_per_gpu` Ray actors, pings them all within 300s,
and drops any that fail init. Requests are round-robined across the live pool
under a threading lock.

### Deployment modes

| `use_extra_gpu` | Ray resource claim | When to use |
|-----------------|-------------------|-------------|
| `false` (default) | `num_gpus=1/actors_per_gpu` | Gym runs its own Ray cluster on dedicated GPU nodes, HTTP-separated from vLLM |
| `true` | `resources={"extra_gpu": 1/actors_per_gpu}`, `num_gpus=0` | Gym joins the vLLM Ray cluster; a separate node is registered with `ray start --resources='{"extra_gpu": N}'` |

In `use_extra_gpu=false` mode, actors are interleaved after init so that
round-robin dispatch spreads across physical GPUs before doubling up on the
same one (creation order is `[GPU0×A, GPU1×A, ...]`; dispatch order becomes
`[GPU0, GPU1, ..., GPU0, GPU1, ...]`).

### Python mirroring for cross-node workers

uv ships python-build-standalone binaries whose absolute paths differ across
containers. `segale_actor.py` copies the venv's Python root to a
shared-FS path at `LONGMT_EVAL_PY_CACHE` (default `/opt/Gym/.cache/longmt-python`)
so remote Ray workers can resolve a stable `py_executable` at runtime. Set
`LONGMT_EVAL_PY_CACHE` to a Lustre path when running across nodes.

## Input JSONL format

Each task row must provide the fields used by `LongmtEvalRunRequest` (passed
through `verifier_metadata`):

```json
{
  "text": "<full source document text>",
  "source_language": "en",
  "target_language": "de_DE",
  "source_lang_name": "English",
  "target_lang_name": "German",
  "doc_id": "my-article-2024",
  "seg_id": 1
}
```

`text` is the tiktoken-truncated source document written by the benchmark's
`prepare.py`. `source_language` and `target_language` are used by the actor
for language-fidelity detection. `doc_id` identifies the document in logs
and output rows.

## Example usage

### Reward profiling from pre-baked rollouts (no GPU, no model server)

```bash
gym eval profile \
    --inputs resources_servers/longmt_eval/data/example_rollouts_materialized_inputs.jsonl \
    --rollouts resources_servers/longmt_eval/data/example_rollouts.jsonl
```

Outputs `example_rollouts_reward_profiling.jsonl` (per-task stats) and
`example_rollouts_agent_metrics.json` (agent-level aggregates) alongside the
rollouts file.

### Full rollout collection (requires model server)

```bash
# Start servers (smoke-test mode — no GPU needed for the verifier)
gym env start \
    --resources-server longmt_eval \
    --model-type vllm_model & \
    ++longmt_eval.resources_servers.longmt_eval.compute_segale=false

# Collect rollouts — also writes results/longmt_eval_rollouts_materialized_inputs.jsonl
gym eval run --no-serve \
    --agent longmt_eval_simple_agent \
    --input resources_servers/longmt_eval/data/example.jsonl \
    --output results/longmt_eval_rollouts.jsonl \
    --num-repeats 1

# Profile rewards from the collected rollouts
gym eval profile \
    --inputs results/longmt_eval_rollouts_materialized_inputs.jsonl \
    --rollouts results/longmt_eval_rollouts.jsonl
```

For a full SLURM run with SEGALE enabled on WMT24++ see
[`benchmarks/longmt_wmt24pp/`](../../benchmarks/longmt_wmt24pp/).

## Config

| Key | Default | Meaning |
|-----|---------|---------|
| `compute_segale` | `true` | Run the full SEGALE pipeline; `false` returns `reward=0.0` without a GPU |
| `comet_model` | `Unbabel/wmt22-cometkiwi-da` | HF repo for the COMETKiwi checkpoint |
| `comet_batch_size` | `8` | Aligned span pairs per COMETKiwi forward pass; 8 is safe for 80 GB GPUs |
| `comet_num_shards` | `8` (`4` for wmt24pp/pg19) | Physical GPUs to use; total actors = `comet_num_shards × actors_per_gpu` |
| `actors_per_gpu` | `1` (`4` for wmt24pp/pg19) | Actors co-placed on each GPU; each claims `1/actors_per_gpu` of the GPU |
| `embed_batch_size` | `512` | Overlap strings per LASER2 `encode_sentences()` call |
| `assert_no_reasoning` | `true` | Assert the generation contains no `<think>...</think>` tags. Reasoning must be parsed by the inference server upstream — a leaked preamble surfaces as an `AssertionError` instead of being silently rescued |
| `use_extra_gpu` | `false` | Actor resource mode; see Deployment modes above |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LASER_HOME` | Path to the LASER2 model weights (required for SEGALE actors) |
| `HF_HOME` / `HF_HUB_CACHE` | HuggingFace cache; COMETKiwi checkpoint resolved here |
| `HF_HUB_OFFLINE` | Set to `1` to prevent any HF Hub network calls |
| `ERSATZ` | Path to the ersatz segmenter model weights |
| `LONGMT_EVAL_PY_CACHE` | Shared-FS path for the mirrored uv Python root used by Ray workers |

## Licensing

- Code: Apache 2.0
- `Unbabel/wmt22-cometkiwi-da`: Apache 2.0 (check model card)
- `SEGALE` pipeline: see [jeffwillette/SEGALE](https://github.com/jeffwillette/SEGALE)
- `laser-encoders` (forked): BSD
- `ersatz` (forked): MIT
- `unbabel-comet`: Apache 2.0
- `langdetect`: Apache 2.0
