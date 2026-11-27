# CG-Bench-hard (single-turn, long_acc)

Single-turn full-video multiple-choice (`long_acc`, without subtitles) evaluation benchmark for the
[CG-Bench-hard](https://huggingface.co/datasets/CG-Bench/CG-Bench) slice (2,417 questions over 1,010 long videos, 5–8
options per question), built on `resources_servers/cgbench` (`video_mcqa`). Clue-clip QA (`clue_acc`), temporal
grounding (`mIoU`), and open-ended QA are excluded. The agentic counterpart is `benchmarks/cgbench_hard_agentic`.

## Data & Gated Access

| Field | Value |
| --- | --- |
| Source | Hugging Face `CG-Bench/CG-Bench`, revision `aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57` (`gated: auto`) |
| Annotations | `cgbench_hard.json` (2,417 questions; SHA-256 verified) |
| Videos | `cg_videos_720p/<video_uid>.mp4` (1,010 unique videos, extracted from `video_chunk_01.zip`..`video_chunk_69.zip`) |
| License | `MIT` (dataset card declaration; video copyrights belong to original creators for academic research use; no data committed) |

Because `CG-Bench/CG-Bench` is `gated: auto` on Hugging Face, first accept the dataset terms at
<https://huggingface.co/datasets/CG-Bench/CG-Bench>, export `HF_TOKEN` (or run `huggingface-cli login`), download the
pinned revision, and extract the full-video zip chunks into `<source_dir>/cg_videos_720p/`:

```bash
export HF_TOKEN=<your_hf_token>
hf download CG-Bench/CG-Bench --repo-type dataset \
    --revision aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57 \
    --include "cgbench_hard.json" --include "video_chunk_*.zip" \
    --local-dir <source_dir>
# Merge and unzip video_chunk_01.zip..video_chunk_69.zip into <source_dir>/cg_videos_720p/
gym eval prepare --benchmark cgbench_hard ++cgbench_hard_source_dir=<source_dir>
```

If `<source_dir>/FILES.sha256` exists, `prepare.py` reads video digests directly without rehashing; otherwise it hashes
`<video_dir>/<video_uid>.mp4`. Set `CGBENCH_HARD_JSON` (or `++prepare_script_args.annotations_path=...`) to read
`cgbench_hard.json` from another directory, or `++prepare_script_args.download=true` to fetch the JSON with `HF_TOKEN`.

## Scoring & Metrics

- **Default**: `cgbench_rule_v1` with the no-subtitle CG-Bench `long_acc` JSON prompt (`prompt="official"`).
- **Boxed mode**: pass `++prepare_script_args.prompt=boxed ++prepare_script_args.answer_policy=boxed_only_v1` (or
  `boxed_or_letter_v1`).
- **LLM extraction**: compose with `resources_servers/cgbench/configs/cgbench_single_turn_llm_extraction.yaml` and
  prepare with `++prepare_script_args.answer_policy=llm_extraction_v1`.
- **Metrics**: `mean/CG_Bench_hard`, `CG_Bench_hard/raw_acc`, `/coverage`, `/is_complete`, plus breakdowns by
  `category` (3), `sub_category` (12), `domain` (14), and `duration` (7 buckets: `0-10m` through `60m+`).

## Running

```bash
gym eval run --benchmark cgbench_hard --model-type vllm_model --split benchmark \
    --output results/cgbench_hard.jsonl \
    ++cgbench_hard_source_dir=<source_dir> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Ensure the model server has `--allowed-local-media-path` covering `cgbench_hard_video_url_dir`.
