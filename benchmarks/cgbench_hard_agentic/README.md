# CG-Bench-hard (agentic, long_acc)

Agentic full-video multiple-choice (`long_acc`, without subtitles) evaluation benchmark for the
[CG-Bench-hard](https://huggingface.co/datasets/CG-Bench/CG-Bench) slice (2,417 questions over 1,010 long videos, 5–8
options per question). Stock OpenCode answers each question inside a sandbox built from
`resources_servers/video_mcqa/sandbox_image` with the shared `watch_video` and `view_image` tools; the `cgbench`
resources server lends the sandbox, checks the video's SHA-256 before the episode starts, and reports metrics under
`CG_Bench_hard_agentic`.

## Data & Gated Access

| Field | Value |
| --- | --- |
| Source | Hugging Face `CG-Bench/CG-Bench`, revision `aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57` (`gated: auto`) |
| Annotations | `cgbench_hard.json` (2,417 questions; SHA-256 verified) |
| Videos | `cg_videos_720p/<video_uid>.mp4` uploaded to `/testbed/videos/<video_uid>.mp4` in each episode's sandbox |
| License | `MIT` (dataset card declaration; video copyrights belong to original creators for academic research use; no data committed) |

Because `CG-Bench/CG-Bench` is `gated: auto` on Hugging Face, first accept the dataset terms at
<https://huggingface.co/datasets/CG-Bench/CG-Bench>, export `HF_TOKEN` (or run `huggingface-cli login`), download the
pinned revision, and extract `video_chunk_01.zip`..`video_chunk_69.zip` into `<source_dir>/cg_videos_720p/`:

```bash
export HF_TOKEN=<your_hf_token>
hf download CG-Bench/CG-Bench --repo-type dataset \
    --revision aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57 \
    --include "cgbench_hard.json" --include "video_chunk_*.zip" \
    --local-dir <source_dir>
gym eval prepare --benchmark cgbench_hard_agentic ++cgbench_hard_agentic_source_dir=<source_dir>
```

## Running

```bash
gym eval run --benchmark cgbench_hard_agentic --model-type vllm_model --split benchmark \
    --output results/cgbench_hard_agentic.jsonl \
    ++cgbench_hard_agentic_source_dir=<source_dir> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Requires `VIDEO_MCQA_SANDBOX_IMAGE` (built from `resources_servers/video_mcqa/sandbox_image`) and
`OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY`. To use `boxed_only_v1`, prepare with
`++prepare_script_args.prompt=boxed ++prepare_script_args.answer_policy=boxed_only_v1`; to use `llm_extraction_v1`,
include `resources_servers/cgbench/configs/cgbench_agentic_llm_extraction.yaml` and prepare with
`++prepare_script_args.answer_policy=llm_extraction_v1`.
