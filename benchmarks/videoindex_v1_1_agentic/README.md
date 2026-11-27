# Video-Index v1.1 (agentic)

Agentic evaluation benchmark for [Video-Index v1.1](https://huggingface.co/datasets/Video-Index/Video-Index-v1-1)
pinned to the latest cleaned release `e7404c468cd0ad83d7c96834c6eb4767fe831a72` (144 multiple-choice questions with
2–10 options across four capability groups). Stock OpenCode answers each question inside a sandbox built from
`resources_servers/video_mcqa/sandbox_image` with the shared `watch_video` and `view_image` tools; the `videoindex`
resources server lends the sandbox, checks the video's SHA-256 before the episode starts, and reports metrics under
`Video_Index_v1_1_agentic`.

## Data

| Field | Value |
| --- | --- |
| Source | Hugging Face `Video-Index/Video-Index-v1-1`, revision `e7404c468cd0ad83d7c96834c6eb4767fe831a72` |
| Items | `items/test.jsonl` (144 questions; cleaned release removing 6 incomplete upstream clips; SHA-256 verified) |
| Videos | `videos/<name>.mp4` uploaded to `/testbed/videos/<name>.mp4` in each episode's sandbox |
| License | `unknown` (aggregated third-party benchmarks; evaluation only; no data committed) |

```bash
hf download Video-Index/Video-Index-v1-1 --repo-type dataset \
    --revision e7404c468cd0ad83d7c96834c6eb4767fe831a72 \
    --local-dir <source_dir>
gym eval prepare --benchmark videoindex_v1_1_agentic ++videoindex_v1_1_agentic_source_dir=<source_dir>
```

## Running

```bash
gym eval run --benchmark videoindex_v1_1_agentic --model-type vllm_model --split benchmark \
    --output results/videoindex_v1_1_agentic.jsonl \
    ++videoindex_v1_1_agentic_source_dir=<source_dir> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

To run multiple repeats, prepare with `++num_repeats=<N>` and pass `--num-repeats <N>` to `gym eval run`.

Requires `VIDEO_MCQA_SANDBOX_IMAGE` (built from `resources_servers/video_mcqa/sandbox_image`) and
`OPENSANDBOX_DOMAIN` / `OPENSANDBOX_API_KEY`. To use `llm_extraction_v1` instead of the default official rule
(`videoindex_rule_v1`), include `resources_servers/videoindex/configs/videoindex_agentic_llm_extraction.yaml` and
prepare with `++prepare_script_args.answer_policy=llm_extraction_v1`.
