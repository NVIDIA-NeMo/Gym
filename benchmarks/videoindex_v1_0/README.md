# Video-Index v1.0 (single-turn)

Single-turn evaluation benchmark for [Video-Index v1.0](https://huggingface.co/datasets/Video-Index/Video-Index)
(840 multiple-choice questions with 2–10 options across four capability groups: `perception`, `temporal`,
`spatial_physical`, and `reasoning_knowledge`), built on `resources_servers/videoindex` (`video_mcqa`).
The agentic counterpart is `benchmarks/videoindex_v1_0_agentic`.

## Data

| Field | Value |
| --- | --- |
| Source | Hugging Face `Video-Index/Video-Index`, revision `fb63b6c93948e78e76aee9c87745bf78358d3011` |
| Items | `items/test.jsonl` (840 questions; SHA-256 verified) |
| Videos | `videos/<name>.mp4` |
| License | `unknown` (aggregated third-party benchmarks; evaluation only; no data committed) |

Download the pinned revision into a local directory before preparing:

```bash
hf download Video-Index/Video-Index --repo-type dataset \
    --revision fb63b6c93948e78e76aee9c87745bf78358d3011 \
    --local-dir <source_dir>
gym eval prepare --benchmark videoindex_v1_0 ++videoindex_v1_0_source_dir=<source_dir>
```

In offline mode (`download=false`), if `<source_dir>/FILES.sha256` exists, `prepare.py` reads video digests directly
without rehashing; otherwise it hashes local `videos/*.mp4` files. With `++prepare_script_args.download=true`,
`prepare.py` always materializes and verifies each video against Hugging Face LFS size and SHA-256 under `video_dir`
even when `FILES.sha256` is present. Set `videoindex_v1_0_video_url_dir` when the model server reads the videos
at a different path or HTTP/file prefix. To run multiple repeats, pass the same count at preparation and run time
(`++num_repeats=<N>` during `gym eval prepare` and `--num-repeats <N>` during `gym eval run`).

## Scoring & Metrics

- **Default**: official Apache-2.0 rule (`videoindex_rule_v1`) adapted from `Espere-1119-Song/Video-Index`
  (`4819d9252ae177f87a134f0ceda74855847afe2a`), requiring no judge model.
- **Optional LLM extraction**: compose with
  `resources_servers/videoindex/configs/videoindex_single_turn_llm_extraction.yaml` and prepare with
  `++prepare_script_args.answer_policy=llm_extraction_v1`.
- **Metrics**: `mean/Video_Index_v1_0`, `Video_Index_v1_0/raw_acc`, `/coverage`, `/is_complete`, and per-group
  `Video_Index_v1_0/capability_group/{perception,temporal,spatial_physical,reasoning_knowledge}/raw_acc`.

## Running

```bash
gym eval run --benchmark videoindex_v1_0 --model-type vllm_model --split benchmark \
    --output results/videoindex_v1_0.jsonl \
    ++videoindex_v1_0_source_dir=<source_dir> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Ensure the model server has `--allowed-local-media-path` covering `videoindex_v1_0_video_url_dir`.
