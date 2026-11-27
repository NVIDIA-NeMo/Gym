# LVBench

LVBench ([THUDM/LVBench](https://huggingface.co/datasets/THUDM/LVBench), **not LongVideoBench**: 1,549 questions over
103 long videos) in a single turn: one model call per question with the video as an `input_video` `file://` URL and no
tools. The environment is the single-turn judge-extracted configuration of the `lvbench` resources server
(`resources_servers/lvbench/configs/lvbench_single_turn_llm_extraction.yaml`; single-turn video inputs and scoring are
described in `resources_servers/video_mcqa/README.md`). The agentic version, with OpenCode and video tools in a
sandbox, is `benchmarks/lvbench_agentic`.

## Data

`prepare.py` reads the authors' annotation, `video_info.meta.jsonl` of the Hugging Face dataset
[`THUDM/LVBench`](https://huggingface.co/datasets/THUDM/LVBench) at revision `0caedb92`, which it downloads and checks
by SHA-256 (`LVBENCH_ANNOTATIONS` names a local copy instead). The source directory (`lvbench_source_dir`) holds the
videos and their checksums:

- `video/<key>.mp4`, the dataset's 103 videos named by their `key`;
- `FILES.sha256`, their SHA-256 as `sha256sum` writes it, with paths relative to the directory
  (`video/Cm73ma6Ibcs.mp4`).

`resources_servers/video_mcqa/prepare.py`'s `prepare_dataset` writes `data/lvbench_benchmark.jsonl` and the frozen
cohort `data/lvbench_cohort.json`. No video is decoded during preparation: each row records the video's SHA-256 in
`verifier_metadata` and passes `file://<lvbench_video_url_dir>/<key>.mp4` as `input_video`. Set
`lvbench_video_url_dir` when the model server reads the `video/` directory at another path.

| Field | Value |
| --- | --- |
| Task ID | `lvbench:<uid>` |
| Input | One `input_video` part (`file://` URL) followed by the benchmark's letter prompt (`prompt="lvbench"` in `resources_servers/video_mcqa/README.md`): the question, the `(A) ...` options, `Answer with the option's letter from the given choices directly.` |
| Answer policy | `llm_extraction_v1`: a judge model reads the chosen option from the final answer |
| Categories | `domain` (annotation `type`) and overlapping capabilities (`task_type`: entity recognition, event understanding, key information retrieval, temporal grounding, reasoning, summarization), as `video_mcqa` reports them under `LVBench` |

The dataset's terms are CC-BY-NC-SA-4.0 with research-only / non-commercial restrictions (see
`resources_servers/lvbench/README.md`).

## Serving and frame sampling

Neither `prepare.py` nor `simple_agent` chooses a frame count or resizes frames; frame sampling belongs to the model
server. With vLLM, start the server with `--allowed-local-media-path <lvbench_video_url_dir>` so it can read the
`file://` video URLs, and pass `--media-io-kwargs` for the desired sampling policy (vLLM defaults to 32 frames if
unset), for example:

```bash
vllm serve <model> --allowed-local-media-path <lvbench_video_url_dir> \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
```

## Judge

`judge_base_url`, `judge_api_key` and `judge_model_name` default to gpt-6.1-sol on the inference API at max reasoning
effort (`JUDGE_BASE_URL`, `JUDGE_API_KEY` or `INFERENCE_API_KEY`, and `JUDGE_MODEL` override them). The judge sees
the question text, its options and the final answer, never the video or the gold label.

## Running

```bash
gym eval run --benchmark lvbench --model-type vllm_model --split benchmark \
    --output results/lvbench.jsonl \
    ++lvbench_source_dir=<directory with FILES.sha256 and video/> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Set `++lvbench_video_url_dir=<path on model server>` when the model server mounts `video/` at a different path than
`<lvbench_source_dir>/video`.
