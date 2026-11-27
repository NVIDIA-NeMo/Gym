# Video-MME

Video-MME (2,700 questions over 900 videos, without subtitles) in a single turn: one model call per question with the
video as an `input_video` `file://` URL and no tools. The environment is the single-turn judge-extracted configuration
of the `videomme_v1` resources server (`resources_servers/videomme_v1/configs/videomme_v1_single_turn_llm_extraction.yaml`;
single-turn video inputs and scoring are described in `resources_servers/video_mcqa/README.md`). The agentic version,
with OpenCode and video tools in a sandbox, is `benchmarks/videomme_v1_agentic`.

## Data

`prepare.py` reads the authors' annotation, `videomme/test-00000-of-00001.parquet` of the Hugging Face dataset
[`lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME) at revision `ead1408f`, which it downloads
and checks by SHA-256 (`VIDEOMME_V1_PARQUET` names a local copy instead). The source directory
(`videomme_v1_source_dir`) holds the videos and their checksums:

- `video/<videoID>.mp4`, the dataset's 900 videos named by their `videoID`;
- `FILES.sha256`, their SHA-256 as `sha256sum` writes it, with paths relative to the directory
  (`video/fFjv93ACGo8.mp4`).

`resources_servers/video_mcqa/prepare.py`'s `prepare_dataset` writes `data/videomme_v1_benchmark.jsonl` and the frozen
cohort `data/videomme_v1_cohort.json`. No video is decoded during preparation: each row records the video's SHA-256 in
`verifier_metadata` and passes `file://<videomme_v1_video_url_dir>/<videoID>.mp4` as `input_video`. Set
`videomme_v1_video_url_dir` when the model server reads the `video/` directory at another path.

| Field | Value |
| --- | --- |
| Task ID | `videomme_v1:<question_id>` |
| Input | One `input_video` part (`file://` URL) followed by the benchmark's letter prompt (`prompt="videomme_v1"` in `resources_servers/video_mcqa/README.md`): `Select the best answer ...`, the question, the `A. ...` options, `The best answer is:` |
| Answer policy | `llm_extraction_v1`: a judge model reads the chosen option from the final answer |
| Categories | duration, domain, sub-category and task type, as `video_mcqa` reports them under `Video_MME` |

The dataset's terms restrict its use to academic research (see `resources_servers/videomme_v1/README.md`).

## Serving and frame sampling

Neither `prepare.py` nor `simple_agent` chooses a frame count or resizes frames; frame sampling belongs to the model
server. With vLLM, start the server with `--allowed-local-media-path <videomme_v1_video_url_dir>` so it can read the
`file://` video URLs, and pass `--media-io-kwargs` for the desired sampling policy (vLLM defaults to 32 frames if
unset), for example:

```bash
vllm serve <model> --allowed-local-media-path <videomme_v1_video_url_dir> \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
```

## Judge

`judge_base_url`, `judge_api_key` and `judge_model_name` default to gpt-6.1-sol on the inference API at max reasoning
effort (`JUDGE_BASE_URL`, `JUDGE_API_KEY` or `INFERENCE_API_KEY`, and `JUDGE_MODEL` override them). The judge sees
the question text, its options and the final answer, never the video or the gold label.

## Running

```bash
gym eval run --benchmark videomme_v1 --model-type vllm_model --split benchmark \
    --output results/videomme_v1.jsonl \
    ++videomme_v1_source_dir=<directory with FILES.sha256 and video/> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Set `++videomme_v1_video_url_dir=<path on model server>` when the model server mounts `video/` at a different path
than `<videomme_v1_source_dir>/video`.
