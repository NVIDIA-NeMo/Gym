# Video-MME v2

Video-MME v2 (3,200 questions over 800 videos, without subtitles) in a single turn: one model call per question with
the video as an `input_video` `file://` URL and no tools. The environment is the single-turn judge-extracted
configuration of the `videomme_v2` resources server
(`resources_servers/videomme_v2/configs/videomme_v2_single_turn_llm_extraction.yaml`; single-turn video inputs and
group scoring are described in `resources_servers/videomme_v2/README.md`). The agentic version, with OpenCode and
video tools in a sandbox, is `benchmarks/videomme_v2_agentic`.

## Data

`prepare.py` reads the authors' annotation, `test.parquet` of the Hugging Face dataset
[`MME-Benchmarks/Video-MME-v2`](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2) at revision `6e4bebb0`,
which it downloads and checks by SHA-256 (`VIDEOMME_V2_PARQUET` names a local copy instead). The source directory
(`videomme_v2_source_dir`) holds the videos and their checksums:

- `video/<video_id>.mp4`, the dataset's 800 videos;
- `FILES.sha256`, their SHA-256 as `sha256sum` writes it, with paths relative to the directory (`video/001.mp4`).

It writes `data/videomme_v2_benchmark.jsonl` and the frozen cohort `data/videomme_v2_cohort.json`. No video is decoded
during preparation: each row records the video's SHA-256 in `verifier_metadata` and passes
`file://<videomme_v2_video_url_dir>/<video_id>.mp4` as `input_video`. Set `videomme_v2_video_url_dir` when the model
server reads the `video/` directory at another path.

| Field | Value |
| --- | --- |
| Task ID | `videommev2-L<level>/<question_id>` |
| Input | One `input_video` part (`file://` URL) followed by the benchmark's letter prompt (`prompt="videomme_v2"` in `resources_servers/video_mcqa/README.md`): the question, the `A. ...` options, `Respond with only the letter (A, ..., or H) of the correct option.`, `Answer: ` |
| Answer policy | `llm_extraction_v1`: a judge model reads the chosen option from the final answer |
| Groups | the four questions of each video in order; a group's level is its fourth question's, and each question keeps its own (as the official script scores them) |

The dataset's terms restrict its use to academic research (see `resources_servers/videomme_v2/README.md`).

## Serving and frame sampling

Neither `prepare.py` nor `simple_agent` chooses a frame count or resizes frames; frame sampling belongs to the model
server. With vLLM, start the server with `--allowed-local-media-path <videomme_v2_video_url_dir>` so it can read the
`file://` video URLs, and pass `--media-io-kwargs` for the desired sampling policy (vLLM defaults to 32 frames if
unset), for example:

```bash
vllm serve <model> --allowed-local-media-path <videomme_v2_video_url_dir> \
    --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
```

## Judge

`judge_base_url`, `judge_api_key` and `judge_model_name` default to gpt-6.1-sol on the inference API at max reasoning
effort (`JUDGE_BASE_URL`, `JUDGE_API_KEY` or `INFERENCE_API_KEY`, and `JUDGE_MODEL` override them). The judge sees
the question text, its options and the final answer, never the video or the gold label.

## Running

```bash
gym eval run --benchmark videomme_v2 --model-type vllm_model --split benchmark \
    --output results/videomme_v2.jsonl \
    ++videomme_v2_source_dir=<directory with FILES.sha256 and video/> \
    ++policy_base_url=<vLLM URL> ++policy_model_name=<served model>
```

Set `++videomme_v2_video_url_dir=<path on model server>` when the model server mounts `video/` at a different path
than `<videomme_v2_source_dir>/video`.
