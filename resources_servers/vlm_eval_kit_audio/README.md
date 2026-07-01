# vlm_eval_kit_audio — generic VLMEvalKit driver (audio-first)

A single generic scorer that dispatches on the VLMEvalKit `DATASET_TYPE`
(MCQ / QA / Y-N / VQA) instead of a per-benchmark `_score_<bench>`, so new
VLMEvalKit benchmarks can be added with little or no code. Audio-first. MVP
benchmark: **MMAU** (pure-audio MCQ/QA, scored generatively with `can_infer`).

## How it works
- **Inference** is done by Gym's `vllm_model` server, not by VLMEvalKit. This
  server only (a) prepares data and (b) scores in `verify()`.
- **Audio transport** uses the `vllm_model` side-channel: rows carry a clip on
  `responses_create_params.metadata.audio_path` (file path, resolved against
  `config.audio_root`) or `audio_data` (inline `data:audio/...;base64,` URI); the
  model server splices it into an `audio_url` content block. No Gym schema change.
- **Scoring** (`app.py`) reuses VLMEvalKit `can_infer` for option-letter
  extraction when `vlmeval` is importable; otherwise a dependency-light local
  fallback keeps `verify()` and the unit tests runnable without the heavy
  `vlmeval`/torch import. `strip_think` strips `<think>…</think>` before parsing
  (thinking models can emit option letters inside the reasoning block).
- Hard types (LLM-judge, IoU-grounding, ASR-WER, circular-MMBench) are out of
  scope here — they stay bespoke.

## Data
- `data/example.jsonl` (committed): 5 self-contained smoke rows with a tiny
  inline silent WAV. Regenerate: `python generate_example_data.py`.
- Real benchmarks: need a VLMEvalKit install that provides the target audio
  datasets, plus the dataset staged under `LMUDataRoot()`:
  ```bash
  uv pip install -e <path-to-your-vlmevalkit> --no-deps
  # stage the dataset (e.g. MMAU_test.json + wav files) under $LMUData
  python prepare_data.py --dataset MMAU_test --out data/MMAU_test_validation.jsonl
  ```

## Test (offline, no model)
```bash
# from the Gym repo root, with the dev venv:
.venv/bin/python -m pytest resources_servers/vlm_eval_kit_audio/tests/ -q
```

## Run (needs an audio-capable model)
End-to-end requires a multimodal, audio-capable model served behind an
OpenAI-compatible chat/completions endpoint, wired via `--model-type vllm_model`
(the audio side-channel lives in that model server). Point `env.yaml`
(`policy_base_url` / `policy_api_key` / `policy_model_name`) at your endpoint, then:
```bash
gym eval run --config resources_servers/vlm_eval_kit_audio/configs/smoke.yaml \
  --model-type vllm_model --agent vlm_eval_kit_audio_simple_agent \
  --output results/audio_smoke.jsonl --split validation \
  --num-repeats 1 --max-output-tokens 8192 --temperature 0.0
```
