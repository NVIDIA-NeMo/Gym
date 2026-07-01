# vlm_eval_kit_audio — generic VLMEvalKit driver (audio-first)

Option **B** from `~/repos/tasks/kit/plan.md`: one generic scorer that dispatches on
the VLMEvalKit `DATASET_TYPE` (MCQ / QA / Y-N / VQA) instead of a per-benchmark
`_score_<bench>`. Audio-first because that is the priority Omni path. MVP benchmark:
**MMAU** (pure-audio MCQ/QA, scored generatively with `can_infer`).

## How it works
- **Inference** is done by Gym's `vllm_model` server, not by VLMEvalKit. This server
  only (a) prepares data and (b) scores in `verify()`.
- **Audio transport** uses the existing `vllm_model` side-channel: rows carry a clip
  on `responses_create_params.metadata.audio_path` (file path, resolved against
  `config.audio_root`) or `audio_data` (inline `data:audio/...;base64,` URI). The
  model server splices it into an `audio_url` content block. No Gym schema change.
- **Scoring** (`app.py`) reuses VLMEvalKit `can_infer` for option-letter extraction
  when `vlmeval` is importable; otherwise a local fallback keeps `verify()` and the
  unit tests runnable without the heavy `vlmeval`/torch import. `strip_think` removes
  `<think>…</think>` before parsing (thinking models leak option letters there).
- Hard types (LLM-judge, IoU-grounding, ASR-WER, circular-MMBench) are **out of scope**
  here — they stay bespoke (see `plan.md`).

## Data
- `data/example.jsonl` (committed): 5 self-contained smoke rows with a tiny inline
  silent WAV. Regenerate: `python generate_example_data.py`.
- Real MMAU: needs the **internal mcore fork** installed and MMAU data staged under
  `LMUDataRoot()`:
  ```bash
  uv pip install -e /home/mj/repos/forks/VLMEvalKitMcore --no-deps
  # stage MMAU_test.json + wavs under $LMUData
  python prepare_data.py --dataset MMAU_test --out data/MMAU_test_validation.jsonl
  ```

## Test (offline, no model)
```bash
# from Gym root, with the dev venv:
.venv/bin/python -m pytest resources_servers/vlm_eval_kit_audio/tests/ -q
```

## Run (needs an audio-capable model — cluster)
End-to-end requires an Omni model served via `vllm_model`. As of 2026-07 there is **no
Nano 3.5 Omni checkpoint yet**; use **Nano V3 Omni GA**
(`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, vLLM 0.20.0 + `--trust-remote-code`).
See `~/repos/tasks/kit/` for the run plan.
