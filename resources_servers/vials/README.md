# vials resources server

Scores one VIALS attempt with the upstream LLM judge. The benchmark, its data and its reference results are described in [`benchmarks/vials/README.md`](../../benchmarks/vials/README.md).

- `app.py`: answer extraction, the text-only judge with up to `judge_attempts` attempts, the image re-grade of `UNVERIFIABLE:` verdicts (`multimodal_judge_fallback`), and the metrics (accuracy, pass@k, pass^k, per domain).
- `upstream_protocol.py`: prompts, answer extraction and judge parsing vendored from the reference harness (MIT, commit `002a0d25`).
- `configs/judge_model_openai.yaml`: the `judge_model` server (OpenAI-compatible Responses endpoint).
- `fixture_judge.py`: a service-free judge for `VERIFIER_FIXTURE` and the tests; it answers the real judge request.
- `data/example.jsonl`: five synthetic tasks with generated images, not VIALS data. `data/example_rollouts.jsonl` holds their rollouts with Gemini 3.7 Flash and the gpt-5-mini judge.

A judge call that fails (transport, timeout, HTTP error) raises `JudgeError`; Gym masks the rollout as `judge_failed` and retries it on resume. An empty or unparseable verdict after every attempt is a received answer and scores 0, as upstream.
