# VIALS resources server

Direct visual question answering with VIALS answer extraction and scientific-answer
judging. Gym's `simple_agent` runs the policy; this server verifies answers without
solver tools or a sandbox.

## Request and score

Required fields: `responses_create_params`, `question`, and `expected_answer`.
Optional `answer_lower_bound` and `answer_upper_bound` are inclusive lead-assigned
bounds. References stay outside policy input. Images are embedded as `input_image`
data URLs in their original panel order.

Verification strips `<think>`/`<thinking>` blocks, then uses upstream `ANSWER:`
extraction. Missing visible answers score zero without a judge call. The text-only
judge compares the candidate to the reference using VIALS's exact grading prompt.
With `multimodal_fallback: true`, `UNVERIFIABLE:` judgments are regraded using all
images from the original policy input. Correct answers score one; incorrect answers
score zero. Usage combines both judge passes and any malformed-output retries.

The prompt, JSON verdict parser, and extraction code are extracted from upstream
commit `002a0d25031f02a760da5d14f2c29bbd26eb06e4` in `upstream.py`, with the
original MIT license in `UPSTREAM_LICENSE`. No external VIALS checkout, LiteLLM
runtime, or temporary image files are needed.

## Judge configuration

`configs/vials.yaml` declares `judge_model_server`, `multimodal_fallback` (default
true), and `max_concurrent_judges` (default four). `configs/judge_model.yaml` binds
the default `vials_judge_model` to Gym's OpenAI model adapter, defaulting to
`gpt-5-mini`. Override its URL, key, and model via `vials_judge_base_url`,
`vials_judge_api_key`, and `vials_judge_model` in Gym's `env.yaml`. Other Gym judge
adapters can be selected by changing `judge_model_server` and loading their config.
Policy and judge bindings are independent.

All inference uses Gym's asynchronous model-server client and `call_judge` helper.
Transport/authentication failures propagate as `JudgeError`; Gym's `/verify`
failsafe records them in the retryable judge-failures sidecar and excludes them
from aggregate scores. Malformed received judge JSON is retried up to three times,
then scores zero with `judge_unparseable: true`. Missing fallback evidence is a
masked `verifier_error`. These failure distinctions follow Gym's contracts and
differ from upstream's treatment of exhausted calls as incorrect attempts.

## Validation

`VERIFIER_FIXTURE` exercises the real verifier with mocked model-server HTTP
responses. Component tests cover parsing, numeric-bound forwarding, panel order,
multimodal fallback, reasoning isolation, API failures, and the HTTP endpoint.
See `benchmarks/vials/README.md` for preparation and evaluation commands. A real
policy/judge smoke evaluation is still required before benchmarking or publication.
