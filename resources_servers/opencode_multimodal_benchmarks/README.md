# Visual benchmark verification

This resources server scores VStar, HRBench8K, VTCBench, Agent-X and BabyVision
responses. Dataset preparation produces Gym task rows with reference labels in
`verifier_metadata`, separate from model inputs and the episode sandbox.

VStar and HRBench8K use multiple-choice extraction with a reference-blind judge
fallback. VTCBench uses the public rule matcher followed by an LLM judge for rule-negative answers. Agent-X uses its Goal
Accuracy rubric and retains fractional subjective scores. BabyVision uses
LLM answer verification. Empty answers score zero.

Supply `judge_model` and either `judge_server` or `judge_endpoint` explicitly.
There is no built-in deployment endpoint or model route. Configure `judge_server`
to route judge calls through a Gym model server with
rollout correlation and capture. Judge failures remain errors that require
resolution before reporting a complete dataset score. The OpenCode evaluation
agent records recognized agent failures as zero-scored episodes with failure
metadata; unclassified infrastructure errors remain unfinished.

See the [benchmark guide](../../benchmarks/opencode_multimodal_benchmarks/ONBOARDING.md) for
input formats, preparation, configuration and validation, and the
[agent README](../../responses_api_agents/opencode_multimodal_agent/README.md)
for execution and tool policies.

VTCBench scoring follows the [public evaluator](https://github.com/zhuzil/VTC-Bench/blob/5e0a5533f597b0c56dae7d9705771498202ab19b/eval/eval/VLMEvalKit/vlmeval/dataset/image_vqa.py).
Rules inspect the whole visible response. The judge extracts option letters for
multiple-choice tasks and checks semantic equivalence for free answers. Set
`vtc_llm_fallback: false` for the public rule-only mode. Results identify the
scorer version, scoring method and judge model. Invalid judge verdicts are retried
five times using the public temperature schedule. Unlike the public script, an
exhausted judge failure remains retryable rather than silently scoring zero or
falling back to rule-only results. Empty agent answers score zero without judging.
