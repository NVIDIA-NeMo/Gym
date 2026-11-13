# Visual benchmark verification

This resources server scores VStar, HRBench8K, VTCBench and ZeroBench
responses. Dataset preparation produces Gym task rows with reference labels in
`verifier_metadata`, separate from model inputs and the episode sandbox.

VStar and HRBench8K use multiple-choice extraction with a reference-blind judge
fallback. VTCBench uses the public rule matcher followed by an LLM judge for rule-negative answers. ZeroBench uses the public last-curly-brace extraction
and case-insensitive exact comparison, without a judge. Empty answers score zero.

For judge-based benchmarks, supply `judge_model` and either `judge_server` or `judge_endpoint` explicitly.
There is no built-in deployment endpoint or model route. Configure `judge_server`
to route judge calls through a Gym model server with
rollout correlation and capture. Judge failures remain errors that require
resolution before reporting a complete dataset score. For agent sessions, the
Environment Server's `agent_outcome_policy` determines whether a failed attempt
is verified, retried or left ungraded. The OpenCode session evaluation
`/run` path scores established model-caused failures as zero; provider 429/5xx,
transport, setup and unclassified process failures remain retryable or unfinished.

See the [benchmark guide](../../benchmarks/opencode_multimodal_benchmarks/ONBOARDING.md) for
input formats, preparation, configuration and validation, and the
[agent README](../../responses_api_agents/opencode_agent/README.md)
for execution and tool policies.

VTCBench scoring follows the [public evaluator](https://github.com/zhuzil/VTC-Bench/blob/5e0a5533f597b0c56dae7d9705771498202ab19b/eval/eval/VLMEvalKit/vlmeval/dataset/image_vqa.py).
Rules inspect the final `Answer:` line, or the last nonempty line, and require
normalized exact free-answer matching. This is a documented scoring deviation
from the pinned evaluator; results use the v3 scorer and require rebaselining. The judge extracts option letters for
multiple-choice tasks and checks semantic equivalence for free answers. Set
`vtc_llm_fallback: false` for the public rule-only mode. Results identify the
scorer version, scoring method and judge model. Invalid judge verdicts are retried
five times using the public temperature schedule. Unlike the public script, an
exhausted judge failure remains retryable rather than silently scoring zero or
falling back to rule-only results. Empty agent answers score zero without judging.

The five color-square examples use varied correct option letters and fresh hosted
Qwen policy replies, recorded with temperature 0.6, top-p 0.95, and a 512-token
output limit. They are input/verifier examples, not a benchmark baseline or an
OpenCode tool demonstration. Wrong-answer, judge, and masked-error cases live in
the offline verifier fixtures and grading-evidence tests.
