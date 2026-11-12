# HealthBench Rubric Resources Server

Stateless rubric grading for HealthBench and HealthBench Professional, composed with Gym's simple agent. The server performs no policy inference itself and owns no tools or sandbox.

The scorer awards signed rubric points for met criteria, normalized by all positive points. Professional applies the reference character-length adjustment. Raw per-row scores are retained, the Gym reward is bounded to [0, 1], and headline benchmark metrics clamp the mean of raw scores. See [the benchmark protocol and onboarding checklist](../../benchmarks/healthbench/onboarding.md).

Judge concurrency, per-call timeout and malformed-grade attempts are declared in the YAML. Transport errors propagate through `call_judge`; malformed grades retry before raising `JudgeError`. Gym masks those failures. Empty policy answers are unmasked zero-score results. Stateless reverification is supported.

`VERIFIER_FIXTURE` exports offline full, zero, and malformed cases. The fixture substitutes only the external judge; it runs the real verifier. Unit tests additionally cover masked errors, timeouts, parsing retries, signed points, length adjustment, judgement provenance, and aggregation. The five committed example tasks are synthetic.

```bash
gym env test healthbench
gym env test healthbench_professional
pytest resources_servers/healthbench_rubric/tests benchmarks/healthbench/tests
```

The grader prompt is copied verbatim from MIT-licensed OpenAI simple-evals at commit `652c89d0ca9df547706735883097e9537d40dc47`; its license is in `prompts/LICENSE`. The implementation remains `verified: false` pending live-run and reference-score evidence.
