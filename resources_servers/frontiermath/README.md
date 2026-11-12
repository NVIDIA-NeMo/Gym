# FrontierMath verifier

Grades the final assistant answer using exact integer or symbolic equivalence.
The last `\boxed{...}` outside reasoning tags must be complete.
Indic decimal digits, bounded finite sums, and supported generating-function
coefficients are accepted. Approximate values and unevaluated variables fail.

The server uses Gym's `SimpleResourcesServer`, `simple_agent`, and `vllm_model`.
Answer extraction and symbolic grading run together in bounded subprocesses with
a timeout. Each result includes
`reward`, `extracted_answer`, and `grading_status`. No LLM judge is used.
Worker failures and timeouts set `mask_sample=true`, `failure_kind=verifier_error`,
and a diagnostic `failure_reason`. They are excluded from accuracy; missing or
invalid model answers remain unmasked zero rewards.

```bash
python -m pytest resources_servers/frontiermath/tests
```
