# GDP.pdf

Native NeMo Gym verifier for [Surge AI GDP.pdf](https://huggingface.co/datasets/surgeai/GDP.pdf), following the Artificial Analysis intelligence-benchmarking methodology v4.3.

The verifier makes one independent LLM-judge call per hidden rubric criterion. The judge receives only the original task, candidate answer, and one criterion. `reward` is the criterion mean-pass value for a dense RL signal. Aggregate evaluation reports both All-pass (the headline metric) and Mean Pass.

Policy and judge transport failures remain distinct from wrong answers through NeMo Gym's standard failure sidecar and retry path. Empty policy output is a scored zero attempt. Malformed judge outputs are retried up to five times, and a valid verdict is required for every criterion before an attempt is accepted.

The public GDP.pdf split is evaluation-only and must not be used for model training. The environment may be used for RL with compatible, independently generated and licensed tasks that use the same schema and verifier.

## Licensing

- Environment code: Apache-2.0
- GDP.pdf public data: MIT; evaluation-only contamination warning and canary apply
- LiteParse: Apache-2.0
