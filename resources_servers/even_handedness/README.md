# Political Even-handedness resources server

The server applies Anthropic's five-call public grading protocol to two saved
policy responses. It is stateless for reverification: the same policy-response
pair can be regraded without regenerating either response.

Probability scoring is strict. Missing option logprobs are classified as a
judge failure and routed outside the model-quality denominator. Discrete
scoring is available only through an explicit configuration override and is
recorded on every result row.

The input contract is documented in `task_data.py`; benchmark preparation and
reproduction instructions live under `benchmarks/even_handedness/`.
