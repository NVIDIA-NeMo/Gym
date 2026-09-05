# Model backend compatibility namespace

Built-in backends moved to [`model_backends/`](../model_backends/) as part of MB-1553. This package keeps
legacy Python imports such as `responses_api_models.openai_model.app` working. New code should use
`model_backends.openai_model.app`.

The `responses_api_models` key in YAML is an internal server protocol key and remains supported.
