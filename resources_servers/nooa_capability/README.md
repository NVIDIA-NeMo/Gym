# NOOA capability verifier

This Resources server preserves the value normalization and binary equality rules of NOOA v0.0.9's `eval_pipeline.scoring.ExactMatchScorer` inside Gym's `/verify` lifecycle.

Rows provide `expected_result` as verifier-only data. The server extracts the last assistant message, parses JSON or Python literal values recursively, unwraps common scalar result envelopes, and returns reward `1.0` for a match or `0.0` otherwise.

It is intentionally separate from `string_match`: that general benchmark verifier performs answer-format, punctuation, LaTeX, list, and near-numeric normalization that would change the source capability score.

The initial consumer is `responses_api_agents/nooa_agent/configs/nooa_calculate_capability.yaml`.
