# fix/responses-output-hosted-tool-items — Test Contract

## Functional Behavior

- Every provider-executed or client-executed output-call variant supported by the pinned OpenAI SDK validates in `NeMoGymResponse.output`.
- Each wrapper preserves the upstream requirement that the `type` discriminator be present; malformed items without it are rejected.
- `split_responses_input_output_items` starts the output partition at every model-output item type, including hosted MCP and provider- or client-executed calls.
- A transcript containing only input items stays entirely on the input side.
- Conversational transcript canonicalization retains a tool-only model response instead of moving it into request input.
- The PR body and commit metadata contain none of the prohibited attribution terms requested by the contributor.

## Unit Tests

- `TestNeMoGymResponseToolCallItems.test_output_call_items_require_type_discriminator` rejects every new item when `type` is absent.
- `test_split_on_model_output_item_type` covers every non-message output discriminator.
- `test_split_input_only_items` covers the no-output boundary.
- `test_canonicalize_run_transcript_preserves_tool_only_model_output` covers tool-only transcript canonicalization.

## Integration / Functional Tests

- The OpenAI model-server tests and response-schema tests pass together.
- The conversational-tool-use simulation tests pass with the shared splitter change.

## Smoke Tests

- `uv run pytest tests/unit_tests/test_openai_utils.py tests/unit_tests/test_responses_converter.py responses_api_models/openai_model/tests/test_app.py responses_api_agents/conversational_tool_use/simulation/tests/test_app.py -q`
- `uv run pre-commit run --files nemo_gym/openai_utils.py nemo_gym/responses_converter.py tests/unit_tests/test_openai_utils.py tests/unit_tests/test_responses_converter.py responses_api_agents/conversational_tool_use/simulation/app.py responses_api_agents/conversational_tool_use/simulation/tests/test_app.py testing/fix-responses-output-hosted-tool-items.md`

## E2E Tests

N/A — the live provider response was already reproduced for issue #2436; this review adds deterministic regressions for the local validation and transcript paths.

## Manual / cURL Tests

- Run a local malformed-item probe and verify the pinned SDK and Gym both reject every new item without `type`.
- Run a local splitter probe and verify `[user, web_search_call, assistant]` becomes one input item and two output items.
- Inspect PR #2438 body and all branch commit metadata for the prohibited attribution terms before pushing.
