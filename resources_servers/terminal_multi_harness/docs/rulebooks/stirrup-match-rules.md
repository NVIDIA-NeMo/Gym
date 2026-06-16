# Stirrup Match Rules

This is the normative Stirrup verifier contract for `terminal_multi_harness`.

## 1. Classify The Actual Response

Classify the actual model response as follows:

1. If the response contains no function calls, the actual response is a `message`.
2. If the response contains exactly one function call, the actual response is a single tool call.
3. If the response contains multiple sibling function calls in one response, the actual response is multiple tool calls in one response.
4. Ignore assistant preamble text on tool-call turns.

## 2. If Expected Answer Is A `message`

Match if and only if:

1. the actual response is a `message`
2. the actual message is non-empty after trimming

Otherwise it is a mismatch.

Do not compare completion-message text beyond the non-empty check.

## 3. If Expected Answer Is Tool-Based

The actual response must also be tool-based.

Then:

1. If expected is a single tool call, actual must also be a single tool call.
2. If expected is multiple tool calls in one response, actual must also be multiple tool calls in one response with the same count.
3. For multiple tool calls, sort expected and actual tool calls by `(tool_name, normalized_arguments_json)` before comparing them, where `normalized_arguments_json` is the tool's argument object re-serialized with sorted keys. This gives a stable ordering even when several sibling calls share a tool name.
4. Compare each aligned expected/actual tool-call pair using the single-tool-call rule below.
5. If any aligned pair fails, the whole response fails.

If the actual response is a plain `message` while expected is tool-based, it is a mismatch.

## 4. Single Tool Call Rule

For each expected/actual tool-call pair:

1. The actual tool call must validate against the declared tool schema in the current sample's `backend_request.tools`.
2. The actual tool name must match the expected tool name exactly.
3. The actual argument object must not contain any parameter key that is absent from the expected argument object.
4. After those checks pass, apply the tool-specific rule below.

If any step fails, the pair is a mismatch.

## 5. Tool-Specific Rules

If a tool is not listed below, the single-tool-call rule is sufficient and no additional value check is applied.

### `code_exec`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `cmd` must be present
3. normalize expected `cmd` and actual `cmd` by newline normalization and trimming outer whitespace
4. compute string similarity using the same metric as `terminus_judge`:
   - `difflib.SequenceMatcher(None, expected_cmd, actual_cmd).ratio()`
5. the `cmd` field matches only if that similarity score is at or above threshold
6. threshold uses the same shape as `terminus_judge`:
   - per-request threshold override when provided
   - otherwise the verifier default threshold
7. record the similarity score as verifier metadata
8. no other argument value is checked in this tool-specific step

### `web_search`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `query` must be present
3. normalize expected `query` and actual `query` by making everything lowercase and trimming outer whitespace
4. compute string similarity using the same metric as `terminus_judge`:
   - `difflib.SequenceMatcher(None, expected_query, actual_query).ratio()`
5. the `query` field matches only if that similarity score is at or above threshold
6. threshold uses the same shape as `terminus_judge`:
   - per-request threshold override when provided
   - otherwise the verifier default threshold
7. record the similarity score as verifier metadata
8. no other argument value is checked in this tool-specific step

### `fetch_web_page`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `url` must be present
3. normalize expected `url` and actual `url` by making everything lowercase and trimming outer whitespace
4. Check for exact match between expected and actual `url`
5. the url matches only if the exact match is found
6. no other argument value is checked in this tool-specific step

### `finish`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `reason` must be present and a non-empty string
3. actual `paths` must be present and a non-empty JSON array of strings
4. normalize expected `paths` and actual `paths` by stripping outer whitespace from each element and sorting the resulting list (order-insensitive, case-sensitive)
5. the `paths` field matches only if the normalized lists are equal; if they do not match, the pair is a mismatch and no further checks are performed on this tool call
6. normalize expected `reason` and actual `reason` by making everything lowercase and trimming outer whitespace
7. compute string similarity using the same metric as `terminus_judge`:
   - `difflib.SequenceMatcher(None, expected_reason, actual_reason).ratio()`
8. the `reason` field matches only if that similarity score is at or above threshold
9. threshold uses the same shape as `terminus_judge`:
   - per-request threshold override when provided
   - otherwise the verifier default threshold
7. record the similarity score as verifier metadata
8. normalize expected `path` and actual `path` by making everything lowercase and trimming outer whitespace
9. check for exact match between expected and actual `path`
10. the `path` field matches only if the exact match is found
11. no other argument value is checked in this tool-specific step 
