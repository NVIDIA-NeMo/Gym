# Agent006 Match Rules

This is the normative Agent006 verifier contract for `terminal_multi_harness`.
These rules apply only when the verify request sets `harness = "agent006"`.
Same-named tools in other harnesses do not inherit Agent006-specific checks.

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

Do not compare message text beyond the non-empty check.

## 3. If Expected Answer Is Tool-Based

The actual response must also be tool-based.

Then:

1. If expected is a single tool call, actual must also be a single tool call.
2. If expected is multiple tool calls in one response, actual must also be multiple tool calls in one response with the same count.
3. For multiple tool calls, sort expected and actual tool calls by tool name before comparing them. If tool names tie, use the canonicalized JSON argument string as a stable tie-breaker.
4. Compare each aligned expected/actual tool-call pair using the single-tool-call rule below.
5. If any aligned pair fails, the whole response fails.

If the actual response is a plain `message` while expected is tool-based, it is a mismatch.

## 4. Single Tool Call Rule

For each expected/actual tool-call pair:

1. The actual tool call must validate against the declared tool schema in the current sample's `declared_tools`.
2. The actual tool name must match the expected tool name exactly.
3. The actual argument object must not contain any parameter key that is absent from the expected argument object.
4. After those checks pass, apply the tool-specific rule below.

If any step fails, the pair is a mismatch.

## 5. Tool-Specific Rules

If a tool is not listed below, the single-tool-call rule is sufficient and no additional value check is applied.

### `execute_python`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `code` must be present and be a non-empty string
3. expected `code` must be present and be a non-empty string
4. compute string similarity:
   - `difflib.SequenceMatcher(None, expected_code, actual_code).ratio()`
5. the `code` field matches only if that similarity score is at or above threshold
6. threshold uses the same shape as other harnesses:
   - per-request threshold override when provided
   - verifier default threshold otherwise
7. record the similarity score as verifier metadata

### `return_result`

After the single-tool-call rule passes:

1. parse the actual argument object
2. actual `result` must be present
3. serialize both expected and actual `result` values to canonical JSON:
   - `json.dumps(value, sort_keys=True)`
4. compute string similarity:
   - `difflib.SequenceMatcher(None, expected_json, actual_json).ratio()`
5. the `result` field matches only if that similarity score is at or above threshold
6. record the similarity score as verifier metadata

## 6. String Similarity Policy

1. Do not use string similarity unless a tool-specific rule explicitly requires it.
2. In this branch, the string-similarity cases are:
   - `execute_python.code`
   - `return_result.result`
3. When string similarity is used, use the same metric as other harnesses:
   - `difflib.SequenceMatcher(None, s1, s2).ratio()`
4. When string similarity is used, pass/fail is determined by threshold:
   - request override first
   - verifier default threshold otherwise
