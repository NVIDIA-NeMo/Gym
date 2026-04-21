# Opencode Match Rules

This is the normative Opencode verifier contract for `terminal_multi_harness`.
These rules apply only when the verify request sets `harness = "opencode"`.
Same-named tools in other harnesses do not inherit Opencode-specific checks.

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
3. For multiple tool calls, sort expected and actual tool calls by `(tool_name, canonicalized_arguments_json_string)` before comparing them.
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

### `bash`

After the single-tool-call rule passes:

1. parse expected and actual argument objects
2. actual `command` must be present
3. normalize expected `command` and actual `command` by newline normalization and trimming outer whitespace
4. compute string similarity using:
   - `difflib.SequenceMatcher(None, expected_command, actual_command).ratio()`
5. the `command` field matches only if that similarity score is at or above threshold
6. threshold uses:
   - per-request threshold override when provided
   - verifier default threshold otherwise
7. record the similarity score as verifier metadata
8. ignore `description`, `workdir`, and `timeout` value differences when those keys are present on both sides

### `read`

After the single-tool-call rule passes:

1. compare `filePath` exactly
2. ignore `offset` value differences when the key is present on both sides
3. ignore `limit` value differences when the key is present on both sides

### `write`

After the single-tool-call rule passes:

1. compare `filePath` exactly
2. ignore `content`, including empty content

### `edit`

After the single-tool-call rule passes:

1. compare `filePath` exactly
2. actual `newString` and actual `oldString` must both be present and must not be equal
3. compare optional `replaceAll` exactly
4. if `replaceAll` is missing on both sides, that is a match

### `glob`

After the single-tool-call rule passes:

1. compare `pattern` exactly
2. compare optional `path` exactly
3. if `path` is missing on both sides, that is a match

### `grep`

After the single-tool-call rule passes:

1. compare `pattern` exactly
2. compare optional `path` exactly
3. compare optional `include` exactly
4. if an optional field is missing on both sides, that field matches

### `webfetch`

After the single-tool-call rule passes:

1. compare `url` exactly
2. compare `format` exactly
3. ignore `timeout` value differences when the key is present on both sides

### `skill`

After the single-tool-call rule passes:

1. compare `name` exactly

### `task`

After the single-tool-call rule passes:

1. compare `subagent_type` exactly
2. ignore `description` value differences when the key is present on both sides
3. ignore `prompt` value differences when the key is present on both sides
4. ignore `task_id` value differences when the key is present on both sides
5. ignore `command` value differences when the key is present on both sides

### `todowrite`

After the single-tool-call rule passes:

1. if expected `todos = []`, actual matches only if actual `todos = []`
2. otherwise expected and actual `todos` must both be lists of the same length
3. compare todo items in order
4. each todo item matches on `status` and `priority` only
5. ignore todo `content`

## 6. String Similarity Policy

1. Do not use string similarity unless a tool-specific rule explicitly requires it.
2. In this branch, the only string-similarity case is `bash.command`.
3. When string similarity is used, use:
   - `difflib.SequenceMatcher(None, s1, s2).ratio()`
4. When string similarity is used, pass/fail is determined by threshold:
   - request override first
   - verifier default threshold otherwise
