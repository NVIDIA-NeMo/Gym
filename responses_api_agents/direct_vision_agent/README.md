# Direct vision agent

An evaluation harness for a single visual question stored in an environment-owned
sandbox. It makes one Chat Completions request through Gym's Model Server, with
`tool_choice: none` and no `tools` field, and writes the visible answer to a file.
It exposes no shell, Python, OCR, MCP, or retrieval tools to the model.

The environment server creates and destroys the sandbox. The agent accepts its
`SandboxAccess` during session seed, reads public inputs by file transfer, and
releases its connection on close. It rejects tool grants, unseeded inference,
and returned tool calls. Completed responses are reused within a session,
including when retrying an answer-file transfer. Infrastructure retries that
start a new episode may make a new model call.

## Public input contract

The configurable `input_path` (default `/task/input.json`) contains:

```json
{"system": "You are a scientist.", "user": "What does the image show?", "images": ["panel.png"]}
```

`system` is optional. Images are ordered, distinct filenames in the manifest's
own directory. Absolute paths, traversal, non-image filenames, and extra
manifest fields are rejected. The agent never reads the task's verifier or
reference solution. `answer_path` defaults to `/task/answer.txt`.

Configuration lives in `configs/direct_vision_agent.yaml`. Set `model_server`
to the Gym Model Server instance. Generation settings are `max_output_tokens`
(including reasoning), `temperature`, `top_p`, `chat_template_kwargs`, and
`request_timeout_sec`. The endpoint must support the selected limits and images.

The first benchmark pairing is VIALS; see `benchmarks/vials/README.md` for its
preparation and native Harbor environment server commands. The input/output
contract is generic; the agent imports no VIALS code and performs no grading.

## Evidence and limitations

The capability level is Evaluation. The returned transcript preserves public
prompt text, ordered image paths, visible answers, separate reasoning, and provider-reported
usage; absent usage remains unknown. Gym's strict ATIF exporter currently supports
text content only. The complete request, including inline image bytes, and raw
response are saved to `policy-input.json` and `policy-response.json` under
`artifacts_dir` (default `/logs/artifacts`, collected by the Harbor environment
server). Session close explicitly reports an `input_media_in_artifact` gap;
the text transcript does not claim to encode the actual image blocks. Enable Gym model-call capture when a raw
request/response audit is needed. Session close currently reports an
`observation_capture_failed` gap because invocation-level observation records
are not implemented. This is not an Observability or Training capability claim.

Shell-tool conformance probes do not apply: no function-calling shell tool is
exposed. The agent offers no host-only path, continuation, task tool context,
or sandbox ownership. Its file transfers are harness operations.

```bash
pytest responses_api_agents/direct_vision_agent/tests
```
