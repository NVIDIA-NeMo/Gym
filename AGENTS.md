# AGENTS.md

General architecture, commands, and code style live in `CLAUDE.md` at the repo root — read it first. This file only adds cloud/environment-specific caveats.

## Cursor Cloud specific instructions

### Environment
- Python 3.12 + `uv` are preinstalled. `uv` lives at `~/.local/bin/uv` and is already on the login-shell `PATH`. The startup update script runs `uv sync --extra dev`, so the root venv at `.venv/` is ready.
- Activate the environment before running anything: `source .venv/bin/activate`.
- This VM is **CPU-only (no GPU / no `nvidia-smi`)**. Do not use GPU model paths (`local_vllm_model`, `genrm_model`, or vLLM-backed servers) — they need a GPU. The core library, resources servers, agents, and the `vllm_model` proxy are all CPU-only.

### Running / testing (see `CLAUDE.md` for the full command list)
- Set `RAY_TMPDIR=/tmp` when running anything that starts Ray (`gym dev test`, `gym eval run`). Ray's AF_UNIX socket path can otherwise exceed the 107-byte limit.
- `gym env test --resources-server <name>` builds an **isolated per-server venv** on first run (slow, one-time). `os.environ` changes do not propagate into these venvs — set env vars (e.g. `RAY_TMPDIR`) in the outer shell.
- `gym dev test` enforces a 96% coverage gate. With only `--extra dev` installed (no `--extra sandbox`), the sandbox provider modules are uncovered and the gate reports failure even though the tests themselves pass — this is expected, not a regression.
- Known non-blocking failure: `tests/unit_tests/test_cli_utils.py::TestPrintRichTable::test_not_truncated_regardless_of_ambient_width` depends on `rich` console-width detection in a non-TTY and fails in this environment; unrelated to setup.

### Running a full rollout without an external LLM API key (no-GPU e2e)
There is no built-in mock model server. To exercise the full `gym env start` + `gym eval run` pipeline offline, point the `vllm_model` proxy at a local OpenAI-compatible Chat Completions mock:
- Create `env.yaml` (gitignored) at the repo root:
  ```yaml
  policy_base_url: http://127.0.0.1:9000/v1
  policy_api_key: dummy_key
  policy_model_name: mock-model
  ```
- Run any OpenAI-compatible server on that URL that implements `POST /v1/chat/completions` (the `vllm_model` proxy, with default `is_responses_native: false`, converts Responses → Chat Completions and calls `{base_url}/chat/completions`; `/tokenize` is only hit when `return_token_id_information: true`).
- Then: `gym env start --resources-server mcqa --model-type vllm_model`, and in another shell `gym eval run --no-serve --agent mcqa_simple_agent --input resources_servers/mcqa/data/example.jsonl --output results/mcqa_rollouts.jsonl --limit 5`.
- With a real hosted model instead, use `--model-type openai_model` and put the provider `policy_base_url` / `policy_api_key` in `env.yaml`.

### Git
- Commits require DCO sign-off (`git commit -s`). Commit signing (`-S`) is configured per-developer; the cloud VM may not have a signing key, so `-s` alone is sufficient here.
