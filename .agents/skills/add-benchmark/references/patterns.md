# NeMo-Gym Benchmark Patterns Reference

Detailed patterns, schemas, and code examples for adding benchmarks to NeMo-Gym. Read this file when implementing a benchmark.

## Table of Contents

1. [Resources Server Pattern](#resources-server-pattern)
2. [YAML Config Pattern](#yaml-config-pattern)
3. [JSONL Data Schema](#jsonl-data-schema)
4. [Agent Patterns](#agent-patterns)
5. [Code Extraction Patterns](#code-extraction-patterns)
6. [Subprocess Execution with Ray](#subprocess-execution-with-ray)
7. [Test Patterns](#test-patterns)
8. [Data Conversion Script Pattern](#data-conversion-script-pattern)
9. [Dataset Registry Pattern](#dataset-registry-pattern)

---

## Resources Server Pattern

### Minimal verify-only server (e.g. `example_single_tool_call`)

```python
from nemo_gym.server_runtime import (
    SimpleResourcesServer, BaseResourcesServerConfig,
    BaseVerifyRequest, BaseVerifyResponse,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint

class MyConfig(BaseResourcesServerConfig):
    pass

class MyServer(SimpleResourcesServer):
    config: MyConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        model_output = body.response.output_text
        expected = (body.verifier_metadata or {}).get("expected_answer")
        reward = 1.0 if model_output.strip() == expected else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

if __name__ == "__main__":
    MyServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = MyServer.run_webserver()  # noqa: F401
```

### Subprocess execution server (e.g. `code_gen`)

```python
from asyncio import Semaphore, get_running_loop
from time import time
from typing import Any, Dict, List, Optional
import ray
from nemo_gym.server_runtime import (
    SimpleResourcesServer, BaseResourcesServerConfig,
    BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse,
)

class MyConfig(BaseResourcesServerConfig):
    num_processes: int = 8
    timeout_secs: int = 30
    debug: bool = False

class MyVerifyRequest(BaseRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None

class MyVerifyResponse(BaseVerifyResponse):
    extracted_code: Optional[str] = None
    # ... benchmark-specific result fields

class MyServer(SimpleResourcesServer):
    config: MyConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
        model_out = body.response.output_text
        if not model_out or not model_out.strip():
            return MyVerifyResponse(**body.model_dump(), reward=0.0)

        code = extract_code(model_out)  # your extraction function
        if not code:
            return MyVerifyResponse(**body.model_dump(), reward=0.0)

        async with self._semaphore:
            future = run_tests_remote.remote(code, body.verifier_metadata)
            result = await future

        return MyVerifyResponse(
            **body.model_dump(),
            reward=1.0 if result["all_passed"] else 0.0,
            extracted_code=code,
        )
```

### Key rules

- `verify()` must be async
- Return `reward` as 0.0 or 1.0 (binary for RL)
- Handle empty/missing model output gracefully (return 0.0, don't crash)
- `verifier_metadata` is the opaque dict from JSONL — define whatever fields your benchmark needs
- Guard optional nested fields: `(body.verifier_metadata or {}).get("key", default)`
- Use `asyncio.Semaphore` to bound concurrent subprocess/external calls
- For Ray remote tasks: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` in async context.

---

## External Tool Auto-Install Pattern

When a benchmark requires an external tool (compiler, runtime, etc.), auto-install it so users don't need manual setup.

### setup module (`setup_<tool>.py`)

```python
import logging, os, shutil, subprocess, sys
from pathlib import Path

LOG = logging.getLogger(__name__)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PREFIX = _SCRIPT_DIR / ".toolname"
_INSTALL_SCRIPT = _SCRIPT_DIR / "scripts" / "install_tool.sh"

def ensure_tool() -> None:
    if shutil.which("tool"):
        LOG.info("tool found: %s", shutil.which("tool"))
        return
    LOG.warning("tool not found on PATH — attempting auto-install")
    if sys.platform == "darwin":
        subprocess.run(["brew", "install", "tool-package"], check=True)
    elif sys.platform == "linux":
        prefix = os.environ.get("TOOL_PREFIX", str(_DEFAULT_PREFIX))
        env = os.environ.copy()
        env["TOOL_PREFIX"] = prefix
        subprocess.run(["bash", str(_INSTALL_SCRIPT)], check=True, env=env)
        os.environ["PATH"] = str(Path(prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(Path(prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")
    # Verify
    if not shutil.which("tool"):
        raise RuntimeError("Install completed but tool still not on PATH")
```

### Server integration (`app.py`)

```python
def model_post_init(self, context):
    ensure_tool()  # auto-install before any requests
    self._semaphore = Semaphore(value=self.config.num_processes)
```

### Test integration (`tests/conftest.py`)

`pytest.mark.skipif` evaluates at **module import time** (during collection), before any fixtures run. To auto-install before skip checks, use `pytest_configure`:

```python
from setup_tool import ensure_tool

def pytest_configure(config):
    try:
        ensure_tool()  # runs before test collection
    except Exception:
        pass
    # Now skipif(shutil.which("tool") is None) will find the tool
```

### Build script (`scripts/install_tool.sh`)

- Versions as variables at the top
- Configurable prefix via env var (default: `../.toolname/`)
- Idempotent: skip steps if artifacts already exist
- Check for prerequisites (`gcc`, `make`, `wget`/`curl`)

### Gitignore

Add `.toolname/` to the server's `.gitignore`.

---

## YAML Config Pattern

A single YAML file in `configs/` defines the resources server, the agent, and the `single_agent_turn` Environment Server that runs each episode:

```yaml
config_paths:
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml

# Route each taskset's rows to the Environment Server
environment_server_routes:
  my_benchmark: my_benchmark_environment_server
  my_benchmark_example: my_benchmark_environment_server

# Resources server instance
my_benchmark:
  resources_servers:
    my_benchmark:                    # must match subdirectory name
      entrypoint: app.py
      domain: coding                 # or: math, other
      # server-specific config fields:
      num_processes: 8
      timeout_secs: 30
      debug: false
      datasets:
      - name: my_dataset
        type: train
        taskset: my_benchmark        # collation turns each row into task_id + task_input
        jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
        gitlab_identifier:
          dataset_name: my_benchmark
          version: 0.0.1
          artifact_fpath: my_dataset.jsonl
        license: MIT                 # required for train/validation
        num_repeats: 1
      - name: example
        type: example
        taskset: my_benchmark_example
        jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl

# The agent that runs the agent turn
my_benchmark_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_benchmark           # matches instance name above
      model_server:
        type: responses_api_models
        name: policy_model           # defined in env.yaml or policy config

# The Environment Server: seed, agent turn, verify, close
my_benchmark_environment_server:
  _inherit_from: single_agent_turn
  environment_servers:
    single_agent_turn:
      resources_server:
        type: resources_servers
        name: my_benchmark
      agent_server:
        type: responses_api_agents
        name: my_benchmark_simple_agent
      resources_tool_transports: [direct_http]   # omit when the model calls no resources-server tools
```

To pair another agent, such as a session harness for a task sandbox, point `agent_server` at it; the resources server and datasets stay as they are.

### Key rules

- The `verified: false` flag is auto-added by pre-commit hook. Set to `true` after baselining.
- `license` is required for `train` and `validation` datasets.
- Valid license values: `Apache 2.0`, `MIT`, `CC-BY-4.0`, etc.
- `domain` should be one of: `coding`, `math`, `other`, or check `config_types.py` for current enum.
- Dataset `type` must be one of: `train`, `validation`, `example`.
- `gitlab_identifier` is required for `train`/`validation` datasets. `jsonl_fpath` is the local download path. Both fields coexist.
- `example` datasets don't have `gitlab_identifier` — they're committed to git directly.
- Every `taskset` needs a route in `environment_server_routes`. Run an `example` dataset by collating it first (`gym dataset collate ... --mode example_validation --output-dir <dir>`) and passing `<dir>/example.jsonl` to `gym eval run --no-serve --input`.
- Flat rows (no `taskset`) are accepted through `single_agent_turn_legacy`: include `environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml`, inherit `single_agent_turn_legacy`, and bind the same `resources_server` and `agent_server`. See `resources_servers/example_single_tool_call/configs/example_single_tool_call_single_agent_turn.yaml`.
- Do not front the agent with a `legacy_agent` relay. `gym env init` generates one as `<name>_environment_server`; replace it.

---

## JSONL Data Schema

Each line is a JSON object with this structure:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "System prompt here"},
      {"role": "user", "content": "Problem description here"}
    ]
  },
  "verifier_metadata": {
    "test_cases": [...],
    "task_id": "unique_id",
    "category": "optional_category"
  }
}
```

### Required fields

- `responses_create_params.input`: list of message dicts with `role` and `content`. Follows OpenAI message format.

### Optional fields

- `responses_create_params.tools`: function tool definitions if the benchmark uses tool calls
- `responses_create_params.temperature`, `max_output_tokens`, etc.
- `verifier_metadata`: arbitrary dict passed through to `verify()`. Define whatever your benchmark needs.
- Any other top-level fields — they pass through to the resources server via `BaseRunRequest`.

### Data conversion

Convert from source format to Gym JSONL using a conversion script. See [Data Conversion Script Pattern](#data-conversion-script-pattern) for the script template and [Dataset Registry Pattern](#dataset-registry-pattern) for the upload workflow.

---

## Agent Patterns

The Environment Server calls the agent only for the agent turn, inside an agent session (`/v1/agent_sessions`, `/v1/responses`, `/v1/agent_sessions/close`). Seed, verify and close belong to the Environment Server, so an agent never calls `/seed_session` or `/verify` itself.

### Simple agent (default — no custom code needed)

For single-turn benchmarks (model generates once, verify) and tool-calling benchmarks. Reference `simple_agent` as the Environment Server's `agent_server`; set `resources_tool_transports: [direct_http]` when the model calls resources-server tools.

### Multi-turn feedback

For benchmarks where the model gets error feedback and retries:

- **Feedback as a tool.** Add a resources-server route that checks an attempt and returns the errors, such as `POST /check_answer`, and declare it in the row's `responses_create_params.tools`. `simple_agent` calls it like any other tool within the agent turn, and `/verify` scores the final answer. The loop stays inside one agent turn, so it needs no custom agent.
- **Feedback the environment decides.** When the environment, not the model, decides each next turn (a user simulator, turn-based feedback), implement that loop in an Environment Server, as `environment_servers/nemo_user_sim` does.

Do not write a custom `run()` that seeds, generates, verifies and retries in a loop. `proof_refinement_agent` works that way, so it runs only behind a `legacy_agent` relay and is not a pattern for new benchmarks.

### Key rules

- Propagate cookies through every server call: `cookies=request.cookies`, then update with `response.cookies`. This is critical for stateful environments that track per-task sessions.
- Call `raise_for_status()` after every inter-server call
- Use `ConfigDict(extra="allow")` on request/response models for flexible field forwarding
- **Token ID propagation**: model responses include `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`. Propagate these from each model response into the next turn's input — required for RL training. See `swe_agents/app.py` for the implementation pattern.
- **Monotonic trajectories**: NeMo RL requires the token sequence across multi-turn rollouts to only grow. Never summarize, truncate, or modify prior turns between steps.
- **Thinking model compatibility**: thinking models emit `<think>`/`<thinking>` blocks. Strip these before parsing tool calls from model output, or parsing will break. See `_parse_think_tags` in `nemo_gym/models/chat_completions_converter.py` for the handling pattern.
- **Concurrency control for external services**: if the agent calls external services (Docker containers, APIs), use `asyncio.Semaphore` to throttle concurrent calls — external services may not handle thousands of simultaneous requests.

---

## Code Extraction Patterns

For benchmarks that need to extract code from model output:

1. Strip `<think>`/`<thinking>` blocks (reasoning traces from thinking models)
2. Extract from markdown code fences (` ```lang ``` `)
3. Fall back to language-specific markers
4. Validate extracted code has required structure

Handle these edge cases:
- Orphaned closing tags (`</think>` without matching `<think>`)
- Unclosed code fences
- Multiple code blocks (pick longest)
- No code fences at all (raw code in response)

---

## Subprocess Execution with Ray

For benchmarks that compile/run code:

```python
import ray
import subprocess
import tempfile
from pathlib import Path

@ray.remote
def run_tests_remote(code, test_cases, timeout=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source, compile, run tests
        src = Path(tmpdir) / "program.ext"
        src.write_text(code)

        try:
            result = subprocess.run(
                ["compiler", str(src)],
                capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"all_passed": False, "error": "timeout"}
        except FileNotFoundError:
            return {"all_passed": False, "error": "compiler not found"}

        # ... run test cases, compare output
        return {"all_passed": all_passed, ...}
```

### Key rules

- Decode subprocess output with `errors="replace"` to handle non-UTF8
- Implement fail-fast after N consecutive timeouts
- Handle `FileNotFoundError` for missing compilers/tools
- Use `tempfile.TemporaryDirectory` for isolation
- Executables must run on Linux

---

## Test Patterns

### Resources server tests

```python
import shutil
import pytest
from unittest.mock import MagicMock
from nemo_gym.client import ServerClient
from resources_servers.my_benchmark.app import MyServer, MyConfig

SKIP_REASON = "tool-name not installed"

@pytest.mark.skipif(
    shutil.which("tool-name") is None, reason=SKIP_REASON
)
class TestMyServer:
    def setup_method(self):
        self.config = MyConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        self.server = MyServer(config=self.config, server_client=MagicMock(spec=ServerClient))

    # Test cases: verify_pass, verify_fail_compile, verify_fail_wrong_output,
    #             verify_no_code, verify_timeout, code_extraction
```

### Agent tests

Mock `server_client.post()` to simulate server responses without running actual servers.

---

## Data Conversion Script Pattern

Conversion scripts and prompt files belong in the **source repo** (e.g. your dataset repository), not in NeMo-Gym. Only the converted JSONL files are uploaded to the GitLab dataset registry.

**Exception**: When there is no external source repo, keep the conversion script in the resources server directory.

Reference script template (for use in the source repo):

```python
import argparse
import json
from pathlib import Path

def convert_problem(problem: dict, system_prompt: str) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem["prompt"]},
            ]
        },
        "verifier_metadata": {
            "test_cases": problem["tests"],
            "task_id": problem["id"],
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", required=True, help="Path to system prompt text file")
    parser.add_argument("--example-output", help="Path to example.jsonl (first 5 entries)")
    args = parser.parse_args()

    system_prompt = Path(args.system_prompt).read_text().strip()

    with open(args.input) as f:
        data = json.load(f)

    with open(args.output, "w") as out:
        for problem in data:
            record = convert_problem(problem, system_prompt)
            out.write(json.dumps(record) + "\n")

    if args.example_output:
        with open(args.input) as f:
            data = json.load(f)
        with open(args.example_output, "w") as out:
            for problem in data[:5]:
                record = convert_problem(problem, system_prompt)
                out.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
```

Externalize system prompts to text files, pass via `--system-prompt` argument. Multiple prompt tiers enable ablation studies.

---

## Dataset Registry Pattern

### Dataset types and where they live

| Type | Location | Committed to git? | `gitlab_identifier`? |
|------|----------|-------------------|---------------------|
| `example` | `data/example.jsonl` | Yes | No |
| `train` | GitLab dataset registry | No | Yes |
| `validation` | GitLab dataset registry | No | Yes |

### `data/.gitignore` default patterns

Generated by `gym env init`:
```
*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
```

If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`).

### Upload workflow

1. **Generate** the JSONL file using your conversion script (in the source repo)
2. **Upload** to GitLab dataset registry:
   ```bash
   gym dataset upload --storage gitlab \
       --name my_benchmark \
       --revision 0.0.1 \
       --input resources_servers/my_benchmark/data/my_dataset.jsonl
   ```
3. **Add `gitlab_identifier`** to the dataset entry in YAML config:
   ```yaml
   - name: my_dataset
     type: train
     jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
     gitlab_identifier:
       dataset_name: my_benchmark
       version: 0.0.1
       artifact_fpath: my_dataset.jsonl
     license: MIT
   ```
4. **Ensure `.gitignore`** covers the filename (add custom pattern if needed)
5. **Remove from git** if previously tracked: `git rm --cached <file>`

### MLflow credentials

Upload/download requires MLflow credentials in `env.yaml`:
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

The tracking URI format is `https://<gitlab-host>/api/v4/projects/<PROJECT_ID>/ml/mlflow`.

### Verification with `gym dataset collate`

Validate example data (for PR submission):
```bash
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir /tmp/prepare \
    --mode example_validation
```

Download and prepare train/validation from GitLab:
```bash
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir data/my_benchmark \
    --mode train_preparation \
    --download \
    +data_source=gitlab
```

---

## External Benchmark Integration

When wrapping a 3rd-party benchmark library (e.g. SWE-bench, BigCodeBench), integrate at the agent server level, as an agent that serves agent sessions:

- `SimpleResponsesAPIAgent` registers `/v1/agent_sessions`, `/v1/responses` and `/v1/agent_sessions/close`. The adapter implements `_seed_agent_session_state()` (keep the tool and sandbox grants for the session), the session branch of `responses()` (pre-process the Gym request into library input, call the library, post-process its result into a `NeMoGymResponse`), and `_close_agent_session_state()` (release what the session holds and return its observations). `responses_api_agents/simple_agent/app.py` implements all three; `responses_api_agents/langchain_deepagents_agent` subclasses it to run an agent framework.
- The `single_agent_turn` Environment Server drives the episode, and the resources server's `verify()` scores the response.
- When the library's tasks bring their own environment and verifier, task folders in the Harbor format, run by the `harbor` Environment Server, are an equal choice.
- Do not wrap the library in `run()` behind a `legacy_agent` relay.
- The [`add-harness`](../../add-harness/SKILL.md) skill and the "Add or Update a Harness" page (`fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`) have the full session contract.

Add the dependency in `pyproject.toml`, then run `uv lock` and commit `uv.lock`.

Reproduction requirement: run the original repo first, reproduce published numbers, then integrate into Gym and reproduce again. This decouples Gym integration bugs from benchmark bugs.

### Replacing httpx with aiohttp in external libraries

Many external libraries use `httpx.AsyncClient` internally. NeMo Gym requires all async HTTP to go through its global aiohttp client (`nemo_gym.client.http_client.request()`) because httpx/httpcore has O(n^2) connection pooling that causes hangs at high concurrency. See `fern/versions/latest/pages/infrastructure/engineering-notes/aiohttp-vs-httpx.mdx`.

When wrapping such a library, create an adapter class that mimics the library's expected HTTP interface but routes calls through aiohttp. The `tavily_search` resources server demonstrates this pattern:

```python
from nemo_gym.client import request

class AIOHTTPAdapter(BaseModel):
    headers: Dict[str, str]
    base_url: str

    async def post(self, endpoint: str, content: str, **kwargs) -> AdapterResponse:
        response = await request(
            method="POST",
            url=f"{self.base_url}{endpoint}",
            headers=self.headers,
            data=content,
        )
        return AdapterResponse(status_code=response.status, data=await response.json())

    @classmethod
    def from_httpx_client(cls, client: AsyncClient) -> "AIOHTTPAdapter":
        return cls(headers=dict(client.headers), base_url=str(client.base_url))
```

Then in `model_post_init()`, replace the library's internal client:
```python
def model_post_init(self, context):
    self._external_client = ExternalLibraryClient(api_key=key)
    self._external_client._http_client = AIOHTTPAdapter.from_httpx_client(
        self._external_client._http_client
    )
```

---

## Reward Profiling Best Practices

### Model selection

Recommended model suite (as of Feb 2026):
- Your policy model of interest
- GPT-5 Nano, GPT-5 (closed-source baseline)
- Qwen 3 30B A3B Instruct 2507 (open-source instruct)
- Qwen 3 30B A3B Thinking 2507 (open-source thinking)
- If 30B models are too weak: Qwen 3 235B A22B variants, Kimi K2 Instruct, or GLM-4.7

### Validation checks

- Open-source models should achieve 30%+ on your benchmark. If not, investigate bugs.
- Closed-source models should score at or above open-source models. The reverse is rare and usually indicates a bug.
- Run both instruct and thinking models — the benchmark should be model-agnostic.
- Inspect actual failure cases in the rollout JSONL, not just aggregate numbers.

### Variance control

Run the highest-scoring open-source model multiple times. Increase `num_repeats` until variance < 1%.

### Training environments

If adding a training environment (not just a benchmark), additionally run training with NeMo RL:
- GRPO algorithm, 64 prompts per step, 16 rollouts per prompt (adjustable)
- Train with both instruct and thinking models
- Include W&B links and train/validation curve screenshots in the PR

---

## Token ID Propagation (Multi-Turn Training)

During training, model responses include `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs` on response messages. When constructing multi-turn input for the next model call, propagate these fields from the previous model response. This is required for RL training to work correctly.
