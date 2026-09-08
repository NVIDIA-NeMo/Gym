# VCQA Agent

Verified Code QA (VCQA) evaluates code-investigation ability on real GitHub
repositories. Each task gives the model a problem statement plus a per-task
working tree: either a repository snapshot tarball or a git bundle checked out
at a target commit. The agent exposes read-only file tools and sandboxed
`bash`, then grades the final assistant answer with an LLM judge against the
task's must-have rubric.

## Dataset

Rows and artifacts live in the private
[appliedcompute/vcqa-v1](https://huggingface.co/datasets/appliedcompute/vcqa-v1)
Hugging Face dataset.

```text
appliedcompute/vcqa-v1/
+-- rows/
|   +-- train.jsonl
|   +-- test.jsonl
+-- fileset/tars/<owner>/<name>/<sha>.tar.gz
+-- githistory/bundles/{feature_removal,bisect}/<owner>_<name>.bundle
```

Set `HF_TOKEN` before downloading rows or running rollouts. The agent also
uses `HF_TOKEN` as an `Authorization` header when it fetches per-task tarballs
and git bundles.

```bash
export HF_TOKEN=<your-hf-token>
```

Download the held-out split:

```bash
gym dataset download \
    --storage hf \
    --repo-id appliedcompute/vcqa-v1 \
    --revision v1.0.0 \
    --artifact rows/test.jsonl \
    --output responses_api_agents/vcqa_agent/data/test.jsonl
```

Download the training split:

```bash
gym dataset download \
    --storage hf \
    --repo-id appliedcompute/vcqa-v1 \
    --revision v1.0.0 \
    --artifact rows/train.jsonl \
    --output responses_api_agents/vcqa_agent/data/train.jsonl
```

`data/example.jsonl` is included as a schema example. The example rows use
representative artifact keys and are not a substitute for the private dataset
artifacts. `data/example_rollouts.jsonl` contains five generated rollout rows
with populated reward and judge fields for quick inspection.

## Input Schema

Each row contains OpenAI Responses create parameters plus verifier metadata:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "<problem statement>"}
    ]
  },
  "verifier_metadata": {
    "task_id": "<unique id>",
    "dataset_kind": "fileset",
    "repo_full_name": "owner/repo",
    "pre_merge_sha": "<sha>",
    "head_sha": "<sha>",
    "artifact_key": "fileset/tars/owner/repo/sha.tar.gz",
    "rubric": {
      "judge": [
        {
          "id": "j01",
          "title": "Required fact the answer must include.",
          "importance": "must have",
          "evidence_required": ["path/to/file.py"]
        }
      ]
    },
    "max_turns": 150
  },
  "agent_ref": {"name": "vcqa_agent"}
}
```

`dataset_kind` controls materialization:

- `fileset`: fetch a `.tar.gz`, extract it, and mount the working tree
  read-only at `/codebase`.
- `githistory`: fetch a git bundle, clone it, and check out `head_sha`.

The reward is `must_pass / must_total` over rubric items marked
`importance: "must have"`. Legacy `{category, description}` rubrics are also
accepted.

## Running Rollouts

Create an `env.yaml` in the Gym repo root with your OpenAI-compatible model
server details:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: <your-api-key>
policy_model_name: gpt-4.1-2025-04-14
```

Start the agent and model server:

```bash
gym env start \
    --config responses_api_agents/vcqa_agent/configs/vcqa_agent.yaml \
    --model-type openai_model
```

In another terminal, collect held-out rollouts:

```bash
gym eval run --no-serve \
    --agent vcqa_agent \
    --input responses_api_agents/vcqa_agent/data/test.jsonl \
    --output results/vcqa_test_rollouts.jsonl \
    --num-repeats 1 \
    --concurrency 2
```

`gym eval run` writes sidecar files next to the output, including
`*_materialized_inputs.jsonl`, `*_failures.jsonl`, and
`*_aggregate_metrics.json`.

## Sandbox Backends

The default backend is `apptainer`, which starts one Apptainer instance per
rollout and runs all tool calls inside that instance. The default image is
`docker://debian:bookworm-slim`; the configured setup command installs
`ripgrep`, `git`, `fd-find`, `tree`, and `ca-certificates`.

For nested-container clusters where `apptainer exec instance://...` is not
available, use `apptainer_exec`. This runs direct `apptainer exec` per tool
call and requires an image with the tool dependencies already installed:

```bash
gym env start \
    --config responses_api_agents/vcqa_agent/configs/vcqa_agent.yaml \
    --model-type openai_model \
    ++vcqa_agent.responses_api_agents.vcqa_agent.sandbox_backend=apptainer_exec \
    ++vcqa_agent.responses_api_agents.vcqa_agent.container_image=/path/to/vcqa-tools.sif \
    ++vcqa_agent.responses_api_agents.vcqa_agent.apptainer_setup_command=""
```

For local development without Apptainer, use `sandbox_backend=local`. This is
not isolated and should only be used with trusted artifacts.

## Tools

The model gets six real tools:

| Tool | Behavior |
| ---- | -------- |
| `read_file` | Read a file under `/codebase`, capped by `max_bytes`. |
| `grep` | Search under `/codebase` with ripgrep. |
| `glob` | Find files under `/codebase` with `fd`. |
| `list_dir` | List a directory under `/codebase`. |
| `write_todos` | Append notes to `/tmp/scratch/todos.md`. |
| `bash` | Run `bash -lc <command>` inside the sandbox. |

Optional distractor tools (`install_package`, `send_pr_review`, `websearch`,
`ask_user`) are enabled by default through `include_distractor_tools: true`.
They exercise off-task tool selection but are not directly penalized by the
judge.

All tool handlers return structured JSON. Invalid arguments, path escapes,
non-zero exits, and timeouts are returned to the model as tool results instead
of crashing the rollout.

## Tests

Run the focused test suite:

```bash
pytest responses_api_agents/vcqa_agent/tests
```

The Apptainer end-to-end smoke test is skipped automatically when `apptainer`
is not on `PATH`. The remaining tests cover path containment, tool schemas,
tool error handling, rubric parsing, judge scoring, artifact fetching, and
failure isolation.
