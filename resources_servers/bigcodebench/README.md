# bigcodebench

Verifies model-generated Python solutions against the
[BigCodeBench](https://github.com/bigcode-project/bigcodebench) unittest
suite. Each task ships its own `unittest.TestCase` class plus an
`entry_point` function name; the model's extracted code is calibrated
(`code_prompt + "\n    pass\n" + extracted`) and run through
`bigcodebench.eval.untrusted_check` in an isolated subprocess.

## Why this server has its own venv

BigCodeBench tests import 70+ third-party libraries with versions pinned
in [`Requirements/requirements-eval.txt`](https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt)
(`numpy==1.21.2`, `keras==2.11.0`, `tensorflow==2.11.0`, ...). Most of
those pins are 3.10-only, while NeMo Gym ships Python 3.12. On startup
the server uses `uv` to build a Python 3.10 venv at `.bcb_venv/` and
installs `bigcodebench` + `requirements-eval.txt` into it. Each
`/verify` shells out to `bcb_runner.py` running under that venv's
`python`. First start is slow (~10 min on a fresh node); subsequent
starts are instant.

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/bigcodebench/configs/bigcodebench.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=bigcodebench_simple_agent \
    +input_jsonl_fpath=resources_servers/bigcodebench/data/example.jsonl \
    +output_jsonl_fpath=results/bigcodebench_rollouts.jsonl \
    +num_repeats=1
```

## Reasoning-parser note

When serving a reasoning model (Nemotron, Qwen3-Thinking, DeepSeek-R1,
GPT-OSS), start vLLM with `--reasoning-parser <name>` so `<think>...</think>`
is stripped at the model edge. Without it, the code-extractor's
`</think>`-stripping rule still fires on the resource-server side, but
post-hoc surgery diverges from Skills' parse-reasoning-True behaviour on
truncated rollouts. See the migration recipe's `run_*.py` for the
canonical setup.

## Calibration

`bigcodebench.evaluate.evaluate(..., calibrated=True)` prepends the
benchmark's `code_prompt` (function signature + docstring stub) plus
`"\n    pass\n"` to the model's solution before running the unittest.
This guarantees the entry_point is defined even when the model returned
only the function body. We replicate it verbatim in `verify()`.
