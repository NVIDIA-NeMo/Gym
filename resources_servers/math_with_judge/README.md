# Description
This is a resources server for verifying the correctness of answers to mathematical problems.  It uses a combination of the Hugging Face Math-Verify library and an LLM as a judge.

The problems in the OpenMathReasoning dataset are taken from the
[OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning)
on Hugging Face.


# Example usage

## Running servers
The following are example commands for running this resources server, along with the simple agent and an OpenAI model:
```bash
gym env start \
    --model-type openai_model \
    --resources-server math_with_judge \
    +math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

The OpenMathReasoning download below and `--download` under "Prepare for trajectory collection" fetch from the
GitLab model registry. They need the `mlflow` extra (`uv sync --extra mlflow`) and `mlflow_tracking_uri` and
`mlflow_tracking_token` in `env.yaml`.

To download the OpenMathReasoning dataset, the following command can be run:
```bash
gym dataset download --storage gitlab \
    --name math_open_math_reasoning \
    --revision 0.0.1 \
    --artifact open_math_reasoning_problems.jsonl \
    --output data/open_math_reasoning_problems.jsonl
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent math_with_judge_simple_agent \
    --input data/open_math_reasoning_problems.jsonl \
    --output results/example_open_math_reasoning_verify_responses.jsonl \
    --limit 5
```

## Prepare for trajectory collection
This writes `validation.jsonl` with each DAPO17k train problem repeated 16 times. NeMo RL reads only the
validation split when `is_trajectory_collection: true`. `--download` fetches `dapo17k` 0.0.1 from the GitLab
model registry.
```bash
gym dataset collate \
    --config resources_servers/math_with_judge/configs/dapo17k_trajectory_collection.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --output-dir data/dapo17k_trajectory_collection \
    --mode train_preparation \
    --download
```

# Sandbox

Math-Verify parses the model's answer with SymPy, so the verifier grades in a sandbox. When the Environment Server
lends one to `/verify` (`sandbox_opt_out.verifier: false`, as in `benchmarks/aime25/config.yaml`), the Math-Verify
check runs there instead of in this server.

AIME opts only the task out (`sandbox_opt_out: {task: true, verifier: false}`): the agent loop runs in the Agent
Server, and the Environment Server starts a sandbox with no network just before `/verify`. Docker, E2B, and
OpenSandbox with `networking.enforces_network_policy: true` can apply `no_network`; Enroot applies only `inherit`, so
such a config refuses to start on it.

How the check runs in the lent sandbox:

- `library_check.py` holds the check. It imports only the standard library and Math-Verify, so the same file runs in
  this server and, uploaded, in the lent sandbox.
- A sandbox that lacks the check's private Python runtime gets it first: a pinned CPython with Math-Verify at this
  server's versions. The sandbox needs no network; this host builds the runtime once (it needs `uv` and access to
  the package index) and uploads it to each sandbox that lacks it, which is every attempt's when each `/verify` gets
  a fresh sandbox. `sandbox_install_timeout_seconds` bounds the build and the install, and at most
  `library_verifier_max_concurrency` checks install or run at once.
- The check scores an answer that times out or crashes its child 0, as this server's own check does. When the sandbox
  cannot run the check (it cannot be borrowed, the runtime does not install, the command fails), the sample is masked
  (`mask_sample`, with a `failure_kind`) instead of scored, and nothing falls back to this server. When this host
  cannot build the runtime, the request fails instead, since every sample would be masked alike.
- With `should_use_judge: true`, an answer Math-Verify rejects still goes to a judge model from this server: a model
  call, not code.

A config that lends no sandbox to `/verify` keeps the check in this server. What reaches code there:

- Math-Verify reads a LaTeX answer with latex2sympy2_extended's ANTLR parser, which builds SymPy objects. Names it
  dispatches on come from fixed lists in its grammar and code; any other name becomes an undefined SymPy function.
- SymPy's evaluating parser (`parse_expr`, which `sympify` and `Number` also call on a string) receives numerals
  (including `1.5E3` and percentages), formulas SymPy builds itself from fixed names (`\gcd` of a decimal goes
  through `nsimplify`), and the text Math-Verify's plain-expression regex admits: digits, a decimal point, whitespace
  and `+ - * × / ^ ( ) ÷`. None of these characters can start a Python identifier, so that text names nothing that
  could be called.

Each check runs in a forked child that is killed after `library_verifier_timeout_seconds`, which bounds the time an
answer can take, such as `9^9^9`. On Linux the child may also map at most `library_verifier_memory_limit_mb` (default
4096) beyond what it inherits, so an answer that asks for a huge object, such as `\operatorname{eye}(1000000000)`,
fails in its child and scores 0; `null` removes the cap. In this server the child is not otherwise isolated: it runs
as this server's user, with its files and network. `test_verifier_does_not_run_code_in_the_answer` checks the
in-server path for the pinned `math-verify`; check it again when you upgrade `math-verify` or `latex2sympy2_extended`.

# Licensing information
Code: Apache 2.0<br>
Data:
- OpenMathReasoning: Creative Commons Attribution 4.0 International
- Math Stack Overflow: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
