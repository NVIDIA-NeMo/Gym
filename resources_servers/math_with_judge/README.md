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

The verifier runs nothing from the answer, so a config may let `/verify` run in this server instead of in the
attempt's sandbox group (`sandbox_opt_out.verifier: true`):

- Math-Verify reads a LaTeX answer with latex2sympy2_extended's ANTLR parser, which builds SymPy objects. Names it
  dispatches on come from fixed lists in its grammar and code; any other name becomes an undefined SymPy function.
- SymPy's evaluating parser (`parse_expr`, which `sympify` and `Number` also call on a string) receives numerals
  (including `1.5E3` and percentages), formulas SymPy builds itself from fixed names (`\gcd` of a decimal goes
  through `nsimplify`), and the text Math-Verify's plain-expression regex admits: digits, a decimal point, whitespace
  and `+ - * × / ^ ( ) ÷`. None of these characters can start a Python identifier, so that text names nothing that
  could be called.
- With `should_use_judge: true`, an answer Math-Verify rejects goes to a judge model: a model call, not code.

Each check runs in a forked child that is killed after `library_verifier_timeout_seconds`, which bounds the time an
answer can take, such as `9^9^9`. On Linux the child may also map at most `library_verifier_memory_limit_mb` (default
4096) beyond what it inherits from the server, so an answer that asks for a huge object, such as
`\operatorname{eye}(1000000000)`, fails in its child and scores 0 instead of exhausting the host; `null` removes the
cap. The child is not otherwise isolated: it runs as this server's user, with its files and network.

`test_verifier_does_not_run_code_in_the_answer` checks this for the pinned `math-verify`; check it again when you
upgrade `math-verify` or `latex2sympy2_extended`.

# Licensing information
Code: Apache 2.0<br>
Data:
- OpenMathReasoning: Creative Commons Attribution 4.0 International
- Math Stack Overflow: Creative Commons Attribution-ShareAlike 4.0 International

Dependencies
- nemo_gym: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
