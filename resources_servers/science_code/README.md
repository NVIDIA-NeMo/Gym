# ScienceCode Resource Server

## Description

This resource server evaluates scientific Python code generation using an LLM as a judge. Given a scientific computing problem description, the model generates Python code that is compared against a reference solution using semantic equivalence checking.

The server extracts Python code from model responses (supporting code blocks and raw code) and uses an LLM judge to determine if the generated solution is functionally equivalent to the reference solution.

Based on the [SciCode benchmark](https://scicode-bench.github.io/) for evaluating scientific code generation capabilities.

## Supported Problem Domains

- **Numerical Methods**: Root finding, integration, differentiation
- **Linear Algebra**: Matrix operations, decompositions, solvers
- **Optimization**: Gradient descent, Newton methods, convex optimization
- **Differential Equations**: ODE/PDE solvers, numerical integration
- **Signal Processing**: FFT, filtering, spectral analysis
- **Statistics/Probability**: Sampling, distributions, hypothesis testing
- **Physics Simulations**: Mechanics, thermodynamics, electromagnetics

## Verification Flow

1. **Code Extraction**: Extract Python code from the model's response (code blocks or raw code)
2. **LLM Judge Evaluation**: Compare extracted code against reference using semantic equivalence
3. **Swap Check** (optional): Run second evaluation with swapped inputs to detect positional bias

## Input Format

Each data sample should include:
- `responses_create_params.input`: User message containing the problem description
- `problem`: Scientific computing problem description (required)
- `solution`: Reference Python solution (required)

### Example Input

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "system",
        "content": "You are an expert scientific coding assistant. \nYou approach each problem thoughtfully, working through the ideas internally until you are confident, and then you share the final polished code.\n\nYou naturally break complex tasks into smaller subproblems, and you like writing a clear function for each part before assembling the final solution. \nYour solutions tend to be clean Python modules without any execution blocks or example usage.\n\nYou're comfortable with a wide range of powerful tools, and especially python libraries numpy, math, scipy, sympy, itertools, and cmath. \nYou often draw on one or more of these libraries, especially when working with STEM problems and when it leads to a more elegant or efficient solution.\n\nWhen you respond, you enjoy presenting the complete solution as a single ```python``` code block — concise, professional, and ready to use. \nYour overall goal is to provide thoughtfully designed, beautifully structured Python solutions that reflect your expertise and experience."
      },
      {
        "role": "user",
        "content": "Examine the problem below. Think step-by-step and provide a solution:\n\nImplement a function `numerical_derivative(f, x, h=1e-5)` that computes the numerical derivative of function f at point x using the central difference method."
      }
    ]
  },
  "problem": "Implement a function `numerical_derivative(f, x, h=1e-5)` that computes the numerical derivative of function f at point x using the central difference method.",
  "solution": "def numerical_derivative(f, x, h=1e-5):\n    return (f(x + h) - f(x - h)) / (2 * h)"
}
```

### Key Fields

| Field | Required | Description |
|-------|----------|-------------|
| `problem` | Yes | Scientific computing problem description |
| `solution` | Yes | Reference Python solution |
| `responses_create_params` | Yes | Model input containing the full prompt |
| `uuid` | Recommended | Unique identifier for tracking the example |

## Usage

### Running Servers

```bash
config_paths="resources_servers/science_code/configs/science_code.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[${config_paths}]" \
  +science_code_resources_server.resources_servers.science_code.judge_responses_create_params.max_output_tokens=1024
```

### Collecting Rollouts

```bash
ng_collect_rollouts +agent_name=science_code_simple_agent \
    +input_jsonl_fpath=resources_servers/science_code/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/science_code/data/example_rollouts.jsonl
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `judge_model_server` | - | Model server to use as the judge |
| `judge_system_message` | (see `prompts.py`) | System message for the judge LLM |
| `judge_equal_label` | `[[A=B]]` | Label indicating equivalent solutions |
| `judge_not_equal_label` | `[[A!=B]]` | Label indicating non-equivalent solutions |
| `check_twice_swap` | `true` | Run swap check to detect positional bias |
| `reward_if_swap_fails` | `0.0` | Reward when swap check fails |

The judge prompts are defined in `prompts.py`:
- `SCIENCE_CODE_JUDGE_SYSTEM_MESSAGE`: Scientific code-specific system instructions with detailed equivalence rules for numerical methods, algorithms, and scientific computing patterns
- `SCIENCE_CODE_JUDGE_PROMPT_TEMPLATE`: User-level template with problem description and solutions labeled as A and B

## Equivalence Criteria

The LLM judge considers solutions equivalent if they:
1. Produce numerically equivalent results (within floating-point tolerance)
2. Implement the correct algorithm/method for the stated problem
3. Handle edge cases consistently
4. Use mathematically equivalent formulations (even if syntactically different)

### Examples of Equivalent Solutions

- Loop vs vectorized NumPy operations achieving same result
- Different numerical solvers converging to same answer
- `scipy.linalg.solve` vs `numpy.linalg.solve` for same system
- Recursive vs iterative implementations of same algorithm

### Examples of Non-Equivalent Solutions

- Different algorithms producing different numerical results
- Missing or incorrect normalization
- Off-by-one errors affecting output
- Wrong formula or sign errors

## Licensing Information

**Code**: Apache 2.0

**Data**: CC-BY-4.0 (synthetic examples)

## Dependencies

- nemo_gym: Apache 2.0
- NumPy, SciPy: BSD-3-Clause
