# Competitive Coding Resources Server

### Overview
Verifies competitive programming solutions by executing submitted code against unit tests. The server consumes agent trajectories and returns a reward based on whether the assistant's code produces the correct outputs for given test inputs.
Data source: [Filtered competitive programming dataset](https://huggingface.co/datasets/Nexusflow/comp_prog_filtered_no_function); split=`train`

### Input schema
- `responses_create_params`: OpenAI Responses create params
  - Use only a user message with the problem statement and instructions (e.g., "You are an expert competitive programmer...").
  - `verifier_metadata` (required):
    - `unit_tests` (required): dict with `inputs` and `outputs` arrays containing test cases.
      - `inputs`: list of strings representing stdin input for each test case
      - `outputs`: list of strings representing expected stdout output for each test case
    - `problem_id` (optional): unique identifier for the problem

**Notes**
- All test cases must pass for a solution to receive a reward of 1.0
- Failed test cases result in a reward of 0.0 with detailed error information

### Test execution (for now)
- Code is executed using Python's `exec()` function in a controlled environment
- Each test case runs with redirected stdin/stdout:
  - `stdin` is populated with the test input
  - `stdout` is captured for comparison with expected output
- Available built-ins include common functions: `input`, `print`, `range`, `len`, `int`, `str`, `list`, etc.
- Newlines in test data are properly handled (converts `\\n` to actual newlines)

### Example dataset row
```json
{
    "responses_create_params": {
        "input": [
            {
                "role": "user",
                "content": "You are an expert competitive programmer. You will be given a problem statement and must output a complete Python solution that reads from stdin and writes to stdout.\n\nPolycarp has $n$ different binary words. A word called binary if it contains only characters '0' and '1'. For example, these words are binary: \"0001\", \"11\", \"0\" and \"0011100\".\n\nPolycarp wants to offer his set of $n$ binary words to play a game \"words\". In this game, players name words and each next word (starting from the second) must start with the last character of the previous word. The first word can be any. For example, these sequence of words can be named during the game: \"0101\", \"1\", \"10\", \"00\", \"00001\".\n\nWord reversal is the operation of reversing the order of the characters. For example, the word \"0111\" after the reversal becomes \"1110\", the word \"11010\" after the reversal becomes \"01011\".\n\nProbably, Polycarp has such a set of words that there is no way to put them in the order correspondent to the game rules. In this situation, he wants to reverse some words from his set so that:  the final set of $n$ words still contains different words (i.e. all words are unique);  there is a way to put all words of the final set of words in the order so that the final sequence of $n$ words is consistent with the game rules. \n\nPolycarp wants to reverse minimal number of words. Please, help him.\n\n\n-----Input-----\n\nThe first line of the input contains one integer $t$ ($1 \\le t \\le 10^4$) — the number of test cases in the input. Then $t$ test cases follow.\n\nThe first line of a test case contains one integer $n$ ($1 \\le n \\le 2\\cdot10^5$) — the number of words in the Polycarp's set. Next $n$ lines contain these words. All of $n$ words aren't empty and contains only characters '0' and '1'. The sum of word lengths doesn't exceed $4\\cdot10^6$. All words are different.\n\nGuaranteed, that the sum of $n$ for all test cases in the input doesn't exceed $2\\cdot10^5$. Also, guaranteed that the sum of word lengths for all test cases in the input doesn't exceed $4\\cdot10^6$.\n\n\n-----Output-----\n\nPrint answer for all of $t$ test cases in the order they appear.\n\nIf there is no answer for the test case, print -1. Otherwise, the first line of the output should contain $k$ ($0 \\le k \\le n$) — the minimal number of words in the set which should be reversed. The second line of the output should contain $k$ distinct integers — the indexes of the words in the set which should be reversed. Words are numerated from $1$ to $n$ in the order they appear. If $k=0$ you can skip this line (or you can print an empty line). If there are many answers you can print any of them.\n\n\n-----Example-----\nInput\n4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n\nOutput\n1\n3 \n-1\n0\n\n2\n1 2"
            }
        ]
    },
    "verifier_metadata": {
        "problem_id": "c69268d8bdb4da0685d7b187c88296c1",
        "unit_tests": {
            "inputs": ["4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n"],
            "outputs": ["1\n3 \n-1\n0\n\n2\n1 2 \n"]
        }
    }
}
```

### Example of rollouts and usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml"

# Running the server
ng_run "+config_paths=[$config_paths]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=comp_coding

# Prepare example data for validation
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=resources_servers/comp_coding/data/ \
    +mode=example_validation

# Download train data from gitlab model registry
ng_download_dataset_from_gitlab \
    +dataset_name=comp_coding \
    +version=0.0.1 \
    +run_id=5a1167ef-3533-486f-9c0e-49d1e97fc887 \
    +artifact_fpath=train.jsonl \
    +output_fpath=resources_servers/comp_coding/data/train.jsonl

# Collect rollouts from example problems
ng_collect_rollouts +agent_name=comp_coding_simple_agent \
    +input_jsonl_fpath=resources_servers/comp_coding/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/comp_coding/data/example_rollouts.jsonl \
    +limit=null
```

### Optional data preperation/validation scripts

```bash
# Build training dataset from collected examples
uv run python resources_servers/comp_coding/scripts/build_examples.py \
    --out resources_servers/comp_coding/data/train.jsonl \
    --split train[:5000]

# Validate and pre-process train dataset
uv run python resources_servers/comp_coding/scripts/validate_dataset.py \
    --in data/comp_coding/train.jsonl --fail-fast
```

### Error handling
The server provides specific error messages for different failure modes:
- `Empty model output`: No text found in the response
- `Missing verifier_metadata.unit_tests`: Required test data not provided
- `Invalid unit_tests`: Malformed test case data
- `Could not extract code`: No valid Python code found in response
- `INVALID_TEST_FORMAT`: Test inputs/outputs length mismatch or empty
- `TEST_CASE_N_FAILED`: Specific test case failed with expected vs actual output
- `TEST_CASE_N_ERROR`: Runtime error during test execution

## Licensing information
TODO: @kbhardwaj to confirm data/code licensing information w Vahid and team
