(tutorial-code-tester-server)=

# Build a Code Testing Server

In previous tutorials, verification compared text outputs. Now you'll build a code testing server that verifies generated code by actually executing it against test cases.

:::{card}

**Goal**: Build a server that executes agent-generated code and scores based on test results.

^^^

**In this tutorial, you will**:

1. Create a code execution sandbox
2. Implement a `run_code` tool that executes Python safely
3. Verify by running test cases against generated code
4. Walk away with a pattern for code verification

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`/tutorials/creating-resource-server`
- ✅ Understanding of code sandboxing (or willingness to learn)
- ✅ Python environment with `subprocess` available

**What you'll build**: A coding challenge server where agents write Python functions, and you verify by running test cases.

:::{tip}
**Reference implementations**:
- `resources_servers/code_gen/` — Code generation with LiveCodeBench
- `resources_servers/math_with_code/` — Math solved via code execution
:::

---

## 1. Understand What We're Building

The code testing flow:

1. User says: "Write a function `add(a, b)` that returns the sum"
2. Agent writes code: `def add(a, b): return a + b`
3. Server executes test cases: `add(2, 3) == 5`, `add(-1, 1) == 0`
4. Verification: 2/2 tests pass → reward = 1.0

```{mermaid}
flowchart LR
    A["Agent Code"] --> B["Sandbox"]
    B --> C["Run Tests"]
    C --> D{"Results"}
    D -->|"5/5 pass"| E["✅ 1.0"]
    D -->|"3/5 pass"| F["⚠️ 0.6"]
    D -->|"0/5 pass"| G["❌ 0.0"]
```

**✅ Success Check**: You understand test-based verification.

---

## 2. Create the Code Server

<!-- SME: Adapt from code_gen/app.py -->

```python
# app.py
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig
import subprocess
import tempfile
import os

class CodeServerConfig(BaseResourcesServerConfig):
    timeout_seconds: int = 5
    
class CodeServer(SimpleResourcesServer):
    
    def __init__(self, config: CodeServerConfig):
        super().__init__(config)
        self.timeout = config.timeout_seconds
        
        # Problems with test cases
        self.problems = {
            "add": {
                "description": "Write a function add(a, b) that returns the sum of a and b",
                "tests": [
                    ("add(2, 3)", 5),
                    ("add(-1, 1)", 0),
                    ("add(0, 0)", 0),
                    ("add(100, 200)", 300),
                ]
            },
            "multiply": {
                "description": "Write a function multiply(a, b) that returns the product",
                "tests": [
                    ("multiply(2, 3)", 6),
                    ("multiply(-1, 5)", -5),
                    ("multiply(0, 100)", 0),
                ]
            }
        }
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_code",
                    "description": "Submit Python code for testing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code defining the required function"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]
```

---

## 3. Implement Safe Code Execution

:::{warning}
Code execution is dangerous. Always sandbox properly in production.
:::

```python
# Add to app.py

def call_tool(self, tool_name: str, tool_args: dict) -> str:
    if tool_name == "submit_code":
        code = tool_args.get("code", "")
        return self._execute_code(code)
    return "Unknown tool"

def _execute_code(self, code: str) -> str:
    """Execute code in a sandboxed subprocess."""
    
    # Create temporary file with the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.write("\n\n# Test execution will be added by verifier\n")
        temp_path = f.name
    
    try:
        # Basic execution to check syntax
        result = subprocess.run(
            ['python', '-c', f'exec(open("{temp_path}").read())'],
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        
        return "Code submitted successfully. Tests will run during verification."
        
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    finally:
        os.unlink(temp_path)
```

---

## 4. Implement Test-Based Verification

```python
# Add to app.py

def verify(self, responses_create_params: dict, output: list) -> float:
    """Verify code by running test cases."""
    
    # Identify which problem was asked
    user_message = responses_create_params["input"][0]["content"].lower()
    
    problem = None
    for prob_name, prob_data in self.problems.items():
        if prob_name in user_message:
            problem = prob_data
            break
    
    if problem is None:
        return 0.0
    
    # Extract submitted code from tool calls
    submitted_code = None
    for msg in output:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                if call.get("function", {}).get("name") == "submit_code":
                    args = call["function"].get("arguments", "{}")
                    import json
                    submitted_code = json.loads(args).get("code", "")
                    break
    
    if not submitted_code:
        return 0.0
    
    # Run tests
    passed = 0
    total = len(problem["tests"])
    
    for test_call, expected in problem["tests"]:
        if self._run_test(submitted_code, test_call, expected):
            passed += 1
    
    return passed / total

def _run_test(self, code: str, test_call: str, expected) -> bool:
    """Run a single test case."""
    
    test_code = f"""
{code}

result = {test_call}
print(result == {expected})
"""
    
    try:
        result = subprocess.run(
            ['python', '-c', test_code],
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        return result.stdout.strip() == "True"
    except:
        return False
```

**✅ Success Check**: Returns pass_rate as score (e.g., 4/5 = 0.8).

---

## 5. Create Test Data

Create `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Write a Python function called 'add' that takes two numbers and returns their sum."}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Create a 'multiply' function in Python that returns the product of two numbers."}], "model": "gpt-4"}}
```

---

## 6. Run and Test

```bash
ng_collect_rollouts +agent_name=code_simple_agent \
    +input_jsonl_fpath=resources_servers/code/data/example.jsonl \
    +output_jsonl_fpath=results/code_rollouts.jsonl \
    +limit=2
```

**✅ Success Check**: Reward reflects test pass rate.

---

## Security Hardening

For production, add:

```python
# Resource limits
import resource
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

# Network isolation (use Docker or similar)
# Filesystem restrictions (use tmpfs or chroot)
# No imports of dangerous modules
```

---

## Learn More

- {doc}`/training/verification/index` — Test-based verification patterns
- {doc}`/training/rollout-collection/configure-sampling` — Sampling for code generation

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train Your Model
:link: /tutorials/integrate-training-frameworks/train-with-trl
:link-type: doc

Use your code rollouts to train better coders.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` Combine with LLM Judge
:link: math-verifier-server
:link-type: doc

Add LLM verification for partial credit.
:::

::::
