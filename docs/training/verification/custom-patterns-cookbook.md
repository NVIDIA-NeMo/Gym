(training-verification-custom-patterns)=

# Custom Verification Patterns

Copy-paste verification patterns for building custom resource servers. Each pattern includes complete working code from NeMo Gym's built-in servers.

:::{card}

**Audience**: Building custom resource servers (10% of users)

^^^

**Quick patterns for**:

1. Exact match verification
2. LLM judge verification
3. Code execution verification
4. Structured output validation
5. Multi-objective verification
6. Hybrid verification (fallback strategies)

:::

**Prerequisites**: Basic understanding from {doc}`../../get-started/verifying-agent-results`.

:::{tip}
**Using existing servers?** You don't need this guide—just pick a server from {doc}`index`.
:::

---

## Pattern 1: Exact Match

Compare extracted answer directly to expected value. Use for tasks with canonical correct answers.

### When to Use

- Multiple choice questions (A/B/C/D)
- Numeric answers with exact values
- Classification tasks
- Any task with unambiguous correct answer

### Complete Example

```python
import re
from typing import Optional
from nemo_gym.base_responses_api_agent import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_resources_server import SimpleResourcesServer

class YourResourcesServer(SimpleResourcesServer):
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # 1. Extract agent's response text
        response_text = ""
        for item in body.response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        response_text += content.text
        
        # 2. Extract expected answer from metadata
        expected = body.verifier_metadata.get("expected_answer", "").strip().upper()
        
        # 3. Extract agent's answer using pattern
        # Look for \\boxed{X} or "Answer: X" patterns
        match = re.search(r'\\boxed\{([A-D])\}', response_text)
        if not match:
            match = re.search(r'Answer:\s*([A-D])', response_text, re.IGNORECASE)
        
        predicted = match.group(1).upper() if match else None
        
        # 4. Compare and compute reward
        is_correct = (predicted == expected) if (predicted and expected) else False
        reward = 1.0 if is_correct else 0.0
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Source**: `resources_servers/mcqa/app.py:259-261`

### Variations

**Lenient matching** (try multiple extraction strategies):

```python
def extract_answer(text: str) -> Optional[str]:
    """Try multiple extraction patterns"""
    patterns = [
        r'\\boxed\{([A-D])\}',           # LaTeX boxed
        r'Answer:\s*([A-D])',            # "Answer: X"
        r'\(([A-D])\)',                  # (X)
        r'^([A-D])$',                    # Just the letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    return None
```

**Per-record regex** (custom patterns per task):

```python
# In verifier_metadata
{
  "expected_answer": "B",
  "output_regex": r"The correct option is ([A-D])"
}

# In verify()
if "output_regex" in body.verifier_metadata:
    pattern = body.verifier_metadata["output_regex"]
    match = re.search(pattern, response_text)
    predicted = match.group(1) if match else None
```

---

## Pattern 2: LLM Judge

Use another LLM to judge semantic equivalence. Use for open-ended questions with multiple valid phrasings.

### When to Use

- Open-ended QA (multiple valid answers)
- Free-form text generation
- Semantic correctness matters more than exact wording
- Paraphrasing is acceptable

### Complete Example

```python
from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

class YourResourcesServer(SimpleResourcesServer):
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # 1. Extract question and answers
        question = body.verifier_metadata.get("question", "")
        expected_answer = body.verifier_metadata.get("expected_answer", "")
        
        # 2. Extract agent's generated answer
        generated_answer = ""
        for item in body.response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        generated_answer += content.text
        
        # 3. Format judge prompt
        judge_prompt = f"""<|Problem|>
{question}

<|Start of Assistant A's Answer|>
{expected_answer}
<|End of Assistant A's Answer|>

<|Start of Assistant B's Answer|>
{generated_answer}
<|End of Assistant B's Answer|>

Are Assistant A and B's answers semantically equivalent? Respond with [[A=B]] if yes, [[A!=B]] if no."""
        
        # 4. Call judge model
        server_client = ServerClient.load_from_global_config()
        judge_request = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,  # Deterministic judging
        )
        
        result = await server_client.post(
            server_name="judge_model",  # Configure in your setup
            url_path="/v1/responses",
            json=judge_request,
        )
        judge_response = await result.json()
        
        # 5. Parse judge verdict
        judge_text = judge_response['output_text']
        is_equivalent = "[[A=B]]" in judge_text
        
        reward = 1.0 if is_equivalent else 0.0
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Source**: `resources_servers/equivalence_llm_judge/app.py:379-411`

### Bias Mitigation: Swap Check

```python
async def verify_with_swap_check(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # First pass: A vs B
    first_equal = await self._judge_equivalence(expected, generated)
    
    # Second pass: B vs A (swap positions)
    second_equal = await self._judge_equivalence(generated, expected)
    
    # If verdicts differ, position bias detected
    if first_equal != second_equal:
        reward = 0.5  # Uncertain
    else:
        reward = 1.0 if first_equal else 0.0
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Configuration options**:

```python
class YourResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: str = "judge_model"
    judge_system_message: str = "You are an expert judge..."
    judge_prompt_template: str = "..."  # Custom template
    check_twice_swap: bool = True       # Enable swap check
```

---

## Pattern 3: Code Execution

Execute generated code and verify against test cases. Use for code generation tasks.

### When to Use

- Code generation challenges
- Programming problems with unit tests
- Computational correctness verification
- Pass/fail based on execution

### Complete Example

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from asyncio import Semaphore

class YourResourcesServer(SimpleResourcesServer):
    
    def model_post_init(self, context):
        self._semaphore = Semaphore(value=4)  # Limit parallel execution
        self._executor = ProcessPoolExecutor(max_workers=4)
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # 1. Extract generated code
        response_text = body.response.output_text
        code = self._extract_code(response_text)
        
        if not code:
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)
        
        # 2. Get test cases
        tests = body.verifier_metadata["unit_tests"]
        
        # 3. Execute code with tests (sandboxed)
        async with self._semaphore:  # Rate limiting
            results = await self._run_tests(code, tests)
        
        # 4. Compute reward based on pass rate
        passed = sum(1 for r in results if r == True)
        total = len(results)
        
        # Binary: all tests must pass
        reward = 1.0 if passed == total else 0.0
        
        # Alternative: partial credit
        # reward = passed / total if total > 0 else 0.0
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from markdown code fences"""
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        return text  # Assume entire response is code
    
    async def _run_tests(self, code: str, tests: list) -> list[bool]:
        """Execute code with test cases in separate process"""
        loop = asyncio.get_event_loop()
        
        # Run in process pool for safety
        results = await loop.run_in_executor(
            self._executor,
            self._execute_code,
            code,
            tests
        )
        
        return results
    
    @staticmethod
    def _execute_code(code: str, tests: list) -> list[bool]:
        """Execute in isolated process"""
        results = []
        
        for test in tests:
            try:
                # Combine code + test
                full_code = f"{code}\n\n{test['test_code']}"
                
                # Execute with timeout
                exec_globals = {}
                exec(full_code, exec_globals)
                
                # Check result
                expected = test['expected_output']
                actual = exec_globals.get('result')
                
                results.append(actual == expected)
                
            except Exception:
                results.append(False)
        
        return results
```

**Source**: `resources_servers/comp_coding/app.py:79-149`

### Safety Considerations

```python
# Timeout protection
import signal

def timeout_handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    exec(code)
finally:
    signal.alarm(0)  # Cancel timeout
```

---

## Pattern 4: Structured Output Validation

Verify response conforms to JSON schema. Use when agent must produce structured data.

### When to Use

- JSON generation tasks
- API response formatting
- Data extraction into schema
- Structured agent outputs

### Complete Example

```python
import json
from jsonschema import validate, ValidationError

class YourResourcesServer(SimpleResourcesServer):
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # 1. Extract agent's response
        response_text = ""
        for item in body.response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        response_text += content.text
        
        # 2. Get expected schema
        schema = json.loads(body.verifier_metadata["schema"])
        
        # 3. Try to parse and validate JSON
        try:
            # Parse JSON
            response_obj = json.loads(response_text)
            
            # Validate against schema
            validate(instance=response_obj, schema=schema)
            
            # Success: valid JSON matching schema
            reward = 1.0
            
        except (json.JSONDecodeError, ValidationError):
            # Failed: invalid JSON or schema mismatch
            reward = 0.0
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

**Source**: `resources_servers/structured_outputs/app.py:50-74`

### Schema Enforcement

```python
def make_schema_strict(self, schema: dict):
    """Enforce strict validation (no extra fields)"""
    if "properties" in schema:
        schema["required"] = list(schema["properties"])
        schema["additionalProperties"] = False
    
    # Recurse for nested objects
    for value in schema.values():
        if isinstance(value, dict):
            self.make_schema_strict(value)

# Usage in verify()
schema = json.loads(body.verifier_metadata["schema"])
self.make_schema_strict(schema)  # Now strict
validate(instance=response_obj, schema=schema)
```

### Partial Credit Variation

```python
def validate_with_partial_credit(self, response_obj: dict, schema: dict) -> float:
    """Give partial credit for partially correct structure"""
    try:
        validate(instance=response_obj, schema=schema)
        return 1.0  # Perfect
    except ValidationError:
        # Check how many required fields are present
        required = schema.get("required", [])
        present = [f for f in required if f in response_obj]
        
        return len(present) / len(required) if required else 0.0
```

---

## Pattern 5: Multi-Objective Verification

Track multiple metrics and combine into composite reward. Use for tasks with multiple success criteria.

### When to Use

- Tasks have multiple quality dimensions
- Want to track correctness AND efficiency
- Need separate metrics for analysis
- Balancing competing objectives

### Complete Example

```python
class YourVerifyResponse(BaseVerifyResponse):
    """Custom response with multiple metrics"""
    accuracy: float
    efficiency: float
    completeness: float

class YourResourcesServer(SimpleResourcesServer):
    
    async def verify(self, body: BaseVerifyRequest) -> YourVerifyResponse:
        # 1. Extract agent response
        response_output = body.response.output
        
        # 2. Compute multiple metrics independently
        
        # Metric 1: Correctness
        expected = body.verifier_metadata["expected_answer"]
        actual = self._extract_answer(response_output)
        accuracy = 1.0 if actual == expected else 0.0
        
        # Metric 2: Efficiency (tool usage)
        optimal_tools = 2
        actual_tools = sum(1 for item in response_output 
                          if item.type == "function_call")
        tool_penalty = max(0, actual_tools - optimal_tools) * 0.1
        efficiency = max(0.0, 1.0 - tool_penalty)
        
        # Metric 3: Completeness (response coverage)
        required_elements = body.verifier_metadata.get("required_elements", [])
        response_text = self._get_response_text(response_output)
        present = sum(1 for elem in required_elements 
                     if elem.lower() in response_text.lower())
        completeness = present / len(required_elements) if required_elements else 1.0
        
        # 3. Combine into primary reward (weighted)
        reward = (
            0.6 * accuracy +      # Primary: 60%
            0.3 * efficiency +    # Secondary: 30%
            0.1 * completeness    # Tertiary: 10%
        )
        
        # 4. Return all metrics (automatically aggregated by NeMo Gym)
        return YourVerifyResponse(
            **body.model_dump(),
            reward=reward,           # Primary training signal
            accuracy=accuracy,       # Track separately
            efficiency=efficiency,   # Track separately
            completeness=completeness  # Track separately
        )
```

**Source**: `resources_servers/multineedle/app.py:86-105`

### Hierarchical Gating

```python
# Correctness must be met before other metrics matter
if accuracy < 0.5:
    # Failed primary objective
    reward = accuracy * 0.3  # Max 0.15
else:
    # Met primary objective
    reward = 0.6 * accuracy + 0.3 * efficiency + 0.1 * completeness
```

### Weight Tuning

```python
class YourResourcesServerConfig(BaseResourcesServerConfig):
    """Make weights configurable"""
    accuracy_weight: float = 0.6
    efficiency_weight: float = 0.3
    completeness_weight: float = 0.1
    
    def model_post_init(self, context):
        # Validate weights sum to 1.0
        total = self.accuracy_weight + self.efficiency_weight + self.completeness_weight
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"

# In verify()
reward = (
    self.config.accuracy_weight * accuracy +
    self.config.efficiency_weight * efficiency +
    self.config.completeness_weight * completeness
)
```

---

## Pattern 6: Hybrid Verification

Combine multiple verification strategies with fallback logic. Use when one method isn't sufficient.

### When to Use

- Fast deterministic check available but not always conclusive
- Expensive LLM judge as fallback for edge cases
- Want to minimize verification costs
- Combining symbolic + learned verification

### Complete Example

```python
class YourResourcesServer(SimpleResourcesServer):
    
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # 1. Extract answers
        expected = body.verifier_metadata["expected_answer"]
        generated = self._extract_answer(body.response)
        
        # 2. Stage 1: Fast deterministic check
        library_reward = self._verify_with_library(expected, generated)
        
        # 3. Stage 2: Expensive judge (only if needed)
        if library_reward > 0.5:
            # Library succeeded - use its result
            reward = library_reward
        elif self.config.should_use_judge:
            # Library inconclusive - call judge
            judge_reward = await self._verify_with_judge(
                question=body.verifier_metadata["question"],
                expected=expected,
                generated=generated
            )
            reward = judge_reward
        else:
            # Library failed, judge disabled
            reward = library_reward
        
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
    
    def _verify_with_library(self, expected: str, generated: str) -> float:
        """Fast symbolic verification (math-verify, exact match, etc.)"""
        try:
            # Example: symbolic math equivalence
            from math_verify import evaluate_equivalence
            
            result = evaluate_equivalence(expected, generated)
            
            if result == "equivalent":
                return 1.0
            elif result == "not_equivalent":
                return 0.0
            else:
                return 0.5  # Inconclusive
                
        except Exception:
            return 0.5  # Library failed
    
    async def _verify_with_judge(self, question: str, expected: str, 
                                 generated: str) -> float:
        """Expensive LLM judge verification"""
        # Use Pattern 2 (LLM Judge) here
        judge_result = await self._call_judge_model(question, expected, generated)
        return 1.0 if judge_result == "equivalent" else 0.0
```

**Source**: `resources_servers/library_judge_math/app.py:145-161`

### Cost Optimization

```python
class HybridVerificationMetrics(BaseVerifyResponse):
    """Track which verification method was used"""
    library_reward: float
    judge_reward: Optional[float]
    verification_method: str  # "library" or "judge"

async def verify(self, body: BaseVerifyRequest) -> HybridVerificationMetrics:
    library_reward = self._verify_with_library(expected, generated)
    
    if library_reward > 0.5:
        # Library succeeded
        return HybridVerificationMetrics(
            **body.model_dump(),
            reward=library_reward,
            library_reward=library_reward,
            judge_reward=None,
            verification_method="library"
        )
    else:
        # Call judge
        judge_reward = await self._verify_with_judge(...)
        return HybridVerificationMetrics(
            **body.model_dump(),
            reward=judge_reward,
            library_reward=library_reward,
            judge_reward=judge_reward,
            verification_method="judge"
        )

# After collection, analyze costs:
# - Count how many used library vs judge
# - Estimate cost savings from hybrid approach
```

---

## Quick Reference

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Pattern
  - Use Case
  - Source Example
* - **Exact Match**
  - Canonical answers (MCQA, classification)
  - mcqa
* - **LLM Judge**
  - Open-ended QA, semantic equivalence
  - equivalence_llm_judge
* - **Code Execution**
  - Code generation with tests
  - comp_coding
* - **Schema Validation**
  - Structured JSON outputs
  - structured_outputs
* - **Multi-Objective**
  - Multiple quality dimensions
  - multineedle
* - **Hybrid**
  - Fast check + expensive fallback
  - library_judge_math
```

---

## Implementation Checklist

When building custom verification:

- [ ] **Choose appropriate pattern** for your task
- [ ] **Extract agent response** from `body.response.output`
- [ ] **Get expected answer** from `body.verifier_metadata`
- [ ] **Compute reward** (0.0–1.0 range by convention)
- [ ] **Return BaseVerifyResponse** (or custom response class)
- [ ] **Test with sample data** before large-scale collection
- [ ] **Validate reward distribution** (not all 0.0 or 1.0)
- [ ] **Add type hints** for request/response classes
- [ ] **Handle edge cases** (empty response, malformed data)
- [ ] **Document configuration options** if any

---

## Next Steps

**After choosing a pattern**:

1. Create resource server: `ng_init_resources_server +entrypoint=resources_servers/your_server/`
2. Implement `verify()` using pattern above
3. Test with {doc}`../../get-started/verifying-agent-results`
4. Collect rollouts with {doc}`../rollout-collection/index`
5. Prepare data with {doc}`../datasets/prepare-for-training`

**For multi-objective details**: See {doc}`multi-objective-scoring`

**For full examples**: Browse `resources_servers/` directory in repo

:::{button-ref} ../rollout-collection/index
:color: primary
:outline:
:ref-type: doc

Start Collecting Rollouts →
:::

