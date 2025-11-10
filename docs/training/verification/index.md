(training-verification)=

# Verification

Validate that your resource server's verification logic produces useful reward signals, or build custom verification for specialized domains.

When you choose a resource server, you're also choosing its verification logic—the `verify()` function that scores agent outputs and returns rewards. This section helps you test that verification works for your training algorithm, or customize it if needed.

:::{seealso}
**Haven't chosen a resource server yet?** Start with {ref}`training-resource-servers` to select one first. Verification is built into each resource server.
:::

---

## Topics

Validation and customization guides for resource server verification logic.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Validate Verification
:link: validate-verification
:link-type: doc

Test your chosen resource server's `verify()` function. Collect sample rollouts and validate that reward signals are useful for your training algorithm.
+++
{bdg-primary}`Start here` {bdg-secondary}`validation` {bdg-secondary}`testing`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Verification
:link: custom-patterns-cookbook
:link-type: doc

Implement your own `verify()` function in a custom resource server using copy-paste patterns and multi-objective techniques.
+++
{bdg-secondary}`Advanced` {bdg-secondary}`custom-patterns` {bdg-secondary}`multi-objective`
:::

::::

---

## How Verification Works

Verification is not a separate component—it lives inside your resource server as a `verify()` method exposed as an HTTP endpoint (`POST /verify`).

**During rollout collection**, the system automatically:

1. Agent completes interaction → produces output
2. Collection system calls `POST /verify` on your resource server
3. Resource server's `verify()` function scores the output
4. Reward signal gets saved in the rollout JSONL

:::{dropdown} **Architecture Details** (click to expand)
:color: secondary
:icon: code

Each resource server implements the `verify()` method:

```python
# From nemo_gym/base_resources_server.py
class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/verify")(self.verify)  # HTTP endpoint
        return app
    
    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass  # Each resource server implements this
```

**Example from `structured_outputs` server** (validates JSON schema):

```python
async def verify(self, body: VerifyRequest) -> BaseVerifyResponse:
    response_text = extract_output_text(body.response)
    
    try:
        response_obj = json.loads(response_text)
        validate_against_schema(response_obj, body.schema_str)
        return BaseVerifyResponse(reward=1.0)
    except Exception:
        return BaseVerifyResponse(reward=0.0)
```

**Example from `mcqa` server** (extracts multiple-choice answers):

```python
async def verify(self, body: MCQAVerifyRequest) -> MCQAVerifyResponse:
    text = extract_last_assistant_text(body)
    
    # Parse answer using configurable grading mode
    if body.grading_mode == "strict_single_letter_boxed":
        predicted = parse_boxed_answer(text, allowed_letters)
    elif body.grading_mode == "lenient_boxed":
        predicted = parse_flexible(text, allowed_letters)
    
    reward = 1.0 if predicted == body.expected_answer else 0.0
    return MCQAVerifyResponse(reward=reward)
```

:::

---

## What Each Server Verifies

Different resource servers implement different verification approaches for their task domains:

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Server
  - Verification Approach
  - Reward Type
* - **mcqa**
  - Extracts answer letters (A/B/C/D) using regex patterns
  - Binary (0.0 or 1.0)
* - **comp_coding**
  - Executes code against unit tests
  - Binary (all tests pass)
* - **library_judge_math**
  - Validates mathematical equivalence using symbolic libraries
  - Continuous (0.0–1.0)
* - **equivalence_llm_judge**
  - Uses LLM to judge semantic similarity
  - Continuous (0.0–1.0)
* - **structured_outputs**
  - Validates JSON schema compliance
  - Binary (valid/invalid)
* - **instruction_following**
  - Checks constraint satisfaction
  - Binary (all constraints met)
* - **python_math_exec**
  - Executes Python code and validates results
  - Binary (correct result)
```

When you choose a resource server, you're choosing its verification logic as a package.

---

## Verification Workflow

The typical verification workflow for training data generation:

```text
[1. Choose Resource Server]  ← Selects verification logic
    ↓
[2. Collect Test Rollouts]   ← Generate 20-50 samples
    ↓
[3. Validate Verification]   ← Check reward distribution
    ↓
[4. Scale Collection]         ← Or customize if needed
```

**Most users**: {doc}`validate-verification` (5-10 min validation)  
**Custom needs**: {doc}`custom-patterns-cookbook` (build your own `verify()`)

---

## Quick Decision Guide

```{list-table}
:header-rows: 1
:widths: 40 60

* - Your Situation
  - Recommended Action
* - **Selected a resource server**
  - Start with {doc}`validate-verification` to test it works
* - **Verification produces bad rewards**
  - Try different resource server or see {doc}`custom-patterns-cookbook`
* - **Need domain-specific verification**
  - Build custom with {doc}`custom-patterns-cookbook`
* - **Want multi-objective scoring**
  - See {doc}`multi-objective-scoring` for tracking dimensions
```

---

## Related Topics

**Before verification**:

- {ref}`training-resource-servers` - Choose which server (and verification)

**After verification**:

- {ref}`training-rollout-collection` - Scale up collection
- {ref}`training-datasets` - Prepare data for training

```{toctree}
:hidden:
:maxdepth: 1

validate-verification
custom-patterns-cookbook
multi-objective-scoring
```
