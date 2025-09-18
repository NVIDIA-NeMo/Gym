## Details 

**Environment**: Added a stateful async python code executor. The python code executor has access to scipy, numpy & pandas.


**Dataset**: Currently works with open_math_reasoning_problems_tool dataset which is a formatted version of open_math_reasoning_problems


### How to run

Download the `open_math_reasoning_problems` dataset (no python tool)

```
ng_download_dataset_from_gitlab \
    +dataset_name=math_open_math_reasoning \
    +version=0.0.1 \
    +artifact_fpath=open_math_reasoning_problems.jsonl \
    +output_fpath=data/open_math_reasoning_problems.jsonl
```

Use this below script to convert it into `open_math_reasoning_problems_tool.jsonl` or (you can also download it directly)

Conversion code (Run it from the data/ folder)
```
import json

def convert_to_math_tool_format(original_data, record_id=0):
    # Enhanced system message with tool instructions
    enhanced_system_message = """# Instructions
You are a math assistant. When you need to perform calculations, use the execute_python tool to write and run Python code. Any result that can be used should be in a print statement so it can be captured by stdout. If there are multiple calculations and they depend on each other, you should use the execute_python tool to perform the calculations step by step. You have access to numpy, scipy, and pandas libraries. After completing all calculations, provide your final answer in the format: \\boxed{your_answer} 

# Available Tools
- execute_python: Execute Python code to perform mathematical calculations

# Example
If asked "What is 5 factorial?", you would use execute_python with code like:
```python
import math
result = math.factorial(5)
print(f"5! = {result}")
```"""

    # Tool configuration
    tools = [
        {
            "type": "function",
            "name": "execute_python",
            "description": "Execute Python code to perform calculations. You have access to numpy, scipy, pandas and basic math operations.",
            "parameters": {
                "properties": {
                    "code": {
                        "type": "string",
                        "title": "Python Code",
                        "description": "The Python code to execute for the calculation."
                    }
                },
                "type": "object",
                "required": ["code"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]

    # Create the new format
    converted = {
        "id": record_id,
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": enhanced_system_message
                },
                {
                    "role": "user", 
                    "content": original_data["question"]
                }
            ],
            "tools": tools,
            "parallel_tool_calls": False
        },
        "expected_result": original_data["expected_answer"],
        "expected_code_contains": ""
    }
    
    return converted

def convert_jsonl_file(input_file, output_file):
    """Convert original JSONL file to math + tool format"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile):
            if line.strip():  # Skip empty lines
                original_data = json.loads(line)
                converted_data = convert_to_math_tool_format(original_data, line_num)
                outfile.write(json.dumps(converted_data) + '\n')

# Usage
convert_jsonl_file('open_math_reasoning_problems.jsonl', 'open_math_reasoning_problems_tool.jsonl')
```

OR

If you want to download directly

```
ng_download_dataset_from_gitlab \
    +dataset_name=open_math_reasoning_problems_tool \
    +version=0.0.1 \
    +artifact_fpath=open_math_reasoning_problems_tool.jsonl \
    +output_fpath=data/open_math_reasoning_problems_tool.jsonl
```


Start server 
```
ng_run "+config_paths=[responses_api_agents/simple_agent/configs/simple_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/python_math_exec/configs/python_math_exec.yaml]"     +simple_agent.responses_api_agents.simple_agent.resources_server.name=python_math_exec
```


Collect trajectories
```
ng_collect_rollouts +agent_name=simple_agent +input_jsonl_fpath=data/open_math_reasoning_problems_tool.jsonl +output_jsonl_fpath=results/open_math_reasoning_problems_tool_output_new.jsonl +limit=1
```



### Additional Details

Currently the system prompt looks like this. In the future this can be changed
```
# Instructions
You are a math assistant. When you need to perform calculations, use the execute_python tool to write and run Python code. Any result that can be used should be in a print statement so it can be captured by stdout. If there are multiple calculations and they depend on each other, you should use the execute_python tool to perform the calculations step by step. You have access to numpy, scipy, and pandas libraries. After completing all calculations, provide your final answer in the format: \\boxed{your_answer} 

# Available Tools
- execute_python: Execute Python code to perform mathematical calculations

# Example
If asked "What is 5 factorial?", you would use execute_python with code like:
```python
import math
result = math.factorial(5)
print(f"5! = {result}")
``` 

### Collected trajectories

```
{"responses_create_params": {"background": null, "include": null, "input": [{"content": "# Instructions\nYou are a math assistant. When you need to perform calculations, use the execute_python tool to write and run Python code. Any result that can be used should be in a print statement so it can be captured by stdout. If there are multiple calculations and they depend on each other, you should use the execute_python tool to perform the calculations step by step. You have access to numpy, scipy, and pandas libraries. After completing all calculations, provide your final answer in the format: \\boxed{your_answer} \n\n# Available Tools\n- execute_python: Execute Python code to perform mathematical calculations\n\n# Example\nIf asked \"What is 5 factorial?\", you would use execute_python with code like:\n```python\nimport math\nresult = math.factorial(5)\nprint(f\"5! = {result}\")\n```", "role": "system", "type": "message"}, {"content": "Determine how many 1000 digit numbers \\( A \\) have the property that when any digit of \\( A \\), aside from the first, is deleted to form a 999 digit number \\( B \\), then \\( B \\) divides \\( A \\).", "role": "user", "type": "message"}], "instructions": null, "max_output_tokens": null, "max_tool_calls": null, "metadata": null, "model": null, "parallel_tool_calls": false, "previous_response_id": null, "prompt": null, "reasoning": null, "service_tier": null, "store": null, "temperature": null, "text": null, "tool_choice": "auto", "tools": [{"name": "execute_python", "parameters": {"properties": {"code": {"type": "string", "title": "Python Code", "description": "The Python code to execute for the calculation."}}, "type": "object", "required": ["code"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Execute Python code to perform calculations. You have access to numpy, scipy, pandas and basic math operations."}], "top_logprobs": null, "top_p": null, "truncation": null, "user": null, "stream": null}, "response": {"id": "resp_021cf35c851b3d560068c46fa6dcb481968eb9bd8e53990061", "created_at": 1757704102.0, "error": null, "incomplete_details": null, "instructions": null, "metadata": {}, "model": "gpt-4.1-2025-04-14", "object": "response", "output": [{"id": "msg_021cf35c851b3d560068c46f84110c8196863d28e0aed85fcb", "content": [{"annotations": [], "text": "Let \\( A \\) be a 1000-digit number. If you remove any digit (other than the first) to obtain a 999-digit number \\( B \\), then \\( B \\) must divide \\( A \\).\n\nLet's formalize what this means.\n\nLet \\( A = a_1 a_2 \\ldots a_{1000} \\) (digits with \\( a_1 \\neq 0 \\)), and for each \\( 2 \\leq k \\leq 1000 \\), let \\( B_k \\) be the number formed by removing the \\( k \\)-th digit from \\( A \\).\n\nWe need \\( B_k | A \\) for every \\( k \\) from 2 to 1000.\n\nLet's find all such 1000-digit numbers \\( A \\).\n\n### Step 1: Try if such numbers exist at all.\n\nConsider divisibility. If such a number \\( A \\) exists, it should be highly divisible by all its 999-digit versions. Let's try with a known digit pattern.\n\n#### (a) All digits equal\n\nSuppose \\( A = 111\\ldots 1 \\) (1000 times the digit 1).\n\nThen deleting the \\( k \\)-th digit gives another number \\( B_k = 111\\ldots 1 \\) (999 times digit 1).\n\nLet's compute if \\( B_k | A \\) for this case:\n\nLet \\( x = 111\\ldots 1 \\) (999 times), and \\( A = x \\times 10 + 1 \\) (removing a digit in the second spot). But in general, the position changes so let's write formally:\n\nThe number with \\( n \\) digits '1' is \\( S_n = \\frac{10^n - 1}{9} \\).\n\nSo,\n- \\( A = S_{1000} = \\frac{10^{1000} - 1}{9} \\)\n- If we remove the \\( k \\)-th digit, we get: \\( B_k = S_{999} = \\frac{10^{999} - 1}{9} \\)\n\nDoes \\( S_{999} \\mid S_{1000} \\)?\n\nLet's check:\n\n\\[\nS_{999} = \\frac{10^{999} - 1}{9}\n\\]\n\\[\nS_{1000} = \\frac{10^{1000} - 1}{9}\n\\]\n\n\\[\nS_{1000} = \\frac{10 \\cdot 10^{999} - 1}{9}\n= \\frac{10^{1000} - 1}{9}\n= \\frac{10 \\cdot 10^{999} - 1}{9}\n= \\frac{10 \\cdot (10^{999} - 1) + 10 - 1}{9}\n= 10 \\cdot \\frac{10^{999} - 1}{9} + \\frac{9}{9}\n= 10S_{999} + 1\n\\]\n\nSo,\n\\[\nS_{1000} = 10 S_{999} + 1\n\\]\n\nIs \\( S_{999} \\mid S_{1000} \\)?\n\n\\[\nS_{1000} \\equiv 1 \\pmod{S_{999}}\n\\]\n\nSo \\( S_{999} \\) does **not** divide \\( S_{1000} \\).\n\nSimilar calculation holds for all digits the same.\n\n---\n\n### Step 2: What about numbers with only two digits, like numbers with 999 nines followed by a zero?\n\nIf \\( A \\) ends with zero, removing zero gives a 999-digit number. But similar arithmetic likely fails.\n\n---\n\n### Step 3: Try **repunit numbers** and see if any pattern works.\n\nLet's compute in a smaller case. Try \\( A = 111 \\), n=3.\n\n- Remove 2nd digit: 11. Does 11 | 111? 111/11 = 10.09, no.\n\nTry consecutive numbers.\n\nLook for palindromes or structure.\n\n---\n\n### Step 4: Try numbers of the form \\( 10^p \\), e.g., \\( 10^{999} \\).\n \n- \\( 10^{999} \\) has one digit 1 followed by all zeros.\n- If we remove any zero digit, the length reduces, but the result is still a power of ten or similar.\n- However, then \\( B \\) would be \\( 10^{k-2} \\times 10^{999-(k-1)} \\), but in general, the result is not necessarily a divisor.\n\n---\n\n### Step 5: Consider divisible by all possible B.\n\nIn summary, for any 1000-digit number \\( A \\), if after deleting any digit except the first, the resulting 999-digit number always divides \\( A \\), this is only possible if *all* such numbers are divisors of \\( A \\). But the number of these divisors is huge, and for most \\( A \\), not the case.\n\n**Let\u2019s check if any number exists at all.**\n\nLet's recall a classic result: it\u2019s only possible when all digits are zero **but that\u2019s not a valid 1000-digit number (leading digit can't be zero).**\n\n#### Let's try with \\( A = 100\\ldots 00 \\) (one followed by 999 zeros):\n\n- Remove k-th digit, most give a number with one less zero.\n- The resulting 999-digit number will be \\( 10^{999-k} \\times 10^{k-2} \\), so \\( B_k \\).\n- Original \\( A = 10^{999} \\).\n- Will \\( B_k \\mid A \\) for all \\( k \\)? Given both are powers of 10 (possibly less a factor of ten), yes, but only if the structure is preserved.\n\nLet\u2019s verify for \\( k=2 \\):\n- Remove second digit: The number is \\( 10^{998} \\).\n- \\( 10^{998} \\mid 10^{999} \\), yes.\n\nSimilarly, for all \\( k \\), the result is a power of ten less than the original, always a divisor.\n\nTherefore, numbers of the form\n\\[\nA = 10^{999}\n\\]\nBut \\( 10^{999} \\) is 1 followed by 999 zeros. Is this a 1000-digit number? Yes.\n\n**Are there any more such numbers?** Suppose \\( A = d \\times 10^{999} \\) with \\( d > 1 \\). For other digits being nonzero, the divisibility will fail.\n\n### Step 6: Final Check\n\nLet's check if **any** 1000-digit number of the form \\( A = 10^{999} \\) has the property.\n\n- Remove any zero: the number becomes a 999-digit number, which is a power of ten, and it always divides \\( 10^{999} \\).\n- Remove the first digit (**not allowed** per the problem).\n- Remove any other digit: always the same.\n\nSo, **only one such 1000-digit number exists.**\n\nLet me check with Python for more systematic confirmation.", "type": "output_text", "logprobs": []}], "role": "assistant", "status": "completed", "type": "message"}, {"arguments": "{\n  \"code\": \"count = 0\\nfor d in range(1, 10):\\n    A = str(d) + '0' * 999\\n    # For all k from 2 to 1000 (1 to 999 in zero-based)\\n    good = True\\n    for k in range(1, 1000):\\n        B = A[:k] + A[k+1:] # Remove digit at k (1-based, zero-based index)\\n        B_int = int(B)\\n        if int(A) % B_int != 0:\\n            good = False\\n            break\\n    if good:\\n        count += 1\\nprint(count)\\n\"\n}", "call_id": "call_RLAOPO9GBdgI4hV9oAD3sC1C", "name": "execute_python", "type": "function_call", "id": "fc_021cf35c851b3d560068c46fa469b48196a1e0683825bad194", "status": "completed"}, {"call_id": "call_RLAOPO9GBdgI4hV9oAD3sC1C", "output": "{\"success\":true,\"stdout\":\"9\\n\",\"stderr\":\"\",\"error_message\":null,\"result\":null,\"session_id\":\"f7b4e540-337b-4bdc-9e61-ba9510b00740\"}", "type": "function_call_output", "id": null, "status": null}, {"id": "msg_021cf35c851b3d560068c46fa73a8c81968d9b70a75d7b8731", "content": [{"annotations": [], "text": "\\[\n\\boxed{9}\n\\]\n\nThere are \\(\\boxed{9}\\) such 1000-digit numbers. These are the numbers where the first digit is any from 1 to 9, followed by 999 zeros.", "type": "output_text", "logprobs": []}], "role": "assistant", "status": "completed", "type": "message"}], "parallel_tool_calls": false, "temperature": 1.0, "tool_choice": "auto", "tools": [{"name": "execute_python", "parameters": {"properties": {"code": {"type": "string", "title": "Python Code", "description": "The Python code to execute for the calculation."}}, "type": "object", "required": ["code"], "additionalProperties": false}, "strict": true, "type": "function", "description": "Execute Python code to perform calculations. You have access to numpy, scipy, pandas and basic math operations."}], "top_p": 1.0, "background": false, "max_output_tokens": null, "max_tool_calls": null, "previous_response_id": null, "prompt": null, "reasoning": {"effort": null, "generate_summary": null, "summary": null}, "service_tier": "default", "status": "completed", "text": {"format": {"type": "text"}, "verbosity": "medium"}, "top_logprobs": 0, "truncation": "disabled", "usage": {"input_tokens": 2007, "input_tokens_details": {"cached_tokens": 1664}, "output_tokens": 50, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2057}, "user": null, "prompt_cache_key": null, "safety_identifier": null, "store": true}, "reward": 0.0, "extracted_answer": "9", "accuracy": false}

```