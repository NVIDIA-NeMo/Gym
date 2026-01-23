import csv
import json
from pathlib import Path


input_fpath = "data/Eval sets - cohesity.csv"
input_fpath = Path(input_fpath)
output_fpath = Path("resources_servers/cohesity_netbackup_rag/data/validation.jsonl")

with input_fpath.open() as f:
    next(f)  # Skip the first line
    data = list(csv.DictReader(f))

with output_fpath.open("w") as f:
    for row in data:
        if row["Brian context contains reference answer?"] != "Yes":
            continue

        output_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert in answering questions using provided context snippets. You will be provided a set of context snippets in the form of several search results and a question to answer. Both the context and question will be enclosed in ```. Please answer the question only using the content in the snippets provided and provide as much detail as possible in your answer. Do not provide any content in the answer that is not in the context provided to you.",
                    },
                    {
                        "role": "user",
                        "content": f"""# Context
```
{row["context"]}
```

# Question
```
{row["question"]}
```""",
                    },
                ],
                "max_output_tokens": 16_384,
            },
            "judge_responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert grader. You will be provided with several context snippets, a question, a correct reference answer to that question, and a candidate answer, each enclosed in ```. Your job is to grade whether the candidate answer matches the correct reference answer in terms of content and whether the candidate answer contains additional content that is not found in the context snippets.",
                    },
                    {
                        "role": "user",
                        "content": f"""# Context
```
{row["context"]}
```

# Question
```
{row["question"]}
```

# Reference answer
```
{row["reference_answer"]}
```

# Candidate answer
```
{{candidate_answer}}
```""",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "grade_candidate_answer",
                        "description": "",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "matches_reference_content": {
                                    "type": "boolean",
                                    "description": "Whether or not the candidate answer matches the correct reference answer in terms of content.",
                                },
                                "contains_additional_content_not_in_context": {
                                    "type": "boolean",
                                    "description": "Whether or not the candidate answer contains additional content that is not found in the context snippets.",
                                },
                            },
                            "required": ["matches_reference_content", "contains_additional_content_not_in_context"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                ],
                "parallel_tool_calls": False,
                "temperature": 0.0,
            },
            # Carryover metadata
            "id": row["id"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "context": row["context"],
            "upstream_customer_metadata": {
                "generated_answer": row["generated_answer"],
                "scores": row["scores"],
                "labels": row["labels"],
                "explanations": row["explanations"],
            },
        }

        f.write(json.dumps(output_row) + "\n")
