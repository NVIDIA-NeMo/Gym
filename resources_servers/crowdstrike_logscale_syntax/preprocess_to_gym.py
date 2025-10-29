import json
from pathlib import Path


input_fpath = "data/crowdstrike_logscaleSyntax.json"
input_fpath = Path(input_fpath)
output_fpath = Path("resources_servers/crowdstrike_logscale_syntax/data/validation.jsonl")


with input_fpath.open() as f:
    data = json.load(f)

with output_fpath.open("w") as f:
    for row in data:
        output_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert in using Crowdstrike LogScale software. If you need to output a LogScale query, please use keyword arguments whenever possible.",
                    },
                    {"role": "user", "content": row["question"]},
                ],
                "max_output_tokens": 16_384,
            },
            "judge_responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert in using Crowdstrike LogScale software. You will be provided with a query, some helpful context, a correct reference answer with proper LogScale syntax for that query, and 3 incorrect answers with improper LogScale syntax. Then, you will be given a candidate answer. Your job is to grade whether the candidate answer has proper syntax or not and whether the candidate answer used as many or more keyword arguments than the reference answer.",
                    },
                    {
                        "role": "user",
                        "content": f"""Query: {row["question"]}
Helpful context: {row["context"]}
Reference answer with correct syntax: {row["referencs"][0]["correct_answer"]}
Example answer 1 with incorrect syntax: {row["referencs"][0]["wrong_answer_1"]}
Example answer 2 with incorrect syntax: {row["referencs"][0]["wrong_answer_2"]}
Example answer 3 with incorrect syntax: {row["referencs"][0]["wrong_answer_3"]}

Candidate answer: {{candidate_answer}}""",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "grade_candidate_answer_syntax",
                        "description": "",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "correct_syntax": {
                                    "type": "boolean",
                                    "description": "Whether or not the candidate answer has correct syntax or not",
                                },
                                "used_keyword_arguments": {
                                    "type": "boolean",
                                    "description": "Whether or not the candidate answer used as many or more keyword arguments as the reference correct answer.",
                                },
                            },
                            "required": ["correct_syntax", "used_keyword_arguments"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "required",
                "temperature": 0.0,
            },
            # Carryover metadata
            "id": row["id"],
            "reference": row["referencs"][0],
        }

        f.write(json.dumps(output_row) + "\n")
