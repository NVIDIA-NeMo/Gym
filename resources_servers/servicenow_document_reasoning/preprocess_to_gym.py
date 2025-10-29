import json
from pathlib import Path


input_fpath = "data/servicenow_servicenow.json"
input_fpath = Path(input_fpath)
output_fpath = Path("resources_servers/servicenow_document_reasoning/data/validation.jsonl")

with input_fpath.open() as f:
    data = json.load(f)

with output_fpath.open("w") as f:
    for row in data:
        kb_documents_str = ""
        for document in row["kb_documents"]:
            kb_document_str = f"""title: {document["title"]}
title: {document["content"]}

"""
            kb_documents_str += kb_document_str

        output_row = {
            "responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert in reasoning and answering questions using provided context snippets. You will be provided a set of context snippets in the form of several search results and a question to answer. Both the context and question will be enclosed in ```. Please answer the question only using the content in the snippets provided and provide as much detail as possible in your answer. Do not provide any content in the answer that is not in the context provided to you.",
                    },
                    {
                        "role": "user",
                        "content": f"""# Context
```
{kb_documents_str}
```

# Question
```
{row["question"]}
```""",
                    },
                ],
            },
            "judge_responses_create_params": {
                "input": [
                    {
                        "role": "developer",
                        "content": "You are an expert grader. You will be provided with a correct reference answer and a candidate answer, each enclosed in ```. Your job is to grade whether the candidate answer matches the correct reference answer.",
                    },
                    {
                        "role": "user",
                        "content": f"""# Reference answer
```
{row["ground_truth"]}
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
                                "matches_reference": {
                                    "type": "boolean",
                                    "description": "Whether or not the candidate answer matches the correct reference answer.",
                                },
                            },
                            "required": ["matches_reference"],
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
            "kb_documents": row["kb_documents"],
            "catalogs": row["catalogs"],
            "question": row["question"],
            "unanswerable": row["unanswerable"],
            "type": row["type"],
            "subtype": row["subtype"],
            "ground_truth": row["ground_truth"],
            "upstream_customer_metadata": {
                "reference": row["reference"],
                "answer_explanation": row["answer_explanation"],
            },
        }

        f.write(json.dumps(output_row) + "\n")
