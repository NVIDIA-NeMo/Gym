import json

from datasets import load_dataset


val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

rows = []
for example in val_ds:
    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "system",
                    "content": "Your task is to solve a math problem.  Make sure to put the answer (and only the answer) inside \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": example["problem"],
                },
            ]
        },
        "question": example["problem"],
        "expected_answer": example["answer"],
    }
    rows.append(json.dumps(row) + "\n")


with open("data/aime24_validation.jsonl", "w") as f:
    f.writelines(rows)
