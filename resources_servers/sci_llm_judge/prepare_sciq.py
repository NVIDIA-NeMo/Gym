import json
import os

from datasets import load_dataset


ds = load_dataset("allenai/sciq", split="validation")

rows = []
for example in ds:
    # Build a simple prompt: prepend instruction string, no options included
    prefix = "Answer the following question. Make sure to put the final answer (and only the final answer) inside \\boxed{}.\n\n"
    user_content = prefix + example["question"]

    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
        },
        "question": example["question"],
        "expected_answer": example["correct_answer"],
    }
    rows.append(json.dumps(row) + "\n")


os.makedirs("data", exist_ok=True)
with open("data/sciq_validation.jsonl", "w") as f:
    f.writelines(rows)
