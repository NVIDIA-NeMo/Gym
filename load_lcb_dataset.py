"""
Run as:
```bash
python load_lcb_dataset.py
```
"""

import json
import os
from argparse import Namespace


def preprocess_livecodebench_chatml_template(data_file):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    code_instruction = """Write Python code to solve the problem. Please place the solution code in the following format:\n```python\n# Your solution code here\n```"""

    with open(data_file, "r") as f:
        data_list = json.load(f)

    prompt_list = []
    qid_list = []
    for item in data_list:
        question = item["question_content"].strip()
        if item["starter_code"] != "":
            question += (
                "\n\n"
                + "Solve the problem starting with the provided function header.\n\nFunction header:\n"
                + "```\n"
                + item["starter_code"]
                + "\n```"
            )

        question += "\n\n" + code_instruction

        final_prompt = instruction + "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"

        prompt_list.append(final_prompt)
        qid_list.append(item["question_id"])

    return prompt_list, qid_list


def get_prompt_list(args):
    ## get input data
    if args.eval_dataset == "livecodebench":
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    elif args.eval_dataset == "livecodebench_v6":
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_v6_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    else:
        raise ValueError("please input a correct eval_dataset name!")

    return prompt_list, qid_list


if __name__ == "__main__":
    args = Namespace(
        benchmark_folder="/lustre/fsw/portfolios/llmservice/users/wdai/data/foundational_qa/test_benchmarks",
        livecodebench_path="livecodebench/test_aug2024tojan2025.json",
        livecodebench_v6_path="livecodebench/test_feb2025toApr2025.json",
        eval_dataset="livecodebench",
    )
    prompt_list, qid_list = get_prompt_list(args)

    with open("temp.json", "w") as f:
        json.dump(prompt_list, f)
