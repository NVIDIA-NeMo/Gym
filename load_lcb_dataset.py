"""
Run as:
```bash
python inference.py \
    --benchmark-folder /lustre/fsw/portfolios/llmservice/users/wdai/data/foundational_qa/test_benchmarks \
    --livecodebench-path livecodebench/test_aug2024tojan2025.json \
    --livecodebench-v6-path livecodebench/test_feb2025toApr2025.json \
    --eval_dataset livecodebench
```
"""

import json
import os


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
    data_list = None

    ## get input data
    if args.eval_dataset == "livecodebench":
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    elif args.eval_dataset == "livecodebench_v6":
        input_datapath = os.path.join(args.benchmark_folder, args.livecodebench_v6_path)
        prompt_list, qid_list = preprocess_livecodebench_chatml_template(input_datapath)

    else:
        raise ValueError("please input a correct eval_dataset name!")

    print("number of total prompt_list:", len(prompt_list))
    if args.start_idx != -1 and args.end_idx != -1:
        print("getting data from %d to %d" % (args.start_idx, args.end_idx))
        prompt_list = prompt_list[args.start_idx : args.end_idx]
        if qid_list:
            qid_list = qid_list[args.start_idx : args.end_idx]

    print("number of test samples in the dataset:", len(prompt_list))

    if data_list is not None:
        return prompt_list, qid_list, data_list
    else:
        return prompt_list, qid_list
