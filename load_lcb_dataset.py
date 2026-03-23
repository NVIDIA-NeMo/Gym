"""
Run as:
```bash
python load_lcb_dataset.py
```

One sample looks like:
```json
<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n<|im_start|>user\nYou are given an integer sequence A=(A_1,\\ldots,A_N) of length N. Here, A_1, A_2, \\ldots, A_N are all distinct.\nWhich element in A is the second largest?\n\nInput\n\nThe input is given from Standard Input in the following format:\nN \r\nA_1 A_2 \\ldots A_{N}\n\nOutput\n\nPrint the integer X such that the X-th element in A is the second largest.\n\nConstraints\n\n\n- 2 \\leq N \\leq 100\n- 1 \\leq A_i \\leq 10^9\n- A_1, A_2, \\ldots, A_N are all distinct.\n- All input values are integers.\n\nSample Input 1\n\n4\r\n8 2 5 1\n\nSample Output 1\n\n3\r\n\nThe second largest element in A is A_3, so print 3.\n\nSample Input 2\n\n8\r\n1 2 3 4 5 10 9 11\n\nSample Output 2\n\n6\n\nWrite Python code to solve the problem. Please place the solution code in the following format:\n```python\n# Your solution code here\n```<|im_end|>\n<|im_start|>assistant\n<think>\n
```
```
<|im_start|>system
You are a helpful and harmless assistant.

You are not allowed to use any tools.<|im_end|>
<|im_start|>user
You are given an integer sequence A=(A_1,\ldots,A_N) of length N. Here, A_1, A_2, \ldots, A_N are all distinct.
Which element in A is the second largest?

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \ldots A_{N}

Output

Print the integer X such that the X-th element in A is the second largest.

Constraints


- 2 \leq N \leq 100
- 1 \leq A_i \leq 10^9
- A_1, A_2, \ldots, A_N are all distinct.
- All input values are integers.

Sample Input 1

4
8 2 5 1

Sample Output 1

3

The second largest element in A is A_3, so print 3.

Sample Input 2

8
1 2 3 4 5 10 9 11

Sample Output 2

6

Write Python code to solve the problem. Please place the solution code in the following format:
```python
# Your solution code here
```<|im_end|>
<|im_start|>assistant
<think>
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
