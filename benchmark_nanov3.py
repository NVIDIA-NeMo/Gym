import json


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
