import json

import pandas


math_instruction = "Please place your final answer inside \\boxed{}."


def preprocess_gpqa_chatml_template(data_file):
    QUERY_TEMPLATE_MULTICHOICE = "Return your final response within \\boxed{{}} and only include the letter choice (e.g., A, B, C, or D) as your final response.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    with open(data_file, "r") as f:
        data_list = json.load(f)

    prompt_list = []
    for item in data_list:
        choices_dict = dict(
            Question=item["question"].strip(),
            choice1=item["choice_A"].strip(),
            choice2=item["choice_B"].strip(),
            choice3=item["choice_C"].strip(),
            choice4=item["choice_D"].strip(),
        )
        final_question = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        final_prompt = (
            instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_aime24_chatml_template(data_file):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    prompt_list = []
    with open(data_file, "r") as f:
        for line in f:
            item = json.loads(line)
            final_question = item["question"].strip()
            final_question += "\n\n" + math_instruction

            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )

            prompt_list.append(final_prompt)

    return prompt_list


def preprocess_aime25_chatml_template(data_file):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    prompt_list = []
    with open(data_file, "r") as f:
        for line in f:
            item = json.loads(line)
            final_question = item["problem"].strip()
            final_question += "\n\n" + math_instruction

            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )

            prompt_list.append(final_prompt)

    return prompt_list


def preprocess_hmmt25_chatml_template(data_file):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    prompt_list = []
    with open(data_file, "r") as f:
        for line in f:
            item = json.loads(line)
            final_question = item["problem"].strip()
            final_question += "\n\n" + math_instruction

            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )

            prompt_list.append(final_prompt)

    return prompt_list


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


def preprocess_mmlu_reasoning_chatml_template(data_file):
    QUERY_TEMPLATE_MULTICHOICE = "Return your final response within \\boxed{{}} and only include the letter choice (e.g., A, B, C, or D) as your final response.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    df = pandas.read_csv(data_file)
    test_list = [row.to_dict() for _, row in df.iterrows()]

    prompt_list = []
    for item in test_list:
        question = item["Question"]
        choice_a = str(item["A"]).strip()
        choice_b = str(item["B"]).strip()
        choice_c = str(item["C"]).strip()
        choice_d = str(item["D"]).strip()

        choices_dict = dict(Question=question, choice1=choice_a, choice2=choice_b, choice3=choice_c, choice4=choice_d)
        final_question = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
        final_prompt = (
            instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_mtbench_firstturn(data_file):
    with open(data_file, "r") as f:
        data_list = json.load(f)

    qid_list = []
    prompt_list = []
    for item in data_list:
        qid_list.append(item["prompt_id"])
        first_question = item["prompt"][0]
        final_prompt = "<|im_start|>user\n" + first_question + "<|im_end|>\n<|im_start|>assistant\n"

        prompt_list.append(final_prompt)

    return prompt_list, qid_list


def preprocess_mtbench_secondturn(data_file, output_file):
    """
    output_file: model output file for the first turn of MT Bench
    """
    with open(data_file, "r") as f:
        data_list = json.load(f)

    id2output = {}
    with open(output_file, "r") as f:
        for line in f:
            item = json.loads(line)
            id2output[item["task_id"]] = item["output"]

    qid_list = []
    prompt_list = []
    for item in data_list:
        qid_list.append(item["prompt_id"])
        first_question = item["prompt"][0]
        second_question = item["prompt"][1]

        model_output = id2output[item["prompt_id"]]

        final_prompt = (
            "<|im_start|>user\n"
            + first_question
            + "<|im_end|>\n<|im_start|>assistant\n"
            + model_output
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + second_question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

        prompt_list.append(final_prompt)

    return prompt_list, qid_list


def preprocess_ifeval_chatml_template(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    with open(data_file, "r") as f:
        data = f.readlines()
    data_list = [json.loads(x) for x in data]

    qid_list = []
    prompt_list = []
    for item in data_list:
        qid_list.append(item["key"])
        final_question = item["prompt"]

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n</think>\n"
            )

        prompt_list.append(final_prompt)

    return prompt_list, qid_list


def preprocess_mmlu_zero_shot_chatml_template(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    # MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Question:\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\n\nConclude your response with the sentence `The answer is \\boxed{{X}}.`, in which X is the correct capital letter of your choice."
    df = pandas.read_csv(data_file)
    test_list = [row.to_dict() for _, row in df.iterrows()]

    prompt_list = []
    for item in test_list:
        choices_dict = dict(
            Question=item["Question"].strip(),
            choice1=str(item["A"]).strip(),
            choice2=str(item["B"]).strip(),
            choice3=str(item["C"]).strip(),
            choice4=str(item["D"]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n</think>\n"
            )

        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_mmlu_redux_zero_shot_chatml_template(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Question:\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\n\nConclude your response with the sentence `The answer is \\boxed{{X}}.`, in which X is the correct capital letter of your choice."
    test_list = json.load(open(data_file))

    prompt_list = []
    for item in test_list:
        choices_dict = dict(
            Question=item["question"].strip(),
            choice1=str(item["options"][0]).strip(),
            choice2=str(item["options"][1]).strip(),
            choice3=str(item["options"][2]).strip(),
            choice4=str(item["options"][3]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n</think>\n"
            )

        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_mmlu_pro_zero_shot_chatml_template(data_file, think=True):
    def _preprocess(data_list):
        output_list = []
        for item in data_list:
            options = []
            for opt in item["options"]:
                if opt == "N/A":
                    continue
                options.append(opt)
            item["options"] = options
            output_list.append(item)
        return output_list

    def _format_each_sample(sample):
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

        question = sample["question"]
        options = sample["options"]

        sample_prompt = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n"
        sample_prompt += question + "\n\nAnswer Choices:"
        for i, opt in enumerate(options):
            sample_prompt += "\n(%s) %s" % (choices[i], opt)
        sample_prompt = sample_prompt.strip() + "\n"

        return sample_prompt

    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    with open(data_file, "r") as f:
        test_list = json.load(f)
    test_list = _preprocess(test_list)

    prompt_list = []
    for test_sample in test_list:
        test_prompt = _format_each_sample(test_sample)
        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = (
                instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n</think>\n"
            )
        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_mmlu_redux_zero_shot_chatml_template(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    test_list = json.load(open(data_file))

    prompt_list = []
    for item in test_list:
        choices_dict = dict(
            Question=item["question"].strip(),
            choice1=str(item["options"][0]).strip(),
            choice2=str(item["options"][1]).strip(),
            choice3=str(item["options"][2]).strip(),
            choice4=str(item["options"][3]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n</think>\n"
            )

        prompt_list.append(final_prompt)

    return prompt_list


def preprocess_mmlu_redux_zero_shot_chatml_template_with_id(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    test_list = json.load(open(data_file))

    prompt_list = []
    for idx, item in enumerate(test_list):
        choices_dict = dict(
            Question=item["question"].strip(),
            choice1=str(item["options"][0]).strip(),
            choice2=str(item["options"][1]).strip(),
            choice3=str(item["options"][2]).strip(),
            choice4=str(item["options"][3]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n"

        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


def preprocess_mmlu_redux_zero_shot_chatml_template_with_id_using_gpqa_prompt(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Return your final response within \\boxed{{}} and only include the letter choice (e.g., A, B, C, or D) as your final response.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    # MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

    test_list = json.load(open(data_file))

    prompt_list = []
    for idx, item in enumerate(test_list):
        choices_dict = dict(
            Question=item["question"].strip(),
            choice1=str(item["options"][0]).strip(),
            choice2=str(item["options"][1]).strip(),
            choice3=str(item["options"][2]).strip(),
            choice4=str(item["options"][3]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n"

        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


def preprocess_mmlu_zero_shot_chatml_template_with_id(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"
    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"
    df = pandas.read_csv(data_file)
    test_list = [row.to_dict() for _, row in df.iterrows()]

    prompt_list = []
    for idx, item in enumerate(test_list):
        choices_dict = dict(
            Question=item["Question"].strip(),
            choice1=str(item["A"]).strip(),
            choice2=str(item["B"]).strip(),
            choice3=str(item["C"]).strip(),
            choice4=str(item["D"]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n"

        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


def preprocess_mmlu_zero_shot_chatml_template_with_id_using_gpqa_prompt(data_file, think=True):
    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    MMLU_QUERY_TEMPLATE_MULTICHOICE = "Return your final response within \\boxed{{}} and only include the letter choice (e.g., A, B, C, or D) as your final response.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

    # MMLU_QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n{Question}\n\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

    df = pandas.read_csv(data_file)
    test_list = [row.to_dict() for _, row in df.iterrows()]

    prompt_list = []
    for idx, item in enumerate(test_list):
        choices_dict = dict(
            Question=item["Question"].strip(),
            choice1=str(item["A"]).strip(),
            choice2=str(item["B"]).strip(),
            choice3=str(item["C"]).strip(),
            choice4=str(item["D"]).strip(),
        )
        final_question = MMLU_QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)

        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + final_question + "<|im_end|>\n<|im_start|>assistant\n"

        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


def preprocess_mmlu_pro_zero_shot_chatml_template_with_id(data_file, think=True):
    def _preprocess(data_list):
        output_list = []
        for item in data_list:
            options = []
            for opt in item["options"]:
                if opt == "N/A":
                    continue
                options.append(opt)
            item["options"] = options
            output_list.append(item)
        return output_list

    def _format_each_sample(sample):
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

        question = sample["question"]
        options = sample["options"]

        # sample_prompt = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n"
        # sample_prompt += question + "\n\nAnswer Choices:"
        # for i, opt in enumerate(options):
        #     sample_prompt += "\n(%s) %s" % (choices[i], opt)
        # sample_prompt = sample_prompt.strip() + "\n"

        sample_prompt = "Question:\n" + question + "\n\nAnswer Choices:"
        for i, opt in enumerate(options):
            sample_prompt += "\n(%s) %s" % (choices[i], opt)
        sample_prompt += "\n\nConclude your response with the sentence `The answer is \\boxed{{X}}.`, in which X is the correct capital letter of your choice."
        sample_prompt = sample_prompt.strip() + "\n"

        return sample_prompt

    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    with open(data_file, "r") as f:
        test_list = json.load(f)
    test_list = _preprocess(test_list)

    prompt_list = []
    for idx, test_sample in enumerate(test_list):
        test_prompt = _format_each_sample(test_sample)
        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


def preprocess_mmlu_pro_zero_shot_chatml_template_with_id_using_gpqa_prompt(data_file, think=True):
    def _preprocess(data_list):
        output_list = []
        for item in data_list:
            options = []
            for opt in item["options"]:
                if opt == "N/A":
                    continue
                options.append(opt)
            item["options"] = options
            output_list.append(item)
        return output_list

    def _format_each_sample(sample):
        choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

        question = sample["question"]
        options = sample["options"]

        # sample_prompt = "Answer the following multiple-choice question. At the end of your response, conclude with the sentence `The answer is \\boxed{{X}}.`, replacing X with the correct capital letter of your choice.\n\n"
        sample_prompt = "Return your final response within \\boxed{{}} and only include the letter choice (e.g., A, B, C, or D) as your final response.\n\n"

        sample_prompt += question + "\n\nAnswer Choices:"
        for i, opt in enumerate(options):
            sample_prompt += "\n(%s) %s" % (choices[i], opt)
        sample_prompt = sample_prompt.strip() + "\n"

        return sample_prompt

    instruction = "<|im_start|>system\nYou are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.<|im_end|>\n"

    with open(data_file, "r") as f:
        test_list = json.load(f)
    test_list = _preprocess(test_list)

    prompt_list = []
    for idx, test_sample in enumerate(test_list):
        test_prompt = _format_each_sample(test_sample)
        if think:
            final_prompt = (
                instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            )
        else:
            final_prompt = instruction + "<|im_start|>user\n" + test_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        prompt_list.append({"id": idx, "prompt": final_prompt})

    return prompt_list


# if __name__ == "__main__":
#     # humaneval_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/datasets/foundational_qa/test_benchmarks/evalplus/humanevalplus/test.json"
#     # preprocess_humaneval_chatml_template_v2(humaneval_path)

#     # mbpp_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/datasets/foundational_qa/test_benchmarks/evalplus/mbppplus/test.json"
#     # preprocess_mbpp_chatml_template_v2(mbpp_path)

#     bigcodebench_hard_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/datasets/foundational_qa/test_benchmarks/bigcodebench/bigcodebench_hard.json"
#     preprocess_bigcodebench_chatml_template(bigcodebench_hard_path)
