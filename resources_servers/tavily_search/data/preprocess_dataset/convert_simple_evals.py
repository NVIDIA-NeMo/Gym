import json
import copy
import os
import sys
import random
# Add parent directory to path so we can import from app.py

random.seed(42)


# SFT_DATA_FILE_PATH="/lustre/fsw/portfolios/llmservice/users/rgala/repos/research-scratch/11_12_search/data/12_14_postprocessed_exa_search_tool_sft_1395samples_with_think_and_tools.jsonl"
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for a query and return up to 10 search results with <link, summary> for each result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to search for"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": False
    }
]

def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def map_sft_sample_to_rl_sample(sft_sample):

    def get_question(messages):
        assert len(messages) >= 2
        assert messages[0]["role"] == "system" or messages[0]["role"] == "developer"
        assert messages[1]["role"] == "user"
        return messages[1]["content"]

    def strip_messages_to_no_asst(messages):
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        messages[0]["role"] = "developer"
        return copy.deepcopy(messages[:2])

    # def reformat_tools(tools):
        # new_tools = []
        # for tool in tools:
        #     new_tools.append({
        #         "type": "function",
        #         **tool["function"]
        #     })
        #     new_tools[-1]["parameters"]["required"] = new_tools[-1]["parameters"].get("required", ["query"])
        #     new_tools[-1]["parameters"]["additionalProperties"] = False
        #     new_tools[-1]["strict"] = False #got 422 error without this.
        # return new_tools


    starting_message_sample = strip_messages_to_no_asst(sft_sample["messages"])
    question = get_question(starting_message_sample)

    responses_create_params = {
        "input": starting_message_sample,
        "tools": TOOLS
    }

    return {
        "responses_create_params": responses_create_params,
        "ground_truth": sft_sample["metadata"]["final_answer_entity"],
        "question": question
    }


def map_benchmark_sample_to_rl_sample(benchmark_sample):

    messages = [
        {
            "role": "system",
            "content": "Please think step by step and reason about the problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return your final answer."
        },
        {
            "role": "user",
            "content": QUERY_TEMPLATE.format(Question=benchmark_sample["problem"])
        }
    ]

    responses_create_params = {
        "input": messages,
        "tools": TOOLS
    }

    return {
        "responses_create_params": responses_create_params,
        "ground_truth": benchmark_sample["answer"],
        "question": benchmark_sample["problem"]
    }

def test_final_sample(sample):
    required_keys = ["responses_create_params", "ground_truth", "question"]
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Key {key} not found in sample")
    return sample

def write_benchmark_samples(rl_samples, output_data_folder, completed_test_set_file_name, n_100_file_name, n_30_file_name):
    random.shuffle(rl_samples)
    with open(os.path.join(output_data_folder, completed_test_set_file_name), "w") as file:
        for rl_sample in rl_samples:
                file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, n_100_file_name), "w") as file:
        for rl_sample in rl_samples[:100]:
                file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, n_30_file_name), "w") as file:
        for rl_sample in rl_samples[:30]:
            file.write(json.dumps(rl_sample) + "\n")

def write_train_validation_samples(rl_samples, output_data_folder, train_file_name, validation_file_name):
    random.shuffle(rl_samples)
    with open(os.path.join(output_data_folder, train_file_name), "w") as file:
        for rl_sample in rl_samples[:-100]:
            file.write(json.dumps(rl_sample) + "\n")
    with open(os.path.join(output_data_folder, validation_file_name), "w") as file:
        for rl_sample in rl_samples[-100:]:
            file.write(json.dumps(rl_sample) + "\n")

if __name__ == "__main__":

    GYM_DATA_FOLDER = "/lustre/fsw/portfolios/llmservice/users/rgala/repos/nemo-gym-gitlab/resources_servers/tavily_search/data"
    DATA_FILE_PATH="/lustre/fsw/portfolios/llmservice/users/rgala/repos/nemo-gym-gitlab/resources_servers/tavily_search/preprocess_dataset/simple_evals_og_dataset/simple_qa_test_set.jsonl"
    benchmark_samples = read_jsonl_file(DATA_FILE_PATH)

    # rl_samples = [map_simple_evals_sample_to_rl_sample(simple_evals_sample) for simple_evals_sample in simpleqa_samples]
    rl_samples = [map_benchmark_sample_to_rl_sample(benchmark_sample) for benchmark_sample in benchmark_samples]

    OUTPUT_DATA_FOLDER = os.path.join(GYM_DATA_FOLDER, "benchmark", "simpleqa")
    os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)

    rl_samples = [test_final_sample(rl_sample) for rl_sample in rl_samples]
    write_benchmark_samples(rl_samples, OUTPUT_DATA_FOLDER, "simpleqa_test_set.jsonl", "simpleqa_test_set_n_100.jsonl", "simpleqa_test_set_n_30.jsonl")
