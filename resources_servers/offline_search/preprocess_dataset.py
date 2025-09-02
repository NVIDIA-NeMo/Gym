import os
import datasets
import json

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def convert_to_hf_dataset(jsonl_file_path):
    data = read_jsonl(jsonl_file_path)
    dataset = datasets.Dataset.from_list(data)
    return dataset

def load_input_dataset(dataset_path, is_local):
    if is_local:
        dataset = convert_to_hf_dataset(dataset_path)
    else:
        dataset = datasets.load_dataset(dataset_path, split="train")
    return dataset

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "name": "search",
        "description": "Search for a query. This returns the top 10 search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to search for",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

def convert_to_responses_api_format(dataset):
    new_dataset = []

    def convert_to_responses_api_format_single(row):
        new_row = {"responses_create_params": {}}
        create_params = json.loads(row["create_params"])

        messages = create_params["messages"]
        assert messages[0]["role"] == "system"
        new_row["responses_create_params"]["instructions"] = \
        """Solve the following problem. You are encouraged to use the tools provided to you to solve the problem, to make sure you can get to the right answer. You must only issue one tool call at a time. Once you are done issuing calls, then return you final answer and put the answer option (and only the option [A, B, C, etc]) inside \\boxed{{}}. The `search` tool helps return search results from the internet. You are encouraged to use tools to have the most grounded answer."""
        assert messages[1]["role"] == "user"
        new_row["responses_create_params"]["input"] = messages[1]["content"]
        new_row["responses_create_params"]["tools"]=TOOL_DESCRIPTIONS
        new_row["responses_create_params"]["max_output_tokens"]=32768 # TODO: max_tokens should change based on teh model, right now we keep it to 1000
        new_row["responses_create_params"]["temperature"]=0.6
        new_row["responses_create_params"]["parallel_tool_calls"]=False
        new_row["expected_answer"] = row["expected_answer"]
        new_row["task_difficulty_qwen3_32b_avg_8"] = row["task_difficulty_qwen3_32b_avg_8"] #TODO: Add this
        return new_row
    
    for sample in dataset:
        new_row = convert_to_responses_api_format_single(sample)
        new_dataset.append(new_row)
    return new_dataset
        
def write_to_jsonl(dataset, file_path):
    count=0
    with open(file_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            count+=1
    print(f"written {count} rows to {file_path}")
if __name__ == "__main__":
    resource_name="offline_search"
    # dataset_path = "Nexusflow/0823_syn_gpqa_hard_samples_only" 
    # output_file_name = f"MCQA_syn_gpqa_1_2_difficulty_filtered"
    dataset_path = "Nexusflow/0823_syn_hle_hard_samples_only"
    output_file_name = f"MCQA_syn_hle_difficulty_filtered"
    folder=os.path.join("resources_servers",resource_name, "data")
    os.makedirs(folder, exist_ok=True)
    
    dataset = load_input_dataset(dataset_path, is_local=False)
    new_dataset = convert_to_responses_api_format(dataset)
    write_to_jsonl(new_dataset, os.path.join(folder,output_file_name+"_responses_api.jsonl"))


'''


ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/offline_search/data/MCQA_syn_gpqa_1_2_difficulty_filtered_responses_api.jsonl \
    +output_jsonl_fpath=resources_servers/offline_search/results/MCQA_syn_gpqa_1_2_difficulty_filtered_trajectory_collection.jsonl \
    +limit=5

ng_upload_dataset_to_gitlab \
    +dataset_name=search_STEM_syn_gpqa_v1_2_difficulty_filtered \
    +version=0.0.1 \
    +input_jsonl_fpath=data/MCQA_syn_gpqa_1_2_difficulty_filtered_responses_api.jsonl
'''