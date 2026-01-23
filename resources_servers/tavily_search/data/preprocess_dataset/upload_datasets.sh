

cd /lustre/fsw/portfolios/llmservice/users/rgala/repos/nemo-gym-gitlab
source .venv/bin/activate

# ng_upload_dataset_to_gitlab \
#     +dataset_name=tavily_search \
#     +version=0.0.1 \
#     +input_jsonl_fpath=resources_servers/tavily_search/data/sft_samples/sft_samples_train.jsonl

# ng_upload_dataset_to_gitlab \
#     +dataset_name=tavily_search \
#     +version=0.0.1 \
#     +input_jsonl_fpath=resources_servers/tavily_search/data/sft_samples/sft_samples_validation.jsonl

# ng_upload_dataset_to_gitlab \
#     +dataset_name=tavily_search \
#     +version=0.0.1 \
#     +input_jsonl_fpath=resources_servers/tavily_search/data/benchmark/browsecomp/browsecomp_test_set.jsonl

# ng_upload_dataset_to_gitlab \
#     +dataset_name=tavily_search \
#     +version=0.0.1 \
#     +input_jsonl_fpath=resources_servers/tavily_search/data/benchmark/simple_qa/simple_qa_test_set.jsonl

ng_upload_dataset_to_gitlab \
    +dataset_name=tavily_search \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/tavily_search/data/example.jsonl